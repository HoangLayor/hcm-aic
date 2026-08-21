import os
import gc
import json
import logging
import subprocess
import re
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.config import config

logger = logging.getLogger("TranscriptExtractor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class TranscriptExtractor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipeline_device = 0 if torch.cuda.is_available() else -1
        self.model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.sample_rate = 16000
        
        logger.info(f"Loading PhoASR ({config.ASR_MODEL_ID})...")
        try:
            self.processor = AutoProcessor.from_pretrained(config.ASR_MODEL_ID)
            self.processor.feature_extractor.return_attention_mask = True
            self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                config.ASR_MODEL_ID,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            ).to(self.device)
            self.asr_model.eval()
            self.asr_model.generation_config.forced_decoder_ids = None

            self.asr_pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.asr_model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=30,
                return_timestamps="word",
                generate_kwargs={"language": "vi", "task": "transcribe", "return_legacy_cache": True},
                torch_dtype=self.model_dtype,
                device=self.pipeline_device,
            )
            logger.info("PhoASR loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load PhoASR: {e}")
            raise

        logger.info(f"Loading Pyannote Diarization ({config.DIARIZATION_MODEL_ID})...")
        try:
            os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
            from pyannote.audio import Pipeline as DiarizationPipeline
            if config.HF_TOKEN:
                self.diarization_pipe = DiarizationPipeline.from_pretrained(
                    config.DIARIZATION_MODEL_ID,
                    use_auth_token=config.HF_TOKEN
                )
                if torch.cuda.is_available():
                    self.diarization_pipe.to(torch.device("cuda"))
                logger.info("Pyannote loaded successfully.")
            else:
                self.diarization_pipe = None
                logger.warning("No HF_TOKEN provided. Diarization will be disabled.")
        except Exception as e:
            logger.error(f"Failed to load Pyannote: {e}")
            self.diarization_pipe = None

    def free_memory(self):
        """Explicitly clear ASR and Diarization models from VRAM."""
        if hasattr(self, 'asr_pipe'):
            del self.asr_pipe
        if hasattr(self, 'asr_model'):
            del self.asr_model
        if hasattr(self, 'processor'):
            del self.processor
        if hasattr(self, 'diarization_pipe'):
            del self.diarization_pipe
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Transcript models memory freed.")

    def extract_audio(self, video_path: Path) -> np.ndarray:
        """Extracts audio to numpy array using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_f32le", "-ar", str(self.sample_rate), "-ac", "1",
            "-f", "f32le", "-"
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            logger.error(f"FFmpeg audio extraction failed: {process.stderr.decode('utf-8', errors='ignore')}")
            return np.array([])
        
        audio = np.frombuffer(process.stdout, dtype=np.float32)
        return audio

    def run_diarization(self, audio_data: np.ndarray) -> list:
        if not self.diarization_pipe or len(audio_data) == 0:
            return []
            
        # Pyannote expects a specific input format
        tensor = torch.from_numpy(audio_data).unsqueeze(0)
        try:
            diarization = self.diarization_pipe(
                {"waveform": tensor, "sample_rate": self.sample_rate},
                min_speakers=config.DIARIZATION_MIN_SPEAKERS,
                max_speakers=config.DIARIZATION_MAX_SPEAKERS
            )
            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(speaker)))
            return turns
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []

    # =========================================================================
    # POST-PROCESSING METHODS (SYNCED FROM RAW SCRIPT)
    # =========================================================================

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        return re.sub(r"\s+([,.;:!?])", r"\1", text)

    @staticmethod
    def round_sec(value: float) -> float:
        return round(float(value), 3)

    def score_sentence_confidence(self, audio: np.ndarray, text: str, start_sec: float, end_sec: float):
        """Teacher-forced token log-probability of PhoASR for a single sentence."""
        padded_start = max(0.0, float(start_sec) - config.CONFIDENCE_AUDIO_PAD_SEC)
        padded_end = min(len(audio) / self.sample_rate, float(end_sec) + config.CONFIDENCE_AUDIO_PAD_SEC)
        start_sample = int(round(padded_start * self.sample_rate))
        end_sample = int(round(padded_end * self.sample_rate))
        sentence_audio = audio[start_sample:end_sample]
        
        if sentence_audio.size == 0 or not text.strip():
            return None

        features = self.processor.feature_extractor(
            sentence_audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features.to(self.device, dtype=self.model_dtype)
        attention_mask = features.attention_mask.to(self.device)

        prompt_pairs = self.processor.get_decoder_prompt_ids(
            language="vi", task="transcribe", no_timestamps=True
        )
        prompt_ids = [int(token_id) for _, token_id in prompt_pairs if token_id is not None]
        text_ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            return None
            
        label_ids = prompt_ids + text_ids + [self.processor.tokenizer.eos_token_id]
        labels = torch.tensor([label_ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            outputs = self.asr_model(
                input_features=input_features,
                attention_mask=attention_mask,
                labels=labels,
            )
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            chosen = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]

        text_scores = chosen[len(prompt_ids):len(prompt_ids) + len(text_ids)]
        if text_scores.numel() == 0:
            return None
            
        avg_logprob = float(text_scores.mean().item())
        return {
            "avg_logprob": round(avg_logprob, 4),
            "confidence": round(float(np.exp(max(-20.0, avg_logprob))), 4),
        }

    def split_words_into_sentences(self, words: list) -> list:
        sentences = []
        current = []

        def flush():
            if not current:
                return
            sentence = {
                "start_sec": self.round_sec(current[0]["start_sec"]),
                "end_sec": self.round_sec(current[-1]["end_sec"]),
                "text": self.clean_text("".join(x.get("text", "") for x in current)),
                "word_count": len(current),
                "words": current,
            }
            speakers = [x.get("speaker") for x in current if x.get("speaker")]
            if speakers:
                sentence["speaker"] = max(set(speakers), key=speakers.count)
            sentences.append(sentence)
            current.clear()

        for word in words:
            if current:
                gap = float(word["start_sec"]) - float(current[-1]["end_sec"])
                previous_speaker = current[-1].get("speaker")
                next_speaker = word.get("speaker")
                if gap >= config.SENTENCE_PAUSE_SEC or (
                    previous_speaker and next_speaker and previous_speaker != next_speaker
                ):
                    flush()
            current.append(dict(word))
            stripped = word.get("text", "").strip().rstrip('"”\')]}')
            duration = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
            if stripped.endswith((".", "!", "?", "…")) or duration >= config.MAX_SENTENCE_SEC:
                flush()
        flush()
        return [x for x in sentences if x["text"]]

    def classify_transcript_quality(self, confidence: float) -> str:
        if confidence is None:
            return "REVIEW"
        if confidence < config.QUALITY_NCR_MAX_CONFIDENCE:
            return "NCR"
        if confidence >= config.QUALITY_RELIABLE_MIN_CONFIDENCE:
            return "RELIABLE"
        return "REVIEW"

    def build_scored_sentences(self, audio: np.ndarray, words: list) -> list:
        sentences = self.split_words_into_sentences(words)
        for sentence in sentences:
            metrics = None
            confidence_error = None
            try:
                metrics = self.score_sentence_confidence(
                    audio, sentence["text"], sentence["start_sec"], sentence["end_sec"]
                )
            except Exception as exc:
                confidence_error = f"{type(exc).__name__}: {exc}"

            if metrics is not None:
                sentence.update(metrics)
            else:
                sentence.update({"avg_logprob": None, "confidence": None})
                
            sentence["quality"] = self.classify_transcript_quality(sentence.get("confidence"))
            if confidence_error:
                sentence["confidence_error"] = confidence_error
        return sentences

    def sentence_payload(self, sentence: dict) -> dict:
        confidence = sentence.get("confidence")
        return {
            "start": self.round_sec(sentence["start_sec"]),
            "end": self.round_sec(sentence["end_sec"]),
            "text": sentence["text"],
            "confidence": None if confidence is None else round(float(confidence), 3),
            "quality": sentence["quality"],
        }

    def group_sentences_into_turns(self, sentences: list) -> list:
        turns = []
        for sentence in sorted(sentences, key=lambda x: (x["start_sec"], x["end_sec"])):
            speaker = sentence.get("speaker", "UNKNOWN")
            payload = self.sentence_payload(sentence)
            can_merge = (
                turns
                and turns[-1]["speaker"] == speaker
                and payload["start"] - turns[-1]["end"] <= config.TURN_MERGE_GAP_SEC
            )
            if can_merge:
                turns[-1]["end"] = payload["end"]
                turns[-1]["sentences"].append(payload)
            else:
                turns.append({
                    "speaker": speaker,
                    "start": payload["start"],
                    "end": payload["end"],
                    "sentences": [payload],
                })
        return turns

    def flatten_transcript(self, turns: list) -> str:
        return self.clean_text(" ".join(
            sentence.get("text", "")
            for turn in turns
            for sentence in turn.get("sentences", [])
        ))

    def assign_speaker(self, start: float, end: float, speaker_turns: list) -> str:
        overlaps = [
            (max(0.0, min(end, turn_end) - max(start, turn_start)), speaker)
            for turn_start, turn_end, speaker in speaker_turns
        ]
        best_overlap, best_speaker = max(overlaps, default=(0.0, "UNKNOWN"))
        if best_overlap > 0:
            return best_speaker
        if not speaker_turns:
            return "UNKNOWN"
        midpoint = (float(start) + float(end)) / 2
        return min(
            speaker_turns,
            key=lambda x: 0.0 if x[0] <= midpoint <= x[1]
            else min(abs(midpoint - x[0]), abs(midpoint - x[1])),
        )[2]

    # =========================================================================

    def transcribe_words(self, audio_data: np.ndarray) -> list:
        if len(audio_data) == 0:
            return []
            
        try:
            result = self.asr_pipe({"array": audio_data, "sampling_rate": self.sample_rate})
            text = self.clean_text(result.get("text", ""))
            
            region_duration = len(audio_data) / self.sample_rate
            words = []
            
            for chunk in result.get("chunks", []):
                timestamp = chunk.get("timestamp") or (None, None)
                word_start = 0.0 if timestamp[0] is None else float(timestamp[0])
                word_end = region_duration if timestamp[1] is None else float(timestamp[1])
                words.append({
                    "text": chunk.get("text", ""),
                    "start_sec": self.round_sec(word_start),
                    "end_sec": self.round_sec(min(word_end, region_duration)),
                })

            if text and not words:
                words = [{
                    "text": text,
                    "start_sec": 0.0,
                    "end_sec": self.round_sec(region_duration),
                }]
                
            return words
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []

    def process_segment(self, segment_meta: dict, video_dir: Path):
        segment_id = segment_meta['segment_id']
        segment_file = Path(segment_meta['file_path'])
        
        logger.info(f"Processing Transcript for {segment_id}...")
        
        audio_data = self.extract_audio(segment_file)
        if len(audio_data) == 0:
            return
            
        # 1. Diarization
        speaker_turns = self.run_diarization(audio_data)
        
        # 2. Transcription (Word Level)
        words = self.transcribe_words(audio_data)
        if not words:
            return
            
        # 3. Assign Speaker to Words
        for word in words:
            word["speaker"] = self.assign_speaker(word["start_sec"], word["end_sec"], speaker_turns)
            
        # 4. Post-processing (Sentences, Confidence, Turns)
        sentences = self.build_scored_sentences(audio_data, words)
        turns = self.group_sentences_into_turns(sentences)
        full_text = self.flatten_transcript(turns)

        # 5. Save to JSON
        transcript_dir = video_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "segment_id": segment_id,
            "global_start_frame": segment_meta.get('global_start_frame', 0),
            "start_time_sec": segment_meta.get('start_time_sec', 0.0),
            "end_time_sec": segment_meta.get('end_time_sec', 0.0),
            "turns": turns,
            "full_text": full_text.strip()
        }
        
        with open(transcript_dir / f"{segment_id}_transcript.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"  -> Transcript extracted: {full_text[:50]}...")

    def process_video(self, video_id: str, force: bool = False):
        """Extracts transcripts for all segments of a video with checkpoint support."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found for video {video_id}.")
            return
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            segments = manifest.get('segments', [])
            logger.info(f"Processing {len(segments)} segments for {video_id}...")
            
            transcript_dir = video_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            
            for seg_meta in segments:
                segment_id = seg_meta['segment_id']
                trans_file = transcript_dir / f"{segment_id}_transcript.json"
                
                # Checkpoint check: skip extraction if transcript file already exists
                if not force and trans_file.exists():
                    try:
                        with open(trans_file, 'r', encoding='utf-8') as f:
                            trans_data = json.load(f)
                        if trans_data.get("full_text"):
                            logger.info(f"  -> [Skip] Transcript for {segment_id} (already extracted: {trans_data['full_text'][:40]}...).")
                            continue
                    except Exception:
                        pass
                
                self.process_segment(seg_meta, video_dir)
                
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            raise

class TranscriptContext:
    def __enter__(self):
        self.extractor = TranscriptExtractor()
        return self.extractor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.extractor.free_memory()
