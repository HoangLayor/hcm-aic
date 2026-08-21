import gc
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.config import config


logger = logging.getLogger("TranscriptExtractor")


class TranscriptExtractor:
    """PhoASR + Silero VAD + pyannote pipeline ported from transcript.ipynb."""

    def __init__(self):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg was not found in PATH.")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipeline_device = 0 if torch.cuda.is_available() else -1
        self.model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.sample_rate = config.ASR_SAMPLE_RATE

        logger.info("Loading PhoASR (%s)...", config.ASR_MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(config.ASR_MODEL_ID)
        # Whisper has pad_token == eos_token, so the feature extractor must emit a mask.
        self.processor.feature_extractor.return_attention_mask = True
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.ASR_MODEL_ID,
            torch_dtype=self.model_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            # Keep this identical to the notebook. SDPA can change PhoASR behaviour.
            attn_implementation="eager",
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
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
            },
            torch_dtype=self.model_dtype,
            device=self.pipeline_device,
        )

        # Use ONNX VAD on CPU first, as in the notebook, and preserve TorchScript fallback.
        try:
            self.vad_model = load_silero_vad(onnx=True)
            vad_backend = "ONNX Runtime (CPU)"
        except Exception as exc:
            logger.warning("Could not initialize ONNX VAD (%s); using TorchScript.", exc)
            self.vad_model = load_silero_vad(onnx=False)
            vad_backend = "TorchScript (CPU)"
        logger.info("PhoASR and Silero VAD loaded [%s].", vad_backend)

        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        from pyannote.audio import Pipeline as DiarizationPipeline

        hf_token = self._resolve_hf_token()
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is required for pyannote speaker-diarization-community-1. "
                "Set AIC_HF_TOKEN/HF_TOKEN or add HF_TOKEN to Kaggle Secrets."
            )

        logger.info("Loading pyannote diarization (%s)...", config.DIARIZATION_MODEL_ID)
        self.diarization_pipe = DiarizationPipeline.from_pretrained(
            config.DIARIZATION_MODEL_ID,
            token=hf_token,
        )
        if self.diarization_pipe is None:
            raise RuntimeError(
                "Could not load pyannote diarization. Check HF_TOKEN and accept the model terms."
            )
        self.diarization_pipe.to(torch.device(self.device))
        logger.info("Pyannote diarization loaded.")

    @staticmethod
    def _resolve_hf_token():
        token = config.HF_TOKEN or os.environ.get("HF_TOKEN")
        if token:
            return token
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            return None

    def free_memory(self):
        """Explicitly clear ASR, VAD and diarization models from memory."""
        for name in ("asr_pipe", "asr_model", "processor", "vad_model", "diarization_pipe"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Transcript models memory freed.")

    def extract_audio(self, video_path: Path) -> np.ndarray:
        """Decode a video directly to mono float32 16 kHz without a temporary WAV."""
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(self.sample_rate),
            "-f", "f32le", "pipe:1",
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            error = process.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg failed for {video_path}: {error}")
        audio = np.frombuffer(process.stdout, dtype=np.float32).copy()
        if audio.size == 0:
            raise ValueError(f"Video has no audio: {video_path}")
        return audio

    @staticmethod
    def collect_speaker_turns(annotation) -> list:
        if hasattr(annotation, "itertracks"):
            return [
                (float(turn.start), float(turn.end), str(speaker))
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
        return [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, speaker in annotation
        ]

    def run_diarization(self, audio_data: np.ndarray) -> list:
        if audio_data.size == 0:
            return []
        output = self.diarization_pipe(
            {
                "waveform": torch.from_numpy(audio_data).unsqueeze(0),
                "sample_rate": self.sample_rate,
            },
            min_speakers=config.DIARIZATION_MIN_SPEAKERS,
            max_speakers=config.DIARIZATION_MAX_SPEAKERS,
        )
        # Community-1 exposes non-overlapping turns, which are preferred by the notebook.
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = output.speaker_diarization
        return self.collect_speaker_turns(annotation)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        return re.sub(r"\s+([,.;:!?])", r"\1", text)

    @staticmethod
    def round_sec(value: float) -> float:
        return round(float(value), 3)

    @staticmethod
    def atomic_write_text(path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)

    @classmethod
    def atomic_write_json(cls, path: Path, data: dict):
        cls.atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))

    def score_sentence_confidence(
        self, audio: np.ndarray, text: str, start_sec: float, end_sec: float
    ):
        """Teacher-forced token log-probability for one PhoASR sentence."""
        padded_start = max(0.0, float(start_sec) - config.CONFIDENCE_AUDIO_PAD_SEC)
        padded_end = min(
            len(audio) / self.sample_rate,
            float(end_sec) + config.CONFIDENCE_AUDIO_PAD_SEC,
        )
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
            nonlocal current
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
            current = []

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
            stripped = word.get("text", "").strip().rstrip('"”’)]}')
            duration = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
            if stripped.endswith((".", "!", "?", "…")) or duration >= config.MAX_SENTENCE_SEC:
                flush()
        flush()
        return [sentence for sentence in sentences if sentence["text"]]

    @staticmethod
    def classify_transcript_quality(confidence: float) -> str:
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

            sentence.update(metrics or {"avg_logprob": None, "confidence": None})
            sentence["quality"] = self.classify_transcript_quality(sentence["confidence"])
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
            key=lambda turn: 0.0 if turn[0] <= midpoint <= turn[1]
            else min(abs(midpoint - turn[0]), abs(midpoint - turn[1])),
        )[2]

    def transcribe_speech_region(
        self, audio_data: np.ndarray, start_sample: int, end_sample: int
    ):
        region = audio_data[start_sample:end_sample]
        region_duration = len(region) / self.sample_rate
        if region_duration < config.ASR_MIN_SPEECH_SECONDS:
            return None

        result = self.asr_pipe({"array": region, "sampling_rate": self.sample_rate})
        text = self.clean_text(result.get("text", ""))
        local_start = start_sample / self.sample_rate
        local_end = end_sample / self.sample_rate

        words = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            word_start = 0.0 if timestamp[0] is None else float(timestamp[0])
            word_end = region_duration if timestamp[1] is None else float(timestamp[1])
            words.append({
                "text": chunk.get("text", ""),
                "start_sec": self.round_sec(local_start + word_start),
                "end_sec": self.round_sec(min(local_start + word_end, local_end)),
            })

        if text and not words:
            words = [{
                "text": text,
                "start_sec": self.round_sec(local_start),
                "end_sec": self.round_sec(local_end),
            }]
        return {"text": text, "words": words}

    def process_segment(self, segment_meta: dict, video_dir: Path) -> dict:
        segment_id = segment_meta["segment_id"]
        segment_file = Path(segment_meta["file_path"])
        logger.info("Processing transcript for %s...", segment_id)

        audio_data = self.extract_audio(segment_file)
        vad_regions = get_speech_timestamps(
            torch.from_numpy(audio_data),
            self.vad_model,
            sampling_rate=self.sample_rate,
            threshold=config.VAD_THRESHOLD,
            min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
            max_speech_duration_s=config.ASR_MAX_SPEECH_SECONDS,
            return_seconds=False,
        )

        speaker_turns = self.run_diarization(audio_data) if vad_regions else []
        words = []
        for region in vad_regions:
            utterance = self.transcribe_speech_region(
                audio_data, int(region["start"]), int(region["end"])
            )
            if utterance is None or not utterance["text"]:
                continue
            for word in utterance["words"]:
                word["speaker"] = self.assign_speaker(
                    word["start_sec"], word["end_sec"], speaker_turns
                )
            words.extend(utterance["words"])

        sentences = self.build_scored_sentences(audio_data, words)
        turns = self.group_sentences_into_turns(sentences)
        full_text = self.flatten_transcript(turns)
        start = self.round_sec(segment_meta.get("start_time_sec", 0.0))
        metadata_end = float(segment_meta.get("end_time_sec", -1.0))
        end = self.round_sec(
            metadata_end if metadata_end >= start else start + len(audio_data) / self.sample_rate
        )
        output_data = {
            "video_id": video_dir.name,
            "segment_id": segment_id,
            "start": start,
            "end": end,
            # Keep legacy fields consumed by the embedding stage.
            "global_start_frame": segment_meta.get("global_start_frame", 0),
            "start_time_sec": start,
            "end_time_sec": end,
            "turns": turns,
            "full_text": full_text,
        }

        transcript_dir = video_dir / "transcripts"
        transcript_path = transcript_dir / f"{segment_id}_transcript.json"
        self.atomic_write_json(transcript_path, output_data)
        error_path = transcript_dir / f"{segment_id}_transcript.error.json"
        if error_path.exists():
            error_path.unlink()
        logger.info("Transcript extracted: %s", full_text[:50])
        return output_data

    @staticmethod
    def _is_valid_checkpoint(data: dict) -> bool:
        required_keys = {"video_id", "segment_id", "start", "end", "turns", "full_text"}
        return required_keys.issubset(data)

    def process_video(self, video_id: str, force: bool = False) -> dict:
        """Extract transcripts for every VAD segment with resumable checkpoints."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for video {video_id}: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        segments = manifest.get("segments", [])
        if not segments:
            raise ValueError(f"Manifest contains no segments: {manifest_path}")
        logger.info("Processing %d segments for %s...", len(segments), video_id)

        transcript_dir = video_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        failures = []

        for segment_meta in segments:
            segment_id = segment_meta["segment_id"]
            transcript_path = transcript_dir / f"{segment_id}_transcript.json"
            if not force and transcript_path.exists():
                try:
                    checkpoint = json.loads(transcript_path.read_text(encoding="utf-8"))
                    if self._is_valid_checkpoint(checkpoint):
                        stats["skipped"] += 1
                        logger.info("[Skip] Transcript for %s already exists.", segment_id)
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

            try:
                self.process_segment(segment_meta, video_dir)
                stats["processed"] += 1
            except Exception as exc:
                stats["errors"] += 1
                failure = {
                    "segment_id": segment_id,
                    "file_path": str(segment_meta.get("file_path", "")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                self.atomic_write_json(
                    transcript_dir / f"{segment_id}_transcript.error.json", failure
                )
                logger.exception("Transcript failed for %s", segment_id)

        logger.info("Transcript summary for %s: %s", video_id, stats)
        if failures:
            failed_ids = ", ".join(item["segment_id"] for item in failures)
            raise RuntimeError(
                f"Transcript extraction failed for {len(failures)} segment(s): {failed_ids}"
            )
        return stats


class TranscriptContext:
    def __enter__(self):
        self.extractor = TranscriptExtractor()
        return self.extractor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.extractor.free_memory()
