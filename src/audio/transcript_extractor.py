import os
import gc
import json
import logging
import subprocess
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
                generate_kwargs={"language": "vi", "task": "transcribe"},
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
                turns.append((turn.start, turn.end, speaker))
            return turns
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []

    def transcribe(self, audio_data: np.ndarray) -> list:
        if len(audio_data) == 0:
            return []
            
        try:
            result = self.asr_pipe(audio_data)
            return result.get("chunks", []) # Whisper chunks
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
        
        # 2. Transcription
        chunks = self.transcribe(audio_data)
        
        # 3. Align Speakers to Chunks
        def assign_speaker(start, end):
            if not speaker_turns:
                return "SPEAKER_00"
            overlaps = [
                (max(0.0, min(end, turn_end) - max(start, turn_start)), speaker)
                for turn_start, turn_end, speaker in speaker_turns
            ]
            best_overlap, best_speaker = max(overlaps, default=(0.0, "SPEAKER_00"))
            return best_speaker if best_overlap > 0 else "SPEAKER_00"
            
        transcript_data = []
        full_text = ""
        
        for chunk in chunks:
            timestamp = chunk.get("timestamp", (0.0, 0.0))
            # Handle possible None in timestamps
            start_t = timestamp[0] if timestamp[0] is not None else 0.0
            end_t = timestamp[1] if timestamp[1] is not None else start_t + 1.0
            
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            speaker = assign_speaker(start_t, end_t)
            
            transcript_data.append({
                "start": start_t,
                "end": end_t,
                "speaker": speaker,
                "text": text
            })
            full_text += f"{speaker}: {text}\n"

        # Save to JSON
        transcript_dir = video_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "segment_id": segment_id,
            "global_start_frame": segment_meta.get('global_start_frame', 0),
            "start_time_sec": segment_meta.get('start_time_sec', 0.0),
            "end_time_sec": segment_meta.get('end_time_sec', 0.0),
            "transcript": transcript_data,
            "full_text": full_text.strip()
        }
        
        out_file = transcript_dir / f"{segment_id}_transcript.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"  -> Transcript generated ({len(transcript_data)} utterances)")

    def process_video(self, video_id: str, force: bool = False):
        """Processes all segments of a video."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found for {video_id}")
            return
            
        # Optional: Skip logic if transcripts already exist
        transcript_dir = video_dir / "transcripts"
        if not force and transcript_dir.exists() and any(transcript_dir.iterdir()):
            logger.info(f"Transcripts already exist for {video_id}. Skipping. (Use --force to override)")
            return
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            segments = manifest.get('segments', [])
            logger.info(f"Generating transcripts for {len(segments)} segments in {video_id}...")
            
            for seg_meta in segments:
                self.process_segment(seg_meta, video_dir)
                
            logger.info(f"Successfully finished Transcripts for {video_id}.")
        except Exception as e:
            logger.error(f"Transcript extraction failed for {video_id}: {e}")

# Context manager to ensure model cleanup
class TranscriptContext:
    def __enter__(self):
        self.extractor = TranscriptExtractor()
        return self.extractor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.extractor.free_memory()
