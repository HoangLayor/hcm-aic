import os
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict

import torch
from src.config import config

logger = logging.getLogger("VadSplitter")
logger.setLevel(logging.INFO)
# Basic console handler
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class VadVideoSplitter:
    def __init__(self):
        self.sample_rate = 16000
        logger.info("Initializing Silero VAD Model...")
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.get_speech_timestamps = utils[0]
            self.read_audio = utils[2]
            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def extract_audio(self, video_path: Path, temp_audio_path: Path):
        """Extracts 16kHz mono audio from video."""
        logger.info(f"Extracting audio from {video_path}...")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", str(self.sample_rate), "-ac", "1",
            str(temp_audio_path)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio extraction completed.")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode('utf-8', errors='ignore')}")
            raise

    def get_split_points(self, audio_path: Path) -> List[float]:
        """Runs VAD and finds mid-point silences."""
        logger.info("Running VAD analysis...")
        try:
            wav = self.read_audio(str(audio_path), sampling_rate=self.sample_rate)
            speech_timestamps = self.get_speech_timestamps(
                wav, self.model, sampling_rate=self.sample_rate,
                min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
                threshold=config.VAD_THRESHOLD,
                min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS
            )
            
            split_points = []
            if len(speech_timestamps) > 1:
                for i in range(len(speech_timestamps) - 1):
                    curr_end = speech_timestamps[i]['end'] / self.sample_rate
                    next_start = speech_timestamps[i+1]['start'] / self.sample_rate
                    # Mid-point split
                    split_points.append(round(curr_end + (next_start - curr_end) / 2.0, 3))
            
            logger.info(f"Found {len(split_points)} split points based on silence.")
            return split_points
        except Exception as e:
            logger.error(f"VAD analysis failed: {e}")
            raise

    def split_video(self, video_path: Path, split_points: List[float], output_dir: Path) -> List[Dict]:
        """Splits video at given points using precise re-encoding to avoid frame drift."""
        logger.info(f"Splitting video into {len(split_points) + 1} segments...")
        segments_meta = []
        points = [0.0] + split_points
        video_id = video_path.stem
        
        for i in range(len(points)):
            start_time = points[i]
            end_time = points[i+1] if i < len(points) - 1 else None
            duration = (end_time - start_time) if end_time else None
            
            # Fallback: if duration > MAX_SEGMENT_DURATION, we just log a warning for now (keep it simple)
            if duration and duration > config.VAD_MAX_SEGMENT_DURATION_SEC:
                logger.warning(f"Segment {i+1} duration ({duration}s) exceeds max ({config.VAD_MAX_SEGMENT_DURATION_SEC}s).")

            output_file = output_dir / f"{video_id}_seg{i+1:03d}.mp4"
            
            # Using precise cutting: re-encode with ultrafast to guarantee accurate timestamps and no frame drift
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ss", str(start_time)]
            if duration:
                cmd.extend(["-t", str(duration)])
            # Re-encode video very fast, copy audio
            cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", str(output_file)])
            
            logger.info(f"Cutting segment {i+1}: start={start_time}s, duration={duration}s")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Calculate precise global start frame assuming fixed TARGET_FPS
                global_start_frame = int(start_time * config.TARGET_FPS)
                
                segments_meta.append({
                    "segment_id": f"{video_id}_seg{i+1:03d}",
                    "file_path": str(output_file),
                    "start_time_sec": start_time,
                    "end_time_sec": end_time if end_time else -1.0,
                    "duration_sec": duration if duration else -1.0,
                    "global_start_frame": global_start_frame
                })
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to cut segment {i+1}: {e.stderr.decode('utf-8', errors='ignore')}")
                raise

        return segments_meta

    def process_video(self, video_path: str):
        """End-to-end processing of a single video."""
        start_t = time.time()
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
            
        output_dir = Path(config.OUTPUT_DIR) / video_path.stem / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_audio = output_dir / "temp_audio.wav"
        
        try:
            self.extract_audio(video_path, temp_audio)
            split_points = self.get_split_points(temp_audio)
            segments_meta = self.split_video(video_path, split_points, output_dir)
            
            # Save manifest
            manifest_path = output_dir.parent / "manifest_vad.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({"video_id": video_path.stem, "segments": segments_meta}, f, indent=2)
            
            logger.info(f"Successfully processed {video_path.name} in {time.time() - start_t:.2f}s")
            return manifest_path
        except Exception as e:
            logger.error(f"Failed to process video {video_path.name}: {e}")
            return None
        finally:
            if temp_audio.exists():
                temp_audio.unlink()
                logger.info("Cleaned up temp audio.")

# For quick testing
if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    splitter = VadVideoSplitter()
    # Replace with a real video path to test
    # splitter.process_video("raw/vid_sample_001.mp4")
