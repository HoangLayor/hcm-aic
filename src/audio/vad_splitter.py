import os
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

from src.config import config

logger = logging.getLogger("VadSplitter")
logger.setLevel(logging.INFO)
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

    def get_video_duration(self, video_path: Path) -> float:
        """Gets video duration accurately using cv2 or ffprobe."""
        if cv2 is not None:
            try:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    if frame_count > 0 and fps > 0:
                        return float(frame_count / fps)
            except Exception:
                pass

        # Fallback to ffprobe
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except Exception:
            pass

        return -1.0

    def extract_audio(self, video_path: Path, temp_audio_path: Path):
        """Extracts 16kHz mono audio from video."""
        logger.info(f"Extracting audio from {video_path.name}...")
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

            logger.info(f"Found {len(split_points)} split points based on silence ({len(split_points) + 1} segments).")
            return split_points
        except Exception as e:
            logger.error(f"VAD analysis failed: {e}")
            raise

    def _cut_single_segment(self, args: dict) -> dict:
        """Helper to cut a single segment with fast seeking and fixed CFR."""
        video_path = args["video_path"]
        output_file = args["output_file"]
        start_time = args["start_time"]
        duration = args["duration"]
        segment_id = args["segment_id"]
        global_start_frame = args["global_start_frame"]

        # Fast Input Seeking: -ss before -i + ultrafast re-encoding + strict 30 FPS CFR
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-i", str(video_path),
        ]
        if duration is not None and duration > 0:
            cmd.extend(["-t", f"{duration:.3f}"])
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-r", str(config.TARGET_FPS),
            "-vsync", "cfr",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            str(output_file)
        ])

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {
            "segment_id": segment_id,
            "file_path": str(output_file),
            "start_time_sec": start_time,
            "end_time_sec": (start_time + duration) if duration else -1.0,
            "duration_sec": duration if duration else -1.0,
            "global_start_frame": global_start_frame
        }

    def split_video(self, video_path: Path, split_points: List[float], output_dir: Path, total_duration: float = -1.0) -> List[Dict]:
        """Splits video in parallel using fast seeking and multi-threading."""
        points = [0.0] + split_points
        video_id = video_path.stem
        cut_tasks = []

        for i in range(len(points)):
            start_time = points[i]
            end_time = points[i+1] if i < len(points) - 1 else (total_duration if total_duration > 0 else None)
            duration = (end_time - start_time) if end_time else None
            output_file = output_dir / f"{video_id}_seg{i+1:03d}.mp4"
            global_start_frame = int(round(start_time * config.TARGET_FPS))

            cut_tasks.append({
                "video_path": video_path,
                "output_file": output_file,
                "start_time": start_time,
                "duration": duration,
                "segment_id": f"{video_id}_seg{i+1:03d}",
                "global_start_frame": global_start_frame,
                "index": i
            })

        max_workers = min(4, os.cpu_count() or 2)
        logger.info(f"Cutting {len(cut_tasks)} segments in parallel (using {max_workers} CPU workers)...")
        results = [None] * len(cut_tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._cut_single_segment, task): task["index"]
                for task in cut_tasks
            }
            for future in as_completed(future_to_task):
                idx = future_to_task[future]
                try:
                    res = future.result()
                    results[idx] = res
                except Exception as e:
                    logger.error(f"Failed to cut segment index {idx+1}: {e}")
                    raise

        return results

    def process_video(self, video_path: str, force: bool = False):
        """End-to-end processing of a single video with checkpoint support."""
        start_t = time.time()
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
            
        output_dir = Path(config.OUTPUT_DIR) / video_path.stem / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir.parent / "manifest_vad.json"

        # Checkpoint check: skip if manifest exists and all segment files are intact
        if not force and manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    cached_manifest = json.load(f)
                cached_segments = cached_manifest.get("segments", [])
                if cached_segments and all(Path(s["file_path"]).exists() and Path(s["file_path"]).stat().st_size > 0 for s in cached_segments):
                    logger.info(f"  -> [Skip] VAD Splitting for {video_path.name} (manifest and {len(cached_segments)} segments already exist).")
                    return manifest_path
            except Exception:
                pass

        temp_audio = output_dir / "temp_audio.wav"
        try:
            total_duration = self.get_video_duration(video_path)
            self.extract_audio(video_path, temp_audio)
            split_points = self.get_split_points(temp_audio)
            segments_meta = self.split_video(video_path, split_points, output_dir, total_duration=total_duration)
            
            # Save manifest
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
    # splitter.process_video("data/vid_sample_001.mp4")
