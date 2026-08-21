import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


# Keep this unit test runnable in a lightweight development environment. Kaggle
# installs the real package before running the pipeline.
if importlib.util.find_spec("silero_vad") is None:
    silero_stub = types.ModuleType("silero_vad")
    silero_stub.get_speech_timestamps = lambda *args, **kwargs: []
    silero_stub.load_silero_vad = lambda *args, **kwargs: None
    sys.modules["silero_vad"] = silero_stub

from src.audio import transcript_extractor as transcript_module
from src.audio.transcript_extractor import TranscriptExtractor
from src.config import config


class TranscriptNotebookParityTests(unittest.TestCase):
    def test_asr_vad_defaults_match_notebook_without_changing_stage_one(self):
        self.assertEqual(config.ASR_VAD_THRESHOLD, 0.50)
        self.assertEqual(config.ASR_VAD_MIN_SPEECH_DURATION_MS, 250)
        self.assertEqual(config.ASR_VAD_MIN_SILENCE_MS, 350)
        self.assertEqual(config.ASR_VAD_SPEECH_PAD_MS, 300)
        self.assertEqual(config.ASR_MAX_SPEECH_SECONDS, 28.0)

        # Stage 1 keeps its existing, independently configured split threshold.
        self.assertEqual(config.VAD_MIN_SILENCE_MS, 1000)

    def test_process_segment_scores_each_vad_utterance_separately(self):
        extractor = TranscriptExtractor.__new__(TranscriptExtractor)
        extractor.sample_rate = 100
        extractor.vad_model = object()
        extractor.vad_kwargs = {
            "threshold": 0.50,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 350,
            "speech_pad_ms": 300,
            "max_speech_duration_s": 28.0,
        }
        extractor.extract_audio = Mock(return_value=np.zeros(200, dtype=np.float32))
        extractor.run_diarization = Mock(return_value=[])
        extractor.assign_speaker = Mock(return_value="UNKNOWN")

        def transcribe_region(_audio, start_sample, end_sample):
            return {
                "start_sec": start_sample / extractor.sample_rate,
                "end_sec": end_sample / extractor.sample_rate,
                "text": f" vùng-{start_sample}",
                "words": [{
                    "text": f" vùng-{start_sample}",
                    "start_sec": start_sample / extractor.sample_rate,
                    "end_sec": end_sample / extractor.sample_rate,
                }],
            }

        extractor.transcribe_speech_region = Mock(side_effect=transcribe_region)

        def score_words(_audio, words):
            word = words[0]
            return [{
                "start_sec": word["start_sec"],
                "end_sec": word["end_sec"],
                "text": word["text"].strip(),
                "speaker": word["speaker"],
                "quality": "RELIABLE",
                "confidence": 0.9,
            }]

        extractor.build_scored_sentences = Mock(side_effect=score_words)

        vad_regions = [{"start": 0, "end": 40}, {"start": 60, "end": 100}]
        segment_meta = {
            "segment_id": "video_seg001",
            "file_path": "unused.mp4",
            "global_start_frame": 0,
            "start_time_sec": 0.0,
            "end_time_sec": 2.0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir) / "video"
            with patch.object(
                transcript_module, "get_speech_timestamps", return_value=vad_regions
            ) as vad_mock:
                output = extractor.process_segment(segment_meta, video_dir)

        self.assertEqual(extractor.build_scored_sentences.call_count, 2)
        self.assertEqual(
            [call.args[1][0]["text"] for call in extractor.build_scored_sentences.call_args_list],
            [" vùng-0", " vùng-60"],
        )
        self.assertEqual(output["full_text"], "vùng-0 vùng-60")
        self.assertEqual(
            set(output),
            {
                "video_id",
                "segment_id",
                "start",
                "end",
                "global_start_frame",
                "start_time_sec",
                "end_time_sec",
                "turns",
                "full_text",
            },
        )
        self.assertEqual(vad_mock.call_args.kwargs["min_silence_duration_ms"], 350)


if __name__ == "__main__":
    unittest.main()
