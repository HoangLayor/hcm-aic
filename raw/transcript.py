# Cài đúng phiên bản mà model card PhoASR khuyến nghị và VAD ổn định.
%pip install -q --upgrade "transformers==4.48.0" "accelerate>=0.26,<2" "silero-vad[onnx-cpu]==6.2.1" "onnxruntime==1.27.0" "pyannote.audio==4.0.7" "soundfile>=0.12"


import csv
import gc
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# ========================= CẤU HÌNH =========================
MODEL_ID = "Qualcomm-AI-Research/PhoASR-whisper-small"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16_000

# Để None: tự tìm folder chứa L21_Vxxx/segment_XXX.mp4 trong /kaggle/input.
# Nếu Kaggle có nhiều dataset, đặt đường dẫn cụ thể, ví dụ:
# VIDEO_ROOT = Path("/kaggle/input/htv9-segments/output_segments")
VIDEO_ROOT = None
OUTPUT_ROOT = Path("/kaggle/working/transcriptions")

# False giúp tiếp tục từ checkpoint; True sẽ chạy lại mọi clip.
OVERWRITE = False
MAX_FILES = None          # Dùng 3 để test nhanh, None để chạy toàn bộ.

# Tham số VAD phù hợp thoại truyền hình có nhạc/tạp âm nền.
VAD_THRESHOLD = 0.50
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 350
SPEECH_PAD_MS = 300       # Thêm ngữ cảnh để hạn chế cắt giữa âm tiết/từ.
MAX_SPEECH_SECONDS = 28.0  # Whisper dùng cửa sổ 30 giây.
MIN_ASR_SECONDS = 0.35

# Diarization: speaker được nhận diện cục bộ trong từng segment.
DIARIZATION_MIN_SPEAKERS = 1
DIARIZATION_MAX_SPEAKERS = 6
TURN_MERGE_GAP_SEC = 1.50

# Chất lượng transcript chỉ dựa trên confidence ASR, không đánh giá ngữ nghĩa.
# confidence >= 0.70: RELIABLE; 0.50 <= confidence < 0.70: REVIEW; < 0.50: NCR.
QUALITY_RELIABLE_MIN_CONFIDENCE = 0.70
QUALITY_NCR_MAX_CONFIDENCE = 0.50
CONFIDENCE_AUDIO_PAD_SEC = 0.15
SENTENCE_PAUSE_SEC = 0.85
MAX_SENTENCE_SEC = 18.0

assert shutil.which("ffmpeg"), "Kaggle runtime không tìm thấy ffmpeg."
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CẢNH BÁO: Không có GPU; notebook vẫn chạy nhưng sẽ rất chậm.")


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def discover_video_root(input_root=Path("/kaggle/input")):
    videos = sorted(input_root.rglob("segment_*.mp4"), key=natural_key)
    if not videos:
        # Fallback nếu tên clip không theo mẫu segment_XXX.mp4.
        videos = sorted(input_root.rglob("*.mp4"), key=natural_key)
    if not videos:
        raise FileNotFoundError("Không tìm thấy MP4 nào trong /kaggle/input.")

    common_parent = Path(os.path.commonpath([str(p.parent) for p in videos]))
    # Khi dataset chỉ chứa một folder L21_Vxxx, lấy folder cha để vẫn giữ L21_Vxxx.
    if re.fullmatch(r"L\d+_V\d+", common_parent.name, flags=re.IGNORECASE):
        common_parent = common_parent.parent
    return common_parent


if VIDEO_ROOT is None:
    VIDEO_ROOT = discover_video_root()
else:
    VIDEO_ROOT = Path(VIDEO_ROOT)

video_files = sorted(VIDEO_ROOT.rglob("segment_*.mp4"), key=natural_key)
if not video_files:
    video_files = sorted(VIDEO_ROOT.rglob("*.mp4"), key=natural_key)
if MAX_FILES is not None:
    video_files = video_files[:MAX_FILES]

assert video_files, f"Không có MP4 trong {VIDEO_ROOT}"
print("VIDEO_ROOT :", VIDEO_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("Số clip    :", len(video_files))
print("Ví dụ      :", video_files[0].relative_to(VIDEO_ROOT))


from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from silero_vad import get_speech_timestamps, load_silero_vad

os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
from pyannote.audio import Pipeline as DiarizationPipeline

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1
model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Đang tải PhoASR...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
# Whisper có pad_token == eos_token, nên phải yêu cầu feature extractor tạo attention_mask.
processor.feature_extractor.return_attention_mask = True
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="eager",
).to(device_name)
asr_model.eval()
# Dùng language/task động bên dưới thay cho forced_decoder_ids cũ trong checkpoint.
asr_model.generation_config.forced_decoder_ids = None

asr_pipe = pipeline(
    task="automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    return_timestamps="word",
    generate_kwargs={
        "language": "vi",
        "task": "transcribe",
        "return_legacy_cache": True,
    },
    torch_dtype=model_dtype,
    device=pipeline_device,
)

# ONNX VAD chạy rất nhanh trên CPU, để GPU dành cho ASR.
# Nếu ONNX Runtime không tương thích với image Kaggle, tự chuyển sang TorchScript.
try:
    vad_model = load_silero_vad(onnx=True)
    vad_backend = "ONNX Runtime (CPU)"
except Exception as exc:
    print(f"Cảnh báo: không khởi tạo được ONNX VAD ({type(exc).__name__}: {exc})")
    print("Đang chuyển sang Silero TorchScript...")
    vad_model = load_silero_vad(onnx=False)
    vad_backend = "TorchScript (CPU)"
print(f"Đã tải xong PhoASR + Silero VAD [{vad_backend}].")

# Community-1 cần Hugging Face token quyền Read trong Kaggle Secrets: HF_TOKEN.
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError(
            "Không đọc được HF_TOKEN. Hãy thêm token quyền Read vào Kaggle Secrets "
            "và chấp nhận điều kiện truy cập Community-1."
        ) from exc

print("Đang tải pyannote Community-1...")
diar_pipeline = DiarizationPipeline.from_pretrained(
    DIARIZATION_MODEL_ID, token=HF_TOKEN
)
if diar_pipeline is None:
    raise RuntimeError("Không tải được pyannote Community-1. Hãy kiểm tra HF_TOKEN.")
diar_pipeline.to(torch.device(device_name))
print("Đã tải xong diarization pipeline.")


_source_metadata_cache = {}


def decode_audio(video_path, sample_rate=SAMPLE_RATE):
    """Giải mã trực tiếp MP4 thành mono float32 16 kHz, không tạo WAV tạm."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        error = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg lỗi với {video_path}: {error}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError(f"Clip không có audio: {video_path}")
    return audio


def clean_text(text):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def round_sec(value):
    return round(float(value), 3)


def score_sentence_confidence(audio, text, start_sec, end_sec):
    """Teacher-forced token log-probability của PhoASR cho đúng một câu."""
    padded_start = max(0.0, float(start_sec) - CONFIDENCE_AUDIO_PAD_SEC)
    padded_end = min(len(audio) / SAMPLE_RATE, float(end_sec) + CONFIDENCE_AUDIO_PAD_SEC)
    start_sample = int(round(padded_start * SAMPLE_RATE))
    end_sample = int(round(padded_end * SAMPLE_RATE))
    sentence_audio = audio[start_sample:end_sample]
    if sentence_audio.size == 0 or not text.strip():
        return None

    features = processor.feature_extractor(
        sentence_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = features.input_features.to(device_name, dtype=model_dtype)
    attention_mask = features.attention_mask.to(device_name)

    # Nhãn gồm prompt tiếng Việt + task transcribe + token nội dung câu.
    prompt_pairs = processor.get_decoder_prompt_ids(
        language="vi", task="transcribe", no_timestamps=True
    )
    prompt_ids = [int(token_id) for _, token_id in prompt_pairs if token_id is not None]
    text_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    if not text_ids:
        return None
    label_ids = prompt_ids + text_ids + [processor.tokenizer.eos_token_id]
    labels = torch.tensor([label_ids], dtype=torch.long, device=device_name)

    with torch.inference_mode():
        outputs = asr_model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
        chosen = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]

    # Chỉ chấm token nội dung, không tính token language/task/eos.
    text_scores = chosen[len(prompt_ids):len(prompt_ids) + len(text_ids)]
    if text_scores.numel() == 0:
        return None
    avg_logprob = float(text_scores.mean().item())
    return {
        "avg_logprob": round(avg_logprob, 4),
        # Đây là geometric mean token probability, không phải xác suất đã calibration.
        "confidence": round(float(np.exp(max(-20.0, avg_logprob))), 4),
    }


def split_words_into_sentences(words):
    """Tách câu theo dấu câu, khoảng nghỉ, speaker change và độ dài tối đa."""
    sentences = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        sentence = {
            "start_sec": round_sec(current[0]["start_sec"]),
            "end_sec": round_sec(current[-1]["end_sec"]),
            "text": clean_text("".join(x.get("text", "") for x in current)),
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
            if gap >= SENTENCE_PAUSE_SEC or (
                previous_speaker and next_speaker and previous_speaker != next_speaker
            ):
                flush()
        current.append(dict(word))
        stripped = word.get("text", "").strip().rstrip('"”’)]}')
        duration = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
        if stripped.endswith((".", "!", "?", "…")) or duration >= MAX_SENTENCE_SEC:
            flush()
    flush()
    return [x for x in sentences if x["text"]]


def classify_transcript_quality(confidence):
    """Ba mức chất lượng chỉ dựa trên confidence ASR, không đọc ngữ nghĩa câu."""
    if confidence is None:
        return "REVIEW"
    if confidence < QUALITY_NCR_MAX_CONFIDENCE:
        return "NCR"
    if confidence >= QUALITY_RELIABLE_MIN_CONFIDENCE:
        return "RELIABLE"
    return "REVIEW"


def build_scored_sentences(audio, words, clip_offset_sec=None):
    sentences = split_words_into_sentences(words)
    for sentence in sentences:
        metrics = None
        confidence_error = None
        try:
            metrics = score_sentence_confidence(
                audio, sentence["text"], sentence["start_sec"], sentence["end_sec"]
            )
        except Exception as exc:
            confidence_error = f"{type(exc).__name__}: {exc}"

        if metrics is not None:
            sentence.update(metrics)
        else:
            sentence.update({"avg_logprob": None, "confidence": None})
        sentence["quality"] = classify_transcript_quality(sentence["confidence"])
        if confidence_error:
            sentence["confidence_error"] = confidence_error
        if clip_offset_sec is not None:
            sentence["source_start_sec"] = round_sec(clip_offset_sec + sentence["start_sec"])
            sentence["source_end_sec"] = round_sec(clip_offset_sec + sentence["end_sec"])
    return sentences


def load_source_timeline(video_path):
    """Đọc start/end của clip trong result.json nếu có; chấp nhận end=-1."""
    metadata_path = video_path.parent / "result.json"
    cache_key = str(metadata_path)
    if cache_key not in _source_metadata_cache:
        mapping = {}
        if metadata_path.exists():
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                for item in raw.get("segments", []):
                    mapping[Path(item.get("file_path", "")).name] = item
            except Exception as exc:
                print(f"Cảnh báo: không đọc được {metadata_path}: {exc}")
        _source_metadata_cache[cache_key] = mapping
    return _source_metadata_cache[cache_key].get(video_path.name)


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def transcribe_speech_region(audio, start_sample, end_sample, clip_offset_sec=None):
    region = audio[start_sample:end_sample]
    region_duration = len(region) / SAMPLE_RATE
    if region_duration < MIN_ASR_SECONDS:
        return None

    result = asr_pipe({"array": region, "sampling_rate": SAMPLE_RATE})
    text = clean_text(result.get("text", ""))
    local_start = start_sample / SAMPLE_RATE
    local_end = end_sample / SAMPLE_RATE

    words = []
    for chunk in result.get("chunks", []):
        timestamp = chunk.get("timestamp") or (None, None)
        word_start = 0.0 if timestamp[0] is None else float(timestamp[0])
        word_end = region_duration if timestamp[1] is None else float(timestamp[1])
        word = {
            "text": chunk.get("text", ""),
            "start_sec": round_sec(local_start + word_start),
            "end_sec": round_sec(min(local_start + word_end, local_end)),
        }
        if clip_offset_sec is not None:
            word["source_start_sec"] = round_sec(clip_offset_sec + word["start_sec"])
            word["source_end_sec"] = round_sec(clip_offset_sec + word["end_sec"])
        words.append(word)

    # Fallback nếu Whisper có text nhưng không trả word timestamp.
    if text and not words:
        words = [{
            "text": text,
            "start_sec": round_sec(local_start),
            "end_sec": round_sec(local_end),
        }]

    utterance = {
        "start_sec": round_sec(local_start),
        "end_sec": round_sec(local_end),
        "text": text,
        "words": words,
    }
    if clip_offset_sec is not None:
        utterance["source_start_sec"] = round_sec(clip_offset_sec + local_start)
        utterance["source_end_sec"] = round_sec(clip_offset_sec + local_end)
    return utterance


def collect_speaker_turns(annotation):
    if hasattr(annotation, "itertracks"):
        return [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    return [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, speaker in annotation
    ]


def run_speaker_diarization(audio):
    output = diar_pipeline(
        {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        },
        min_speakers=DIARIZATION_MIN_SPEAKERS,
        max_speakers=DIARIZATION_MAX_SPEAKERS,
    )
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = output.speaker_diarization
    return collect_speaker_turns(annotation)


def assign_speaker(start, end, speaker_turns):
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


def sentence_payload(sentence):
    confidence = sentence.get("confidence")
    return {
        "start": round_sec(sentence["start_sec"]),
        "end": round_sec(sentence["end_sec"]),
        "text": sentence["text"],
        "confidence": None if confidence is None else round(float(confidence), 3),
        "quality": sentence["quality"],
    }


def group_sentences_into_turns(sentences):
    """Gom các câu liền nhau của cùng speaker; timestamp bên trong là local theo segment."""
    turns = []
    for sentence in sorted(sentences, key=lambda x: (x["start_sec"], x["end_sec"])):
        speaker = sentence.get("speaker", "UNKNOWN")
        payload = sentence_payload(sentence)
        can_merge = (
            turns
            and turns[-1]["speaker"] == speaker
            and payload["start"] - turns[-1]["end"] <= TURN_MERGE_GAP_SEC
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


def get_segment_bounds(source_meta, duration_sec):
    start = 0.0
    if source_meta is not None and float(source_meta.get("start_time_sec", -1)) >= 0:
        start = float(source_meta["start_time_sec"])
    metadata_end = -1.0 if source_meta is None else float(source_meta.get("end_time_sec", -1))
    end = metadata_end if metadata_end >= start else start + float(duration_sec)
    return round_sec(start), round_sec(end)


def make_segment_id(video_id, start, end):
    return f"{video_id}_{int(round(start)):06d}_{int(round(end)):06d}"


def flatten_transcript(record):
    return clean_text(" ".join(
        sentence.get("text", "")
        for turn in record.get("turns", [])
        for sentence in turn.get("sentences", [])
    ))


def process_video(video_path):
    relative_path = video_path.relative_to(VIDEO_ROOT)
    output_dir = OUTPUT_ROOT / relative_path.parent
    json_path = output_dir / f"{video_path.stem}.json"
    txt_path = output_dir / f"{video_path.stem}.txt"
    error_path = output_dir / f"{video_path.stem}.error.json"

    if json_path.exists() and not OVERWRITE:
        try:
            saved_record = json.loads(json_path.read_text(encoding="utf-8"))
            required_keys = {"video_id", "segment_id", "start", "end", "turns"}
            if required_keys.issubset(saved_record):
                # Tự phục hồi TXT nếu checkpoint schema mới còn nhưng TXT bị thiếu.
                if not txt_path.exists():
                    saved_text = flatten_transcript(saved_record)
                    atomic_write_text(txt_path, saved_text + ("\n" if saved_text else ""))
                return "skipped", saved_record
        except (json.JSONDecodeError, OSError):
            # Checkpoint hỏng/ghi dở: xử lý lại clip này.
            pass

    audio = decode_audio(video_path)
    duration_sec = len(audio) / SAMPLE_RATE

    vad_regions = get_speech_timestamps(
        torch.from_numpy(audio),
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS,
        max_speech_duration_s=MAX_SPEECH_SECONDS,
        return_seconds=False,
    )

    source_meta = load_source_timeline(video_path)
    segment_start, segment_end = get_segment_bounds(source_meta, duration_sec)
    clip_offset_sec = segment_start
    speaker_turns = run_speaker_diarization(audio) if vad_regions else []

    utterances = []
    for region in vad_regions:
        utterance = transcribe_speech_region(
            audio, int(region["start"]), int(region["end"]), clip_offset_sec
        )
        if utterance is not None and utterance["text"]:
            for word in utterance.get("words", []):
                word["speaker"] = assign_speaker(
                    word["start_sec"], word["end_sec"], speaker_turns
                )
            utterances.append(utterance)

    sentences = []
    for utterance in utterances:
        utterance_sentences = build_scored_sentences(
            audio, utterance.get("words", []), clip_offset_sec
        )
        sentences.extend(utterance_sentences)

    turns = group_sentences_into_turns(sentences)
    video_id = video_path.parent.name
    record = {
        "video_id": video_id,
        "segment_id": make_segment_id(video_id, segment_start, segment_end),
        "start": segment_start,
        "end": segment_end,
        "turns": turns,
    }

    transcript = flatten_transcript(record)
    atomic_write_json(json_path, record)
    atomic_write_text(txt_path, transcript + ("\n" if transcript else ""))
    if error_path.exists():
        error_path.unlink()
    return "processed", record


# ================= TEST DIARIZATION TRÊN MỘT CLIP =================
# Cell 4 đã tải sẵn Community-1 bằng HF_TOKEN; cell này chỉ chạy thử segment_004.
from IPython.display import Audio, display

# segment_004 dài khoảng một phút: đủ ngắn để test nhưng có cơ hội chứa đổi người nói.
# Có thể đổi thành segment_001.mp4, segment_003.mp4...
TEST_SEGMENT_NAME = "segment_004.mp4"
test_matches = [
    p for p in VIDEO_ROOT.rglob(TEST_SEGMENT_NAME) if p.parent.name == "L21_V001"
]
if not test_matches:
    raise FileNotFoundError(f"Không tìm thấy L21_V001/{TEST_SEGMENT_NAME}")
test_video = test_matches[0]
print("Clip test:", test_video.relative_to(VIDEO_ROOT))

test_audio = decode_audio(test_video)
display(Audio(test_audio, rate=SAMPLE_RATE))

print("Đang chạy pyannote Community-1...")
speaker_turns = run_speaker_diarization(test_audio)
print(f"Pyannote phát hiện {len(set(x[2] for x in speaker_turns))} speaker:")
for start, end, speaker in speaker_turns:
    print(f"  [{start:7.2f} - {end:7.2f}] {speaker}")

# Chạy PhoASR trên đúng các vùng có tiếng người như pipeline chính.
test_vad_regions = get_speech_timestamps(
    torch.from_numpy(test_audio),
    vad_model,
    sampling_rate=SAMPLE_RATE,
    threshold=VAD_THRESHOLD,
    min_speech_duration_ms=MIN_SPEECH_MS,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS,
    max_speech_duration_s=MAX_SPEECH_SECONDS,
    return_seconds=False,
)
test_words = []
for region in test_vad_regions:
    utt = transcribe_speech_region(
        test_audio, int(region["start"]), int(region["end"]), None
    )
    if utt:
        test_words.extend(utt["words"])

def assign_speaker(start, end, turns):
    # Chọn speaker có thời lượng giao lớn nhất với từ hiện tại.
    overlaps = [
        (max(0.0, min(end, turn_end) - max(start, turn_start)), speaker)
        for turn_start, turn_end, speaker in turns
    ]
    best_overlap, best_speaker = max(overlaps, default=(0.0, "UNKNOWN"))
    if best_overlap > 0:
        return best_speaker
    # Timestamp ASR đôi khi lệch nhẹ: dùng speaker gần trung điểm từ nhất.
    if not turns:
        return "UNKNOWN"
    midpoint = (start + end) / 2
    return min(
        turns,
        key=lambda x: 0.0 if x[0] <= midpoint <= x[1] else min(abs(midpoint-x[0]), abs(midpoint-x[1])),
    )[2]

for word in test_words:
    word["speaker"] = assign_speaker(word["start_sec"], word["end_sec"], speaker_turns)

# Tách câu, chấm confidence và gom thành đúng schema batch.
test_sentences = build_scored_sentences(test_audio, test_words, None)
test_source_meta = load_source_timeline(test_video)
test_start, test_end = get_segment_bounds(
    test_source_meta, len(test_audio) / SAMPLE_RATE
)
test_video_id = test_video.parent.name
test_record = {
    "video_id": test_video_id,
    "segment_id": make_segment_id(test_video_id, test_start, test_end),
    "start": test_start,
    "end": test_end,
    "turns": group_sentences_into_turns(test_sentences),
}

print("\n========== JSON TEST ĐÚNG SCHEMA BATCH ==========")
print(json.dumps(test_record, ensure_ascii=False, indent=2))


stats = {"processed": 0, "skipped": 0, "errors": 0}
failed = []

for index, video_path in enumerate(tqdm(video_files, desc="Phiên âm"), start=1):
    try:
        status, record = process_video(video_path)
        stats[status] += 1
        sentence_count = sum(len(turn["sentences"]) for turn in record["turns"])
        ncr_count = sum(
            sentence["quality"] == "NCR"
            for turn in record["turns"]
            for sentence in turn["sentences"]
        )
        tqdm.write(
            f"[{index}/{len(video_files)}] {video_path.parent.name}/{video_path.name} | "
            f"turns={len(record['turns'])} sentences={sentence_count} NCR={ncr_count}"
        )
    except Exception as exc:
        stats["errors"] += 1
        relative_path = video_path.relative_to(VIDEO_ROOT)
        error_path = OUTPUT_ROOT / relative_path.parent / f"{video_path.stem}.error.json"
        error_record = {
            "relative_video_path": relative_path.as_posix(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(error_path, error_record)
        failed.append(error_record)
        tqdm.write(f"LỖI {relative_path}: {type(exc).__name__}: {exc}")

    if index % 20 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("Hoàn tất:", stats)
if failed:
    print("Các clip lỗi được lưu thành *.error.json để dễ kiểm tra và chạy lại.")


# Tổng hợp mọi checkpoint thành CSV + JSONL, kể cả kết quả từ phiên Kaggle trước.
record_paths = sorted(
    [p for p in OUTPUT_ROOT.rglob("segment_*.json") if not p.name.endswith(".error.json")],
    key=natural_key,
)
records = [json.loads(p.read_text(encoding="utf-8")) for p in record_paths]

jsonl_path = OUTPUT_ROOT / "all_segments.jsonl"
atomic_write_text(
    jsonl_path,
    "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
)

summary_path = OUTPUT_ROOT / "summary.csv"
temp_summary = summary_path.with_suffix(".csv.tmp")
with temp_summary.open("w", encoding="utf-8-sig", newline="") as file:
    fieldnames = [
        "video_id", "segment_id", "start", "end",
        "turn_count", "sentence_count", "reliable_count",
        "review_count", "ncr_count", "transcript",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for item in records:
        sentences = [
            sentence
            for turn in item.get("turns", [])
            for sentence in turn.get("sentences", [])
        ]
        quality_counts = {quality: 0 for quality in ("RELIABLE", "REVIEW", "NCR")}
        for sentence in sentences:
            quality = sentence.get("quality", "REVIEW")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        writer.writerow({
            "video_id": item.get("video_id"),
            "segment_id": item.get("segment_id"),
            "start": item.get("start"),
            "end": item.get("end"),
            "turn_count": len(item.get("turns", [])),
            "sentence_count": len(sentences),
            "reliable_count": quality_counts["RELIABLE"],
            "review_count": quality_counts["REVIEW"],
            "ncr_count": quality_counts["NCR"],
            "transcript": flatten_transcript(item),
        })
os.replace(temp_summary, summary_path)

archive_path = shutil.make_archive(
    "/kaggle/working/htv9_transcriptions", "zip", root_dir=OUTPUT_ROOT
)

quality_totals = {quality: 0 for quality in ("RELIABLE", "REVIEW", "NCR")}
for item in records:
    for turn in item.get("turns", []):
        for sentence in turn.get("sentences", []):
            quality = sentence.get("quality", "REVIEW")
            quality_totals[quality] = quality_totals.get(quality, 0) + 1

print(f"Đã tổng hợp {len(records)} clip")
print("Chất lượng câu:", quality_totals)
print("CSV      :", summary_path)
print("JSONL    :", jsonl_path)
print("Tải về  :", archive_path)

# Xem thử một kết quả.
if records:
    preview = records[0]
    print("\nVí dụ:", preview["segment_id"])
    print(json.dumps(preview, ensure_ascii=False, indent=2))


## Cấu trúc đầu ra
