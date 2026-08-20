# %%
# CELL 1: Cài đặt và Import thư viện
# !pip install torch torchaudio pydub pydantic

import os
import subprocess
import json
import time
import abc
import logging
from typing import List, Optional, Dict, Any
from pprint import pprint

import torch
import torchaudio
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field

# --- THIẾT LẬP LOGGER ---
logger = logging.getLogger("VadSplitter")
logger.setLevel(logging.DEBUG) # Bắt mọi level

# 1. Console Handler: Chỉ in ra INFO trở lên (dành cho log ngắn)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(ch_formatter)

# 2. File Handler: Ghi toàn bộ DEBUG và INFO vào file (dành cho log dài)
log_file = 'splitter_process.log'
fh = logging.FileHandler(log_file, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
fh.setFormatter(fh_formatter)

logger.addHandler(ch)
logger.addHandler(fh)

logger.info("Đã import thành công các thư viện & thiết lập Logger!")

# %%
# CELL 2: Schema Định nghĩa (I/O & Internal DTO)

# --- EXTERNAL SCHEMAS ---
class SplitStrategy(str, Enum):
    MID_POINT = "MID_POINT"

class LogConfig(BaseModel):
    save_metadata_json: bool = False
    save_vad_timestamps_json: bool = False
    save_split_points_json: bool = False
    save_ffmpeg_commands_txt: bool = False
    save_final_response_json: bool = True

class VADConfig(BaseModel):
    min_silence_duration_ms: int = Field(default=1000, ge=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    speech_pad_ms: int = Field(default=30, ge=0)
    split_strategy: SplitStrategy = Field(default=SplitStrategy.MID_POINT)

class SplitVideoRequest(BaseModel):
    job_id: str
    video_id: str
    video_input_path: str
    output_dir: str
    vad_config: VADConfig = VADConfig()
    log_config: LogConfig = LogConfig()

class SegmentResult(BaseModel):
    segment_index: int
    file_path: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float

class ProcessingMetrics(BaseModel):
    total_processing_time_sec: float = 0.0
    audio_extract_time_sec: float = 0.0
    vad_inference_time_sec: float = 0.0
    video_split_time_sec: float = 0.0

class SplitVideoResponse(BaseModel):
    job_id: str
    status: str
    metrics: ProcessingMetrics = ProcessingMetrics()
    segments: List[SegmentResult] = []
    error_message: Optional[str] = None

# --- INTERNAL STATE (DTO) ---
@dataclass
class VideoSplitPoint:
    split_time_sec: float

@dataclass
class SplitContext:
    request: SplitVideoRequest
    temp_audio_path: str = "temp_audio.wav"
    
    # Metadata đầu vào (vừa inspect được)
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    
    split_points_sec: List[float] = field(default_factory=list)
    segments_output: List[SegmentResult] = field(default_factory=list)
    
    time_audio_extraction: float = 0.0
    time_vad: float = 0.0
    time_split: float = 0.0
    
    @property
    def final_output_dir(self) -> str:
        return os.path.join(self.request.output_dir, self.request.video_id)

    def save_artifact(self, filename: str, data_str: str):
        if not os.path.exists(self.final_output_dir):
            os.makedirs(self.final_output_dir)
        filepath = os.path.join(self.final_output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data_str)
            
logger.info("Khởi tạo Schema thành công!")


# %%
# CELL 3: Interfaces (Abstract Base Classes)

class IVideoInspector(abc.ABC):
    @abc.abstractmethod
    def inspect(self, ctx: SplitContext) -> None:
        """Quan sát dữ liệu đầu vào (lấy metadata video bằng ffprobe)"""
        pass

class IAudioExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, ctx: SplitContext) -> None:
        """Trích xuất audio từ video"""
        pass

class IVadAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, ctx: SplitContext) -> None:
        """Phân tích VAD tìm điểm cắt"""
        pass

class IVideoSplitter(abc.ABC):
    @abc.abstractmethod
    def split(self, ctx: SplitContext) -> None:
        """Cắt video"""
        pass

logger.info("Khởi tạo Interfaces (ABC) thành công!")


# %%
# CELL 4: Concrete Implementations

class FFmpegVideoInspector(IVideoInspector):
    def inspect(self, ctx: SplitContext) -> None:
        logger.info(f"Đang quan sát dữ liệu đầu vào: {ctx.request.video_input_path}...")
        command = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", ctx.request.video_input_path
        ]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            metadata = json.loads(result.stdout.decode('utf-8'))
            ctx.video_metadata = metadata
            
            # Log ngắn ra console
            format_info = metadata.get("format", {})
            duration = format_info.get("duration", "N/A")
            size = format_info.get("size", "N/A")
            logger.info(f"  -> Video Duration: {duration}s | Size: {size} bytes")
            
            # Log cục JSON khổng lồ vào FILE, không in ra terminal
            if ctx.request.log_config.save_metadata_json:
                ctx.save_artifact("metadata.json", json.dumps(metadata, indent=2))
                logger.info("  -> Đã lưu chi tiết metadata vào file metadata.json")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi đọc metadata video bằng ffprobe.")
            raise RuntimeError(f"FFprobe Error: {e.stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            logger.error(f"Lỗi không xác định khi parse metadata: {e}")
            raise e

class FFmpegAudioExtractor(IAudioExtractor):
    def extract(self, ctx: SplitContext) -> None:
        logger.info(f"Đang tách audio từ {ctx.request.video_input_path}...")
        start_t = time.time()
        command = [
            "ffmpeg", "-y", "-i", ctx.request.video_input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            ctx.temp_audio_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ctx.time_audio_extraction = time.time() - start_t
            logger.info(f"  -> Hoàn tất tách audio (mất {ctx.time_audio_extraction:.2f}s)")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi dùng FFmpeg Extract Audio: {e.stderr.decode('utf-8', errors='ignore')}")
            raise RuntimeError("Lỗi trích xuất audio, vui lòng xem log_file.")

class SileroVadAnalyzer(IVadAnalyzer):
    def __init__(self):
        logger.info("Khởi tạo mô hình Silero VAD (Pytorch)...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        self.read_audio = utils[2]
        self.sampling_rate = 16000
        logger.info("  -> Tải mô hình thành công.")

    def analyze(self, ctx: SplitContext) -> None:
        logger.info(f"Đang phân tích VAD file {ctx.temp_audio_path}...")
        
        start_t = time.time()
        wav = self.read_audio(ctx.temp_audio_path, sampling_rate=self.sampling_rate)
        cfg = ctx.request.vad_config
        
        speech_timestamps = self.get_speech_timestamps(
            wav, self.model, sampling_rate=self.sampling_rate,
            min_silence_duration_ms=cfg.min_silence_duration_ms,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_duration_ms,
            speech_pad_ms=cfg.speech_pad_ms
        )
        
        speech_segments_sec = [
            {'start': ts['start'] / self.sampling_rate, 'end': ts['end'] / self.sampling_rate} 
            for ts in speech_timestamps
        ]
        
        # Log toàn bộ mảng timestamp (rất dài) vào file, console bỏ qua
        if ctx.request.log_config.save_vad_timestamps_json:
            ctx.save_artifact("timestamps.json", json.dumps(speech_segments_sec, indent=2))
            logger.info("  -> Đã lưu mảng timestamps thô vào file timestamps.json")
        
        split_points = []
        if len(speech_segments_sec) > 1:
            for i in range(len(speech_segments_sec) - 1):
                curr_end = speech_segments_sec[i]['end']
                next_start = speech_segments_sec[i+1]['start']
                silence_duration = next_start - curr_end
                
                if cfg.split_strategy == SplitStrategy.MID_POINT:
                    split_point = curr_end + (silence_duration / 2.0)
                    split_points.append(round(split_point, 3))
                    
        ctx.split_points_sec = split_points
        ctx.time_vad = time.time() - start_t
        
        if ctx.request.log_config.save_split_points_json:
            ctx.save_artifact("split_points.json", json.dumps(split_points, indent=2))
            logger.info("  -> Đã lưu mảng điểm cắt vào file split_points.json")
            
        logger.info(f"  -> Hoàn tất tìm {len(split_points)} điểm cắt (mất {ctx.time_vad:.2f}s)")

class FFmpegVideoSplitter(IVideoSplitter):
    def split(self, ctx: SplitContext) -> None:
        logger.info(f"Tiến hành cắt video (FFmpeg) tại {len(ctx.split_points_sec)} điểm...")
        start_t = time.time()
        
        # Thư mục lưu riêng cho từng video_id
        final_output_dir = os.path.join(ctx.request.output_dir, ctx.request.video_id)
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
            
        points = [0.0] + ctx.split_points_sec
        ffmpeg_commands_log = []
        
        for i in range(len(points)):
            start_time = points[i]
            end_time = points[i+1] if i < len(points) - 1 else None
            duration = (end_time - start_time) if end_time else None
            
            output_file = os.path.join(final_output_dir, f"segment_{i+1:03d}.mp4")
            command = ["ffmpeg", "-y", "-i", ctx.request.video_input_path, "-ss", str(start_time)]
            
            if duration:
                command.extend(["-t", str(duration)])
                
            command.extend(["-c", "copy", output_file])
            ffmpeg_commands_log.append(' '.join(command))
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                ctx.segments_output.append(SegmentResult(
                    segment_index=i+1,
                    file_path=output_file,
                    start_time_sec=round(start_time, 3),
                    end_time_sec=round(end_time, 3) if end_time else -1.0,
                    duration_sec=round(duration, 3) if duration else -1.0
                ))
            except subprocess.CalledProcessError as e:
                logger.error(f"Lỗi khi cắt đoạn {i+1}: {e.stderr.decode('utf-8', errors='ignore')}")
                raise RuntimeError(f"FFmpeg Split Error at segment {i+1}")
                
        ctx.time_split = time.time() - start_t
        
        if ctx.request.log_config.save_ffmpeg_commands_txt:
            ctx.save_artifact("ffmpeg_commands.txt", "\n".join(ffmpeg_commands_log))
            logger.info("  -> Đã lưu các lệnh FFmpeg vào file ffmpeg_commands.txt")
            
        logger.info(f"  -> Hoàn tất cắt {len(ctx.segments_output)} đoạn (mất {ctx.time_split:.2f}s)")

logger.info("Cài đặt Concrete Classes thành công!")

# %%
# CELL 5: Application Service (Orchestrator)

class VideoSplitterService:
    def __init__(self, inspector: IVideoInspector, extractor: IAudioExtractor, analyzer: IVadAnalyzer, splitter: IVideoSplitter):
        self._inspector = inspector
        self._extractor = extractor
        self._analyzer = analyzer
        self._splitter = splitter
        
    def process(self, request: SplitVideoRequest) -> SplitVideoResponse:
        logger.info(f"--- BẮT ĐẦU XỬ LÝ JOB: {request.job_id} ---")
        total_start_t = time.time()
        ctx = SplitContext(request=request)
        response = SplitVideoResponse(job_id=request.job_id, status="PENDING")
        
        try:
            # 1. Quan sát đầu vào
            self._inspector.inspect(ctx)
            
            # 2. Tách âm thanh
            self._extractor.extract(ctx)
            
            # 3. Phân tích VAD
            self._analyzer.analyze(ctx)
            
            # 4. Cắt video
            if len(ctx.split_points_sec) > 0:
                self._splitter.split(ctx)
            else:
                logger.info("Bỏ qua cắt video do không tìm thấy điểm cắt nào.")
                
            response.status = "SUCCESS"
            response.segments = ctx.segments_output
            logger.info("--- HOÀN TẤT XỬ LÝ THÀNH CÔNG ---")
            
        except Exception as e:
            response.status = "FAILED"
            response.error_message = str(e)
            logger.error(f"--- XỬ LÝ THẤT BẠI: {e} ---")
            
        finally:
            response.metrics = ProcessingMetrics(
                total_processing_time_sec=round(time.time() - total_start_t, 3),
                audio_extract_time_sec=round(ctx.time_audio_extraction, 3),
                vad_inference_time_sec=round(ctx.time_vad, 3),
                video_split_time_sec=round(ctx.time_split, 3)
            )
            # Response JSON khổng lồ cũng ghi vào file log thay vì terminal
            if ctx.request.log_config.save_final_response_json:
                ctx.save_artifact("result.json", response.model_dump_json(indent=2))
                logger.info("--- ĐÃ LƯU KẾT QUẢ CUỐI CÙNG VÀO FILE result.json ---")
            
            if os.path.exists(ctx.temp_audio_path):
                os.remove(ctx.temp_audio_path)
                
        return response

logger.info("Khởi tạo VideoSplitterService thành công!")

# %%
# CELL 6: Dependency Injection & Thực thi (Integration Test)

# 1. Khởi tạo Component dependencies
inspector = FFmpegVideoInspector()
extractor = FFmpegAudioExtractor()
analyzer = SileroVadAnalyzer()
splitter = FFmpegVideoSplitter()

# 2. Bơm dependencies vào Service
video_service = VideoSplitterService(
    inspector=inspector,
    extractor=extractor, 
    analyzer=analyzer, 
    splitter=splitter
)

request_payload = SplitVideoRequest(
    job_id="job-solid-logger-003",
    video_id="vid_sample_001",
    video_input_path="sample_video.mp4", # <--- ĐỔI TÊN FILE TẠI ĐÂY
    output_dir="output_segments",
    vad_config=VADConfig(
        min_silence_duration_ms=1000,
        threshold=0.5
    ),
    log_config=LogConfig(
        save_metadata_json=False,
        save_vad_timestamps_json=False,
        save_split_points_json=False,
        save_ffmpeg_commands_txt=False,
        save_final_response_json=True
    )
)

if not os.path.exists(request_payload.video_input_path):
    logger.error(f"Không tìm thấy file {request_payload.video_input_path}")
else:
    logger.info("🚀 GỌI COMPONENT THÔNG QUA SERVICE...")
    final_response = video_service.process(request_payload)
    
    # Chỉ in log ngắn báo cáo trạng thái ra terminal. Json đầy đủ đã ghi vào file.
    logger.info(f"Trạng thái Job: {final_response.status}")
    logger.info(f"Tổng số đoạn cắt: {len(final_response.segments)}")
    logger.info(f"Chi tiết đầy đủ (Metadata, Timestamp, Lệnh FFmpeg, Response) xin xem tại file: {log_file}")
