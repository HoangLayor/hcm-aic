# AIC 2026 Video Retrieval Pipeline (Kaggle Edition)

Hệ thống rút trích đặc trưng đa phương thức (Multi-modal Feature Extraction) và Tìm kiếm Video (Video Retrieval) phục vụ cuộc thi **Ho Chi Minh City AI Challenge (HCM AI Challenge 2026)**. Hệ thống được thiết kế theo kiến trúc **Staged Execution Pipeline** để hoạt động tối ưu trên môi trường **Kaggle Notebooks** / **Google Colab** (giới hạn tài nguyên < 16GB VRAM, 20GB Disk).

---

## 1. Kiến trúc Hệ thống (Pipeline Architecture)

Pipeline xử lý theo quy trình 5 bước tuần tự, tải từng model vào GPU, xử lý batch video và giải phóng bộ nhớ (VRAM & RAM) triệt để giữa các Stage:

```
[Raw MP4 Videos]
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 1: Silero VAD + FFmpeg                                │
 │ Tách video thành các phân đoạn (segments) theo giọng nói    │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 1.5: PhoASR & Pyannote (Transcript)                   │
 │ Trích xuất âm thanh thành văn bản & phân tách người nói     │
 └─────────────────────────────┬───────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
 ┌───────────────────────────┐                   ┌───────────────────────────┐
 │ STAGE 2: DINOv2           │                   │ STAGE 3: Qwen VLM         │
 │ Gom cụm & trích Keyframes │                   │ Sinh Dense Captions       │
 └─────────────┬─────────────┘                   └─────────────┬─────────────┘
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 4: Qwen3-VL-Embedding (2B)                            │
 │ Nhúng đa phương thức (Ảnh Keyframe + Caption) -> Vector     │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 5: Local Qdrant Vector Database (Dim: 2048)           │
 │ Lưu trữ vector nhúng & metadata phục vụ truy vấn thời gian  │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ SEARCH & RETRIEVAL ENGINE (KIS, TRAKE, Q&A)                 │
 └─────────────────────────────────────────────────────────────┘
```

### Chi tiết các thành phần trong `src/`:
- [`src/config.py`](file:///d:/Projects/aic/src/config.py): Cấu hình toàn cục tập trung (hỗ trợ Pydantic v1, v2 và biến môi trường `AIC_*`).
- [`src/audio/vad_splitter.py`](file:///d:/Projects/aic/src/audio/vad_splitter.py): Tách video dựa vào khoảng lặng (Silero VAD) và FFmpeg `libx264 -preset ultrafast` chống lệch frame.
- [`src/audio/transcript_extractor.py`](file:///d:/Projects/aic/src/audio/transcript_extractor.py): Trích xuất âm thanh thành văn bản (PhoASR) và nhận diện người nói (Pyannote Diarization).
- [`src/vision/keyframe_extractor.py`](file:///d:/Projects/aic/src/vision/keyframe_extractor.py): Dùng DINOv2 ViT-Base (`dinov2_vitb14`) gom cụm cosine và trích frame đại diện (medoid).
- [`src/vlm/segment_captioner.py`](file:///d:/Projects/aic/src/vlm/segment_captioner.py): Dùng Qwen VLM (`Qwen/Qwen3.5-2B`) mô tả nội dung video ngắn.
- [`src/embedding/qwen3_embedder.py`](file:///d:/Projects/aic/src/embedding/qwen3_embedder.py): Dùng `Qwen/Qwen3-VL-Embedding-2B` nhúng đồng thời cả ảnh keyframe và text caption vào cùng một không gian vector đa phương thức.
- [`src/storage/qdrant_manager.py`](file:///d:/Projects/aic/src/storage/qdrant_manager.py): Quản lý Vector DB Qdrant cục bộ (File-based/In-Memory, không cần cài đặt Docker server).
- [`src/search/retriever.py`](file:///d:/Projects/aic/src/search/retriever.py): Cỗ máy tìm kiếm tích hợp cho Known-Item Search (KIS), Video QA, và chuỗi sự kiện liên tiếp (TRAKE).

---

## 2. Hướng dẫn Chạy trên Kaggle / Colab

### Bước 2.1: Cài đặt Môi trường (Notebook Cell đầu tiên)
Chọn Accelerator là **GPU T4x2** hoặc **P100**:

```bash
# 1. Cài đặt thư viện AI và dependencies tương thích
!pip install -q --upgrade transformers accelerate qwen-vl-utils sentence-transformers
!pip install -q "silero-vad[onnx-cpu]==6.2.1" "onnxruntime==1.27.0" "pyannote.audio==4.0.7" "soundfile>=0.12" pydantic-settings qdrant-client opencv-python pytest
!apt-get install -y ffmpeg

# 2. Clone repository về Kaggle
!git clone https://github.com/HoangLayor/hcm-aic.git /kaggle/working/hcm-aic
%cd /kaggle/working/hcm-aic
```

---

### Bước 2.2: Các chế độ chạy với `run_kaggle.py`

File [`run_kaggle.py`](file:///d:/Projects/aic/run_kaggle.py) hỗ trợ đầy đủ các cờ dòng lệnh (CLI Flags):

#### 1. Kiểm tra môi trường (Dry-Run Mode)
Nạp thử 5 mô hình vào GPU/RAM và giải phóng bộ nhớ để đảm bảo môi trường không bị lỗi mà không tốn thời gian xử lý video:
```bash
!python run_kaggle.py --dry-run
```

#### 2. Chạy thử nghiệm trên 1 - 2 video thật (Quick Test)
```bash
!python run_kaggle.py --raw-dir /kaggle/input/datasets/dotrantu/aic-10-video/Segment_Video --limit 2
```

#### 3. Chạy THẬT toàn bộ Video Dataset (Mặc định chạy Batch 5 video/lần)
Hệ thống sẽ tự động xử lý trọn gói (End-to-End) 5 video mỗi đợt, nạp thẳng vào Qdrant rồi mới qua 5 video tiếp theo để chống tràn ổ cứng và có kết quả tìm kiếm sớm:
```bash
!python run_kaggle.py --raw-dir /kaggle/input/datasets/dotrantu/aic-10-video/Segment_Video --batch-size 5
```

#### 4. Chạy riêng lẻ từng Stage nếu cần Debug:
```bash
!python run_kaggle.py --stage vad         # Chỉ chạy Stage 1 (VAD Splitting)
!python run_kaggle.py --stage transcript  # Chỉ chạy Stage 1.5 (Transcript PhoASR)
!python run_kaggle.py --stage dino        # Chỉ chạy Stage 2 (Keyframe DINOv2)
!python run_kaggle.py --stage vlm         # Chỉ chạy Stage 3 (Qwen Dense Captioning)
!python run_kaggle.py --stage embed     # Chỉ chạy Stage 4 (Qwen3-VL Embedding)
!python run_kaggle.py --stage qdrant    # Chỉ chạy Stage 5 (Qdrant Ingestion)
```

---

## 3. Cấu hình & Biến môi trường (Environment Variables)

Hệ thống hỗ trợ cấu hình động thông qua file [`src/config.py`](file:///d:/Projects/aic/src/config.py) hoặc thiết lập biến môi trường với tiền tố `AIC_`:

| Tên biến môi trường | Giá trị mặc định | Mô tả |
| :--- | :--- | :--- |
| `AIC_RAW_DIR` | `./data` | Thư mục chứa các file video gốc (`.mp4`, `.mkv`,...). |
| `AIC_OUTPUT_DIR` | `./output` | Thư mục lưu các artifact trung gian (segments, keyframes, captions, embeddings). |
| `AIC_DB_DIR` | `./qdrant_db` | Thư mục lưu trữ database vector Qdrant cục bộ. |
| `AIC_VECTOR_DIM` | `2048` | Kích thước vector nhúng của Qwen3-VL-Embedding. |
| `AIC_TARGET_FPS` | `30` | FPS mục tiêu cố định chống trôi frame timestamp. |
| `AIC_ASR_MODEL_ID` | `Qualcomm-AI-Research/PhoASR-whisper-small` | Model nhận diện giọng nói (ASR). |
| `AIC_USE_TRANSCRIPT_BRANCH` | `True` | Chạy Stage 1.5 khi dùng `--stage all`. |
| `AIC_ASR_MAX_SPEECH_SECONDS` | `28.0` | Độ dài tối đa của mỗi vùng tiếng nói đưa vào PhoASR. |
| `AIC_ASR_MIN_SPEECH_SECONDS` | `0.35` | Bỏ qua vùng tiếng nói ngắn hơn ngưỡng này. |
| `AIC_HF_TOKEN` | `(rỗng)` | HuggingFace Token bắt buộc nếu muốn bật Pyannote Diarization (nhận diện người nói). |
| `AIC_DINO_MODEL_ID` | `dinov2_vitb14` | Model DINOv2 trích đặc trưng hình ảnh. |
| `AIC_DINO_SIMILARITY_THRESHOLD` | `0.65` | Ngưỡng Cosine gom cụm frame keyframe. |
| `AIC_VLM_MODEL_ID` | `Qwen/Qwen3.5-2B` | Model Qwen VLM sinh caption video. |
| `AIC_EMBEDDER_MODEL_ID` | `Qwen/Qwen3-VL-Embedding-2B` | Model Multi-modal Embedding. |

---

## 4. Cấu trúc Dữ liệu lưu trữ

Mỗi video sau khi qua pipeline sẽ được lưu trữ có cấu trúc trong thư mục `OUTPUT_DIR/<video_id>/`:

```text
output/
└── <video_id>/                          # Ví dụ: L01_V001
    ├── manifest_vad.json                # Thông tin metadata các phân đoạn cắt VAD
    ├── segments/                        # Các đoạn video cắt nhỏ (.mp4)
    │   ├── <video_id>_seg001.mp4
    │   └── <video_id>_seg002.mp4
    ├── keyframes/                       # Ảnh keyframe & metadata frame index
    │   ├── <video_id>_seg001_kf_0.jpg
    │   └── <video_id>_seg001_meta.json
    ├── captions/                        # Caption text do Qwen VLM sinh ra
    │   └── <video_id>_seg001_caption.json
    └── embeddings/                      # Vector nhúng và metadata nén
        ├── keyframe_vectors.pt
        ├── keyframe_metadata.json
        ├── caption_vectors.pt
        └── caption_metadata.json
```

---

## 5. Hướng dẫn Kiểm thử (Testing & Retrieval)

Hệ thống có sẵn bộ test suite `pytest` mô phỏng các dạng truy vấn chính thức của AIC 2026:

### Chạy Unit Test & E2E Queries:
```bash
!pytest tests/test_e2e_queries.py -v
```

**Chi tiết các Test Case:**
- `test_kis_query`: Kiểm tra khả năng nhận diện truy vấn Known-Item Search. Hệ thống sẽ test việc parse Text Query và gọi hàm search Vector trên Qdrant.
- `test_trake_query`: Kiểm tra xem hệ thống có xử lý mảng $N$ truy vấn độc lập cho $N$ mốc thời gian ($E_1 \rightarrow E_4$) hay không.
- `test_qa_search_context`: Kiểm tra việc tìm Top-K Video Context trước khi giao cho VLM đọc OCR và sinh câu trả lời.

### Thử nghiệm truy vấn tìm kiếm bằng mã Python:
```python
from src.search.retriever import SearchRetriever

# Khởi tạo retriever kết nối tới local Qdrant
retriever = SearchRetriever()

# Tìm kiếm Known-Item Search (KIS)
query = "A chef cooking noodles in a busy kitchen with steam rising"
results = retriever.kis_search(query=query, top_k=5)

for res in results:
    print(f"Rank {res['rank']} | Score: {res['score']:.4f} | Video: {res['video_id']} | Frame: {res['frame_index']}")
```

---

## 6. Chiến lược Tăng tốc Hệ thống (Acceleration Strategies)

Nếu bạn có quỹ thời gian nộp bài eo hẹp và cần xử lý hàng ngàn video nhanh nhất có thể, hãy áp dụng các kỹ thuật sau để tăng tốc:

1. **Dọn dẹp bộ nhớ đệm Qdrant cũ khi đổi chiều vector**:
   Nếu bạn từng chạy với cấu hình cũ, hãy xóa thư mục database cũ trước khi chạy lại:
   ```bash
   !rm -rf /kaggle/working/qdrant_db
   ```
2. **Xóa video trung gian sau khi đã nạp Qdrant để tiết kiệm ổ đĩa Kaggle**:
   Sau khi hoàn tất Stage 5 cho toàn bộ một batch, pipeline tự động xóa các file `output/<video_id>/segments/*.mp4` của batch đó để giải phóng dung lượng đĩa. File `manifest_vad.json` và các metadata khác vẫn được giữ lại.
3. **Multiprocessing cho VAD (Sử dụng 100% CPU)**:
   Quá trình dùng FFmpeg cắt video có thể sử dụng `concurrent.futures.ProcessPoolExecutor` trong file `vad_splitter.py` để cắt nhiều đoạn video cùng lúc tận dụng 4 lõi CPU của Kaggle.
4. **Lượng tử hóa Mô hình (Quantization 4-bit / 8-bit)**:
   Mô hình `Qwen3.5-2B` đang tiêu tốn ~8GB VRAM ở chuẩn FP16. Nếu cài thêm thư viện `bitsandbytes` và cấu hình `load_in_4bit=True` trong `segment_captioner.py`, dung lượng VRAM sẽ giảm xuống chỉ còn ~3GB. Nhờ đó bạn có thể tăng `BATCH_SIZE` lên gấp 3 lần, tăng tốc độ xử lý video đáng kể.
5. **Sử dụng Flash Attention 2 / SDPA**:
   Khi load các mô hình LLM, việc kích hoạt `attn_implementation="sdpa"` hoặc `"flash_attention_2"` giúp tối ưu hóa ma trận self-attention trên GPU của Kaggle, tăng tốc độ sinh caption nhanh hơn 20-30%.
6. **Compile mô hình bằng TensorRT / ONNX**:
   Các mô hình tĩnh như `DINOv2` và `Qwen3-VL-Embedder` có thể được biên dịch sang định dạng TensorRT để tăng tốc độ nhúng vector gấp 3-5 lần so với chạy PyTorch native.

---

## 7. Đề xuất Kiến trúc Nâng cao (Advanced Proposals)

Để lọt vào **Top bảng xếp hạng (Leaderboard) AIC**, hệ thống cần độ chính xác cực cao để phân biệt các chi tiết tinh vi. Dưới đây là các định hướng nâng cấp mã nguồn:

### A. RRF (Reciprocal Rank Fusion) cho Text-Image Match
Hiện tại KIS Search tìm kiếm đồng thời trên Text Vector và Image Vector dựa vào điểm Cosine trung bình. Thay vì lấy điểm Cosine đơn thuần, hãy áp dụng thuật toán **RRF**. Thuật toán này cộng gộp thứ hạng (Rank) của khung hình đó ở cả 2 không gian Vector lại với nhau:
$$\text{Score} = \frac{1}{k + \text{rank}_{\text{text}}} + \frac{1}{k + \text{rank}_{\text{image}}}$$
RRF được chứng minh là tăng độ chính xác lên 10-15% trong các hệ thống lai (Hybrid Search).

### B. Multi-Crop Object Detection (Nhận diện vật thể nhỏ)
Nhiều truy vấn yêu cầu tìm vật thể rất nhỏ (Ví dụ: "Biển báo cấm rẽ trái ở góc màn hình"). Vector của một ảnh toàn cảnh sẽ bị nhiễu và dễ bỏ qua biển báo này. 
- **Đề xuất:** Tích hợp mô hình Zero-shot Object Detection (như **YOLO-World** hoặc **GroundingDINO**). Với mỗi Keyframe, crop các vật thể con ra, sau đó dùng Qwen-Embedder nhúng các ảnh Crop này thành các vector độc lập đính kèm với `frame_index` gốc.

### C. Tối ưu hóa Thuật toán Cửa sổ trượt (Sliding Window) cho TRAKE
Truy vấn TRAKE yêu cầu 4 khoảnh khắc liên tiếp. Đôi khi DINOv2 trích xuất thiếu mất 1 khoảnh khắc quan trọng do ngưỡng Similar quá cao. Hãy nội suy (interpolate) các khung hình giữa 2 keyframe nếu điểm số của chuỗi sự kiện bị đứt gãy.
