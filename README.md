# AIC 2026 Video Retrieval Pipeline (Kaggle Edition)

Hệ thống rút trích đặc trưng (Feature Extraction) và Tìm kiếm Video (Video Retrieval) được tối ưu hóa đặc biệt để chạy trên môi trường **Kaggle Notebooks** hoặc **Google Colab** với giới hạn tài nguyên khắt khe (16GB VRAM, 20GB Disk).

## 1. Cấu trúc Hệ thống

Hệ thống được chia thành các phân hệ xử lý tuần tự (Staged Execution) để chống quá tải RAM/VRAM:
- `src/config.py`: File cấu hình toàn cục (Hyperparameters, ngưỡng cắt, VRAM limits).
- `src/audio/vad_splitter.py`: Dùng Silero VAD và FFmpeg `libx264 -ultrafast` cắt video thành các đoạn nhỏ dựa vào khoảng lặng, không bị lỗi trôi frame.
- `src/vision/keyframe_extractor.py`: Sử dụng DINOv2 để gom cụm và trích xuất Frame đại diện (Medoid) cho mỗi cảnh.
- `src/vlm/segment_captioner.py`: Sử dụng `Qwen3.5-2B` để mô tả ngữ nghĩa (Dense Captioning) cho từng đoạn video.
- `src/embedding/qwen3_embedder.py`: Dùng `Qwen3-VL-Embedding` để nhúng (encode) cả ảnh keyframe và text caption thành vector.
- `src/storage/qdrant_manager.py`: Lưu trữ vector cục bộ (In-Memory/File) không cần cài đặt Docker.
- `src/search/retriever.py`: Cỗ máy tìm kiếm tích hợp cho KIS, TRAKE và Q&A.

---

## 2. Hướng dẫn Chạy trên Kaggle

### Bước 2.1: Chuẩn bị Môi trường
Mở Kaggle Notebook, chọn Accelerator là **GPU T4x2** hoặc **P100**. Trong ô code đầu tiên, chạy lệnh cài đặt:

```bash
!pip install -q torch torchvision torchaudio sentence-transformers transformers
!pip install -q qdrant-client pydantic opencv-python
!apt-get install -y ffmpeg
```

### Bước 2.2: Tải Source Code & Dữ liệu
Upload toàn bộ thư mục `src/` và file `run_kaggle.py` lên thư mục `/kaggle/working/` của bạn.
Đặt các video MP4 raw vào thư mục `/kaggle/working/raw/`.

### Bước 2.3: Chiến lược Chạy không tràn ổ cứng (Batch Execution)
Do Kaggle chỉ có 20GB ổ cứng, nếu bạn có hàng trăm video, **KHÔNG ĐƯỢC chạy toàn bộ Data một lúc**. Bạn cần chỉnh sửa file `run_kaggle.py` để chạy theo từng Batch (VD: 5 video một lần) và xóa sạch thư mục `output` sau khi đã đưa vector vào Qdrant.

Chạy Pipeline:
```bash
!python run_kaggle.py
```
Quá trình này sẽ xử lý tự động, in log ra màn hình và giải phóng GPU liên tục sau mỗi Stage.

---

## 3. Hướng dẫn Test (Kiểm thử)

Hệ thống đã có sẵn các test case mẫu trong thư mục `tests/` dùng thư viện `pytest`.

### 3.1. Cài đặt Pytest
```bash
!pip install pytest
```

### 3.2. Cách Test từng phần (Unit Tests)
*Note: Các kịch bản test này mô phỏng các truy vấn phức tạp của AIC 2026.*

**Chạy toàn bộ kịch bản test E2E (KIS, TRAKE, QA):**
```bash
!pytest tests/test_e2e_queries.py -v
```

**Chi tiết các Test Case đang được bảo vệ:**
- `test_kis_query`: Kiểm tra khả năng nhận diện truy vấn Known-Item Search. Hệ thống sẽ test việc parse Text Query và gọi hàm search Vector trên Qdrant.
- `test_trake_query`: Kiểm tra xem hệ thống có xử lý mảng $N$ truy vấn độc lập cho $N$ mốc thời gian ($E_1 \rightarrow E_4$) hay không.
- `test_qa_search_context`: Kiểm tra việc tìm Top-K Video Context trước khi giao cho VLM đọc OCR và sinh câu trả lời.

### 3.3. Tự test với Query của riêng bạn
Bạn có thể mở `tests/test_e2e_queries.py`, thay đổi đoạn Text trong biến `query = "..."` thành câu hỏi của bạn để xem cỗ máy Retriever trả về `video_id` và `frame_index` như thế nào.

---

## 4. Tùy chỉnh Nâng cao (Hyperparameters)

Mở file `src/config.py` để tinh chỉnh:
- **`TARGET_FPS` = 30**: Giữ nguyên thông số này để frame index khi nộp bài thi khớp 100% với video BTC.
- **`DINO_SIMILARITY_THRESHOLD` = 0.65**: Nếu muốn lấy nhiều ảnh Keyframe chi tiết hơn, hãy tăng số này lên (0.8). Nếu muốn nén dung lượng lại, giảm xuống (0.5).
- **`IMAGE_RESIZE_PX` = 512**: Ảnh đẩy vào mô hình Qwen3-VL sẽ bị resize về 512x512. Để cao hơn sẽ nét hơn (bắt OCR chữ nhỏ tốt hơn) nhưng tiêu tốn RAM theo hàm mũ.

---

## 5. Chiến lược Tăng tốc hệ thống (Acceleration Strategies)

Nếu bạn có quỹ thời gian nộp bài eo hẹp và cần xử lý hàng ngàn video nhanh nhất có thể, hãy áp dụng các kỹ thuật sau để tăng tốc hệ thống:

1. **Multiprocessing cho VAD (Sử dụng 100% CPU)**:
   - Quá trình dùng FFmpeg cắt video đang chạy tuần tự. Bạn có thể sử dụng `concurrent.futures.ProcessPoolExecutor` trong file `vad_splitter.py` để cắt nhiều đoạn video cùng lúc. Băng thông ổ cứng và số nhân CPU (Kaggle có 4 lõi) sẽ quyết định tốc độ.
2. **Lượng tử hóa Mô hình (Quantization 4-bit / 8-bit)**:
   - Mô hình `Qwen3.5-2B` đang ngốn ~8GB VRAM ở chuẩn FP16. Nếu cài thêm thư viện `bitsandbytes` và cấu hình `load_in_4bit=True` trong `segment_captioner.py`, dung lượng VRAM sẽ giảm xuống chỉ còn ~3GB. Nhờ đó bạn có thể tăng `BATCH_SIZE` lên gấp 3 lần, tăng tốc độ xử lý video lên đáng kể.
3. **Sử dụng Flash Attention 2**:
   - Khi load các mô hình LLM, hãy thêm tham số `attn_implementation="flash_attention_2"` thay vì `"sdpa"`. Thư viện này tối ưu hóa ma trận self-attention trên GPU T4 của Kaggle, giúp tốc độ sinh chữ (generate) nhanh hơn 20-30%.
4. **Compile mô hình bằng TensorRT / ONNX**:
   - Các mô hình tĩnh như `DINOv2` và `Qwen3-VL-Embedder` có thể được biên dịch (compile) sang định dạng TensorRT. Tốc độ nhúng Vector sẽ tăng gấp 3-5 lần so với chạy PyTorch native.

---

## 6. Đề xuất Kiến trúc Nâng cao (Advanced Proposals)

Để lọt vào **Top bảng xếp hạng (Leaderboard) AIC**, hệ thống của bạn sẽ cần độ chính xác cực cao để phân biệt các chi tiết tinh vi. Dưới đây là các định hướng nâng cấp mã nguồn:

### A. RRF (Reciprocal Rank Fusion) cho Text-Image Match
Hiện tại KIS Search tìm kiếm đồng thời trên Text Vector và Image Vector dựa vào điểm Cosine trung bình. Tuy nhiên, thay vì lấy điểm Cosine, hãy áp dụng thuật toán **RRF**. Thuật toán này cộng gộp thứ hạng (Rank) của khung hình đó ở cả 2 không gian Vector lại với nhau (Ví dụ: `Score = 1/(k + rank_text) + 1/(k + rank_image)`). RRF được chứng minh là tăng độ chính xác lên 10-15% trong các hệ thống lai (Hybrid Search).

### B. Multi-Crop Object Detection (Nhận diện vật thể nhỏ)
Nhiều truy vấn yêu cầu tìm vật thể rất nhỏ (Ví dụ: "Biển báo cấm rẽ trái ở góc màn hình"). Vector của một ảnh toàn cảnh sẽ bị nhiễu và dễ bỏ qua biển báo này. 
**Đề xuất:** Cài thêm một mô hình Zero-shot Object Detection (như **YOLO-World** hoặc **GroundingDINO**). Với mỗi Keyframe, crop các vật thể con ra, sau đó dùng Qwen-Embedder nhúng các ảnh Crop này thành các vector độc lập đính kèm với `frame_index` gốc.

### C. Khôi phục luồng Whisper ASR (Âm thanh)
Đối với các câu hỏi Q&A liên quan đến lời nói của nhân vật (Ví dụ: "MC đang đọc câu thơ gì?"), mô hình hình ảnh sẽ hoàn toàn "mù". Hãy bật lại cờ `USE_TRANSCRIPT_BRANCH = True` trong file config để tích hợp PhoASR vào kho dữ liệu Qdrant.

### D. Tối ưu hóa Thuật toán Cửa sổ trượt (Sliding Window) cho TRAKE
Truy vấn TRAKE yêu cầu 4 khoảnh khắc liên tiếp. Đôi khi DINOv2 trích xuất thiếu mất 1 khoảnh khắc quan trọng do ngưỡng Similar quá cao. Hãy nội suy (interpolate) các khung hình giữa 2 keyframe nếu điểm số của chuỗi sự kiện bị đứt gãy.
