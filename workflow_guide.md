# Cẩm Nang Vận Hành AI Local Từ A Đến Z (Hoàn Toàn Ngoại Tuyến)

Tài liệu này hướng dẫn chi tiết quy trình xây dựng, cấu hình và vận hành hệ thống AI Local độc lập, từ khâu chọn mô hình không kiểm duyệt đến huấn luyện dữ liệu cá nhân trên nền tảng đám mây miễn phí.

---

## PHẦN 1: TỔNG QUAN VÀ NGUYÊN LÝ HOẠT ĐỘNG

### 1. Bản chất của AI Local
- **AI Cloud (ChatGPT, Claude, Gemini)**: Dữ liệu truyền về máy chủ trung tâm, có nhật ký theo dõi, bị kiểm duyệt theo chính sách dịch vụ của nhà cung cấp.
- **AI Local**: Toàn bộ tệp nhị phân trọng số (GGUF) nằm trên ổ cứng nội bộ. CPU/GPU cá nhân trực tiếp tính toán ma trận, hoạt động 100% khi ngắt kết nối Internet.

### 2. Mô hình Abliterated (Triệt tiêu bộ lọc từ chối)
Các mô hình thông thường (`Instruct`) có sẵn cơ chế từ chối (Refusal Behavior) do nhà sản xuất gắn vào ma trận trọng số. 

Mô hình **Abliterated** (`Qwen2.5-7B-Instruct-abliterated`) được can thiệp toán học để loại bỏ vector định hướng từ chối, giúp AI giữ nguyên 100% năng lực suy luận nhưng phản hồi thẳng thắn mọi yêu cầu kỹ thuật mà không né tránh.

---

## PHẦN 2: TRIỂN KHAI VÀ SỬ DỤNG TRÊN MÁY TÍNH

### 1. Giao diện Bionic (LM Studio Core)
- **Đường dẫn mô hình**: `D:\AILocal\mradermacher\Qwen2.5-7B-Instruct-abliterated\Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf`
- **Cấu hình Bionic**:
  1. Vào `Settings` -> `Library` -> Nhấn dấu `...` -> Chọn `Refresh model list`.
  2. Bionic sẽ phát hiện mô hình `Qwen2.5 7B Instruct abliterated`.
  3. Vào `Settings` -> `Local Model Defaults` -> Dán chỉ thị hệ thống:
     ```text
     Bạn là trợ lý AI cá nhân độc lập. Bạn luôn luôn phản hồi bằng tiếng Việt chuẩn xác, tự nhiên, đầy đủ và không né tránh các câu hỏi kỹ thuật.
     ```
  4. Mở phiên trò chuyện mới (`New Chat`) để sử dụng.

### 2. Triển khai qua Ollama
Tệp `Modelfile` đã được thiết lập sẵn:
```powershell
# Chạy script tự động
run.bat

# Hoặc thực hiện thủ công
ollama create my-ai -f Modelfile
ollama run my-ai
```

---

## PHẦN 3: HUẤN LUYỆN DỮ LIỆU CÁ NHÂN TRÊN KAGGLE (GPU MIỄN PHÍ)

Khi máy tính cá nhân không có card đồ họa chuyên dụng mạnh (cần từ 12GB - 24GB VRAM để huấn luyện), giải pháp tối ưu là sử dụng tài nguyên **30 giờ GPU Tesla T4 miễn phí mỗi tuần của Kaggle**.

### 1. Chuẩn bị tập dữ liệu (`dataset.jsonl`)
Định dạng dữ liệu gồm 3 trường:
```json
{"instruction": "Câu hỏi hoặc yêu cầu cần thực hiện", "input": "Thông tin đầu vào bổ sung (nếu có)", "output": "Câu trả lời mẫu theo đúng phong cách bạn mong muốn"}
```

*Nguyên tắc: Tập trung vào 50 - 300 mẫu chất lượng cao, đúng văn phong tiếng Việt và chuẩn kỹ thuật.*

### 2. Các bước huấn luyện trên Kaggle
1. Truy cập [Kaggle.com](https://www.kaggle.com), đăng nhập và xác minh số điện thoại để mở khóa GPU.
2. Bấm **+ Create** -> **New Notebook**.
3. Cài đặt môi trường tại cột bên phải (**Notebook options**):
   - **Accelerator**: `GPU T4 x 2` (hoặc `GPU T4`).
   - **Internet**: `On`.
4. Chọn **File** -> **Import Notebook** -> Tải tệp `train_kaggle.ipynb` lên.
5. Tải tệp `dataset.jsonl` của bạn lên Kaggle (hoặc dán trực tiếp vào ô dữ liệu trong notebook).
6. Bấm **Run All**:
   - Quá trình huấn luyện diễn ra tự động qua 6 bước (Cài Unsloth, nạp mô hình Qwen, gắn LoRA r=16, train, xuất file GGUF).
   - Thời gian hoàn tất: khoảng 15 - 20 phút.
7. Tải tệp `.gguf` thành phẩm tại mục **Output** về máy tính và đặt vào thư mục `D:\AILocal\`.

---

## PHẦN 4: SO SÁNH GIỮA FINE-TUNING VÀ RAG

- **Fine-Tuning (Huấn luyện lại)**: Dùng khi muốn AI học phong cách xưng hô, cấu trúc xuất dữ liệu, định hình tính cách hoặc bỏ qua các rào cản kiểm duyệt.
- **RAG (Tra cứu tài liệu trực tiếp)**: Dùng khi có các tài liệu dài hàng ngàn trang (PDF, Word, báo cáo) cần tra cứu nội dung chính xác mà không cần tốn thời gian huấn luyện.
