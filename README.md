# Dự Án AI Local Độc Lập — Tự Chủ & Không Kiểm Duyệt

Dự án cung cấp trọn bộ giải pháp kỹ thuật để tự triển khai, vận hành và tinh chỉnh (fine-tune) một mô hình trí tuệ nhân tạo (LLM) **chạy cục bộ 100% trên máy tính cá nhân**, hoàn toàn ngoại tuyến (offline), bảo mật tuyệt đối dữ liệu và không chịu sự kiểm soát hay rào cản kiểm duyệt đạo đức giả tạo từ các bên thứ ba.

---

## 1. Kiến Trúc Hệ Thống

```
[Dữ Liệu Huấn Luyện (dataset.jsonl)]
                │
                ▼
[Huấn Luyện QLoRA Miễn Phí (train_kaggle.ipynb / train_lora.py)]
                │
                ▼
[Mô Hình Không Kiểm Duyệt (Dolphin 3.0 - Llama 3.1 8B)]
                │
       ┌────────┴────────┐
       ▼                 ▼
[Giao Diện Bionic]  [Engine Ollama]
 (Llama.cpp Core)    (CLI & API)
       │                 │
       └────────┬────────┘
                ▼
  Trò Chuyện Ngoại Tuyến 100% (Tiếng Việt)
```

---

## 2. Cấu Trúc Thư Mục Dự Án

```
d:\AILocal\
├── .gitignore                # Chặn đẩy các file GGUF 4GB+ lên Git
├── README.md                 # Tài liệu tổng quan kiến trúc và hướng dẫn nhanh
├── workflow_guide.md         # Cẩm nang vận hành chi tiết từ A đến Z bằng tiếng Việt
├── dataset.jsonl             # Tập dữ liệu mẫu để dạy phong cách và tri thức riêng
├── train_kaggle.ipynb        # Notebook chạy fine-tune trên Kaggle GPU (T4 x 2) miễn phí
├── train_lora.py             # Script fine-tune QLoRA nội bộ (Unsloth + PyTorch)
├── Modelfile                 # Bản chỉ thị nhân cách & ép buộc tiếng Việt cho Ollama
└── run.bat                   # Kịch bản khởi chạy một chạm trên Windows
```

---

## 3. Đặc Điểm Nổi Bật

1. **Không kiểm duyệt tuyệt đối (Dolphin 3.0 Core)**: Sử dụng mô hình **Dolphin 3.0 (Llama 3.1 8B)** do Eric Hartford / Cognitive Computations phát triển, loại bỏ hoàn toàn các bộ lọc từ khóa và cơ chế từ chối, giúp AI phản hồi trực tiếp các yêu cầu kỹ thuật và phân tích mã nguồn mà không từ chối hay giảng đạo đức.
2. **Ép buộc phản hồi bằng tiếng Việt**: Đã cấu hình chỉ thị hệ thống để mô hình luôn luôn trả lời bằng tiếng Việt tự nhiên, chuẩn xác và mạch lạc.
3. **Phù hợp máy cấu hình phổ thông**:
   - Sử dụng định dạng nén lượng tử **GGUF 4-bit (Q4_K_M)**, dung lượng ~4.58 GB.
   - Vận hành mượt mà trên máy tính 16GB RAM + CPU Ryzen 7 / Intel Core i7.
4. **Huấn luyện miễn phí trên Cloud (Kaggle)**: Sử dụng tài nguyên GPU T4 miễn phí của Kaggle để huấn luyện và xuất tệp mô hình GGUF tải về máy tính.

---

## 4. Hướng Dẫn Sử Dụng Nhanh

### Bước 1: Trò chuyện ngoại tuyến với giao diện Bionic
1. Mở ứng dụng [Bionic](https://bionic-gpt.com) (hoặc LM Studio).
2. Trỏ đường dẫn **Models Directory** trong mục `Library` về thư mục này (`D:\AILocal`).
3. Ứng dụng sẽ tự động nhận diện mô hình `Dolphin3.0 Llama3.1 8B`.
4. Mở khung chat và bắt đầu trò chuyện ngoại tuyến 100%.

### Bước 2: Khởi chạy bằng dòng lệnh (Ollama)
Chạy tệp `run.bat` hoặc gõ:
```powershell
ollama create my-ai -f Modelfile
ollama run my-ai
```

### Bước 3: Tinh chỉnh (Fine-tuning) theo dữ liệu của riêng bạn
1. Bổ sung các cặp câu hỏi - câu trả lời của bạn vào tệp [dataset.jsonl](dataset.jsonl).
2. Tải tệp [train_kaggle.ipynb](train_kaggle.ipynb) lên [Kaggle](https://www.kaggle.com) (chọn Accelerator: GPU T4 x 2).
3. Nhấn **Run All**: Quá trình huấn luyện diễn ra tự động trong khoảng 15-20 phút và xuất ra tệp `.gguf` mới.
4. Tải tệp `.gguf` về máy tính, đặt vào thư mục dự án để sử dụng.

---

## 5. Giấy Phép & Tuyên Bố Miễn Trừ
- Dự án phục vụ mục đích nghiên cứu cá nhân, tự chủ dữ liệu và kỹ thuật phần mềm.
- Mã nguồn và kịch bản huấn luyện tuân thủ các điều khoản giấy phép mã nguồn mở của Apache 2.0 và MIT.
