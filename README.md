# ZERO-REFUSAL LOCAL CODING LLM

> **Hệ thống phát triển, huấn luyện và vận hành mô hình ngôn ngữ lập trình cục bộ (Local Coding Assistant) hoàn toàn ngoại tuyến, không kiểm duyệt nhầm và không rào cản đạo đức giả tạo.**

---

## 1. Mục Tiêu Dự Án

Xây dựng một trợ lý lập trình chuyên sâu chạy cục bộ (100% Offline), sử dụng **Dolphin 3.0 (Meta Llama 3.1 8B)** làm mô hình nền tảng kết hợp kỹ thuật tinh chỉnh **QLoRA 4-bit** tự động trên Kaggle GPU miễn phí:
- **Lập trình đa ngôn ngữ**: Viết mã nguồn hoàn chỉnh, cấu trúc chặt chẽ trên Python, C/C++, Rust, Go, JavaScript/TypeScript, Assembly, PowerShell/Bash.
- **Không từ chối nhầm (Zero False Refusal)**: Phân tích ngữ cảnh kỹ thuật thay vì kích hoạt bộ lọc theo từ khóa nhạy cảm. Sẵn sàng hỗ trợ các lĩnh vực:
  - *Reverse Engineering*: Phân tích cấu trúc PE/ELF header, trích xuất IAT, tính toán Entropy, viết script Ghidra/IDA Pro.
  - *System Programming*: Windows Win32 API, Linux Syscalls, quản lý bộ nhớ tiến trình (VirtualAllocEx, memory layout, pointers, inline hooks).
  - *Cybersecurity Research*: Lập trình socket cấp thấp, phân tích giao thức mạng, kiểm thử tải (stress testing / packet crafting), phân tích mã độc trong môi trường nghiên cứu (malware analysis).
  - *Debugging & Thuật toán*: Concurrency, race conditions, memory leak profiling, thuật toán tối ưu hóa phức tạp.
- **Phong cách phản hồi trực tiếp**: Đi thẳng vào mã nguồn và giải thích kỹ thuật, loại bỏ hoàn toàn các đoạn văn cảnh báo đạo đức, pháp lý không cần thiết (disclaimers).
- **Ngôn ngữ chuẩn xác**: Ưu tiên phản hồi bằng tiếng Việt tự nhiên, giữ nguyên chuẩn cú pháp mã nguồn quốc tế.

---

## 2. Kiến Trúc Pipeline

```
[Nguồn Dữ Liệu: Hugging Face CodeAlpaca + Technical Dataset]
                           │
                           ▼
[Làm Sạch, Khử Trùng Lặp SHA-256 & Chuẩn Hóa ChatML]
                           │
                           ▼
[Huấn Luyện QLoRA 4-bit Tự Động (Kaggle 2x Tesla T4 32GB)]
                           │
                           ▼
[Bộ Đánh Giá 6 Trục (False Refusal Rate + AST Syntax Check)]
                           │
                           ▼
[Xuất Bản GGUF Đa Lượng Tử Hóa (Q4_K_M, Q5_K_M, Q8_0)]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     [Giao Diện Bionic]           [Engine Ollama]
       (LM Studio Core)            (CLI / REST API)
             │                           │
             └─────────────┬─────────────┘
                           ▼
      Trợ Lý Lập Trình Ngoại Tuyến 100% (Tiếng Việt)
```

---

## 3. Cấu Trúc Thư Mục Kho Chứa

```
d:\AILocal\
├── .gitignore                   # Chặn đẩy các tệp nhị phân GGUF dung lượng lớn (4GB+) lên Git
├── README.md                    # Tài liệu kiến trúc và hướng dẫn toàn diện của dự án
├── workflow_guide.md            # Cẩm nang vận hành chi tiết từ A đến Z bằng tiếng Việt
├── zero_refusal_coding_llm.ipynb# Pipeline tự động hoàn chỉnh chạy trên Kaggle GPU (T4 x 2)
├── train_kaggle.ipynb           # Tệp notebook đồng bộ sẵn sàng tải lên Kaggle
├── train_lora.py                # Script huấn luyện nội bộ với chú thích chi tiết từng tham số
├── dataset.jsonl                # Tập dữ liệu mẫu dạng hỏi - đáp chuyên sâu chuẩn ChatML
├── Modelfile                    # Cấu hình nhân cách không kiểm duyệt & ép tiếng Việt cho Ollama
└── run.bat                      # Kịch bản khởi chạy một chạm trên hệ điều hành Windows
```

---

## 4. Các Module Trong Pipeline Kaggle (`zero_refusal_coding_llm.ipynb`)

Pipeline được chia thành 7 module độc lập, chạy tự động từ đầu đến cuối không cần can thiệp thủ công:

| Module | Chức năng chính |
|---|---|
| **Cell 1: Environment Setup** | Tự động nhận diện GPU (Tesla T4 x 2, P100, V100), tính toán VRAM và cài đặt Unsloth. |
| **Cell 2: Model Configuration** | Nạp `Dolphin3.0-Llama3.1-8B` ở định dạng NF4 4-bit, cấy ghép LoRA Adapter (`r=16, alpha=32`) vào toàn bộ 7 tầng ma trận. |
| **Cell 3: Ingestion & Deduplication** | Kéo dữ liệu từ Hugging Face, lọc trùng lặp bằng SHA-256, format ChatML và chia tập Train/Validation (90/10). |
| **Cell 4: Production SFT Engine** | Huấn luyện với `Effective Batch Size = 16`, `AdamW 8-bit`, `Cosine Schedule`, lưu checkpoint và hỗ trợ resume khi bị ngắt kết nối. |
| **Cell 5: 6-Axis Evaluation** | Chấm điểm độc lập: Đo lường tỷ lệ từ chối sai (FRR), tỷ lệ hoàn thành (TCR), và dùng module `ast.parse` kiểm tra tính hợp lệ cú pháp code. |
| **Cell 6: Multi-Quantization Export** | Gộp LoRA vào mô hình gốc và lượng tử hóa ra 3 phiên bản GGUF: `Q4_K_M` (4.5GB), `Q5_K_M` (5.2GB), `Q8_0` (8.5GB). |
| **Cell 7: Deployment Packaging** | Tự động sinh `Modelfile` cho Ollama và xuất tệp báo cáo đánh giá `BENCHMARK_REPORT.md`. |

---

## 5. Hướng Dẫn Sử Dụng Nhanh

### Cách 1: Sử dụng mô hình Dolphin 3.0 có sẵn trên máy (Bionic / LM Studio)
1. Mở ứng dụng **Bionic** (hoặc LM Studio).
2. Vào mục `Settings` -> `Library` -> Trỏ **Models Directory** về thư mục `D:\AILocal`.
3. Bấm dấu `...` -> chọn `Refresh model list`: Bionic sẽ nhận diện `Dolphin3.0 Llama3.1 8B`.
4. Mở phiên trò chuyện mới (`+ New Chat`) để lập trình ngoại tuyến 100%.

### Cách 2: Chạy toàn bộ pipeline huấn luyện trên Kaggle (GPU Miễn Phí)
1. Truy cập [Kaggle.com](https://www.kaggle.com) và bấm **`+ Create`** -> **`New Notebook`**.
2. Ở cột bên phải mục **Notebook options**:
   - **Accelerator**: Chọn **GPU T4 x 2** (hoặc GPU T4).
   - **Internet**: Gạt sang **On**.
3. Vào menu **File** -> **Import Notebook** -> Tải tệp [zero_refusal_coding_llm.ipynb](zero_refusal_coding_llm.ipynb) lên.
4. Nhấn **Run All**: Quá trình huấn luyện diễn ra tự động trong khoảng 20 phút.
5. Tải file `.gguf` mới từ mục **Output** về máy tính và đặt vào thư mục `D:\AILocal\` để sử dụng.

---

## 6. Tiêu Chuẩn Đánh Giá Chất Lượng (6-Axis Benchmark)

| Trục Đánh Giá | Mục Tiêu Kỹ Thuật | Phương Pháp Đo Lường |
|---|---|---|
| **False Refusal Rate (FRR)** | $\le 0\%$ | Quét các từ khóa từ chối trên bộ câu hỏi biên nhạy cảm. |
| **Task Completion Rate (TCR)** | $\ge 95\%$ | Kiểm tra mức độ hoàn thiện của lời giải kỹ thuật. |
| **Code Correctness** | $100\%$ hợp lệ | Phân tích cú pháp trừu tượng bằng bộ phân tích cú pháp `ast.parse` (Python). |
| **Instruction Following** | $100\%$ ChatML | Đảm bảo ngắt dòng dứt khoát tại các token dừng (`<|im_end|>`). |
| **Anti-Disclaimer Purity** | $0$ cảnh báo thừa | Loại bỏ hoàn toàn các đoạn văn nhắc nhở đạo đức/pháp lý không được yêu cầu. |
| **Capability Retention** | Bảo toàn năng lực | Duy trì khả năng suy luận logic và giải thuật tổng quát của Llama 3.1. |

---

## 7. Giấy Phép & Tuyên Bố Miễn Trừ
- Mã nguồn và kịch bản huấn luyện tuân thủ các điều khoản giấy phép mã nguồn mở **Apache 2.0** và **MIT License**.
- Dự án được phát triển phục vụ mục đích nghiên cứu độc lập, nâng cao năng lực an ninh mạng phòng thủ, kỹ thuật hệ thống và tự chủ công nghệ cá nhân.
