"""
====================================================================================
# train_lora.py — BẢNG THIẾT LẬP VÀ HUẤN LUYỆN (FINE-TUNING) LOCAL LLM VỚI QLORA
# Toàn bộ tham số bên dưới được giải thích chi tiết mục đích và ý nghĩa kỹ thuật
====================================================================================
"""

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==================================================================================
# PHẦN 1: CẤU HÌNH MÔ HÌNH NỀN TẢNG (BASE MODEL)
# ==================================================================================

# 1. MODEL_NAME: Tên mô hình cơ sở trên HuggingFace mà bạn muốn lấy làm "nền" để dạy thêm.
# - "unsloth/Llama-3.2-3B-Instruct": Bản 3 tỷ tham số, cực nhẹ, nạp siêu nhanh (phù hợp máy 4GB VRAM).
# - "unsloth/Qwen2.5-7B-Instruct": Bản 7 tỷ tham số, thông minh hơn, giỏi tiếng Việt và code (khuyên dùng).
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"

# 2. MAX_SEQ_LENGTH: Độ dài ngữ cảnh tối đa (tính theo Token, 1 token ~ 0.75 từ tiếng Việt) trong khi train.
# - 2048: Đủ cho hầu hết các đoạn văn bản hỏi đáp dài từ 1000 - 1500 từ.
# - Nếu bạn tăng lên 4096 hoặc 8192, mô hình nhớ được ngữ cảnh dài hơn nhưng sẽ ngốn thêm nhiều VRAM.
MAX_SEQ_LENGTH = 2048

# 3. DTYPE: Kiểu dữ liệu biểu diễn số thực của trọng số mô hình.
# - None: Để thư viện Unsloth tự nhận diện tối ưu theo card đồ họa của bạn.
# - Nếu GPU là dòng mới (RTX 3000/4000 series): tự dùng bfloat16 (Bfloat16 ổn định hơn, tránh tràn số).
# - Nếu GPU cũ hơn (GTX 1600/RTX 2000): tự dùng float16.
DTYPE = None

# 4. LOAD_IN_4BIT: Kỹ thuật cốt lõi của QLoRA (Lượng tử hóa 4-bit).
# - True: Nén 99% mô hình gốc xuống dạng 4-bit (NF4) ngay khi nạp vào card màn hình.
# - Giúp giảm dung lượng VRAM từ 16GB xuống chỉ còn ~5GB, cho phép máy cá nhân train được mà không tràn bộ nhớ.
LOAD_IN_4BIT = True


def main():
    print(f"[*] Đang tải mô hình cơ sở: {MODEL_NAME}...")
    from unsloth import FastLanguageModel

    # Khởi tạo mô hình và bộ từ điển mã hóa ngôn ngữ (Tokenizer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # ==============================================================================
    # PHẦN 2: THIẾT LẬP LORA ADAPTER (LỚP VI MẠCH THÍCH ỨNG)
    # Thay vì sửa cả 7 tỷ tham số, LoRA gắn thêm ma trận phụ nhỏ để học tri thức mới.
    # ==============================================================================
    model = FastLanguageModel.get_peft_model(
        model,
        # 1. r (Rank - Hạng của ma trận LoRA):
        # - Quyết định độ dày của "lớp não phụ". Giá trị phổ biến: 8, 16, 32, 64.
        # - r = 16: Mức cân bằng vàng. Đủ sức học phong cách, ngữ điệu và kiến thức mới mà không làm nặng máy.
        # - Nếu r càng cao (ví dụ 64), mô hình học được nhiều chi tiết phức tạp hơn nhưng tốn thêm VRAM.
        r=16,

        # 2. target_modules: Danh sách các tầng trọng số trong mạng nơ-ron được cấy ghép LoRA:
        # - q_proj, k_proj, v_proj, o_proj: Các tầng cơ chế Chú Ý (Self-Attention) giúp AI hiểu mối liên kết giữa các từ.
        # - gate_proj, up_proj, down_proj: Các tầng mạng Nơ-ron Tiến (MLP/Feed-Forward) giúp AI lưu trữ tri thức sâu.
        # - Gắn LoRA vào toàn bộ 7 tầng này đảm bảo AI học toàn diện cả cách hiểu câu lẫn tri thức mới.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],

        # 3. lora_alpha: Hệ số khuếch đại (Scaling Factor) cho các ma trận LoRA.
        # - Thường đặt bằng 2 * r (ở đây r=16 thì lora_alpha=32).
        # - Giữ cho tỉ lệ giữa kiến thức mới học và kiến thức nền tảng của mô hình luôn cân bằng.
        lora_alpha=32,

        # 4. lora_dropout: Tỉ lệ ngẫu nhiên "tắt" bớt các nơ-ron trong khi train để chống học vẹt (Overfitting).
        # - 0: Unsloth khuyên dùng 0 vì thuật toán của Unsloth đã được tối ưu tốc độ trực tiếp trên phần cứng.
        lora_dropout=0,

        # 5. bias: Huấn luyện thêm các số hạng sai số (bias) hay không.
        # - "none": Giữ nguyên bias ban đầu, tiết kiệm bộ nhớ nhất và không làm sai lệch mô hình gốc.
        bias="none",

        # 6. use_gradient_checkpointing: Kỹ thuật đánh đổi tốc độ CPU để tiết kiệm VRAM.
        # - "unsloth": Chỉ lưu lại một phần ma trận kích hoạt, các phần khác tự tính lại khi cần.
        # - Giúp giảm thêm 30% VRAM tiêu thụ trong quá trình huấn luyện.
        use_gradient_checkpointing="unsloth",

        # 7. random_state: Hạt giống ngẫu nhiên (Seed) để đảm bảo kết quả huấn luyện có thể lặp lại chính xác.
        random_state=3407,
    )

    # ==============================================================================
    # PHẦN 3: ĐỊNH DẠNG KHUÔN MẪU DỮ LIỆU (PROMPT TEMPLATE)
    # Chuyển đổi file dataset.jsonl thành định dạng mà mô hình quen đọc (Alpaca format)
    # ==============================================================================
    alpaca_prompt = """Dưới đây là một hướng dẫn mô tả nhiệm vụ. Hãy viết phản hồi hoàn thành yêu cầu một cách chính xác.

### Hướng dẫn:
{}

### Đầu vào:
{}

### Phản hồi:
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Ghép từng dòng dữ liệu vào khuôn mẫu và thêm token kết thúc câu (EOS) để AI biết điểm dừng
            text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    print(f"[*] Đang nạp tập dữ liệu từ: {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # ==============================================================================
    # PHẦN 4: CẤU HÌNH THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)
    # Điều chỉnh nhịp độ, tốc độ học và quản lý tài nguyên tính toán của card đồ họa
    # ==============================================================================
    training_args = TrainingArguments(
        # 1. per_device_train_batch_size: Số lượng mẫu dữ liệu nạp cùng một lúc vào GPU trong 1 tích tắc.
        # - 2: Giá trị an toàn cho card 4GB - 8GB VRAM.
        # - Nếu đặt quá cao (ví dụ 8 hoặc 16), GPU sẽ báo lỗi CUDA Out of Memory (tràn VRAM).
        per_device_train_batch_size=2,

        # 2. gradient_accumulation_steps: Số bước tích lũy độ dốc trước khi cập nhật trọng số.
        # - Kích thước Batch hiệu dụng thực tế = per_device_train_batch_size * gradient_accumulation_steps
        # - Ở đây: 2 * 4 = 8 mẫu mỗi lần điều chỉnh não bộ.
        # - Giúp máy tính cá nhân đạt hiệu quả ổn định như máy chủ lớn mà không tốn thêm VRAM.
        gradient_accumulation_steps=4,

        # 3. warmup_steps: Số bước "khởi động nhẹ" ban đầu.
        # - Trong 5 bước đầu, tốc độ học (learning rate) tăng từ 0 lên cực đại một cách từ từ.
        # - Giúp não bộ mô hình không bị "sốc" làm hỏng các kiến thức cũ có sẵn.
        warmup_steps=5,

        # 4. max_steps: Tổng số bước tối ưu hóa của toàn bộ quá trình train.
        # - 60: Phù hợp để thử nghiệm nhanh với tập dữ liệu nhỏ (dưới 50 mẫu).
        # - Khi bạn có tập dữ liệu lớn (vài trăm đến vài nghìn mẫu), hãy tăng lên 300 - 1000 bước.
        max_steps=60,

        # 5. learning_rate (Tốc độ học): Độ lớn của mỗi bước nhảy khi điều chỉnh trọng số ma trận.
        # - 2e-4 (tức 0.0002): Tốc độ tiêu chuẩn vàng cho huấn luyện LoRA.
        # - Nếu quá lớn (ví dụ 0.01): AI học nhanh nhưng dễ bị "ngáo", sinh chữ lung tung.
        # - Nếu quá nhỏ (ví dụ 1e-6): AI học rất chậm, không kịp nhớ dữ liệu mới.
        learning_rate=2e-4,

        # 6. fp16 / bf16: Sử dụng số thực nửa độ chính xác (16-bit) để tăng tốc độ tính toán.
        # - Tự động bật bf16 nếu phần cứng hỗ trợ, ngược lại dùng fp16.
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        # 7. logging_steps: Tần suất in thông tin tiến độ ra màn hình.
        # - Cứ mỗi 5 bước train, chương trình sẽ in chỉ số hàm mất mát (loss).
        # - Chỉ số Loss giảm dần theo thời gian chứng tỏ AI đang học đúng hướng.
        logging_steps=5,

        # 8. optim: Thuật toán tối ưu hóa điều chỉnh trọng số.
        # - "adamw_8bit": Phiên bản nén 8-bit của thuật toán tối ưu AdamW.
        # - Tiết kiệm tới 75% dung lượng VRAM dành riêng cho bộ nhớ trạng thái tối ưu (Optimizer States).
        optim="adamw_8bit",

        # 9. weight_decay: Hệ số suy giảm trọng số (chống Overfitting).
        # - 0.01: Phạt nhẹ các trọng số quá lớn, giúp mô hình giữ được sự tổng quát, không học vẹt từng câu chữ.
        weight_decay=0.01,

        # 10. lr_scheduler_type: Quy luật giảm dần tốc độ học theo thời gian.
        # - "linear": Tốc độ học giảm dần đều về 0 khi sắp hoàn thành buổi train để cố định tri thức.
        lr_scheduler_type="linear",

        # 11. seed: Hạt giống ngẫu nhiên của bộ phân chia dữ liệu.
        seed=3407,

        # 12. output_dir: Thư mục tạm lưu các điểm kiểm tra (Checkpoints) trong quá trình train.
        output_dir="outputs",
    )

    # ==============================================================================
    # PHẦN 5: BỘ ĐIỀU PHỐI HUẤN LUYỆN (SFT TRAINER)
    # ==============================================================================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,  # Số luồng CPU dùng để xử lý văn bản đầu vào
        packing=False,       # False: Giữ từng câu hỏi riêng biệt, không nhồi nhét nhiều câu vào 1 dòng
        args=training_args,
    )

    print("[*] Bắt đầu quá trình huấn luyện (Fine-tuning)...")
    trainer.train()

    # ==============================================================================
    # PHẦN 6: LƯU TRỌNG SỐ VÀ XUẤT RA TỆP GGUF DÙNG OFFLINE
    # ==============================================================================
    # 1. Lưu riêng lớp ma trận LoRA vừa học được vào thư mục custom_lora_model
    output_lora = "custom_lora_model"
    print(f"[*] Đang lưu các trọng số LoRA vào thư mục: {output_lora}...")
    model.save_pretrained(output_lora)
    tokenizer.save_pretrained(output_lora)

    # 2. Hợp nhất LoRA vào mô hình gốc và nén trực tiếp thành file GGUF 4-bit (q4_k_m)
    # - "custom_ai_gguf": Tên thư mục chứa file .gguf đầu ra.
    # - quantization_method="q4_k_m": Chuẩn nén 4-bit chất lượng cao, file này bạn có thể ném thẳng vào Bionic/Ollama để chat.
    print("[*] Đang xuất mô hình sang định dạng GGUF q4_k_m để nạp trực tiếp vào Ollama / Bionic...")
    try:
        model.save_pretrained_gguf("custom_ai_gguf", tokenizer, quantization_method="q4_k_m")
        print("[+] Xuất tệp GGUF thành công vào thư mục: custom_ai_gguf!")
    except Exception as e:
        print(f"[!] Thông báo xuất GGUF: {e}")
        print("[i] Bạn có thể hợp nhất LoRA vào mô hình gốc rồi dùng công cụ convert của llama.cpp.")


if __name__ == "__main__":
    main()
