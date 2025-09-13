# ---- Example invocation ----
from src.trainer import train

if __name__ == "__main__":
    DB = "data/pony_prompts.db"
    MODEL = "dphn/Dolphin-Mistral-24B-Venice-Edition"  # replace with the exact HF id you use
    LORA_OUT = "models/mistral24b_lora_danbooru"

    train(
        db_path=DB,
        model_name_or_path=MODEL,
        lora_path=LORA_OUT,
        n_epochs=3,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,  # simulates batch of 8
        learning_rate=2e-4,
        max_seq_len=512,
        use_4bit=True,
    )


