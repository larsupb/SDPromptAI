import argparse

from src.training.hf_trainer import train

if __name__ == "__main__":
    DB = "data/pony_prompts.db"
    MODEL = "/raid/oobabooga/models/hf/dphn/Dolphin-Mistral-24B-Venice-Edition"  # replace with the exact HF id you use
    LORA_OUT = "models/loras/mistral24b_lora_danbooru_v5"

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train(db_path=DB, model_name_or_path=MODEL, lora_path=LORA_OUT,
          n_epochs=3, max_seq_len=512,
          lora_r=16, lora_alpha=32, lora_dropout=0.05)


