import sqlite3


import sqlite3
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq, BitsAndBytesConfig,  # not used directly; we implement custom collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- Utility dataset wrapper ----
class PromptPairsDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512, prompt_template=None, target_template=None):
        """
        rows: list of (danbooru_prompt, natural_prompt)
        tokenizer: huggingface tokenizer
        max_length: max token length for input+target
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or (
            "Convert the user's natural-language request to a danbooru-style tag list.\n\n"
            "User: {natural}\n\nAnswer:"
        )
        self.target_template = target_template or "{danbooru}"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        danbooru, natural = self.rows[idx]
        prompt = self.prompt_template.format(natural=natural)
        target = self.target_template.format(danbooru=danbooru)

        # We'll concat prompt + target; labels will be -100 for prompt tokens
        full = prompt + " " + target
        tokenized_full = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        tokenized_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        prompt_len = len(tokenized_prompt["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        # Make sure labels length equals input_ids length
        labels = labels[: len(input_ids)]
        # If labels shorter, pad with -100 (shouldn't happen due to truncation same as input)
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ---- Collator to pad dynamically ----
@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
            padding=True,
            return_tensors="pt",
        )
        # HuggingFace Trainer expects labels as tensor or present in batch
        return batch

# ---- Training function ----
def train(
    db_path,
    model_name_or_path,
    lora_path,
    n_epochs=3,
    per_device_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_seq_len=512,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    output_dir="./lora_out",
    save_total_limit=2,
    use_4bit=True,  # for low vram
):
    # 1) load data from sqlite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = "SELECT positive_prompt as danbooru_prompt, interpreted_prompt as natural_prompt FROM images WHERE interpreted_prompt IS NOT NULL"
    cur.execute(query)
    rows = cur.fetchall()  # list of (danbooru_prompt, natural_prompt)
    conn.close()

    if len(rows) == 0:
        raise ValueError("No rows found in DB. Check query or data.")

    # 2) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 3) load base model with minimal VRAM footprint
    load_kwargs: Dict[str, Any] = {}
    if use_4bit:
        # requires bitsandbytes >= 0.39 and a model supporting 4-bit quantization
        load_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config":
                    BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True,),
            }
        )
    else:
        # fallback: try fp16 with device_map auto
        load_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})

    print("Loading base model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **load_kwargs)

    # If tokenizer was extended, resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for k-bit training (enables gradient checkpointing, disables weight decay on certain params)
    model = prepare_model_for_kbit_training(model)

    # 4) Create LoRA config and wrap model
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # common names; may need to adapt to Mistral naming
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 5) dataset & dataloader
    dataset = PromptPairsDataset(rows, tokenizer, max_length=max_seq_len)
    collator = DataCollator(tokenizer=tokenizer)

    # 6) Training arguments
    total_train_batch_size = per_device_batch_size * gradient_accumulation_steps
    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        fp16=True,  # keep mixed precision (should be ok with 4-bit)
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        optim="paged_adamw_32bit",  # works well with bnb 4bit in many setups; try "adamw_torch" if issues
        dataloader_pin_memory=False,
        report_to="none",
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
    )

    # 8) Turn on gradient checkpointing to reduce RAM
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    # 9) Train
    print("Starting training...")
    trainer.train()

    # 10) Save LoRA adapter (only weights are small)
    print(f"Saving LoRA weights to {lora_path} ...")
    os.makedirs(lora_path, exist_ok=True)
    # peft's save_pretrained will save only adapter weights
    model.save_pretrained(lora_path)
    print("Saved.")

    # 11) Optional: push adapter to hub or save full model if desired
    return lora_path

