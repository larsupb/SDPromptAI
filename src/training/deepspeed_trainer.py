import json
import os

# import deepspeed
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer

from .utility import load_data
from .datasets import PromptPairsDataset

"""
NOT WORKING: Needs to be rewritten!
"""


# ---- Training function ----
def train(
    db_path,
    model_name_or_path,
    lora_path,
    n_epochs=3,
    max_seq_len=256,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    output_dir="./lora_out",
    save_total_limit=2,
    **kwargs,
):
    # 1) load data
    rows = load_data(db_path)

    # 2) DeepSpeed config
    with open("deepspeed_config.json") as f:
        ds_config = json.load(f)

    # 3) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 4) model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{kwargs['local_rank']}",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    )

    # If tokenizer was extended, resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    # 4) LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for p in model.parameters():
        if p.requires_grad:
            print("Found trainable param:", p.shape)

    # 5) Prepare DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        model_parameters=model.parameters()
    )

    # 6) dataset & dataloader
    dataset = PromptPairsDataset(rows, tokenizer, max_length=max_seq_len)

    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        save_total_limit=save_total_limit,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        deepspeed=ds_config,
        bf16=ds_config["bf16"]["enabled"],
        per_device_train_batch_size=ds_config["train_micro_batch_size_per_gpu"],
        gradient_accumulation_steps=ds_config["gradient_accumulation_steps"],
        weight_decay=ds_config["optimizer"]["params"]["weight_decay"],
    )

    def collate_fn(batch):
        return tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collate_fn,
        args=train_args
    )

    # 8) Train
    trainer.train()

    # 9) Save LoRA adapter (only weights are small)
    print(f"Saving LoRA weights to {lora_path} ...")
    os.makedirs(lora_path, exist_ok=True)
    # peft's save_pretrained will save only adapter weights
    model.save_pretrained(lora_path)
    print("Saved.")

    # 11) Optional: push adapter to hub or save full model if desired
    return lora_path