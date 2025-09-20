import os

import torch

torch.backends.cuda.enable_flash_sdp(True)        # enable FlashAttention kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from ..db import DB
from ..settings import get_settings
from .training_callbacks import SampleEvalCallback
from .utility import load_data, load_danbooru_tags
from .datasets import MultiDataset, DanbooruTagsDataset, PromptPairsDataset


def train(
        db_path, model_name_or_path, lora_path, n_epochs=3, max_seq_len=256, lora_r=8, lora_alpha=16, lora_dropout=0.05,
        output_dir="./workspace/lora_out", save_total_limit=5):
    """
        Train a LoRA adapter on a pre-trained causal language model using 4-bit quantization and optional CPU offloading.

        This method performs the following steps:
            1. Loads training data from the specified database.
            2. Loads and configures a tokenizer, adding a padding token if missing.
            3. Loads a pre-trained model with BitsAndBytes, offloading frozen weights to CPU,
               and places it on the GPU corresponding to the current process.
            4. Resizes token embeddings if the tokenizer was extended.
            5. Enables gradient checkpointing and freezes base model parameters.
            6. Applies LoRA adapters to the attention projection layers for efficient low-rank fine-tuning.
            7. Prepares the model for k-bit training.
            8. Creates a dataset and data collator for causal language modeling.
            9. Configures HuggingFace TrainingArguments for BF16 computation, gradient accumulation, and logging.
           10. Trains the LoRA adapters using HuggingFace Trainer.
           11. Saves the trained LoRA adapter weights to the specified path.

        Args:
            db_path (str): Path to the database containing training prompts.
            model_name_or_path (str): Path or name of the pre-trained HuggingFace model.
            lora_path (str): Directory path to save the trained LoRA adapter weights.
            n_epochs (int, optional): Number of training epochs. Defaults to 3.
            max_seq_len (int, optional): Maximum sequence length for tokenization. Defaults to 256.
            lora_r (int, optional): LoRA rank. Defaults to 8.
            lora_alpha (int, optional): LoRA alpha scaling factor. Defaults to 16.
            lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.05.
            output_dir (str, optional): Directory to store training outputs and checkpoints. Defaults to "./lora_out".
            save_total_limit (int, optional): Maximum number of checkpoints to keep. Defaults to 2.

        Returns:
            str: Path to the saved LoRA adapter weights.

        Notes:
            - Supports multi-GPU training via `accelerate launch --multi_gpu`.
            - Efficient memory usage via 4-bit quantization and CPU offloading.
            - LoRA adapters remain trainable in FP32, while frozen model weights are offloaded.
        """

    # 1) load data
    db = DB(db_path)
    prompts = load_data(db)
    danbooru_db = DB(get_settings().danbooru_db_path)
    tags = load_danbooru_tags(danbooru_db)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # 3) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        offload_folder=None,            # None = use system RAM, not NVMe
        offload_state_dict=True,        # offload frozen weights to CPU
        #bnb_4bit_use_double_quant = True,
        #bnb_4bit_compute_dtype = torch.bfloat16
    )

    # 4) model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map={"": device},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config
    )

    # If tokenizer was extended, resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Enable int8 training
    model = prepare_model_for_kbit_training(model)

    # 4) LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ----------------------------
    # 5) DDP / gradient checkpointing safety
    # ----------------------------
    # Disable gradient checkpointing for LoRA layers to avoid RuntimeError
    model.gradient_checkpointing_disable()

    # 6) dataset & dataloader
    dataset = MultiDataset([PromptPairsDataset(prompts, tokenizer, max_length=max_seq_len),
                            DanbooruTagsDataset(tags, tokenizer, max_length=max_seq_len)])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Split into train and eval (e.g. 95/5 split)
    eval_size = int(0.05 * len(dataset))
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        save_total_limit=save_total_limit,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",  # run eval during training
        eval_steps=100,  # how often to eval
        save_steps=100,
        fp16=False,
        bf16=True,  # compute in bf16
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # add eval dataset
        data_collator=collator,
        args=train_args,
        callbacks=[SampleEvalCallback(tokenizer, eval_dataset, num_samples=2)],
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


