import threading

import torch

torch.backends.cuda.enable_flash_sdp(True)        # enable FlashAttention kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class TransformersChat:
    _instance = None  # singleton instance

    model_name_or_path = "/raid/oobabooga/models/hf/dphn/Dolphin-Mistral-24B-Venice-Edition"
    lora_path = "./models/loras/mistral24b_lora_danbooru_v4"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._attach_lora = False
            cls._instance._system_prompt = None

        # Clear messages for new instance
        cls._instance.messages = []
        return cls._instance

    def __init__(self, system_prompt: str = None, attach_lora: bool = False, device_map="auto", max_history: int = -1):
        if getattr(self, "_initialized", False):
            if self._attach_lora != attach_lora:
                self.handle_lora_change(attach_lora)
            if self._system_prompt != system_prompt:
                self.system_prompt = system_prompt
                self.clear_history()
            return  # Already initialized

        self._initialized = True
        self._attach_lora = attach_lora
        self._system_prompt = system_prompt

        self.messages = []
        self.max_history = max_history
        if self.system_prompt:
            self.messages.append({'role': 'system', 'content': self._system_prompt})

        # Model
        self.load_base_model(device_map)
        # Handle LoRA
        self.handle_lora_change(attach_lora)

        # Set model to eval mode
        self.model.eval()

    def load_base_model(self, device_map):
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.tokenizer = tokenizer
        print("🚀 Loaded tokenizer from", self.model_name_or_path)

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.resize_token_embeddings(len(tokenizer))
        print("🚀 Loaded base model from", self.model_name_or_path)

    def handle_lora_change(self, attach_lora):
        if attach_lora:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("➕ Loaded PEFT adapters from", self.lora_path)
        else:
            if isinstance(self.model, PeftModel):
                base_model = self.model.get_base_model()
                self.model = base_model
                print("❌ Unloaded PEFT adapters, reverted to base model")
        self._attach_lora = attach_lora


    def chat(self, user_message: str, temperature: float = 0.7,
             top_k: int = 50, top_p: float = 1.0, repetition_penalty: float = 1.1) -> str:

        if self.max_history > 0 and len(self.messages) >= self.max_history * 2 + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]

        self.messages.append({'role': 'user', 'content': user_message})

        # Prepare inputs
        model_inputs = self.tokenizer.apply_chat_template(
            self.messages,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        # Ensure pad_token_id is set
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Streamer setup
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Run generate in a background thread
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer
            )
        )
        thread.start()

        # Consume the stream as tokens arrive
        generated_text = ""
        for token in streamer:
            generated_text += token
            print(token, end='', flush=True)
        print()  # for newline after completion

        self.messages.append({'role': 'assistant', 'content': generated_text})
        return generated_text

    def clear_history(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({'role': 'system', 'content': self.system_prompt})

    def get_latest_response(self) -> str:
        for message in reversed(self.messages):
            if message['role'] == 'assistant':
                return message['content']
        return ""