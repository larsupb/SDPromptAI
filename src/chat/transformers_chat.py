import threading

import torch

from .chat_engine import IChatEngine

torch.backends.cuda.enable_flash_sdp(True)        # enable FlashAttention kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class TransformersChat(IChatEngine):
    _instance = None  # singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._lora_attached = False
            cls._instance._system_prompt = None
            cls._instance.model_name_or_path = None
            cls._instance.lora_path = None

        # Clear messages for new instance
        cls._instance.messages = []
        return cls._instance

    def __init__(self,
                 model_name_or_path: str,
                 lora_path: str = None,
                 system_prompt: str = None,
                 attach_lora: bool = False,
                 device_map="auto",
                 max_history: int = -1):
        if getattr(self, "_initialized", False):
            # Allow dynamic reloading of LoRA or system prompt
            if self._lora_attached != attach_lora:
                self.handle_lora_change(attach_lora)
            if self._system_prompt != system_prompt:
                self.system_prompt = system_prompt
                self.clear_history()
            return  # Already initialized

        self._initialized = True
        self._lora_attached = attach_lora
        self._system_prompt = system_prompt

        self.model_name_or_path = model_name_or_path
        self.lora_path = lora_path

        self.messages = []
        self.max_history = max_history
        if self._system_prompt:
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
            dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.resize_token_embeddings(len(tokenizer))
        print("🚀 Loaded base model from", self.model_name_or_path)

    def handle_lora_change(self, attach_lora):
        if attach_lora and self.lora_path:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            self._lora_attached = True
            print("➕ Loaded PEFT adapters from", self.lora_path)
        else:
            self._lora_attached = False
            print("➖ LoRA adapters detached, using base model")

    def chat(self, user_message: str, max_new_tokens: int = 512, do_sample=True, temperature: float = 0.7,
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

        # --- Safe EOS/PAD handling ---
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
            self.model.config.eos_token_id = eos_token_id

        pad_token_id = getattr(self.model.config, "pad_token_id", None)
        if pad_token_id is None:
            if self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                pad_token_id = eos_token_id
            self.model.config.pad_token_id = pad_token_id

        # Streamer setup
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Pick correct target (base or LoRA-wrapped)
        if self._lora_attached:
            target = self.model.generate
        else:
            if hasattr(self.model, "get_base_model"):
                target = self.model.get_base_model().generate
            else:
                target = self.model.generate

        # Run generate in a background thread
        thread = threading.Thread(
            target=target,
            kwargs=dict(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                streamer=streamer
            )
        )
        thread.start()

        # Consume the stream as tokens arrive
        generated_text = ""
        for token in streamer:
            generated_text += token
            print(token, end='', flush=True)
        print()  # newline after completion

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
