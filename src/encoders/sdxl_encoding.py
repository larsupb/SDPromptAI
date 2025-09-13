import logging
import os
from typing import List

import torch
import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

encoder_dir = "/home/lars/SD/scripts/danbooru/models/autismmixSDXL_autismmixPony"

class SDXLEmbedder:

    def __init__(self, device: str = "cuda"):
        self.tok1 = None  # CLIP-L tokenizer
        self.enc1 = None  # CLIP-L encoder
        self.tok2 = None  # bigG tokenizer
        self.enc2 = None  # bigG encoder
        self.device = device

        self.load_model(encoder_dir)

    def load_model(self, model_file: str):
        logging.info("DiffusersEncoding: Loading text encoders from %s", model_file)

        # CLIP-L
        self.tok1 = CLIPTokenizer.from_pretrained(os.path.join(model_file, "tokenizer"))
        self.enc1 = CLIPTextModel.from_pretrained(
            os.path.join(model_file, "text_encoder"),
            torch_dtype=torch.float16
        ).to(self.device)

        # bigG
        self.tok2 = CLIPTokenizer.from_pretrained(os.path.join(model_file, "tokenizer_2"))
        self.enc2 = CLIPTextModel.from_pretrained(
            os.path.join(model_file, "text_encoder_2"),
            torch_dtype=torch.float16
        ).to(self.device)

    def encode(self, prompts: List[str], batch_size: int = 16):
        logging.info(f"DiffusersEncoding: Encoding {len(prompts)} prompts")

        all_embeddings = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts", unit="batch"):
            batch_prompts = prompts[i:i + batch_size]

            clip_l, bigG = self._encode_batch(batch_prompts)

            # Cat both embeddings
            all_embeddings.append(torch.cat([clip_l, bigG], dim=1).cpu())  # (batch, 768+1280=2048)

        return torch.cat(all_embeddings, dim=0).numpy()


    def _encode_batch(self, prompts: List[str], normalize: bool = True):
        # CLIP-L
        inputs1 = self.tok1(
            prompts,
            padding="max_length",
            max_length=self.tok1.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out1 = self.enc1(**inputs1, output_hidden_states=True)
        emb1 = out1.hidden_states[-2].mean(dim=1)  # (batch, 768)
        if normalize:
            # Normalize for cosine similarity
            emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)

        # bigG
        inputs2 = self.tok2(
            prompts,
            padding="max_length",
            max_length=self.tok2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out2 = self.enc2(**inputs2, output_hidden_states=True)
        emb2 = out2.hidden_states[-2].mean(dim=1)  # (batch, 1280)
        if normalize:
            # Normalize for cosine similarity
            emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)

        return emb1, emb2
