import torch
import tqdm
from transformers import AutoTokenizer, AutoModel

# -------------------
# Config
# -------------------
#EMBED_MODEL = "intfloat/e5-large-v2"  # lightweight, can swap for better
#EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"


class TransformersEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", batch_size=4, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16
        )
        self.batch_size = batch_size

    def encode(self, prompts):
        all_embeddings = []
        for i in tqdm.tqdm(range(0, len(prompts), self.batch_size), desc="Encoding prompts", unit="batch"):
            batch = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())  # still tensor

        # make sure everything is a tensor before cat
        all_embeddings = [torch.as_tensor(e) for e in all_embeddings]

        # Concatenate all embeddings and convert to numpy
        out = torch.cat(all_embeddings, dim=0).numpy()

        # Unload everything from GPU
        del self.model
        del inputs
        torch.cuda.empty_cache()

        # Print cuda:0 memory usage
        if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

        return out
