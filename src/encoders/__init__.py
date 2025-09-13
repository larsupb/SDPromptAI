from .transformer_encoding import TransformersEmbedder
from .sdxl_encoding import SDXLEmbedder

def get_encoder(encoder_name: str, device: str):
    if encoder_name == "sdxl":
        encoder = SDXLEmbedder(device)
        return encoder
    elif encoder_name == "transformers_embedder":
        encoder = TransformersEmbedder(device=device)
        return encoder
    else:
        print(f"Unknown encoder: {encoder_name}")
        return None