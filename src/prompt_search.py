from typing import List

from .danbooru.prompt_interpreter import curate_prompt_list
from .db import DB
from .encoders import get_encoder
from .faiss_storage import FaissStorage


def search_prompts(db: DB, faiss_path, query, top_k=10, encoder_name="transformers_embedder", debug=False) -> List[str]:
    # Get encoder
    encoder = get_encoder(encoder_name=encoder_name, device="cpu")

    # Encode query
    embeddings = encoder.encode([query])

    # Search
    faiss = FaissStorage(faiss_path)

    D, I = faiss.search(embeddings, top_k)

    # Get prompts for the found image ids
    prompts: list[str] = db.fetch_prompts(I[0])

    # Curate prompts to have a proper format
    curate_prompt_list(prompts)

    if debug:
        print("Results from faiss index:")
        for dist, idx, prompt in zip(D[0], I[0], prompts):
            print(f"ID: {idx}, Distance: {dist:.4f}, Prompt: {prompt}")

    return prompts
