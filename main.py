import json
import os

import tqdm
import yaml

from prompt_search import search_prompts
from settings import get_settings

from src.civitai import fetch_loop
from src.prompting.improve import prompt_improve_with_similar, prompt_improve
from src.prompting.randomly_inspired import random_prompt
from src.danbooru import interpret
from src.danbooru.prompt_interpreter import curate
from src.db import DB
from src.encoders import get_encoder
from src.faiss_storage import FaissStorage
from src.rag import generate_faiss_query


def interpret_prompts(db: DB):
    rows = db.fetch_all_uninterpreted_prompts()

    for image_id, prompt in tqdm.tqdm(rows, desc="Interpreting prompts", unit="prompt"):
        interpreted_prompt = interpret(settings.get_llm, curate(prompt))
        db.update_image_interpreted_prompt(image_id, interpreted_prompt)


def calculate_embeddings(db: DB, prompt_column, encoder_name, faiss_path):
    # Get encoder
    encoder = get_encoder(encoder_name, device="cuda")

    # Read prompts from DB
    prompts = db.fetch_curated_prompts(prompt_column=prompt_column)

    # Calculate embeddings
    embeddings = encoder.encode([p for _, p in prompts], batch_size=32)

    # Create Faiss index
    FaissStorage.create([i for i, _ in prompts], embeddings, faiss_path)


def prompt_generator_loop(db: DB, faiss_path, mode, prompting_style: dict):
    if mode == "random":
        # Generate initial random prompt in mostly natural language
        chat = random_prompt(db, prompting_style["random"], prompt_count=10)
        user_prompt = chat.get_latest_response()
        # Retrieve similar prompts from the database
        similar_prompts = search_prompts(db, faiss_path, user_prompt, top_k=10)
        # Convert into a more structured prompt
        prompt_improve_with_similar(db, user_prompt, similar_prompts, prompting_style["improve_similar"])
    elif mode == "improve_similar":
        user_prompt = input("Please enter the prompt you would like to use: ")
        # Generate search query for FAISS
        search_query = generate_faiss_query(user_prompt)
        # Retrieve similar prompts from the database
        similar_prompts = search_prompts(db, faiss_path, search_query, top_k=10)
        chat = prompt_improve_with_similar(db, user_prompt, similar_prompts, prompting_style["improve_similar"])
    else: ## mode == "improve":
        user_prompt = input("Please enter the prompt you would like to use: ")
        chat = prompt_improve(user_prompt, prompting_style["improve"])

    # Loop to get user modifications
    while True:
        user_prompt = input("Modifications: ")
        # if empty quit
        if not user_prompt.strip():
            print("Exiting prompt generator.")
            break
        print("Querying the llm to generate a new prompt...")
        print(user_prompt)

        chat.chat(user_prompt, temperature=1)


if __name__ == "__main__":
    # Initialize DB
    # db_path = input("Enter path to SQLite DB (default: data/pony_prompts.db): ").strip() or "data/pony_prompts.db"
    db_path = get_settings().db_path
    db = DB(db_path)

    mode = input("Enter mode fetch/prune/rate/calc/prompt (default: prompt): ").strip().lower() or "prompt"
    if mode == "fetch":
        if not os.getenv("CIVITAI_API_KEY"):
            print("CIVITAI_API_KEY not set. Exiting.")
            exit(1)
        model_hashes_input = input(
            "Enter json file with model hashes (default: model/pony_models.json]): ").strip() or "models/pony_models.json"
        if not model_hashes_input or not os.path.exists(model_hashes_input):
            print("Model hashes file not found.")
            exit(1)
        with open(model_hashes_input, "r") as f:
            model_hashes = json.load(f)

        max_pages_per_model = input("Enter max pages to fetch per model (default: 10): ").strip() or "10"
        if not max_pages_per_model.isdigit() or int(max_pages_per_model) <= 0:
            print("Invalid number of pages. Exiting.")
            exit(1)
        fetch_loop(db, model_hashes, max_pages_per_model=int(max_pages_per_model), api_key=os.getenv("CIVITAI_API_KEY"))
        exit(0)
    elif mode == "prune":
        db.prune_images()
        exit(0)
    elif mode == "rate":
        print("Rating and interpreting prompts...")
        #rate_prompts(db_path)
        interpret_prompts(db)
        exit(0)
    elif mode == "calc":
        calc_column = input("Enter mode: (interpreted_prompt, positive_prompt), default: interpreted_prompt: ").strip() or "interpreted_prompt"
        encoder = input("Enter encoder (transformers_embedder, sdxl), default: transformers_embedder: ").strip() or "transformers_embedder"
        calculate_embeddings(db, calc_column, encoder, get_settings().faiss_path)
        exit(0)
    elif mode == "prompt":
        prompting_style = input("Enter the name of the prompting style (default: pony_sdxl): ").strip() or "pony_sdxl"
        with open(f"models/prompting_instructions.yml", "r") as f:
            prompting_instruction = yaml.safe_load(f)
            if prompting_style not in prompting_instruction:
                print(f"Prompting style {prompting_style} not found in prompting_instructions.yml")
                exit(1)
        mode = input("Enter mode (improve/improve_similar/random) (default: improve): ").strip().lower() or "improve"
        if mode not in ["improve", "improve_similar", "random"]:
            print("Invalid mode. Exiting.")
            exit(1)

        prompt_generator_loop(db, get_settings().faiss_path, mode, prompting_instruction[prompting_style])
        exit(0)
    else:
        print("Unknown mode. Exiting.")
        exit(1)