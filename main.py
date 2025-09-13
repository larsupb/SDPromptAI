# Read image metadata from Civitai and create a dataset for llm training prompts
import json
import os
import re
from typing import List, Dict

import requests
import tqdm
import yaml

from src.ollama_chat import OllamaChat
from src.faiss_storage import FaissStorage
from src.rag import generate_faiss_query
from src.db import DB, curate
from src.encoders import get_encoder
from src.danbooru import interpret, rate


def fetch_civitai_images(model_version_id=None, limit=100, next_page=None, api_key=None):
    if not next_page:
        url = "https://civitai.com/api/v1/images"
        params = {"nsfw": "X",
                  "sort": "Most Reactions",
                  "period":"AllTime",
                  "limit": limit,
                  "token": api_key}
        if model_version_id:
            params["modelVersionId"] = model_version_id
    else:
        url = next_page
        params = {}
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None


def fetch_images_from_model_version(model_version_id, image_ids, image_prompts, api_key, max_queries=10):
    next_page = None
    query_count = 0

    db = DB(db_path)

    image_data_combined = []
    while True:
        images_data = fetch_civitai_images(model_version_id=model_version_id, limit=100,
                                           next_page=next_page, api_key=api_key)
        images_data_filtered = [img for img in images_data['items']
                                if img.get('id') not in image_ids
                                and img.get('meta') and img['meta'].get('prompt')
                                and img.get('meta').get('prompt') not in image_prompts]

        # Append to combined list
        image_data_combined.extend(images_data_filtered)

        # Add new ids and prompts to the set
        image_ids.update(img.get('id') for img in images_data_filtered)
        image_prompts.update(img.get('meta').get('prompt') for img in images_data_filtered)

        if 'nextPage' not in images_data['metadata']:
            break
        else:
            next_page = images_data['metadata']['nextPage']
        query_count += 1
        if query_count >= max_queries:
            break

    # Insert into DB
    db.insert_tags(image_data_combined)

    return len(image_data_combined)


def interpret_prompts(db_path):
    db = DB(db_path)
    rows = db.fetch_all_uninterpreted_prompts()

    for image_id, prompt in tqdm.tqdm(rows, desc="Interpreting prompts", unit="prompt"):
        interpreted_prompt = interpret(curate(prompt))
        db.update_image_interpreted_prompt(image_id, interpreted_prompt)


def rate_prompts(db_path):
    """
    Rate prompts in the database that have a rating of 0
    Currently not in use, as interpret also rates the prompts
    """
    db = DB(db_path)
    rows = db.fetch_all_zero_rating_prompts()

    # Split into batches of 10
    batch_size = 10
    pbar = tqdm.tqdm(total=len(rows), desc="Rating images", unit="image")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        # Get ratings from LLM
        ratings = rate(batch)

        image_ids = [image_id for image_id, _ in batch]
        for image_id, rating in zip(image_ids, ratings):
            try:
                rating_value = int(rating)
                if 1 <= rating_value <= 10:
                    db.update_image_rating(image_id, rating_value)
                else:
                    print(f"Invalid rating. Value ({rating}) not between 1 and 10. Skipping image_id ", image_id)
            except ValueError:
                print(f"Invalid rating. Value {rating} is not a number. Skipping image_id ", image_id)
        pbar.update(len(batch))


def calculate_embeddings(db_path, prompt_column, encoder_name, faiss_path):
    db = DB(db_path)

    # Get encoder
    encoder = get_encoder(encoder_name, device="cuda")

    # Read prompts from DB
    image_ids, image_prompts = db.fetch_all_prompts(prompt_column=prompt_column)

    # Calculate embeddings
    embeddings = encoder.encode(image_prompts)

    # Create Faiss index
    FaissStorage.create(image_ids, embeddings, faiss_path)

def search_prompts(db_path, faiss_path, query, top_k=10, encoder_name="transformers_embedder", debug=False) -> List[Dict]:
    # Get encoder
    encoder = get_encoder(encoder_name=encoder_name, device="cpu")

    # Encode query
    embeddings = encoder.encode([query])

    # Search
    faiss = FaissStorage(faiss_path)

    D, I = faiss.search(embeddings, top_k)

    # Fetch from SQLite
    db = DB(db_path)

    # Get prompts for the found image ids
    prompts = db.fetch_prompts(I[0])

    if debug:
        print("Results from faiss index:")
        for dist, idx, prompt in zip(D[0], I[0], prompts):
            print(f"ID: {idx}, Distance: {dist:.4f}, Prompt: {prompt}")

    return prompts

def prompt_improve(user_prompt, faiss_path, prompting_instruction) -> OllamaChat:
    # Retrieve similar prompts from the database
    print("Receiving similar prompts from the database...")
    similar_prompts = search_prompts(db_path, faiss_path, user_prompt, top_k=10)

    # Create a list of similar prompts
    prompt_list = ""
    for sp in similar_prompts:
        prompt_list += f"- {sp}\n"
    guidance_prompt = ("As guidance, here are some similar prompts. You should use them as inspiration, "
                       "but do not copy them directly. You can combine elements from multiple prompts, but make sure the final "
                       "prompt is unique and tailored to the user's description.")

    prompt = prompting_instruction["system_prompt"]
    prompt = prompt.replace("{guidance_prompt}", guidance_prompt)
    prompt = prompt.replace("{prompt_list}", prompt_list)

    print("Querying the llm to generate a new prompt...")
    print(prompt)
    chat = OllamaChat(system_prompt=prompt)
    chat.chat(user_prompt, temperature=1)
    return chat

def random_prompt(prompting_instruction, prompt_count=10) -> OllamaChat:
    system_prompt = prompting_instruction["system_prompt"]

    # Create a list of similar prompts
    # Retrieve random prompt from the database
    db = DB(db_path)
    prompts = db.fetch_random_prompts(prompt_count)
    prompt_list = ""
    for prompt in prompts:
        # Remove line breaks and excessive spaces
        prompt_list += f"- {prompt}\n"

    user_prompt = prompting_instruction["user_prompt"]
    user_prompt = user_prompt.replace("{prompt_list}", prompt_list)

    print("Querying the llm to generate a new prompt...")
    print(user_prompt)

    chat = OllamaChat(system_prompt=system_prompt)
    chat.chat(user_prompt, temperature=1)
    return chat


def prompt_generator_loop(faiss_path, mode, prompting_style):
    if mode == "random":
        # Generate initial random prompt in mostly natural language
        chat = random_prompt(prompting_style["random_prompt"], prompt_count=10)
        user_prompt = chat.get_latest_response()
        # Convert into a more structured prompt
        prompt_improve(user_prompt, faiss_path, prompting_style["prompt"])
    else:
        user_prompt = input("Please enter the prompt you would like to use: ")
        chat = prompt_improve(user_prompt, faiss_path, prompting_style["prompt"])

    while True:
        user_prompt = input("Modifications: ")
        # if empty quit
        if not user_prompt.strip():
            print("Exiting prompt generator.")
            break

        add_prompts = input("Search similar prompts? (y/n, default n): ").strip().lower() or "n"
        if add_prompts == "y":
            # Generate search query for FAISS
            search_query = generate_faiss_query(user_prompt)
            # Retrieve similar prompts from the database
            similar_prompts = search_prompts(db_path, faiss_path, search_query, top_k=10)
            prompt_list = ""
            for sp in similar_prompts:
                prompt_list += f"- {sp}\n"
            guidance_prompt = (
                "As guidance, here are some similar prompts. You should use them as inspiration, "
                "but do not copy them directly. You can combine elements from multiple prompts, but make sure the final "
                "prompt is unique and tailored to the user's description.")
            user_prompt += "\n" + guidance_prompt + "\n" + prompt_list

        print("Querying the llm to generate a new prompt...")
        print(user_prompt)

        chat.chat(user_prompt, temperature=1)

def fetch_loop(db_path, model_hashes:Dict[str, List[str]], api_key, max_pages_per_model=10):
    db = DB(db_path)

    # Fetch images from Civitai
    image_ids = db.get_image_ids_from_db()
    image_prompts = db.get_image_prompts_from_db()

    pattern = re.compile(r"civitai:(?P<model_id>\d+)@(?P<model_version_id>\d+)")

    total_model_hashes = sum(len(hashes) for hashes in model_hashes.values())
    pbar = tqdm.tqdm(total=total_model_hashes, desc="Fetching images", unit="page")

    for model, hashes in model_hashes.items():
        for model_hash in hashes:
            match = pattern.search(model_hash)
            if match:
                pbar.write(f"model_id: {match.group('model_id')}, model_version_id: {match.group('model_version_id')}")
                new_image_count = fetch_images_from_model_version(match.group("model_version_id"), image_ids,
                                                                  image_prompts, max_queries=max_pages_per_model, api_key=api_key)
                pbar.write(f"Fetched new images: {new_image_count}")
                pbar.update(1)



if __name__ == "__main__":
    # Initialize DB
    db_path = input("Enter path to SQLite DB (default: data/pony_prompts.db): ").strip() or "data/pony_prompts.db"

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
        fetch_loop(db_path, model_hashes, max_pages_per_model=int(max_pages_per_model), api_key=os.getenv("CIVITAI_API_KEY"))
        exit(0)
    elif mode == "prune":
        db = DB(db_path)
        db.prune_images()
        exit(0)
    elif mode == "rate":
        print("Rating and interpreting prompts...")
        #rate_prompts(db_path)
        interpret_prompts(db_path)
        exit(0)
    elif mode == "calc":
        calc_column = input("Enter mode: (interpreted_prompt, positive_prompt), default: interpreted_prompt: ").strip() or "interpreted_prompt"
        encoder = input("Enter encoder (transformers_embedder, sdxl), default: transformers_embedder: ").strip() or "transformers_embedder"
        faiss_path = input("Enter path to Faiss DB (default: data/pony_prompts.index): ").strip() or "data/pony_prompts.index"
        calculate_embeddings(db_path, calc_column, encoder, faiss_path)
        exit(0)
    elif mode == "prompt":
        prompting_style = input("Enter the name of the prompting style (default: pony_sdxl): ").strip() or "pony_sdxl"
        with open(f"models/prompting_instructions.yml", "r") as f:
            prompting_instruction = yaml.safe_load(f)
            if prompting_style not in prompting_instruction:
                print(f"Prompting style {prompting_style} not found in prompting_instructions.yml")
                exit(1)
        mode = input("Enter mode (prompt/random) (default: prompt): ").strip().lower() or "prompt"
        if mode not in ["prompt", "random", ]:
            print("Invalid mode. Exiting.")
            exit(1)

        faiss_path = input("Enter path to Faiss DB (default: data/pony_prompts.index): ").strip() or "data/pony_prompts.index"
        prompt_generator_loop(faiss_path, mode, prompting_instruction[prompting_style])
        exit(0)
    else:
        print("Unknown mode. Exiting.")
        exit(1)