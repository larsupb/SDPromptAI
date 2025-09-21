import yaml

from src.db import DB
from src.prompt_search import search_prompts
from src.prompting.improve import prompt_improve_with_similar, prompt_improve
from src.rag import generate_faiss_query
from src.settings import get_settings


def test_transformers_prompt_improve():
    prompting_style = "pony_sdxl"

    with open(f"./models/prompting_instructions.yml", "r") as f:
        prompting_instruction = yaml.safe_load(f)
    style_dict = prompting_instruction[prompting_style]["improve"]

    user_prompt = "a woman taking a selfie in front of a mirror"

    prompt_improve(user_prompt, style_dict)

def test_transformers_prompt_improve_similar():
    prompting_style = "pony_sdxl"

    db_path = get_settings().db_path
    db = DB(db_path)

    with open(f"./models/prompting_instructions.yml", "r") as f:
        prompting_instruction = yaml.safe_load(f)
    style_dict = prompting_instruction[prompting_style]["improve_similar"]

    user_prompt = "a woman taking a selfie in front of a mirror"

    # Generate search query for FAISS
    search_query = generate_faiss_query(user_prompt)
    # Retrieve similar prompts from the database
    similar_prompts = search_prompts(db, get_settings().faiss_path, search_query, top_k=10)
    # Improve the prompt using similar prompts
    prompt_improve_with_similar(db, user_prompt, similar_prompts, style_dict)