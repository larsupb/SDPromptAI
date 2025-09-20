import yaml

from src.prompt_search import search_prompts
from src.chat import TransformersChat
from src.rag import generate_faiss_query
from src.db import DB
from src.settings import get_settings
from src.prompting.improve import prompt_improve_with_similar


def test_transformers_chat():
    chat = TransformersChat()
    chat.chat("Hello, how are you?")

def test_transformers_prompt_improve():
    prompting_style = "pony_sdxl"

    with open(f"./models/prompting_instructions.yml", "r") as f:
        prompting_instruction = yaml.safe_load(f)
    style_dict = prompting_instruction[prompting_style]["improve"]

    system_prompt = style_dict["system_prompt"]
    user_prompt = style_dict["user_prompt"]
    user_prompt = user_prompt.replace("{user_prompt}", "a woman taking a selfie in front of a mirror")

    chat = TransformersChat(system_prompt=system_prompt)
    chat.chat(user_message=user_prompt, temperature=1, top_k=50, top_p=0.9)

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



