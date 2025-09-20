
from src.db import DB
from src.danbooru.prompt_interpreter import curate_description, curate


def test_curate():
    print("###############\nTesting Prompt Curation\n###############")
    db = DB(db_path="data/pony_prompts.db")
    sample_prompts = db.fetch_curated_prompts(n_top=5, prompt_column="positive_prompt")

    for prompt in sample_prompts:
        curated_prompt = curate(prompt, remove_loras=True)
        print(f"Original: {prompt}\nCurated: {curated_prompt}\n")

def test_danbooru_curate():
    print("###############\nTesting Danbooru Tag Curation\n###############")
    db = DB(db_path="models/danbooru/danbooru_tags.db")
    sample_prompts = db.fetch_danbooru_tags(n_top=3)

    for tag, tag_description in sample_prompts:
        curated_prompt = curate(tag)
        curated_desc = curate_description(tag_description)
        print(f"Original: {curated_prompt}\nCurated: {curated_desc}\n")
