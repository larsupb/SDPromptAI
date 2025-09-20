import re
import sqlite3
from typing import List


def ask_wiki(prompt: str) -> str:
    """Connect to danbooru_tags.db sql lite database and fetch tags."""
    conn = sqlite3.connect("models/danbooru/danbooru_tags.db")
    cur = conn.cursor()

    # split prompt by commas and fetch each tag
    tags = [tag.strip() for tag in prompt.split(",")]
    # Remove duplicate tags
    tags = list(set(tags))
    # Remove quality tags
    quality_tags = ["score_9", "score_8_up", "score_7_up", "masterpiece", "8k"]
    tags = [tag for tag in tags if tag not in quality_tags]

    wiki = []
    for tag in tags:
        tag_proper = tag.replace(" ", "_").lower()
        cur.execute("SELECT name, wiki FROM tags WHERE name=?", (tag_proper,))
        out = cur.fetchone()
        if out:
            name = curate(out[0])
            description = curate_description(out[1])
            wiki.append(f"{name}: {description}")

    return "\n".join(wiki)

def curate_prompt_list(prompts: List[str], remove_loras=True, remove_breaks=True) -> List[str]:
    curated_prompts = []
    for prompt in prompts:
        curated_prompt = curate(prompt, remove_loras=remove_loras, remove_breaks=remove_breaks)
        curated_prompts.append(curated_prompt)
    return curated_prompts

def curate(prompt, remove_loras=True, remove_breaks=True):
    if remove_loras:
        # Remove lora tags <lora:name[:weight]>
        prompt = re.sub(r'<lora:[^>]+>', '', prompt)
    if remove_breaks:
        # Remove BREAK keywords
        prompt = re.sub(r'\bBREAK\b', '', prompt, flags=re.IGNORECASE)

    # Remove leading/trailing whitespace and convert to lowercase
    prompt = prompt.strip().lower()
    prompt = re.sub(r',\s*,', ',', prompt)  # Remove empty tags between commas
    prompt = re.sub(r'\(\s*\)', '', prompt) # Remove empty parentheses
    # Remove empty commas at the begging or end
    prompt = re.sub(r'^\s*,\s*', '', prompt)  # Remove empty commas at the beginning
    prompt = re.sub(r'\s*,\s*$', '', prompt)  # Remove empty commas at the end
    # Remove line breaks and excessive spaces
    prompt = re.sub(r'\s+', ' ', prompt) # Replace multiple spaces with a single space
    prompt = re.sub(r'\n', ' ', prompt)  # Replace line breaks with a space

    # Remove duplicate tags
    tags = [tag.strip() for tag in prompt.split(',')]
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    prompt = ', '.join(unique_tags)

    return prompt


def curate_description(description: str) -> str:
    # Remove rest of text when "examples" is found. "examples" itself should also be removed.
    description = re.split(r'\bexamples\b', description, flags=re.IGNORECASE)[0].strip()
    # Remove rest of text when "see also" is found. "see also" itself should also be removed.
    description = re.split(r'\bsee also\b', description, flags=re.IGNORECASE)[0].strip()
    # Remove rest of text when "related tags" is found. "related tags" itself should also be removed.
    description = re.split(r'\brelated tags\b', description, flags=re.IGNORECASE)[0].strip()
    # Remove "h4." "h6#" and similar wiki markup
    description = re.sub(r'h[1-6][.#]?', '', description)
    # Remove [i] and [/i]
    description = re.sub(r'\[/?i\]', '', description)
    # Remove [b] and [/b] and <b> and </b>
    description = re.sub(r'\[/?b\]|</?b>', '', description)
    # Remove [[ and ]]
    description = re.sub(r'\[\[|]]', '', description)
    # Remove *
    description = re.sub(r'\*', '', description)
    # Remove line breaks
    description = re.sub(r'\n', ' ', description)
    # Truncate to 250 characters
    description = description[:512]
    return description


def interpret(chat_engine, prompt: str) -> str:
    """
    Use LLM to interpret danbooru style prompts and create a concise summary.
    """
    chat = chat_engine(f"""
    Interpret the danbooru style prompts given by the user and create a natural-language based prompt in continuous text style. 
    The user will also provide tag definitions to help you understand the tags better.
    
    Guidelines:
    - Expect danbooru style prompts with tags separated by commas.
    - Quality tags (e.g. score_9, score_8_up, score_7_up, masterpiece, 8k) should be ignored.
    - Describe the main subject(s) and their actions.
    - Describe the environment or setting if mentioned.
    - Avoid mentioning art styles, quality, or artist names.
    - Keep the summary concise and to the point. 
    - Do not write in commanding tone, like "create ..", "imagine .." or "draw ...". It should be a description only.
    - Do not set apostrophes around the prompt.
    Output:
    **Important**: Your answer is processed automatically, do only return the generated prompt. Do not add any additional text.
    """)

    user_message = f"Create a brief natural-language based prompt in continuous text for this:\n{prompt}\n\nTag definitions:\n{ask_wiki(prompt)}"

    return chat.chat(user_message, temperature=0.7)