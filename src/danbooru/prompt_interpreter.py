import json
import re
import sqlite3
from typing import List

from ..db import curate
from ..ollama_chat import OllamaChat


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
            description = curate(out[1])
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
            # Truncate to 250 characters
            description = description[:250]
            wiki.append(f"{out[0]}: {description}")

    return "\n".join(wiki)


def interpret(prompt: str) -> str:
    """
    Use Ollama to interpret danbooru style prompts and create a concise summary.
    """
    chat = OllamaChat(f"""
    Interpret the prompts given by the user and create a concise summary of the main content. 
    Additionally, the user will provide tag definitions from a danbooru_tags database to help you understand the tags better.
    
    Guidelines:
    - Expect danbooru style prompts with tags separated by commas.
    - Quality tags (e.g. score_9, score_8_up, score_7_up, masterpiece, 8k) should be ignored.
    - Focus on the main subject(s) and their actions.
    - Describe the environment or setting if mentioned.
    - Avoid mentioning art styles, quality, or artist names.
    - Keep the summary concise and to the point. 
    
    Output:
    Only return the interpreted prompt without any additional text.""")

    user_message = f"Prompt:\n {prompt}\n\nTag definitions:\n{ask_wiki(prompt)}"

    print(user_message)

    return chat.chat(user_message, temperature=0.5)


def rate(prompts: List[str]) -> str:
    chat = OllamaChat(system_prompt="""
        You are an expert prompt rater. Rate each of the 10 given prompts on a scale from 1 to 10, where 1 is the worst and 10 is the best. 
        Consider aspects like creativity, detail, and potential for generating high-quality images. 
        A prompt with just the subject or a very short description should receive a low rating.
        A prompt has a high rating if it has a subject, a detailed description, an environment, lighting, colors, and quality enhancing terms.
        Respond in json with a list of the ratings only, where the first rating corresponds to the first prompt, 
        the second to the second prompt, and so on.
        The json should be an array of integers, e.g. { "ratings": [ 8, 1, 9,... ] }
        """, max_history=1)

    prompts = ""
    for count, (_, prompt) in enumerate(prompts):
        prompt_clean = prompt.strip().lower()
        prompt_clean = re.sub(r'\s+', ' ', prompt_clean)  # Remove line breaks and excessive spaces
        prompt_clean = prompt_clean[:1000]  # truncate to 1000 chars
        prompts += f"**Prompt {count + 1}**\n{prompt_clean}\n"
    prompt = f"Rate the following prompts on a scale from 1 to 10: " + prompts
    print("Sending batch to rater:", prompt)

    response = chat.chat(prompt, temperature=0.5, format="json")
    print("Response from rater:", response)

    ratings = json.loads(response)['ratings']

    return ratings