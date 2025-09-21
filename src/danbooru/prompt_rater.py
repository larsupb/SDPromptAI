import json
import re
from typing import List

import tqdm

def rate_prompts(db):
    """
    Rate prompts in the database that have a rating of 0
    Currently not in use, as interpret also rates the prompts
    """
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


def rate(llm_engine, prompts: List[str]) -> str:
    chat = llm_engine(system_prompt="""
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

    response = chat.chat(prompt, 512, temperature=0.5)
    print("Response from rater:", response)

    ratings = json.loads(response)['ratings']

    return ratings
