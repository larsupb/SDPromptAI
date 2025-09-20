import re
from typing import Dict, List

import requests
import tqdm


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


def fetch_images_from_model_version(db, model_version_id, image_ids, image_prompts, api_key, max_queries=10):
    next_page = None
    query_count = 0

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


def fetch_loop(db, model_hashes:Dict[str, List[str]], api_key, max_pages_per_model=10):
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
                new_image_count = fetch_images_from_model_version(db, match.group("model_version_id"), image_ids,
                                                                  image_prompts, max_queries=max_pages_per_model, api_key=api_key)
                pbar.write(f"Fetched new images: {new_image_count}")
                pbar.update(1)
