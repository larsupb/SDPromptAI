from ..danbooru.prompt_interpreter import curate_description, curate


def load_data(db) -> list[tuple[str, str]]:
    """
    Loads data from sqlite database
    """
    rows = db.fetch_prompt_duos()
    if len(rows) == 0:
        raise ValueError("No rows found in DB. Check query or data.")

    # Print stats to console in pretty format
    print(f"🚀 Loaded {len(rows)} rows from database. Sample:")
    for i, (danbooru, natural) in enumerate(rows[:3]):
        print(f"--- Sample {i+1} ---")
        print(f"Natural: {natural}")
        print(f"Danbooru: {danbooru}")
        print()
    return rows


def load_danbooru_tags(db) -> list[tuple[str, str]]:
    """
    Loads all unique danbooru tags from the database
    """
    tags = db.fetch_danbooru_tags()
    if len(tags) == 0:
        raise ValueError("No tags found in DB. Check query or data.")
    # Print stats to console in pretty format
    print(f"🚀 Loaded {len(tags)} unique danbooru tags from db.")
    for i, (name, wiki) in enumerate(tags[:3]):
        print(f"--- Tag {i+1} ---")
        print(f"Name: {curate(name)}")
        print(f"Wiki: {curate_description(wiki)}")
        print()
    return tags
