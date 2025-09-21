from ..settings import get_llm


def random_prompt(db, prompting_instruction, prompt_count=10):
    system_prompt = prompting_instruction["system_prompt"]

    # Create a list of similar prompts
    # Retrieve random prompt from the database
    prompts = db.fetch_curated_prompts(n_top=prompt_count, random=True)
    prompt_list = ""
    for prompt in prompts:
        # Remove line breaks and excessive spaces
        prompt_list += f"- {prompt}\n"

    user_prompt = prompting_instruction["user_prompt"]
    user_prompt = user_prompt.replace("{prompt_list}", prompt_list)

    print("Querying the llm to generate a new prompt...")
    print(user_prompt)

    chat = get_llm(system_prompt=system_prompt)
    chat.chat(user_prompt, 512, temperature=1)
    return chat
