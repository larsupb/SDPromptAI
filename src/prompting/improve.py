from ..settings import get_llm


def prompt_improve_with_similar(db, user_prompt:str, similar_prompts:list[str], prompting_instruction):
    """
    Improve the user prompt with similar prompts from the database.
    """
    system_prompt = prompting_instruction["system_prompt"]
    chat = get_llm(system_prompt=system_prompt, attach_lora=True)

    user_prompt_ = prompting_instruction["user_prompt"].replace("{user_prompt}", user_prompt)
    prompt_list = ""
    for sp in similar_prompts:
        prompt_list += f"- {sp}\n"
    user_prompt_ = user_prompt_.replace("{similar_prompts}", prompt_list)
    chat.chat(user_prompt_, temperature=1)
    return chat


def prompt_improve(user_prompt, prompting_instruction):
    chat = get_llm(system_prompt=prompting_instruction["system_prompt"], attach_lora=True)

    prompt = prompting_instruction["user_prompt"]
    prompt = prompt.replace("{user_prompt}", user_prompt)

    chat.chat(prompt, temperature=.9)
    return chat
