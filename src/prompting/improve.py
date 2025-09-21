from ..settings import get_llm


def prompt_improve_with_similar(user_prompt:str, similar_prompts:list[str], prompting_instruction):
    """
    Improve the user prompt with similar prompts from the database.
    """
    system_prompt = prompting_instruction["system_prompt"]
    user_prompt_ = prompting_instruction["user_prompt"].replace("{user_prompt}", user_prompt)
    prompt_list = ""
    for sp in similar_prompts:
        prompt_list += f"- {sp}\n"
    user_prompt_ = user_prompt_.replace("{prompt_list}", prompt_list)

    print("🖥️ System prompt:\n", system_prompt)
    print("👤 User prompt:\n", user_prompt_)

    chat = get_llm(system_prompt=system_prompt, attach_lora=False)
    chat.chat(user_prompt_, 512, temperature=1)
    return chat


def prompt_improve(user_prompt, prompting_instruction):
    """
    Improve the user prompt with an LLM plus a LoRA model.
    """
    chat = get_llm(system_prompt=prompting_instruction["system_prompt"], attach_lora=True)

    prompt = prompting_instruction["user_prompt"]
    prompt = prompt.replace("{user_prompt}", user_prompt)

    print("👤 User prompts the LLM to improve the prompt")
    generated = chat.chat(prompt, max_new_tokens=180, do_sample=False, temperature=.7)

    print("🖥️ RAG is going to polish the generated prompt")
    # Ask another unbiased LLM to polish the generated prompt
    polish_generated_prompt(generated)

    return chat


def polish_generated_prompt(user_prompt: str) -> str:
    """
    Polish the generated prompt with another unbiased LLM.
    This function checks for contradictions, balances the tags, and orders them properly.
    Returns the polished prompt.

    Args:
        user_prompt (str): The prompt to be polished.
    Returns:
        str: The polished prompt.
    """
    rag = get_llm(
        system_prompt=""
                      "You are an AI assistant that helps to polish and improve prompts for Stable Diffusion XL (SDXL) models."
                      "In this case, the model is a so-called PonyDiffusion model, which was trained with prompts consisting of danbooru tags. "
                      "The tags are separated by commas. "
                      "The user will provide you with the prompt and specific tasks.")

    # Check tags for contradictions
    print("👤 User prompts for contradiction checks")
    rag.chat(""
             "Check the tags in this prompt for contradictions. "
             "Contradictions to check:"
             "1. Subject: There should be only one main subject (e.g. '1girl' -> woman or '1boy' -> man). "
             "2. Background: There should be exactly one background setting (e.g., beach, forest, cityscape). The "
             "background may be described with multiple tags, but they should not contradict each other "
             "(e.g., 'office' and 'night club' or 'sunny beach' and 'snowy beach' are contradictory). "
             "3. Lighting: There should be only one primary lighting condition (e.g., daylight, sunset, night). "
             "Return only the corrected prompt with contradictions resolved. "
             "If there are no contradictions, return the prompt as is.\n\n" + user_prompt, 512, temperature=0.7)


    # Balance the tags in the prompt - there should be at most 5 quality tags and 1 camera tag
    print("👤 User prompts for tags balance checks")
    rag.chat(""
             "Check the tags in this prompt for balance. "
             "There should be at most 5 quality tags (score_9, score_8_up, score_7_up, etc.)"
             "and at most 1 camera tag (e.g. DSLR, wide_angle, etc.). "
             "If there are more than 5 quality tags, remove the lowest quality ones. "
             "If there are more than 1 camera tag, remove the extra ones. "
             "Return only the balanced prompt - nothing else!\n\n" + user_prompt, 512, temperature=0.7)

    # Order the tags in the prompt
    print("👤 User prompts for ordering tags")
    rag.chat(""
             "Check if the prompt has the following structure and reorder the tags if necessary:"
             " a) quality tags "
             " b) the main subject (mainly '1girl' -> woman or '1boy' -> man) followed by the action in which the subject is engaged"
             " c) details about the subject (e.g., look, clothing, accessories, expression, pose)"
             " d) details about the environment (background, setting)"
             " e) tags about the lighting and color scheme and eventually an artistic style and camera details "
             "Return only the reordered prompt. "
             "If the prompt is already in the correct order, return it as is.\n\n" + user_prompt, 512, temperature=0.7)

    return rag.get_latest_response()
