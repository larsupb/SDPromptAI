from .ollama_chat import OllamaChat


def generate_faiss_query(user_prompt: str) -> str:
    rag = OllamaChat(system_prompt=
                             "You are an AI assistant that created optimal search queries for a FAISS index. "
                             "FAISS is a vector database that can be searched with text queries. FAISS returns the most "
                             "similar items to the query based on their vector representation. "
                             "Your task is to analyze the user's prompt and create a concise search query that captures "
                             "the main themes and keywords of the prompt. "
                             "**Important:** Return only the search query, without any additional text or explanation. ")
    search_query = rag.chat(f"Create a concise search query for the following prompt, "
                            f"focusing on the main themes and keywords. \n" + user_prompt, temperature=0.7)
    return search_query