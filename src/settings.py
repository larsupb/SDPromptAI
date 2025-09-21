from functools import lru_cache
from pydantic_settings import BaseSettings

from src.chat import IChatEngine, TransformersChat, OllamaChat


class Settings(BaseSettings):
    llm_engine:str = "transformers"
    danbooru_db_path: str = "models/danbooru/danbooru_tags.db"
    db_path: str = "data/pony_prompts.db"
    faiss_path: str = "data/pony_prompts.index"

    model_name_or_path: str
    lora_path: str
    civitai_api_key: str

    # Read environment variables from a .env file if present
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton-Instanz (cached)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

def get_llm(system_prompt: str = None, **kwargs) -> IChatEngine:
    settings = get_settings()
    if settings.llm_engine not in ["transformers", "ollama"]:
        raise ValueError(f"Unknown llm_engine: {settings.llm_engine}")

    if settings.llm_engine == "transformers":
        return TransformersChat(model_name_or_path=settings.model_name_or_path,
                                lora_path=settings.lora_path, system_prompt=system_prompt, **kwargs)
    elif settings.llm_engine == "ollama":
        return OllamaChat(system_prompt=system_prompt, **kwargs)
    else:
        raise ValueError(f"Unknown llm_engine: {settings.llm_engine}")

