import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()


class Config:
    llm_provider: Literal["openrouter", "openai", "ollama"] = os.getenv("LLM_PROVIDER", "openrouter")

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    top_k_initial: int = 10
    top_k_reranked: int = 5
    max_retries: int = 2
    recursion_limit: int = 25


config = Config()
