from openai import OpenAI
from src.config import config


def get_client():
    if config.llm_provider == "openrouter":
        return OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        )
    elif config.llm_provider == "openai":
        return OpenAI(
            api_key=config.openai_api_key,
        )
    elif config.llm_provider == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=config.ollama_base_url,
        )
    raise ValueError(f"Unknown LLM provider: {config.llm_provider}")


def get_model_name():
    if config.llm_provider == "openrouter":
        return config.openrouter_model
    elif config.llm_provider == "openai":
        return config.openai_model
    elif config.llm_provider == "ollama":
        return config.ollama_model
    return "gpt-4o"


def get_llm():
    client = get_client()
    model = get_model_name()

    def call_llm(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    return call_llm
