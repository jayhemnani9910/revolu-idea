"""Application settings using Pydantic."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === API LLM Configuration (Groq / OpenAI-compatible) ===
    # Groq endpoint: LLM_BASE_URL=https://api.groq.com/openai/v1
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # Use "auto" for the built-in Groq pool, or provide a comma-separated list.
    llm_model: str = "auto"

    # === Ollama Configuration (kept for local fallback) ===
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    temperature: float = 0.0
    max_tokens: int = 4096

    # === Search Configuration ===
    search_provider: str = "tavily"
    tavily_api_key: str = ""
    exa_api_key: str = ""

    # === Research Parameters ===
    max_recursion_depth: int = 5
    max_investigations_per_edge: int = 2

    # === Storage Configuration ===
    output_dir: str = "output"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
