"""Dependency Injection Container - Wires up the application."""
from pathlib import Path

from config.settings import Settings
from ports.llm import LLMPort
from ports.search import SearchPort
from ports.storage import StoragePort
from graph.cag_graph import ParallelCAGGraphBuilder


DEFAULT_GROQ_CHAT_MODEL_POOL: list[str] = [
    # General-purpose / higher-quality chat models
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    # Faster / smaller fallbacks
    "llama-3.1-8b-instant",
    "allam-2-7b",
    "groq/compound",
    "groq/compound-mini",
    # OSS models (often slower / stricter limits)
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


class Container:
    """Dependency Injection Container."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._llm: LLMPort | None = None
        self._searcher: SearchPort | None = None
        self._storage: StoragePort | None = None

    def _round_robin_start_index(self, pool_size: int) -> int:
        if pool_size <= 1:
            return 0

        try:
            output_dir = Path(self.settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            idx_file = output_dir / ".llm_model_index"

            current = 0
            if idx_file.exists():
                try:
                    current = int((idx_file.read_text(encoding="utf-8") or "0").strip())
                except Exception:
                    current = 0

            start = current % pool_size
            idx_file.write_text(str((start + 1) % pool_size), encoding="utf-8")
            return start
        except Exception:
            return 0

    @property
    def llm(self) -> LLMPort:
        if self._llm is None:
            if (self.settings.llm_provider or "").strip().lower() == "mock":
                from adapters.mock_adapters import MockLLMAdapter
                print("Using Mock LLM")
                self._llm = MockLLMAdapter()
                return self._llm

            from adapters.openai_compatible_adapter import OpenAICompatibleAdapter
            from adapters.fallback_llm_adapter import FallbackLLMAdapter

            if not self.settings.llm_api_key:
                raise ValueError(
                    "Missing LLM_API_KEY in revolu_idea/.env. "
                    "Set LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL), "
                    "or switch back to Ollama by re-enabling the commented code in container.py."
                )

            raw_models = (self.settings.llm_model or "").strip()
            if raw_models.lower() in {"auto", "all"} and self.settings.llm_provider.strip().lower() == "groq":
                models = list(DEFAULT_GROQ_CHAT_MODEL_POOL)
            else:
                models = [m.strip() for m in raw_models.split(",") if m.strip()]

            # Safety: don't accidentally route research calls to guard/safeguard models.
            models = [
                m
                for m in models
                if "guard" not in m.lower() and "safeguard" not in m.lower()
            ]
            models = list(dict.fromkeys(models))  # de-dupe, preserve order
            if not models:
                raise ValueError("LLM_MODEL is empty. Set LLM_MODEL in revolu_idea/.env.")

            start_index = self._round_robin_start_index(len(models))
            if len(models) == 1:
                print(f"Using API LLM ({self.settings.llm_provider}) with model: {models[0]}")
            else:
                print(
                    f"Using API LLM ({self.settings.llm_provider}) with model pool ({len(models)}). "
                    f"Start model: {models[start_index]}"
                )

            adapters = [
                OpenAICompatibleAdapter(
                    api_key=self.settings.llm_api_key,
                    model_name=model,
                    base_url=self.settings.llm_base_url,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    provider_name=self.settings.llm_provider,
                )
                for model in models
            ]

            self._llm = (
                adapters[0]
                if len(adapters) == 1
                else FallbackLLMAdapter(adapters, start_index=start_index)
            )

            # --- Ollama (local) ---
            # from adapters.ollama_adapter import OllamaAdapter
            # print(f"Using Ollama with model: {self.settings.ollama_model}")
            # self._llm = OllamaAdapter(
            #     model_name=self.settings.ollama_model,
            #     base_url=self.settings.ollama_base_url,
            #     temperature=self.settings.temperature,
            #     max_tokens=self.settings.max_tokens,
            # )
        return self._llm

    @property
    def searcher(self) -> SearchPort:
        if self._searcher is None:
            provider = (self.settings.search_provider or "").strip().lower()

            if provider == "exa" and self.settings.exa_api_key:
                from adapters.exa_adapter import ExaSearchAdapter
                print("Using Exa search")
                self._searcher = ExaSearchAdapter(api_key=self.settings.exa_api_key)
            elif provider == "tavily" and self.settings.tavily_api_key:
                from adapters.tavily_adapter import TavilySearchAdapter
                print("Using Tavily search")
                self._searcher = TavilySearchAdapter(api_key=self.settings.tavily_api_key)
            elif provider == "mock":
                from adapters.mock_adapters import MockSearchAdapter
                print("Using mock search")
                self._searcher = MockSearchAdapter()
            elif provider == "duckduckgo":
                from adapters.duckduckgo_adapter import DuckDuckGoSearchAdapter
                print("Using DuckDuckGo search (Free)")
                self._searcher = DuckDuckGoSearchAdapter()
            else:
                # Auto-select based on available keys
                if self.settings.exa_api_key:
                    from adapters.exa_adapter import ExaSearchAdapter
                    print("Exa key found, using Exa search")
                    self._searcher = ExaSearchAdapter(api_key=self.settings.exa_api_key)
                elif self.settings.tavily_api_key:
                    from adapters.tavily_adapter import TavilySearchAdapter
                    print("Tavily key found, using Tavily search")
                    self._searcher = TavilySearchAdapter(api_key=self.settings.tavily_api_key)
                else:
                    from adapters.mock_adapters import MockSearchAdapter
                    print("No search API key found, using mock search")
                    self._searcher = MockSearchAdapter()
        return self._searcher

    @property
    def storage(self) -> StoragePort:
        if self._storage is None:
            from adapters.local_storage import LocalStorageAdapter
            self._storage = LocalStorageAdapter(base_path=self.settings.output_dir)
        return self._storage

    def get_graph(self):
        builder = ParallelCAGGraphBuilder(
            llm=self.llm,
            searcher=self.searcher,
            max_depth=self.settings.max_recursion_depth,
            max_investigations_per_edge=self.settings.max_investigations_per_edge,
        )
        return builder.build()
