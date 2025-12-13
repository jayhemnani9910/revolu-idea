"""Dependency Injection Container - Wires up the application."""
from config.settings import Settings
from ports.llm import LLMPort
from ports.search import SearchPort
from ports.storage import StoragePort
from graph.cag_graph import ParallelCAGGraphBuilder


class Container:
    """Dependency Injection Container."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._llm: LLMPort | None = None
        self._searcher: SearchPort | None = None
        self._storage: StoragePort | None = None

    @property
    def llm(self) -> LLMPort:
        if self._llm is None:
            from adapters.ollama_adapter import OllamaAdapter
            print(f"Using Ollama with model: {self.settings.ollama_model}")
            self._llm = OllamaAdapter(
                model_name=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        return self._llm

    @property
    def searcher(self) -> SearchPort:
        if self._searcher is None:
            if self.settings.tavily_api_key:
                from adapters.tavily_adapter import TavilySearchAdapter
                print("Using Tavily search")
                self._searcher = TavilySearchAdapter(api_key=self.settings.tavily_api_key)
            else:
                from adapters.mock_adapters import MockSearchAdapter
                print("No Tavily key, using mock search")
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
