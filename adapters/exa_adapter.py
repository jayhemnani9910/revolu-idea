"""Exa (formerly Metaphor) search adapter implementation."""
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from ports.search import SearchPort
from domain.models import Citation
from domain.exceptions import AdapterError


class ExaSearchAdapter(SearchPort):
    """
    Adapter for Exa Search API (formerly Metaphor).
    Exa uses neural/semantic search for better understanding of queries.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Exa adapter.

        Args:
            api_key: Exa API key
        """
        try:
            from exa_py import Exa
        except ImportError:
            raise ImportError(
                "exa-py is required. Install with: pip install exa-py"
            )

        self.client = Exa(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "exa"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> list[Citation]:
        """Execute a search and return normalized Citation objects."""
        try:
            # Exa supports different search types
            # Use neural search for semantic understanding
            response = self.client.search_and_contents(
                query=query,
                num_results=max_results,
                text={"max_characters": 1000},
                use_autoprompt=True,  # Let Exa optimize the query
            )

            citations = []
            for result in response.results:
                url = result.url or ""
                title = result.title or "Untitled"
                snippet = (result.text or "")[:500]

                # Exa provides a score
                score = getattr(result, "score", 0.5)
                domain_score = self.calculate_credibility(url, title)
                final_score = (score + domain_score) / 2

                citations.append(
                    Citation(
                        url=url,
                        title=title,
                        snippet=snippet,
                        credibility_score=min(final_score, 1.0),
                        access_date=datetime.now(),
                    )
                )

            return citations

        except Exception as e:
            raise AdapterError("ExaSearchAdapter", "search", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_news(
        self,
        query: str,
        max_results: int = 5,
        days_back: int = 7,
    ) -> list[Citation]:
        """Search specifically for recent news articles."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            response = self.client.search_and_contents(
                query=query,
                num_results=max_results,
                text={"max_characters": 1000},
                start_published_date=start_date.strftime("%Y-%m-%d"),
                end_published_date=end_date.strftime("%Y-%m-%d"),
                category="news",
            )

            citations = []
            for result in response.results:
                url = result.url or ""
                title = result.title or "Untitled"
                snippet = (result.text or "")[:500]

                score = getattr(result, "score", 0.5)
                domain_score = self.calculate_credibility(url, title)
                final_score = min((score + domain_score) / 2 + 0.1, 1.0)

                citations.append(
                    Citation(
                        url=url,
                        title=title,
                        snippet=snippet,
                        credibility_score=final_score,
                        access_date=datetime.now(),
                    )
                )

            return citations

        except Exception as e:
            raise AdapterError("ExaSearchAdapter", "search_news", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_academic(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Citation]:
        """Search academic/scholarly sources."""
        try:
            # Exa has category support for research papers
            response = self.client.search_and_contents(
                query=query,
                num_results=max_results,
                text={"max_characters": 1000},
                category="research paper",
                use_autoprompt=True,
            )

            citations = []
            for result in response.results:
                url = result.url or ""
                title = result.title or "Untitled"
                snippet = (result.text or "")[:500]

                score = getattr(result, "score", 0.5)
                domain_score = self.calculate_credibility(url, title)
                # Academic boost
                final_score = min((score + domain_score) / 2 + 0.15, 1.0)

                citations.append(
                    Citation(
                        url=url,
                        title=title,
                        snippet=snippet,
                        credibility_score=final_score,
                        access_date=datetime.now(),
                    )
                )

            return citations

        except Exception as e:
            raise AdapterError("ExaSearchAdapter", "search_academic", e)

    async def find_similar(
        self,
        url: str,
        max_results: int = 5,
    ) -> list[Citation]:
        """
        Find similar pages to a given URL.
        Unique Exa capability for expanding research.
        """
        try:
            response = self.client.find_similar_and_contents(
                url=url,
                num_results=max_results,
                text={"max_characters": 1000},
            )

            citations = []
            for result in response.results:
                result_url = result.url or ""
                title = result.title or "Untitled"
                snippet = (result.text or "")[:500]

                score = getattr(result, "score", 0.5)
                domain_score = self.calculate_credibility(result_url, title)
                final_score = (score + domain_score) / 2

                citations.append(
                    Citation(
                        url=result_url,
                        title=title,
                        snippet=snippet,
                        credibility_score=min(final_score, 1.0),
                        access_date=datetime.now(),
                    )
                )

            return citations

        except Exception as e:
            raise AdapterError("ExaSearchAdapter", "find_similar", e)
