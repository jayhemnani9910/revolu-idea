"""Tavily search adapter implementation."""
import asyncio
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from ports.search import SearchPort
from domain.models import Citation
from domain.exceptions import AdapterError


class TavilySearchAdapter(SearchPort):
    """
    Adapter for Tavily Search API.
    Tavily is optimized for AI agents, returning clean text rather than raw HTML.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Tavily adapter.

        Args:
            api_key: Tavily API key
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python is required. Install with: pip install tavily-python"
            )

        self.client = TavilyClient(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "tavily"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> list[Citation]:
        """Execute a search and return normalized Citation objects."""
        try:
            # Tavily's search method (sync, but we'll wrap it)
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth=search_depth if search_depth in ("basic", "advanced") else "basic",
                max_results=max_results,
                include_raw_content=False,
            )

            citations = []
            for result in response.get("results", []):
                url = result.get("url", "")
                title = result.get("title", "Untitled")
                snippet = result.get("content", "")[:500]  # Truncate long snippets

                # Calculate credibility score
                score = result.get("score", 0.5)
                # Combine Tavily's score with our domain heuristics
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
            raise AdapterError("TavilySearchAdapter", "search", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_news(
        self,
        query: str,
        max_results: int = 5,
        days_back: int = 7,
    ) -> list[Citation]:
        """Search specifically for recent news articles."""
        try:
            # Tavily supports topic filtering
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth="advanced",
                max_results=max_results,
                topic="news",
                days=days_back,
            )

            citations = []
            for result in response.get("results", []):
                url = result.get("url", "")
                title = result.get("title", "Untitled")
                snippet = result.get("content", "")[:500]

                # News sources get slightly higher credibility
                score = result.get("score", 0.5)
                domain_score = self.calculate_credibility(url, title)
                # Boost news credibility slightly
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
            raise AdapterError("TavilySearchAdapter", "search_news", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_academic(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Citation]:
        """
        Search academic/scholarly sources.
        Uses advanced search with academic-focused query modification.
        """
        try:
            # Enhance query for academic focus
            academic_query = f"{query} site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov OR site:scholar.google.com OR site:semanticscholar.org"

            response = await asyncio.to_thread(
                self.client.search,
                query=academic_query,
                search_depth="advanced",
                max_results=max_results,
            )

            citations = []
            for result in response.get("results", []):
                url = result.get("url", "")
                title = result.get("title", "Untitled")
                snippet = result.get("content", "")[:500]

                # Academic sources get higher base credibility
                score = result.get("score", 0.5)
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
            raise AdapterError("TavilySearchAdapter", "search_academic", e)
