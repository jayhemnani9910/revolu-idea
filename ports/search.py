"""Abstract interface for Web Search operations."""
from abc import ABC, abstractmethod
from domain.models import Citation


class SearchPort(ABC):
    """
    Port for Web Search operations.
    Abstracts away Tavily, Exa, Google, Bing, etc.

    This interface allows the research system to search the web
    without knowing which specific search provider is being used.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> list[Citation]:
        """
        Execute a search and return normalized Citation objects.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_depth: 'basic' or 'advanced' (provider-specific)

        Returns:
            List of Citation objects with url, title, snippet, credibility_score
        """
        raise NotImplementedError

    @abstractmethod
    async def search_news(
        self,
        query: str,
        max_results: int = 5,
        days_back: int = 7,
    ) -> list[Citation]:
        """
        Search specifically for recent news articles.

        Args:
            query: Search query string
            max_results: Maximum number of results
            days_back: How many days back to search

        Returns:
            List of Citation objects from news sources
        """
        raise NotImplementedError

    @abstractmethod
    async def search_academic(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Citation]:
        """
        Search academic/scholarly sources.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of Citation objects from academic sources
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the search provider name."""
        raise NotImplementedError

    def calculate_credibility(self, url: str, title: str) -> float:
        """
        Calculate credibility score based on domain heuristics.
        Default implementation - adapters can override.

        Args:
            url: Source URL
            title: Page title

        Returns:
            Credibility score between 0.0 and 1.0
        """
        # Extract domain
        try:
            domain = url.split("//")[-1].split("/")[0].lower()
        except (IndexError, AttributeError):
            return 0.3

        # High credibility domains
        high_cred = [
            ".gov", ".edu", "nature.com", "science.org", "pubmed",
            "arxiv.org", "ieee.org", "acm.org", "springer.com",
            "wiley.com", "reuters.com", "apnews.com", "bbc.com",
        ]

        # Medium credibility
        medium_cred = [
            "wikipedia.org", "medium.com", "github.com",
            "stackoverflow.com", "nytimes.com", "wsj.com",
        ]

        # Low credibility indicators
        low_cred = [
            "blog", "forum", "reddit.com", "quora.com",
            "facebook.com", "twitter.com", "tiktok.com",
        ]

        for indicator in high_cred:
            if indicator in domain:
                return 0.9

        for indicator in medium_cred:
            if indicator in domain:
                return 0.7

        for indicator in low_cred:
            if indicator in domain:
                return 0.4

        # Default medium-low credibility for unknown domains
        return 0.5
