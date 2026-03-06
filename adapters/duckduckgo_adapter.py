"""DuckDuckGo Search Adapter (Free & Open Source)."""
import asyncio
from datetime import datetime
from typing import List

from ports.search import SearchPort
from domain.models import Citation


class DuckDuckGoSearchAdapter(SearchPort):
    """
    Adapter for DuckDuckGo Search.
    
    COMPLETELY FREE. No API key required.
    Uses the 'duckduckgo_search' library to scrape results.
    """

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "duckduckgo_search is required for DuckDuckGoSearchAdapter. "
                "Install with: pip install duckduckgo_search"
            ) from e

        self._ddgs = DDGS()
        self._max_retries = max_retries
        self._delay = delay

    @property
    def provider_name(self) -> str:
        return "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> List[Citation]:
        """
        Execute a standard web search using DuckDuckGo.
        """
        # DuckDuckGo rate limiting handling (simple backoff)
        for attempt in range(self._max_retries):
            try:
                # DDGS operations are synchronous (and often generator-based), so run in a thread and materialize.
                results = await asyncio.to_thread(
                    lambda: list(self._ddgs.text(keywords=query, max_results=max_results))
                )

                # Fallback: if no results, try aggressive keyword extraction
                simplified_query = ""
                if not results:
                    words = query.split()
                    # simple stoplist
                    stoplist = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "that", "which", "who", "whom", "this", "these", "those", "hypothesis", "causal", "relationship", "between", "leading", "contributes", "indicates"}
                    keywords = [w for w in words if w.lower().strip(".,:;") not in stoplist]
                    # take up to 6 key terms
                    simplified_query = " ".join(keywords[:6])
                    
                    if simplified_query and simplified_query != query:
                        print(f"  -> Fallback search (DDG): '{simplified_query}'")
                        results = await asyncio.to_thread(
                            lambda: list(
                                self._ddgs.text(keywords=simplified_query, max_results=max_results)
                            )
                        )

                # Final Fallback: Wikipedia
                if not results:
                    try:
                        import wikipedia
                        wiki_query = simplified_query or query
                        print(f"  -> Fallback search (Wikipedia): '{wiki_query}'")
                        # Search for pages
                        page_titles = wikipedia.search(wiki_query, results=1)
                        if page_titles:
                            page = wikipedia.page(page_titles[0], auto_suggest=False)
                            # Create a mock result structure matching DDG
                            results = [{
                                "href": page.url,
                                "title": page.title,
                                "body": page.summary[:1000] # Take first 1000 chars
                            }]
                    except Exception as e:
                        print(f"Wikipedia fallback failed: {e}")
                
                citations = []
                for res in results:
                    citations.append(Citation(
                        url=res.get("href", ""),
                        title=res.get("title", "Untitled"),
                        snippet=res.get("body", "")[:500],  # DDG returns 'body' as snippet
                        credibility_score=0.7,  # Default score for web results
                        access_date=datetime.now(),
                        domain=self._extract_domain(res.get("href", ""))
                    ))
                
                await asyncio.sleep(self._delay) # Be polite
                return citations

            except Exception as e:
                print(f"DDG Search warning (attempt {attempt+1}): {e}")
                await asyncio.sleep(2 * (attempt + 1))
        
        return []

    async def search_news(
        self,
        query: str,
        max_results: int = 5,
        days_back: int = 7,
    ) -> List[Citation]:
        """
        Execute a news search using DuckDuckGo.
        """
        for attempt in range(self._max_retries):
            try:
                # 'd' parameter controls time range (e.g., 'w' for week, 'm' for month)
                # We map days_back to roughly 'w' or 'm'
                time_range = "w" if days_back <= 7 else "m"
                
                results = await asyncio.to_thread(
                    lambda: list(
                        self._ddgs.news(
                            keywords=query,
                            max_results=max_results,
                            timelimit=time_range,
                        )
                    )
                )

                citations = []
                for res in results:
                    citations.append(Citation(
                        url=res.get("url", ""),
                        title=res.get("title", "Untitled"),
                        snippet=res.get("body", "")[:500],
                        credibility_score=0.8, # News tends to be higher credibility
                        access_date=datetime.now(),
                        domain=res.get("source", "News")
                    ))
                
                await asyncio.sleep(self._delay)
                return citations

            except Exception as e:
                print(f"DDG News warning (attempt {attempt+1}): {e}")
                await asyncio.sleep(2 * (attempt + 1))
        
        return []

    async def search_academic(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Citation]:
        """
        Simulate academic search by appending 'site:arxiv.org OR site:edu ...'
        """
        academic_query = f"{query} (site:arxiv.org OR site:edu OR site:ac.uk OR filetype:pdf)"
        return await self.search(academic_query, max_results)

    def _extract_domain(self, url: str) -> str:
        try:
            return url.split("//")[-1].split("/")[0]
        except:
            return "unknown"
