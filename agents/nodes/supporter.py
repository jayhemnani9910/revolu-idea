"""Supporter Researcher Node (Blue Team) - Searches for supporting evidence."""
from typing import Any

from pydantic import BaseModel, Field

from ports.llm import LLMPort
from ports.search import SearchPort
from agents.state import ResearchState
from domain.models import Evidence


SYSTEM_PROMPT = """You are a Research Advocate searching for supporting evidence.
Your job is to find credible evidence that SUPPORTS causal hypotheses.

Key strategies:
1. Search for peer-reviewed studies demonstrating the relationship
2. Look for controlled experiments establishing causality
3. Find meta-analyses or systematic reviews on the topic
4. Search for mechanistic explanations of the causal pathway
5. Look for replicated findings across multiple studies

Be thorough but honest. Prefer high-quality, credible sources."""


class SupportQueries(BaseModel):
    """Structured output for support queries."""

    queries: list[str] = Field(
        ...,
        description="Search queries designed to find supporting evidence",
        min_length=1,
        max_length=5,
    )
    search_strategy: str = Field(
        ..., description="Brief explanation of the search strategy"
    )


class SupporterResearcherNode:
    """
    The Supporter (Blue Team) - Searches for evidence to SUPPORT hypotheses.

    Works in parallel with the Adversary to provide balanced evidence.
    Focuses on finding credible, peer-reviewed sources that establish
    the proposed causal relationship.
    """

    def __init__(self, llm: LLMPort, searcher: SearchPort, max_queries: int = 3):
        """
        Initialize the supporter node.

        Args:
            llm: LLM port for query generation
            searcher: Search port for evidence retrieval
            max_queries: Maximum number of support queries to generate
        """
        self.llm = llm
        self.searcher = searcher
        self.max_queries = max_queries

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Execute supporting research.

        Args:
            state: Current research state with focus_edge

        Returns:
            State updates with supporting_evidence
        """
        edge = state.get("focus_edge")
        if not edge:
            return {"audit_feedback": ["Supporter: No edge to investigate"]}

        print(f"--- Supporter (Blue Team): Supporting '{edge.edge_label}' ---")

        # 1. Generate support queries
        support_queries = await self._generate_support_queries(edge, state)

        # 2. Execute searches
        all_evidence = []
        for query in support_queries:
            evidence = await self._search_and_process(query, edge)
            all_evidence.extend(evidence)

        print(f"Found {len(all_evidence)} pieces of supporting evidence")

        return {
            "supporting_evidence": all_evidence,
            "audit_feedback": [
                f"Supporter: Found {len(all_evidence)} supporting evidence for '{edge.source_id}->{edge.target_id}'"
            ],
        }

    async def _generate_support_queries(
        self,
        edge,
        state: ResearchState,
    ) -> list[str]:
        """Generate queries to find supporting evidence."""
        graph = state.get("causal_graph")
        source_node = graph.get_node(edge.source_id) if graph else None
        target_node = graph.get_node(edge.target_id) if graph else None

        source_label = source_node.label if source_node else edge.source_id
        target_label = target_node.label if target_node else edge.target_id

        prompt = f"""
Generate search queries to find evidence that SUPPORTS this causal hypothesis:

HYPOTHESIS: {source_label} {edge.hypothesis} {target_label}

Your queries should search for:
1. Peer-reviewed studies demonstrating causal relationship
2. Controlled experiments or RCTs testing this link
3. Meta-analyses on the relationship between these variables
4. Mechanistic explanations of how {source_label} causes {target_label}
5. Longitudinal studies tracking the causal pathway

Generate {self.max_queries} specific, searchable queries targeting academic sources.
"""

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=SupportQueries,
                system_prompt=SYSTEM_PROMPT,
            )
            print(f"Search strategy: {result.search_strategy[:100]}...")
            return result.queries[: self.max_queries]

        except Exception as e:
            print(f"Query generation failed: {e}")
            # Fallback queries
            return [
                f"causal relationship {source_label} {target_label} study",
                f"{source_label} causes {target_label} evidence",
                f"mechanism {source_label} {target_label} research",
            ][: self.max_queries]

    async def _search_and_process(
        self,
        query: str,
        edge,
    ) -> list[Evidence]:
        """Execute search and convert to Evidence objects."""
        try:
            # Prioritize academic sources for supporting evidence
            citations = await self.searcher.search_academic(query, max_results=3)

            # Supplement with general search if needed
            if len(citations) < 2:
                general_citations = await self.searcher.search(
                    query, max_results=2, search_depth="advanced"
                )
                citations.extend(general_citations)

            evidence_list = []
            for citation in citations:
                evidence = Evidence(
                    content=citation.snippet,
                    source=citation,
                    supports_hypothesis=True,  # This is supporting evidence
                    relevance_score=citation.credibility_score,
                    extraction_method="support_search",
                )
                evidence_list.append(evidence)

            return evidence_list

        except Exception as e:
            print(f"Search failed for '{query}': {e}")
            return []
