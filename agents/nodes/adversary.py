"""Adversarial Researcher Node (Red Team) - Searches for disproving evidence."""
from typing import Any

from pydantic import BaseModel, Field

from ports.llm import LLMPort
from ports.search import SearchPort
from agents.state import ResearchState, compute_action_hash
from domain.models import Evidence


SYSTEM_PROMPT = """You are a Critical Skeptic and Research Adversary.
Your job is to DISPROVE causal hypotheses by finding counter-evidence.

Key strategies:
1. Search for studies showing NO correlation between the variables
2. Look for CONFOUNDING variables that explain the relationship
3. Find COUNTER-EXAMPLES where the cause is present but effect is absent
4. Search for ALTERNATIVE EXPLANATIONS for the observed relationship
5. Look for methodological critiques of supporting research

Be ruthless but fair. The goal is truth, not confirmation."""


class AttackQueries(BaseModel):
    """Structured output for attack queries."""

    queries: list[str] = Field(
        ...,
        description="Search queries designed to find counter-evidence",
        min_length=1,
        max_length=5,
    )
    attack_strategy: str = Field(
        ..., description="Brief explanation of the attack strategy"
    )


class AdversarialResearcherNode:
    """
    The Adversary (Red Team) - Searches for evidence to DISPROVE hypotheses.

    This is the critical component of the Popperian falsification engine.
    It generates queries specifically designed to find counter-evidence
    and contradictions to the proposed causal relationship.
    """

    def __init__(self, llm: LLMPort, searcher: SearchPort, max_queries: int = 3):
        """
        Initialize the adversary node.

        Args:
            llm: LLM port for query generation
            searcher: Search port for evidence retrieval
            max_queries: Maximum number of attack queries to generate
        """
        self.llm = llm
        self.searcher = searcher
        self.max_queries = max_queries

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Execute adversarial research.

        Args:
            state: Current research state with focus_edge

        Returns:
            State updates with contradicting_evidence
        """
        edge = state.get("focus_edge")
        if not edge:
            return {"audit_feedback": ["Adversary: No edge to investigate"]}

        print(f"--- Adversary (Red Team): Attacking '{edge.edge_label}' ---")

        # 1. Generate attack queries
        attack_queries = await self._generate_attack_queries(edge, state)

        # 2. Execute searches
        all_evidence = []
        action_deltas: dict[str, int] = {}
        action_counts = dict(state.get("action_hashes", {}) or {})
        skipped_repeats = 0
        for query in attack_queries:
            action_key = compute_action_hash(
                "search",
                {"edge_id": edge.id, "query": query},
            )
            if action_counts.get(action_key, 0) >= 2:
                skipped_repeats += 1
                continue
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            action_deltas[action_key] = action_deltas.get(action_key, 0) + 1

            evidence = await self._search_and_process(query, edge)
            all_evidence.extend(evidence)

        print(f"Found {len(all_evidence)} pieces of counter-evidence")

        feedback = (
            f"Adversary: Found {len(all_evidence)} counter-evidence for '{edge.source_id}->{edge.target_id}'"
        )
        if skipped_repeats:
            feedback += f" (skipped {skipped_repeats} repeated query/edge searches)"

        return {
            "contradicting_evidence": all_evidence,
            "action_hashes": action_deltas,
            "audit_feedback": [feedback],
        }

    async def _generate_attack_queries(
        self,
        edge,
        state: ResearchState,
    ) -> list[str]:
        """Generate queries to find disproving evidence."""
        graph = state.get("causal_graph")
        source_node = graph.get_node(edge.source_id) if graph else None
        target_node = graph.get_node(edge.target_id) if graph else None

        source_label = source_node.label if source_node else edge.source_id
        target_label = target_node.label if target_node else edge.target_id

        prompt = f"""
Generate search queries to find evidence that CONTRADICTS this causal hypothesis:

HYPOTHESIS: {source_label} {edge.hypothesis} {target_label}

Your queries should search for:
1. Studies showing NO relationship between these variables
2. Evidence of confounding variables that explain the correlation
3. Counter-examples where {source_label} is present but {target_label} is absent
4. Alternative causes for {target_label}
5. Methodological criticisms of studies supporting this link

Generate {self.max_queries} specific, searchable queries.
"""

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=AttackQueries,
                system_prompt=SYSTEM_PROMPT,
            )
            print(f"Attack strategy: {result.attack_strategy[:100]}...")
            return result.queries[: self.max_queries]

        except Exception as e:
            print(f"Query generation failed: {e}")
            # Fallback queries
            return [
                f"no correlation {source_label} {target_label}",
                f"confounding variables {source_label} {target_label}",
                f"{target_label} without {source_label} evidence",
            ][: self.max_queries]

    async def _search_and_process(
        self,
        query: str,
        edge,
    ) -> list[Evidence]:
        """Execute search and convert to Evidence objects."""
        try:
            # Use general search first for broader coverage
            citations = await self.searcher.search(query, max_results=3)

            # Fall back to academic if needed (or if configured)
            if not citations:
                citations = await self.searcher.search_academic(query, max_results=3)

            evidence_list = []
            for citation in citations:
                evidence = Evidence(
                    content=citation.snippet,
                    source=citation,
                    supports_hypothesis=False,  # This is counter-evidence
                    relevance_score=citation.credibility_score,
                    extraction_method="adversarial_search",
                )
                evidence_list.append(evidence)

            return evidence_list

        except Exception as e:
            print(f"Search failed for '{query}': {e}")
            return []
