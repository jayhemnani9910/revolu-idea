"""Edge Selector Node - Selects the next edge to investigate."""
from typing import Any
from uuid import UUID

from agents.state import ResearchState
from domain.causal_models import CausalEdge


class EdgeSelectorNode:
    """
    Edge Selector - Chooses which causal hypothesis to investigate next.

    Selection strategy:
    1. Prioritize edges with status 'PROPOSED' (never investigated)
    2. Consider edges marked 'UNCLEAR' (need more evidence)
    3. Skip 'VERIFIED' and 'FALSIFIED' edges
    4. Use investigation_count to avoid over-investigating

    Returns None when all edges are resolved -> triggers report writing.
    """

    def __init__(self, max_investigations_per_edge: int = 2):
        """
        Initialize edge selector.

        Args:
            max_investigations_per_edge: Max times to investigate same edge
        """
        self.max_investigations_per_edge = max_investigations_per_edge

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Select the next edge to investigate.

        Args:
            state: Current research state

        Returns:
            State updates with focus_edge and focus_edge_id
        """
        print("--- Edge Selector: Choosing next hypothesis ---")

        graph = state.get("causal_graph")
        if not graph or not graph.edges:
            print("No graph or edges to investigate")
            return {
                "focus_edge": None,
                "focus_edge_id": None,
                "audit_feedback": ["Selector: No edges in graph"],
            }

        # Get candidate edges
        candidates = self._get_candidate_edges(graph)

        if not candidates:
            print("All edges resolved - ready for report")
            return {
                "focus_edge": None,
                "focus_edge_id": None,
                "audit_feedback": ["Selector: All edges resolved, ready for synthesis"],
            }

        # Select best candidate
        selected = self._select_best_edge(candidates, state)

        # Mark as investigating
        selected.status = "INVESTIGATING"
        graph.update_edge(selected)

        print(f"Selected edge: {selected.edge_label}")

        return {
            "focus_edge": selected,
            "focus_edge_id": selected.id,
            "causal_graph": graph,
            "supporting_evidence": [],  # Clear evidence buffer for new investigation
            "contradicting_evidence": [],
            "audit_feedback": [f"Selector: Investigating '{selected.edge_label}'"],
        }

    def _get_candidate_edges(self, graph) -> list[CausalEdge]:
        """Get edges that can be investigated."""
        candidates = []

        for edge in graph.edges:
            # Skip resolved edges
            if edge.status in ("VERIFIED", "FALSIFIED"):
                continue

            # Skip over-investigated edges
            if edge.investigation_count >= self.max_investigations_per_edge:
                continue

            candidates.append(edge)

        return candidates

    def _select_best_edge(
        self,
        candidates: list[CausalEdge],
        state: ResearchState,
    ) -> CausalEdge:
        """
        Select the best edge to investigate next.

        Scoring:
        - PROPOSED edges get priority (never investigated)
        - Lower investigation_count preferred
        - Edges connecting to OUTCOME nodes get priority
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        # Score each candidate
        scored = []
        graph = state["causal_graph"]

        for edge in candidates:
            score = 0

            # Priority by status
            if edge.status == "PROPOSED":
                score += 100
            elif edge.status == "UNCLEAR":
                score += 50

            # Prefer less investigated edges
            score -= edge.investigation_count * 20

            # Boost edges connected to outcome nodes
            target_node = graph.get_node(edge.target_id)
            if target_node and target_node.node_type == "OUTCOME":
                score += 30

            source_node = graph.get_node(edge.source_id)
            if source_node and source_node.node_type == "OUTCOME":
                score += 20

            scored.append((score, edge))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[0][1]


def should_continue_investigating(state: ResearchState) -> str:
    """
    Routing function for LangGraph conditional edges.

    Returns:
        "investigate" - if there's an edge to investigate
        "synthesize" - if all edges resolved, go to writer
        "error" - if something went wrong
    """
    if state.get("error"):
        return "error"

    focus_edge = state.get("focus_edge")

    if focus_edge is None:
        # No more edges to investigate
        return "synthesize"

    return "investigate"
