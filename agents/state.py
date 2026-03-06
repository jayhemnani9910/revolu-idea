"""LangGraph state definitions for the CAG research system."""
from typing import Annotated, TypedDict
from uuid import UUID

from domain.models import Evidence, ResearchReport, AuditResult
from domain.causal_models import CausalGraph, CausalEdge


def replace_evidence(_existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """
    Reducer function for evidence buffers.

    Evidence in state is intended to represent the CURRENT edge's evidence only.
    Replacing (not merging) prevents evidence leakage across investigations.
    """
    # De-duplicate within the provided update to avoid prompt bloat.
    seen: set[UUID] = set()
    deduped: list[Evidence] = []
    for evidence in new:
        if evidence.id in seen:
            continue
        seen.add(evidence.id)
        deduped.append(evidence)
    return deduped


def merge_audit_feedback(existing: list[str], new: list[str]) -> list[str]:
    """Reducer function to append audit feedback."""
    return existing + new


def increment_counter(existing: int, new: int) -> int:
    """Reducer for counters - takes the max."""
    return max(existing, new)


def merge_counter_map(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Reducer for per-key counters (treats updates as deltas)."""
    merged = dict(existing)
    for key, delta in new.items():
        merged[key] = merged.get(key, 0) + int(delta)
    return merged


def merge_action_hashes(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Reducer for action hashes (treats updates as deltas)."""
    return merge_counter_map(existing, new)


class ResearchState(TypedDict):
    """
    The global state of the CAG research process.
    This is the main state that flows through the LangGraph.
    """

    # === Query & Goal ===
    root_query: str  # The original user query
    research_goal: str  # Refined research objective

    # === Causal Graph (The World Model) ===
    causal_graph: CausalGraph  # The evolving causal DAG

    # === Current Focus ===
    focus_edge: CausalEdge | None  # The edge currently being investigated
    focus_edge_id: UUID | None  # ID for serialization

    # === Evidence Buffers (with reducers for parallel merging) ===
    supporting_evidence: Annotated[list[Evidence], replace_evidence]
    contradicting_evidence: Annotated[list[Evidence], replace_evidence]

    # === Results ===
    final_report: ResearchReport | None
    audit_results: list[AuditResult]

    # === Safety & Control ===
    recursion_depth: Annotated[int, increment_counter]
    max_depth: int
    stop_reason: str | None
    total_edges_investigated: int
    node_visit_counts: Annotated[dict[str, int], merge_counter_map]  # Track visits to each node type

    # === Audit Trail ===
    audit_feedback: Annotated[list[str], merge_audit_feedback]
    action_hashes: Annotated[dict[str, int], merge_action_hashes]  # For loop detection

    # === Session ===
    session_id: str
    error: str | None  # Set if an error occurred


def create_initial_state(
    query: str,
    max_depth: int = 3,
    session_id: str | None = None,
) -> ResearchState:
    """
    Factory function to create a properly initialized ResearchState.

    Args:
        query: The research query from the user
        max_depth: Maximum recursion depth for investigation cycles
        session_id: Optional session identifier for checkpointing

    Returns:
        Initialized ResearchState
    """
    from uuid import uuid4

    return ResearchState(
        root_query=query,
        research_goal=query,  # Will be refined by planner
        causal_graph=CausalGraph(root_query=query),
        focus_edge=None,
        focus_edge_id=None,
        supporting_evidence=[],
        contradicting_evidence=[],
        final_report=None,
        audit_results=[],
        recursion_depth=0,
        max_depth=max_depth,
        stop_reason=None,
        total_edges_investigated=0,
        node_visit_counts={},
        audit_feedback=[],
        action_hashes={},
        session_id=session_id or str(uuid4()),
        error=None,
    )


def compute_action_hash(action: str, params: dict) -> str:
    """
    Compute a hash for an action to detect loops.

    Args:
        action: The action type (e.g., "search", "investigate")
        params: Action parameters

    Returns:
        Hash string for deduplication
    """
    import hashlib
    import json

    # Sort params for consistent hashing
    param_str = json.dumps(params, sort_keys=True, default=str)
    content = f"{action}:{param_str}"
    return hashlib.md5(content.encode()).hexdigest()[:16]
