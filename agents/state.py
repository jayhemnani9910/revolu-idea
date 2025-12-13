"""LangGraph state definitions for the CAG research system."""
from typing import Annotated, TypedDict
from uuid import UUID

from domain.models import Evidence, ResearchReport, AuditResult
from domain.causal_models import CausalGraph, CausalEdge


def merge_evidence(existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """
    Reducer function to merge evidence lists.
    Deduplicates by evidence ID.
    """
    existing_ids = {e.id for e in existing}
    merged = list(existing)
    for evidence in new:
        if evidence.id not in existing_ids:
            merged.append(evidence)
            existing_ids.add(evidence.id)
    return merged


def merge_audit_feedback(existing: list[str], new: list[str]) -> list[str]:
    """Reducer function to append audit feedback."""
    return existing + new


def increment_counter(existing: int, new: int) -> int:
    """Reducer for counters - takes the max."""
    return max(existing, new)


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
    supporting_evidence: Annotated[list[Evidence], merge_evidence]
    contradicting_evidence: Annotated[list[Evidence], merge_evidence]

    # === Results ===
    final_report: ResearchReport | None
    audit_results: list[AuditResult]

    # === Safety & Control ===
    recursion_depth: Annotated[int, increment_counter]
    max_depth: int
    total_edges_investigated: int
    node_visit_counts: dict[str, int]  # Track visits to each node type

    # === Audit Trail ===
    audit_feedback: Annotated[list[str], merge_audit_feedback]
    action_hashes: set[str]  # For loop detection

    # === Session ===
    session_id: str
    error: str | None  # Set if an error occurred


class WorkerState(TypedDict):
    """
    State passed to parallel worker nodes (Adversary/Supporter) via Send API.
    Contains only what's needed for the specific research task.
    """

    # The edge being investigated
    edge: CausalEdge

    # Search mode
    is_adversarial: bool  # True for Red Team, False for Blue Team

    # Results (written back to main state)
    evidence: list[Evidence]


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
        total_edges_investigated=0,
        node_visit_counts={},
        audit_feedback=[],
        action_hashes=set(),
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
