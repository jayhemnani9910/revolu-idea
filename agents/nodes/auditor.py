"""Auditor Node - Safety valve and quality checks."""
from typing import Any

from agents.state import ResearchState, compute_action_hash


class AuditorNode:
    """
    The Auditor - Safety valve for the research system.

    Responsibilities:
    1. Prevent infinite loops by tracking action hashes
    2. Enforce recursion depth limits
    3. Monitor resource usage (node visits)
    4. Check for stuck states
    5. Force convergence when necessary

    This node runs before major decisions to ensure system safety.
    """

    def __init__(
        self,
        max_depth: int = 5,
        max_node_visits: int = 10,
        max_same_action: int = 2,
    ):
        """
        Initialize the auditor node.

        Args:
            max_depth: Maximum recursion depth allowed
            max_node_visits: Maximum visits to any single node type
            max_same_action: Maximum times same action can repeat
        """
        self.max_depth = max_depth
        self.max_node_visits = max_node_visits
        self.max_same_action = max_same_action

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Perform safety checks on the current state.

        Args:
            state: Current research state

        Returns:
            State updates (may include error flag to halt execution)
        """
        print("--- Auditor: Performing safety checks ---")

        updates = {}
        issues = []

        # 1. Check recursion depth
        depth_msg = self._check_depth(state)
        if depth_msg and "Max recursion depth" in depth_msg:
            print(f"Auditor: {depth_msg}. Stopping research.")
            return {"stop_reason": "max_depth", "audit_feedback": [depth_msg]}
        if depth_msg:
            issues.append(depth_msg)

        # 2. Check for loops
        loop_check = self._check_loops(state)
        if loop_check:
            issues.append(loop_check)

        # 3. Check node visit counts
        visit_check = self._check_visits(state)
        if visit_check:
            issues.append(visit_check)

        # 4. Check graph progress
        progress_check = self._check_progress(state)
        if progress_check:
            issues.append(progress_check)

        # Compile results
        if issues:
            print(f"Auditor found {len(issues)} issue(s)")
            updates["audit_feedback"] = issues

            # Check for critical issues that should halt execution
            critical = [i for i in issues if "CRITICAL" in i]
            if critical:
                updates["error"] = "; ".join(critical)

        else:
            updates["audit_feedback"] = ["Auditor: All checks passed"]

        return updates

    def _check_depth(self, state: ResearchState) -> str | None:
        """Check recursion depth limit."""
        current_depth = state.get("recursion_depth", 0)
        max_depth = state.get("max_depth", self.max_depth)

        if current_depth >= max_depth:
            return f"Max recursion depth reached ({current_depth}/{max_depth})"

        if current_depth >= max_depth - 1:
            return f"WARNING: Approaching max depth ({current_depth}/{max_depth})"

        return None

    def _check_loops(self, state: ResearchState) -> str | None:
        """Check for repeated actions (loop detection)."""
        # This would be called with current action context
        # For now, we check the state for any loop indicators
        action_hashes = state.get("action_hashes", {})

        # If we have many repeated patterns
        if len(action_hashes) > 50:
            return "WARNING: Large number of distinct actions, possible inefficiency"

        if action_hashes:
            worst_hash, worst_count = max(action_hashes.items(), key=lambda kv: kv[1])
            if worst_count > self.max_same_action:
                return (
                    f"WARNING: Detected repeated action (hash {worst_hash}) "
                    f"executed {worst_count} times"
                )

        return None

    def _check_visits(self, state: ResearchState) -> str | None:
        """Check node visit counts."""
        visit_counts = state.get("node_visit_counts", {})

        for node_name, count in visit_counts.items():
            if count >= self.max_node_visits:
                return f"CRITICAL: Node '{node_name}' visited {count} times (max: {self.max_node_visits})"
            if count >= self.max_node_visits - 2:
                return f"WARNING: Node '{node_name}' approaching visit limit ({count}/{self.max_node_visits})"

        return None

    def _check_progress(self, state: ResearchState) -> str | None:
        """Check if research is making progress."""
        graph = state.get("causal_graph")
        if not graph:
            return "WARNING: No causal graph initialized"

        summary = graph.get_verification_summary()

        # Check if we have edges
        if summary["total_edges"] == 0:
            return "WARNING: No causal edges to investigate"

        # Check for stuck state (many investigations but no resolutions)
        investigated = state.get("total_edges_investigated", 0)
        resolved = summary["verified"] + summary["falsified"]

        if investigated > 5 and resolved == 0:
            return "WARNING: Multiple investigations but no verdicts reached"

        return None


def audit_action(
    state: ResearchState,
    action: str,
    params: dict,
    *,
    max_repeats: int = 2,
) -> dict[str, Any]:
    """
    Utility function to audit a specific action before execution.

    Args:
        state: Current state
        action: Action type (e.g., "search", "investigate")
        params: Action parameters

    Returns:
        State updates (may include error to block action)
    """
    action_hash = compute_action_hash(action, params)
    existing_hashes = state.get("action_hashes", {})

    # Check if this exact action was already done
    seen_count = int(existing_hashes.get(action_hash, 0))
    if seen_count >= max_repeats:
        return {
            "error": (
                f"Loop detected: Action '{action}' with same params repeated "
                f"{seen_count + 1} times"
            ),
            "audit_feedback": [f"BLOCKED: Repeated action '{action}'"],
        }

    # Return delta update (merged by reducer).
    return {"action_hashes": {action_hash: 1}}


def increment_node_visit(state: ResearchState, node_name: str) -> dict[str, Any]:
    """
    Utility function to track node visits.

    Args:
        state: Current state
        node_name: Name of the node being visited

    Returns:
        State updates with incremented visit count
    """
    # Return delta update (merged by reducer).
    return {"node_visit_counts": {node_name: 1}}
