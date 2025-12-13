"""Custom domain exceptions for the CAG Research System."""


class ResearchSystemError(Exception):
    """Base class for all research system exceptions."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class MaxRecursionError(ResearchSystemError):
    """Raised when the research agent hits the maximum allowed depth."""

    def __init__(self, current_depth: int, max_depth: int):
        super().__init__(
            f"Maximum recursion depth reached: {current_depth}/{max_depth}",
            context={"current_depth": current_depth, "max_depth": max_depth},
        )


class EmptySearchResultsError(ResearchSystemError):
    """Raised when a search query returns no useful results."""

    def __init__(self, query: str, attempts: int = 1):
        super().__init__(
            f"No results found for query: '{query}' after {attempts} attempt(s)",
            context={"query": query, "attempts": attempts},
        )


class HallucinationDetectedError(ResearchSystemError):
    """Raised when the auditor detects unsupported claims."""

    def __init__(self, claim: str, reason: str):
        super().__init__(
            f"Hallucination detected: '{claim[:100]}...' - {reason}",
            context={"claim": claim, "reason": reason},
        )


class InvalidStateError(ResearchSystemError):
    """Raised when the agent state violates domain rules."""

    def __init__(self, state_field: str, expected: str, actual: str):
        super().__init__(
            f"Invalid state for '{state_field}': expected {expected}, got {actual}",
            context={"field": state_field, "expected": expected, "actual": actual},
        )


class LoopDetectedError(ResearchSystemError):
    """Raised when the system detects an infinite loop pattern."""

    def __init__(self, action_hash: str, node_name: str):
        super().__init__(
            f"Loop detected in node '{node_name}': action hash {action_hash} already executed",
            context={"action_hash": action_hash, "node": node_name},
        )


class AdapterError(ResearchSystemError):
    """Raised when an adapter fails to communicate with external service."""

    def __init__(self, adapter_name: str, operation: str, original_error: Exception):
        super().__init__(
            f"Adapter '{adapter_name}' failed during '{operation}': {str(original_error)}",
            context={
                "adapter": adapter_name,
                "operation": operation,
                "original_error": str(original_error),
            },
        )
        self.original_error = original_error


class ConflictResolutionError(ResearchSystemError):
    """Raised when the judge cannot resolve conflicting evidence."""

    def __init__(self, edge_id: str, pro_count: int, con_count: int):
        super().__init__(
            f"Cannot resolve conflict for edge '{edge_id}': {pro_count} supporting vs {con_count} opposing",
            context={"edge_id": edge_id, "pro_count": pro_count, "con_count": con_count},
        )
