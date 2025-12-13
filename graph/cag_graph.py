"""CAG (Causal-Adversarial Graph) workflow using LangGraph."""
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ports.llm import LLMPort
from ports.search import SearchPort
from agents.state import ResearchState
from agents.nodes.causal_planner import CausalPlannerNode
from agents.nodes.edge_selector import EdgeSelectorNode, should_continue_investigating
from agents.nodes.adversary import AdversarialResearcherNode
from agents.nodes.supporter import SupporterResearcherNode
from agents.nodes.judge import DialecticalJudgeNode
from agents.nodes.writer import WriterNode
from agents.nodes.auditor import AuditorNode, increment_node_visit


class CAGGraphBuilder:
    """
    Builder for the Causal-Adversarial Graph (CAG) workflow.

    The workflow implements a Popperian falsification engine:
    1. Planner: Creates causal DAG from query
    2. Selector: Picks next edge to investigate
    3. Adversary + Supporter: Parallel research (Red/Blue team)
    4. Judge: Resolves conflict, updates graph
    5. Loop back to Selector until all edges resolved
    6. Writer: Synthesizes final report

    Safety mechanisms:
    - Max recursion depth
    - Loop detection via action hashes
    - Node visit counting
    - Auditor safety checks
    """

    def __init__(
        self,
        llm: LLMPort,
        searcher: SearchPort,
        max_depth: int = 5,
        max_investigations_per_edge: int = 2,
    ):
        """
        Initialize the graph builder.

        Args:
            llm: LLM port for all nodes
            searcher: Search port for research nodes
            max_depth: Maximum investigation cycles
            max_investigations_per_edge: Max times to investigate same edge
        """
        self.llm = llm
        self.searcher = searcher
        self.max_depth = max_depth

        # Initialize nodes
        self.planner = CausalPlannerNode(llm)
        self.selector = EdgeSelectorNode(max_investigations_per_edge)
        self.adversary = AdversarialResearcherNode(llm, searcher)
        self.supporter = SupporterResearcherNode(llm, searcher)
        self.judge = DialecticalJudgeNode(llm)
        self.writer = WriterNode(llm)
        self.auditor = AuditorNode(max_depth)

    def build(self) -> StateGraph:
        """
        Build and compile the LangGraph workflow.

        Returns:
            Compiled StateGraph ready for execution
        """
        # Create the graph
        workflow = StateGraph(ResearchState)

        # === Add Nodes ===
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("auditor", self._run_auditor)
        workflow.add_node("selector", self._run_selector)
        workflow.add_node("adversary", self._run_adversary)
        workflow.add_node("supporter", self._run_supporter)
        workflow.add_node("judge", self._run_judge)
        workflow.add_node("writer", self._run_writer)
        workflow.add_node("error_handler", self._handle_error)

        # === Define Edges ===

        # Start -> Planner
        workflow.add_edge(START, "planner")

        # Planner -> Auditor (safety check)
        workflow.add_edge("planner", "auditor")

        # Auditor -> Selector or Error
        workflow.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {
                "continue": "selector",
                "error": "error_handler",
            },
        )

        # Selector -> Investigate or Synthesize
        workflow.add_conditional_edges(
            "selector",
            self._route_after_selection,
            {
                "investigate": "adversary",  # Will also trigger supporter in parallel
                "synthesize": "writer",
                "error": "error_handler",
            },
        )

        # Adversary -> Judge (waits for supporter via fan-in)
        workflow.add_edge("adversary", "judge")

        # Supporter -> Judge (parallel with adversary)
        workflow.add_edge("supporter", "judge")

        # Judge -> Auditor (loop back with safety check)
        workflow.add_edge("judge", "auditor")

        # Writer -> END
        workflow.add_edge("writer", END)

        # Error Handler -> END
        workflow.add_edge("error_handler", END)

        return workflow.compile()

    # === Node Wrappers ===
    # These wrappers add tracking and error handling

    async def _run_planner(self, state: ResearchState) -> dict:
        """Run planner with tracking."""
        updates = increment_node_visit(state, "planner")
        result = await self.planner(state)
        result.update(updates)
        return result

    async def _run_auditor(self, state: ResearchState) -> dict:
        """Run auditor."""
        return await self.auditor(state)

    async def _run_selector(self, state: ResearchState) -> dict:
        """Run edge selector with tracking."""
        updates = increment_node_visit(state, "selector")
        result = await self.selector(state)
        result.update(updates)
        return result

    async def _run_adversary(self, state: ResearchState) -> dict:
        """Run adversary with tracking."""
        updates = increment_node_visit(state, "adversary")
        result = await self.adversary(state)
        result.update(updates)
        return result

    async def _run_supporter(self, state: ResearchState) -> dict:
        """Run supporter with tracking."""
        updates = increment_node_visit(state, "supporter")
        result = await self.supporter(state)
        result.update(updates)
        return result

    async def _run_judge(self, state: ResearchState) -> dict:
        """Run judge with tracking and depth increment."""
        updates = increment_node_visit(state, "judge")
        # Increment recursion depth after each full investigation cycle
        updates["recursion_depth"] = state.get("recursion_depth", 0) + 1
        result = await self.judge(state)
        result.update(updates)
        return result

    async def _run_writer(self, state: ResearchState) -> dict:
        """Run writer."""
        return await self.writer(state)

    async def _handle_error(self, state: ResearchState) -> dict:
        """Handle errors gracefully."""
        error = state.get("error", "Unknown error")
        print(f"--- Error Handler: {error} ---")

        # Try to produce partial report if possible
        if state.get("causal_graph"):
            try:
                result = await self.writer(state)
                result["audit_feedback"] = [f"Error occurred, partial report generated: {error}"]
                return result
            except Exception:
                pass

        return {
            "audit_feedback": [f"FATAL ERROR: {error}"],
        }

    # === Routing Functions ===

    def _route_after_audit(self, state: ResearchState) -> Literal["continue", "error"]:
        """Route based on auditor results."""
        if state.get("error"):
            return "error"
        return "continue"

    def _route_after_selection(
        self,
        state: ResearchState,
    ) -> Literal["investigate", "synthesize", "error"]:
        """Route based on selector results."""
        if state.get("error"):
            return "error"

        focus_edge = state.get("focus_edge")
        if focus_edge is None:
            # No more edges to investigate
            return "synthesize"

        return "investigate"


def build_cag_graph(
    llm: LLMPort,
    searcher: SearchPort,
    max_depth: int = 5,
) -> StateGraph:
    """
    Convenience function to build the CAG graph.

    Args:
        llm: LLM port
        searcher: Search port
        max_depth: Maximum investigation cycles

    Returns:
        Compiled LangGraph workflow
    """
    builder = CAGGraphBuilder(llm, searcher, max_depth)
    return builder.build()


# === Alternative: Parallel Investigation Graph ===
# This version uses Send API for true parallel execution of Red/Blue teams

class ParallelCAGGraphBuilder(CAGGraphBuilder):
    """
    Alternative builder that uses LangGraph's Send API for
    true parallel execution of Adversary and Supporter nodes.
    """

    def build(self) -> StateGraph:
        """Build graph with parallel investigation."""
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("auditor", self._run_auditor)
        workflow.add_node("selector", self._run_selector)
        workflow.add_node("investigate", self._run_parallel_investigation)
        workflow.add_node("judge", self._run_judge)
        workflow.add_node("writer", self._run_writer)
        workflow.add_node("error_handler", self._handle_error)

        # Edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "auditor")

        workflow.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {"continue": "selector", "error": "error_handler"},
        )

        workflow.add_conditional_edges(
            "selector",
            self._route_after_selection,
            {
                "investigate": "investigate",
                "synthesize": "writer",
                "error": "error_handler",
            },
        )

        workflow.add_edge("investigate", "judge")
        workflow.add_edge("judge", "auditor")
        workflow.add_edge("writer", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile()

    async def _run_parallel_investigation(self, state: ResearchState) -> dict:
        """
        Run both adversary and supporter, merge results.

        Note: For true parallelism with Send API, you would use:
        return [
            Send("adversary", state),
            Send("supporter", state),
        ]

        But this requires different graph structure. Here we simulate
        by running both sequentially (still faster than sync).
        """
        import asyncio

        # Run both concurrently
        adversary_task = self._run_adversary(state)
        supporter_task = self._run_supporter(state)

        adversary_result, supporter_result = await asyncio.gather(
            adversary_task, supporter_task
        )

        # Merge results
        merged = {}
        merged["contradicting_evidence"] = adversary_result.get("contradicting_evidence", [])
        merged["supporting_evidence"] = supporter_result.get("supporting_evidence", [])
        merged["audit_feedback"] = (
            adversary_result.get("audit_feedback", []) +
            supporter_result.get("audit_feedback", [])
        )
        merged["node_visit_counts"] = {
            **state.get("node_visit_counts", {}),
            **adversary_result.get("node_visit_counts", {}),
            **supporter_result.get("node_visit_counts", {}),
        }

        return merged
