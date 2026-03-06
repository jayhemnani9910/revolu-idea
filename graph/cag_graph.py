"""CAG (Causal-Adversarial Graph) workflow using LangGraph."""
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from ports.llm import LLMPort
from ports.search import SearchPort
from agents.state import ResearchState
from agents.nodes.causal_planner import CausalPlannerNode
from agents.nodes.edge_selector import EdgeSelectorNode
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
        workflow.add_node("investigate", self._run_parallel_investigation)
        workflow.add_node("judge", self._run_judge)
        workflow.add_node("writer", self._run_writer)
        workflow.add_node("error_handler", self._handle_error)

        # === Define Edges ===
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "auditor")
        # NOTE: Avoid `add_conditional_edges` due to LangGraph 1.0.x hangs observed
        # in some environments. Nodes return `Command(goto=...)` instead.

        workflow.add_edge("investigate", "judge")

        workflow.add_edge("judge", "auditor")
        workflow.add_edge("writer", END)
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
        result = await self.auditor(state)
        if state.get("error") or result.get("error"):
            return Command(update=result, goto="error_handler")
        if result.get("stop_reason") == "max_depth":
            print("--- Max depth reached, proceeding to synthesis ---")
            return Command(update=result, goto="writer")
        return Command(update=result, goto="selector")

    async def _run_selector(self, state: ResearchState) -> dict:
        """Run edge selector with tracking."""
        updates = increment_node_visit(state, "selector")
        result = await self.selector(state)
        result.update(updates)
        if result.get("error"):
            return Command(update=result, goto="error_handler")
        if result.get("focus_edge") is None:
            return Command(update=result, goto="writer")
        return Command(update=result, goto="investigate")

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

    async def _run_parallel_investigation(self, state: ResearchState) -> dict:
        """Run adversary + supporter concurrently and merge their outputs."""
        adversary_result, supporter_result = await asyncio.gather(
            self._run_adversary(state),
            self._run_supporter(state),
        )

        merged: dict = {
            "contradicting_evidence": adversary_result.get("contradicting_evidence", []),
            "supporting_evidence": supporter_result.get("supporting_evidence", []),
            "audit_feedback": (
                adversary_result.get("audit_feedback", [])
                + supporter_result.get("audit_feedback", [])
            ),
        }

        # Merge delta maps (reducers in ResearchState will apply them).
        node_visit_counts: dict[str, int] = {}
        for counts in (
            adversary_result.get("node_visit_counts", {}) or {},
            supporter_result.get("node_visit_counts", {}) or {},
        ):
            for key, value in counts.items():
                node_visit_counts[key] = node_visit_counts.get(key, 0) + int(value)
        if node_visit_counts:
            merged["node_visit_counts"] = node_visit_counts

        action_hashes: dict[str, int] = {}
        for counts in (
            adversary_result.get("action_hashes", {}) or {},
            supporter_result.get("action_hashes", {}) or {},
        ):
            for key, value in counts.items():
                action_hashes[key] = action_hashes.get(key, 0) + int(value)
        if action_hashes:
            merged["action_hashes"] = action_hashes

        # Propagate errors (if any) from either side.
        error_parts = [
            part
            for part in (
                adversary_result.get("error"),
                supporter_result.get("error"),
            )
            if part
        ]
        if error_parts:
            merged["error"] = "; ".join(error_parts)

        return merged

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

class ParallelCAGGraphBuilder(CAGGraphBuilder):
    """Backward-compatible alias for the default CAG graph builder."""
    pass
