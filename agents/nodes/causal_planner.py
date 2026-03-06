"""Causal Planner Node - Generates the Causal DAG from user query."""
from typing import Any

from pydantic import BaseModel, Field

from ports.llm import LLMPort
from agents.state import ResearchState
from domain.causal_models import CausalGraph, CausalNode, CausalEdge


SYSTEM_PROMPT = """You are a Principal Investigator and Causal Inference Expert.
Your task is to analyze research queries and construct Causal Directed Acyclic Graphs (DAGs).

Key principles:
1. Identify VARIABLES (causes, effects, mediators, confounders)
2. Establish CAUSAL MECHANISMS (not just correlations)
3. Consider ALTERNATIVE EXPLANATIONS
4. Think about CONFOUNDERS that might create spurious relationships

Output a structured causal graph with:
- Nodes: The key variables/concepts
- Edges: The proposed causal relationships with clear hypotheses

Remember: Correlation does not imply causation. Be rigorous."""


class PlannerOutput(BaseModel):
    """Structured output from the planner."""

    research_goal: str = Field(
        ..., description="Refined, specific research objective"
    )
    nodes: list[dict] = Field(
        ..., description="List of causal nodes with id, label, description, node_type"
    )
    edges: list[dict] = Field(
        ..., description="List of causal edges with source_id, target_id, hypothesis"
    )
    reasoning: str = Field(
        ..., description="Explanation of the causal structure"
    )


class CausalPlannerNode:
    """
    The Causal Planner (Hypothesis Generator).
    Converts user query into a structured Causal DAG.

    This is the "System 2" thinking component that provides
    the research structure before any evidence gathering.
    """

    def __init__(self, llm: LLMPort):
        """
        Initialize the planner node.

        Args:
            llm: LLM port for generation
        """
        self.llm = llm

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Execute the planning step.

        Args:
            state: Current research state

        Returns:
            State updates with causal_graph and research_goal
        """
        print(f"--- Causal Planner: Analyzing '{state['root_query'][:50]}...' ---")

        # Check if we already have a graph (re-planning scenario)
        existing_graph = state.get("causal_graph")
        if existing_graph and existing_graph.edges:
            # Re-planning: enhance existing graph based on findings
            return await self._enhance_graph(state)

        # Initial planning: create new graph
        return await self._create_initial_graph(state)

    async def _create_initial_graph(self, state: ResearchState) -> dict[str, Any]:
        """Create the initial causal graph from the query."""
        prompt = f"""
Analyze this research query and construct a Causal DAG:

QUERY: {state['root_query']}

Instructions:
1. Identify 3-7 key VARIABLES (concepts/factors) relevant to this query
2. Determine which variables are:
   - OUTCOME: The main effect/result we want to understand
   - VARIABLE: Regular causal factors
   - CONFOUNDER: Variables that might cause spurious correlations
   - MEDIATOR: Variables that transmit causal effects

3. Define CAUSAL EDGES between variables:
   - Each edge should have a clear causal hypothesis
   - Consider both direct and indirect effects
   - Think about what mechanisms connect the variables

4. Refine the research goal into a specific, testable statement

Example format:
- Nodes: [Factor A (VARIABLE), Factor B (MEDIATOR), Outcome C (OUTCOME)]
- Edges: [A -> B: "increases", B -> C: "leads to"]
"""

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=PlannerOutput,
                system_prompt=SYSTEM_PROMPT,
            )

            # Build the CausalGraph from planner output
            graph = CausalGraph(root_query=state["root_query"])

            # Add nodes
            for node_data in result.nodes:
                node = CausalNode(
                    id=node_data.get("id", f"node_{len(graph.nodes)}"),
                    label=node_data.get("label", "Unknown"),
                    description=node_data.get("description", ""),
                    node_type=node_data.get("node_type", "VARIABLE"),
                )
                graph.add_node(node)

            # Add edges
            skipped_edges: list[str] = []
            cycle_edges: list[str] = []
            for edge_data in result.edges:
                source_id = (edge_data.get("source_id") or "").strip()
                target_id = (edge_data.get("target_id") or "").strip()
                if not source_id or not target_id:
                    skipped_edges.append(f"{source_id or '?'} -> {target_id or '?'} (missing ids)")
                    continue
                if not graph.get_node(source_id) or not graph.get_node(target_id):
                    skipped_edges.append(f"{source_id} -> {target_id} (unknown node id)")
                    continue

                edge = CausalEdge(
                    source_id=source_id,
                    target_id=target_id,
                    hypothesis=edge_data.get("hypothesis", "influences"),
                    status="PROPOSED",
                )
                if graph.add_edge(edge):
                    # Maintain DAG invariant by rejecting edges that introduce cycles.
                    if not graph.is_dag():
                        graph.edges.pop()
                        cycle_edges.append(edge.edge_label)

            if cycle_edges:
                print("Warning: Planner proposed cyclic edges; removed to preserve DAG.")
            if skipped_edges:
                print("Warning: Planner proposed invalid edges; skipped.")

            print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

            extra_feedback: list[str] = []
            if skipped_edges:
                extra_feedback.append(f"Planner: Skipped {len(skipped_edges)} invalid edge(s)")
            if cycle_edges:
                extra_feedback.append(f"Planner: Removed {len(cycle_edges)} cyclic edge(s)")

            return {
                "causal_graph": graph,
                "research_goal": result.research_goal,
                "audit_feedback": [f"Planner: Created DAG - {result.reasoning[:200]}"] + extra_feedback,
            }

        except Exception as e:
            print(f"Planner error: {e}")
            # Return minimal graph on error
            return {
                "causal_graph": CausalGraph(root_query=state["root_query"]),
                "research_goal": state["root_query"],
                "error": f"Planner failed: {str(e)}",
            }

    async def _enhance_graph(self, state: ResearchState) -> dict[str, Any]:
        """Enhance existing graph based on investigation results."""
        existing_graph = state["causal_graph"]
        summary = existing_graph.get_verification_summary()

        prompt = f"""
Review the current research progress and enhance the causal graph if needed.

ORIGINAL QUERY: {state['root_query']}
CURRENT GOAL: {state.get('research_goal', state['root_query'])}

CURRENT GRAPH STATUS:
- Total edges: {summary['total_edges']}
- Verified: {summary['verified']}
- Falsified: {summary['falsified']}
- Unclear: {summary['unclear']}
- Completion: {summary['completion_rate']:.1f}%

RECENT FINDINGS:
{chr(10).join(state.get('audit_feedback', [])[-5:])}

Should we:
1. Add new nodes/edges based on discovered relationships?
2. Refine existing hypotheses?
3. The graph is sufficient - proceed to synthesis?

If enhancement needed, provide new nodes and edges.
If graph is sufficient, return empty lists.
"""

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=PlannerOutput,
                system_prompt=SYSTEM_PROMPT,
            )

            # Add any new nodes
            for node_data in result.nodes:
                if not existing_graph.get_node(node_data.get("id")):
                    node = CausalNode(
                        id=node_data.get("id"),
                        label=node_data.get("label", "Unknown"),
                        description=node_data.get("description", ""),
                        node_type=node_data.get("node_type", "VARIABLE"),
                    )
                    existing_graph.add_node(node)

            # Add any new edges
            cycle_edges: list[str] = []
            for edge_data in result.edges:
                source = (edge_data.get("source_id") or "").strip()
                target = (edge_data.get("target_id") or "").strip()
                if not source or not target:
                    continue
                if not existing_graph.get_node(source) or not existing_graph.get_node(target):
                    continue
                if not existing_graph.get_edge(source, target):
                    edge = CausalEdge(
                        source_id=source,
                        target_id=target,
                        hypothesis=edge_data.get("hypothesis", "influences"),
                        status="PROPOSED",
                    )
                    if existing_graph.add_edge(edge):
                        if not existing_graph.is_dag():
                            existing_graph.edges.pop()
                            cycle_edges.append(edge.edge_label)

            return {
                "causal_graph": existing_graph,
                "research_goal": result.research_goal or state.get("research_goal"),
                "audit_feedback": [f"Planner (enhance): {result.reasoning[:200]}"]
                + (
                    [f"Planner (enhance): Removed {len(cycle_edges)} cyclic edge(s)"]
                    if cycle_edges
                    else []
                ),
            }

        except Exception as e:
            # On error, keep existing graph
            return {
                "audit_feedback": [f"Planner enhance failed: {str(e)}"],
            }
