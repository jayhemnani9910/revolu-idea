"""Dialectical Judge Node - Resolves conflicts between supporting and contradicting evidence."""
from typing import Any, Literal

from pydantic import BaseModel, Field

from ports.llm import LLMPort
from agents.state import ResearchState
from domain.causal_models import CausalEdge


SYSTEM_PROMPT = """You are an Impartial Scientific Judge and Arbiter of Evidence.
Your role is to weigh competing evidence and reach a verdict on causal claims.

Judging principles:
1. CREDIBILITY: Academic peer-reviewed sources > news > blogs
2. RECENCY: Recent studies may override older findings
3. METHODOLOGY: Controlled experiments > observational studies > anecdotes
4. REPLICATION: Findings replicated across studies are stronger
5. EFFECT SIZE: Consider magnitude, not just statistical significance
6. CONFOUNDERS: Evidence accounting for confounders is more reliable

Your verdicts:
- VERIFIED: Strong, consistent evidence supports the causal link
- FALSIFIED: Strong evidence contradicts or disproves the link
- UNCLEAR: Evidence is mixed, insufficient, or requires more investigation

Be rigorous but fair. Acknowledge uncertainty when it exists."""


class JudgmentOutput(BaseModel):
    """Structured output from the judge."""

    verdict: Literal["VERIFIED", "FALSIFIED", "UNCLEAR"] = Field(
        ..., description="Final verdict on the causal hypothesis"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0-1)",
    )
    reasoning: str = Field(
        ..., description="Detailed reasoning for the verdict"
    )
    key_supporting_points: list[str] = Field(
        default_factory=list,
        description="Key points from supporting evidence",
    )
    key_contradicting_points: list[str] = Field(
        default_factory=list,
        description="Key points from contradicting evidence",
    )
    methodological_concerns: list[str] = Field(
        default_factory=list,
        description="Any methodological concerns identified",
    )


class DialecticalJudgeNode:
    """
    The Dialectical Judge - Resolves conflict between Thesis (Blue Team)
    and Antithesis (Red Team) to produce Synthesis (Verified Fact).

    This is the core of the Hegelian dialectic in the CAG system:
    - Thesis: The proposed causal hypothesis
    - Antithesis: Counter-evidence from adversarial research
    - Synthesis: The judge's verdict integrating both perspectives
    """

    def __init__(self, llm: LLMPort, min_evidence_for_verdict: int = 2):
        """
        Initialize the judge node.

        Args:
            llm: LLM port for reasoning
            min_evidence_for_verdict: Minimum evidence pieces for strong verdict
        """
        self.llm = llm
        self.min_evidence_for_verdict = min_evidence_for_verdict

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Execute judgment on the current edge.

        Args:
            state: Current research state with evidence

        Returns:
            State updates with updated causal_graph
        """
        edge = state.get("focus_edge")
        if not edge:
            return {"audit_feedback": ["Judge: No edge to adjudicate"]}

        supporting = state.get("supporting_evidence", [])
        contradicting = state.get("contradicting_evidence", [])

        print(f"--- Judge: Adjudicating '{edge.edge_label}' ---")
        print(f"    Evidence: {len(supporting)} supporting, {len(contradicting)} contradicting")

        # Check if we have enough evidence
        total_evidence = len(supporting) + len(contradicting)
        if total_evidence < self.min_evidence_for_verdict:
            # Insufficient evidence - mark as unclear
            return self._insufficient_evidence(edge, state)

        # Generate judgment
        judgment = await self._adjudicate(edge, supporting, contradicting, state)

        # Update the edge in the graph
        updated_edge = self._update_edge(edge, judgment, supporting, contradicting)
        graph = state["causal_graph"]
        graph.update_edge(updated_edge)

        print(f"Verdict: {judgment.verdict} (confidence: {judgment.confidence:.2f})")

        return {
            "causal_graph": graph,
            "focus_edge": None,  # Clear focus after judgment
            "focus_edge_id": None,
            "total_edges_investigated": state.get("total_edges_investigated", 0) + 1,
            "audit_feedback": [
                f"Judge: {edge.source_id}->{edge.target_id} = {judgment.verdict} "
                f"(conf: {judgment.confidence:.2f}) - {judgment.reasoning[:100]}..."
            ],
        }

    async def _adjudicate(
        self,
        edge: CausalEdge,
        supporting: list,
        contradicting: list,
        state: ResearchState,
    ) -> JudgmentOutput:
        """Generate judgment on the evidence."""
        graph = state.get("causal_graph")
        source_node = graph.get_node(edge.source_id) if graph else None
        target_node = graph.get_node(edge.target_id) if graph else None

        source_label = source_node.label if source_node else edge.source_id
        target_label = target_node.label if target_node else edge.target_id

        # Format evidence for the prompt
        supporting_text = self._format_evidence(supporting, "Supporting")
        contradicting_text = self._format_evidence(contradicting, "Contradicting")

        prompt = f"""
Adjudicate this causal hypothesis based on the evidence presented:

HYPOTHESIS: {source_label} {edge.hypothesis} {target_label}

=== EVIDENCE FOR (Blue Team) ===
{supporting_text}

=== EVIDENCE AGAINST (Red Team) ===
{contradicting_text}

Instructions:
1. Evaluate the CREDIBILITY of each source (academic > news > blog)
2. Consider the METHODOLOGY (experiments > correlations > anecdotes)
3. Check for CONSISTENCY across sources
4. Identify any CONFOUNDING variables mentioned
5. Weigh the overall STRENGTH of each side

Reach a verdict:
- VERIFIED: Evidence strongly supports causality
- FALSIFIED: Evidence clearly contradicts the hypothesis
- UNCLEAR: Evidence is mixed or insufficient

Provide your reasoning and confidence level.
"""

        try:
            judgment = await self.llm.generate_structured(
                prompt=prompt,
                schema=JudgmentOutput,
                system_prompt=SYSTEM_PROMPT,
            )
            return judgment

        except Exception as e:
            print(f"Judgment generation failed: {e}")
            # Return uncertain verdict on error
            return JudgmentOutput(
                verdict="UNCLEAR",
                confidence=0.3,
                reasoning=f"Judgment process encountered error: {str(e)}",
                key_supporting_points=[],
                key_contradicting_points=[],
                methodological_concerns=["Unable to complete full analysis"],
            )

    def _format_evidence(self, evidence_list: list, label: str) -> str:
        """Format evidence list for the prompt."""
        if not evidence_list:
            return f"No {label.lower()} evidence found."

        lines = []
        for i, ev in enumerate(evidence_list[:5], 1):  # Limit to 5 pieces
            source_url = ev.source.url if ev.source else "Unknown"
            credibility = ev.source.credibility_score if ev.source else 0.5
            lines.append(
                f"{i}. [{credibility:.1f} credibility] {ev.content[:300]}..."
                f"\n   Source: {source_url}"
            )

        return "\n\n".join(lines)

    def _update_edge(
        self,
        edge: CausalEdge,
        judgment: JudgmentOutput,
        supporting: list,
        contradicting: list,
    ) -> CausalEdge:
        """Update the edge with judgment results."""
        # Update status and confidence
        edge.status = judgment.verdict
        edge.confidence = judgment.confidence
        edge.judge_reasoning = judgment.reasoning
        edge.investigation_count += 1

        # Add evidence to the edge
        for ev in supporting:
            edge.add_evidence(ev)
        for ev in contradicting:
            edge.add_evidence(ev)

        return edge

    def _insufficient_evidence(
        self,
        edge: CausalEdge,
        state: ResearchState,
    ) -> dict[str, Any]:
        """Handle case with insufficient evidence."""
        edge.status = "UNCLEAR"
        edge.confidence = 0.2
        edge.judge_reasoning = "Insufficient evidence for definitive judgment"
        edge.investigation_count += 1

        graph = state["causal_graph"]
        graph.update_edge(edge)

        return {
            "causal_graph": graph,
            "focus_edge": None,
            "focus_edge_id": None,
            "total_edges_investigated": state.get("total_edges_investigated", 0) + 1,
            "audit_feedback": [
                f"Judge: {edge.source_id}->{edge.target_id} = UNCLEAR (insufficient evidence)"
            ],
        }
