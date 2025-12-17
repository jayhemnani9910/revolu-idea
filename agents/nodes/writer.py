"""Writer Node - Synthesizes verified findings into a research report."""
from typing import Any
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from ports.llm import LLMPort
from agents.state import ResearchState
from domain.models import ResearchReport, ResearchSection, ResearchFinding, Evidence
from domain.causal_models import CausalGraph


SYSTEM_PROMPT = """You are a Technical Research Writer.
Your task is to synthesize verified findings into a comprehensive research report.

Writing principles:
1. ACCURACY: Only include claims supported by evidence
2. CLARITY: Write for an informed but non-expert audience
3. BALANCE: Present both verified and contested findings
4. CITATIONS: Every factual claim must reference its source
5. STRUCTURE: Organize logically with clear sections

Report structure:
- Executive Summary
- Key Findings (with verification status)
- Detailed Analysis by causal relationship
- Methodology note
- Limitations and caveats"""


class SectionContent(BaseModel):
    """Structured output for a report section."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content with inline citations")
    key_points: list[str] = Field(
        default_factory=list, description="Bullet points of key findings"
    )


class ReportOutline(BaseModel):
    """Structured output for report outline."""

    summary: str = Field(..., description="Executive summary (2-3 paragraphs)")
    sections: list[SectionContent] = Field(
        ..., description="Report sections in order"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Limitations and caveats"
    )


class WriterNode:
    """
    The Writer - Synthesizes all verified findings into a coherent research report.

    This is the final synthesis step that produces the deliverable.
    It only uses information from the verified causal graph and
    explicitly notes unverified or contested relationships.
    """

    def __init__(self, llm: LLMPort):
        """
        Initialize the writer node.

        Args:
            llm: LLM port for generation
        """
        self.llm = llm

    async def __call__(self, state: ResearchState) -> dict[str, Any]:
        """
        Generate the final research report.

        Args:
            state: Current research state

        Returns:
            State updates with final_report
        """
        print("--- Writer: Synthesizing Final Report ---")

        graph = state.get("causal_graph")
        if not graph:
            return {"error": "No causal graph to synthesize"}

        # Generate the report
        report = await self._generate_report(graph, state)

        print(f"Generated report: {report.topic[:50]}...")
        print(f"  Sections: {len(report.sections)}")
        print(f"  Total findings: {report.total_findings}")

        return {
            "final_report": report,
            "audit_feedback": [
                f"Writer: Generated report with {len(report.sections)} sections, "
                f"{report.total_findings} findings"
            ],
        }

    async def _generate_report(
        self,
        graph: CausalGraph,
        state: ResearchState,
    ) -> ResearchReport:
        """Generate the full research report."""
        # Gather all evidence and verdicts
        context = self._build_context(graph, state)

        prompt = f"""
Write a SHORT research summary based on these verified findings:

QUERY: {state['root_query']}

=== FINDINGS ===
{context['verified'] if context['verified'] else 'No verified relationships found.'}

=== UNVERIFIED HYPOTHESES ===
{context['unclear'] if context['unclear'] else 'None.'}

Instructions:
1. Summarize the key findings in 2-3 paragraphs.
2. State clearly what is known and what remains unclear.
3. Do not invent information.
"""

        try:
            # Removed audit trail and complex instructions to save tokens
            outline = await self.llm.generate_structured(
                prompt=prompt,
                schema=ReportOutline,
                system_prompt=SYSTEM_PROMPT,
            )

            # Build the report object
            report = ResearchReport(
                topic=state["root_query"],
                summary=outline.summary,
                verification_status=self._determine_status(graph),
            )

            # Add sections with findings
            for section_content in outline.sections:
                section = ResearchSection(
                    title=section_content.title,
                    content=section_content.content,
                    findings=self._extract_findings(section_content, graph),
                )
                report.add_section(section)

            # Add deterministic findings based on the current causal graph so the
            # report's verification metrics reflect actual edge verdicts.
            report.add_section(self._build_detailed_findings_section(graph))

            # Add methodology section
            methodology_section = ResearchSection(
                title="Methodology",
                content=self._generate_methodology(state),
                findings=[],
            )
            report.add_section(methodology_section)

            # Add limitations section if we have any
            if outline.limitations:
                limitations_section = ResearchSection(
                    title="Limitations",
                    content="\n".join(f"- {lim}" for lim in outline.limitations),
                    findings=[],
                )
                report.add_section(limitations_section)

            return report

        except Exception as e:
            print(f"Report generation failed (likely rate limit): {e}")
            print("Generating fallback manual report...")
            
            # Fallback: Construct report manually from graph data
            summary = f"Research into '{state['root_query']}' was conducted. " \
                      f"Due to API rate limits, this is a structured summary of findings."
            
            report = ResearchReport(
                topic=state["root_query"],
                summary=summary,
                verification_status=self._determine_status(graph),
            )

            report.add_section(self._build_detailed_findings_section(graph))
            report.add_section(
                ResearchSection(
                    title="Methodology",
                    content=self._generate_methodology(state),
                    findings=[],
                )
            )
            
            return report

    def _build_context(self, graph: CausalGraph, state: ResearchState) -> dict:
        """Build context string from graph for the prompt."""
        context = {
            "verified": [],
            "falsified": [],
            "unclear": [],
        }

        for edge in graph.edges:
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)

            source_label = source.label if source else edge.source_id
            target_label = target.label if target else edge.target_id

            entry = (
                f"- {edge.source_id} -> {edge.target_id}: {edge.status} (Conf: {edge.confidence:.2f})"
            )
            
            # Ultra-minimal context for free tier rate limits
            # Removed reasoning and evidence details entirely
            
            if edge.status == "VERIFIED":
                context["verified"].append(entry)
            elif edge.status == "FALSIFIED":
                context["falsified"].append(entry)
            else:
                context["unclear"].append(entry)

        # Convert lists to strings
        for key in context:
            if context[key]:
                context[key] = "\n\n".join(context[key])
            else:
                context[key] = "None found."

        return context

    def _determine_status(self, graph: CausalGraph) -> str:
        """Determine overall report verification status."""
        summary = graph.get_verification_summary()

        if summary["total_edges"] == 0:
            return "DRAFT"

        # If all edges are verified or falsified
        resolved = summary["verified"] + summary["falsified"]
        if resolved == summary["total_edges"]:
            return "VERIFIED"

        # If at least half are resolved
        if resolved >= summary["total_edges"] / 2:
            return "PARTIALLY_VERIFIED"

        return "DRAFT"

    def _extract_findings(
        self,
        section_content: SectionContent,
        graph: CausalGraph,
    ) -> list[ResearchFinding]:
        """Extract findings from section content."""
        findings = []

        # Create findings from key points
        for point in section_content.key_points:
            # Try to match with graph edges
            verdict = "UNVERIFIED"
            for edge in graph.edges:
                if edge.source_id.lower() in point.lower() or edge.target_id.lower() in point.lower():
                    verdict = edge.status
                    # Map generic graph status to report verdict
                    if verdict in ("UNCLEAR", "PROPOSED", "INVESTIGATING"):
                        verdict = "UNVERIFIED"
                    break

            finding = ResearchFinding(
                claim=point,
                verdict=verdict,
                confidence=0.7 if verdict == "VERIFIED" else 0.5,
            )
            findings.append(finding)

        return findings

    def _generate_methodology(self, state: ResearchState) -> str:
        """Generate methodology section content."""
        summary = state["causal_graph"].get_verification_summary() if state.get("causal_graph") else {}
        investigated = state.get("total_edges_investigated", 0)

        return f"""This research was conducted using the Causal-Adversarial Graph (CAG) methodology:

1. **Causal Graph Construction**: The research query was decomposed into a directed acyclic graph
   of causal hypotheses linking key variables.

2. **Adversarial Investigation**: Each causal edge was investigated by two parallel processes:
   - **Blue Team (Supporter)**: Searched for evidence supporting the hypothesis
   - **Red Team (Adversary)**: Searched for evidence contradicting the hypothesis

3. **Dialectical Judgment**: An impartial judge weighed the evidence from both teams to reach
   a verdict (VERIFIED, FALSIFIED, or UNCLEAR) for each causal relationship.

4. **Research Statistics**:
   - Total edges in graph: {summary.get('total_edges', 'N/A')}
   - Total edges investigated: {investigated}
   - Verified: {summary.get('verified', 'N/A')}
   - Falsified: {summary.get('falsified', 'N/A')}
   - Unclear: {summary.get('unclear', 'N/A')}
   - Completion rate: {summary.get('completion_rate', 0):.1f}%
   - Session ID: {state.get('session_id', 'N/A')}
"""

    def _build_detailed_findings_section(self, graph: CausalGraph) -> ResearchSection:
        """Build a deterministic findings section directly from the causal graph."""
        if not graph.edges:
            return ResearchSection(
                title="Detailed Causal Findings",
                content="No causal edges were proposed for investigation.",
                findings=[],
            )

        lines: list[str] = []
        findings: list[ResearchFinding] = []

        for edge in graph.edges:
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)

            source_label = source.label if source else edge.source_id
            target_label = target.label if target else edge.target_id

            verdict = (
                "UNVERIFIED"
                if edge.status in ("PROPOSED", "INVESTIGATING", "UNCLEAR")
                else edge.status
            )

            claim = f"{source_label} {edge.hypothesis} {target_label}"

            lines.append(f"### {edge.source_id} -> {edge.target_id}")
            lines.append(f"**Claim:** {claim}")
            lines.append(f"**Status:** {edge.status} (confidence {edge.confidence:.2f})")
            lines.append(
                f"**Evidence:** {len(edge.supporting_evidence)} supporting, {len(edge.contradicting_evidence)} contradicting"
            )
            if edge.judge_reasoning:
                lines.append(f"**Reasoning:** {edge.judge_reasoning}")
            lines.append("")

            findings.append(
                ResearchFinding(
                    claim=claim,
                    verdict=verdict,  # type: ignore[arg-type]
                    confidence=edge.confidence,
                    reasoning=edge.judge_reasoning,
                    supporting_evidence=edge.supporting_evidence[:3],
                    contradicting_evidence=edge.contradicting_evidence[:3],
                )
            )

        return ResearchSection(
            title="Detailed Causal Findings",
            content="\n".join(lines).strip(),
            findings=findings,
        )
