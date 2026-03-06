"""Core domain entities and value objects."""
from typing import Literal
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """
    Represents a specific source used to back a claim.
    Immutable value object.
    """

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Page/article title")
    snippet: str = Field(..., description="Relevant text excerpt")
    access_date: datetime = Field(default_factory=datetime.now)
    credibility_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0.0 to 1.0 score of source reliability",
    )
    domain: str = Field(default="", description="Extracted domain name")

    def model_post_init(self, __context) -> None:
        """Extract domain from URL if not provided."""
        if not self.domain and self.url:
            try:
                self.domain = self.url.split("//")[-1].split("/")[0]
            except (IndexError, AttributeError):
                self.domain = "unknown"


class Evidence(BaseModel):
    """
    Data retrieved to support OR attack a hypothesis.
    Core unit of the adversarial research process.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., description="The evidence text")
    source: Citation = Field(..., description="Source citation")
    supports_hypothesis: bool = Field(
        ..., description="True = supporting evidence, False = contradicting evidence"
    )
    relevance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How relevant to the hypothesis"
    )
    extraction_method: str = Field(
        default="search", description="How this evidence was obtained"
    )


class ResearchFinding(BaseModel):
    """
    A verified atom of information after judge adjudication.
    """

    id: UUID = Field(default_factory=uuid4)
    claim: str = Field(..., description="The factual claim")
    verdict: Literal["VERIFIED", "FALSIFIED", "CONTESTED", "UNVERIFIED"] = Field(
        default="UNVERIFIED"
    )
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the verdict"
    )
    reasoning: str = Field(default="", description="Judge's reasoning for verdict")

    @property
    def total_evidence_count(self) -> int:
        return len(self.supporting_evidence) + len(self.contradicting_evidence)


class AuditResult(BaseModel):
    """
    The result of an auditor's review of research quality.
    """

    passed: bool = Field(..., description="Whether the audit passed")
    score: int = Field(ge=0, le=10, description="Quality score 0-10")
    feedback: str = Field(..., description="Detailed feedback")
    hallucination_check: Literal["CLEAN", "SUSPECT", "CONFIRMED_HALLUCINATION"] = Field(
        default="CLEAN"
    )
    issues_found: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ResearchSection(BaseModel):
    """
    A section of the final research report.
    """

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    findings: list[ResearchFinding] = Field(default_factory=list)

    @property
    def citation_count(self) -> int:
        total = 0
        for f in self.findings:
            total += len(f.supporting_evidence) + len(f.contradicting_evidence)
        return total

    @property
    def verified_findings_count(self) -> int:
        return sum(1 for f in self.findings if f.verdict == "VERIFIED")


class ResearchReport(BaseModel):
    """
    The Aggregate Root representing the final research output.
    """

    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(..., description="Research topic/query")
    summary: str = Field(default="", description="Executive summary")
    sections: list[ResearchSection] = Field(default_factory=list)
    methodology: str = Field(
        default="Causal-Adversarial Graph (CAG) methodology with Red/Blue team verification"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    total_cost: float = Field(default=0.0, description="Total API cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    verification_status: Literal["DRAFT", "VERIFIED", "PARTIALLY_VERIFIED"] = Field(
        default="DRAFT"
    )

    def add_section(self, section: ResearchSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    @property
    def total_findings(self) -> int:
        return sum(len(s.findings) for s in self.sections)

    @property
    def total_citations(self) -> int:
        return sum(s.citation_count for s in self.sections)

    @property
    def verified_percentage(self) -> float:
        if not self.total_findings:
            return 0.0
        verified = sum(s.verified_findings_count for s in self.sections)
        return (verified / self.total_findings) * 100

    def to_markdown(self) -> str:
        """Export report as markdown."""
        lines = [
            f"# {self.topic}",
            "",
            f"**Generated:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Methodology:** {self.methodology}",
            f"**Status:** {self.verification_status}",
            f"**Findings:** {self.total_findings} ({self.verified_percentage:.1f}% verified)",
            "",
            "## Summary",
            self.summary or "_No summary generated_",
            "",
        ]

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

            if section.findings:
                lines.append("### Key Findings")
                for finding in section.findings:
                    status_emoji = {
                        "VERIFIED": "[VERIFIED]",
                        "FALSIFIED": "[FALSIFIED]",
                        "CONTESTED": "[CONTESTED]",
                        "UNVERIFIED": "[UNVERIFIED]",
                    }.get(finding.verdict, "?")
                    lines.append(f"- {status_emoji} {finding.claim}")
                lines.append("")

        return "\n".join(lines)
