"""Mock adapters for testing without API costs."""
import json
import random
from datetime import datetime
from typing import TypeVar, Type
from uuid import UUID, uuid4
from pydantic import BaseModel

from ports.llm import LLMPort
from ports.search import SearchPort
from ports.storage import StoragePort
from domain.models import Citation, ResearchReport
from domain.causal_models import CausalGraph, CausalNode, CausalEdge

T = TypeVar("T", bound=BaseModel)


class MockLLMAdapter(LLMPort):
    """
    Mock LLM adapter for testing.
    Returns deterministic responses based on prompt patterns.
    """

    def __init__(self, model_name: str = "mock-gpt-4", delay: float = 0.0):
        """
        Initialize mock LLM.

        Args:
            model_name: Name to report as model
            delay: Simulated delay in seconds (for testing async)
        """
        self._model_name = model_name
        self._delay = delay
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate mock text response."""
        import asyncio
        if self._delay:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        # Pattern matching for different prompt types
        if "causal" in prompt.lower() or "dag" in prompt.lower():
            return "Based on analysis, the key causal relationships identified are A->B and B->C."

        if "disprove" in prompt.lower() or "contradict" in prompt.lower():
            return "Counter-evidence suggests this relationship may be spurious due to confounding variables."

        if "support" in prompt.lower() or "prove" in prompt.lower():
            return "Multiple studies support this causal mechanism through controlled experiments."

        if "judge" in prompt.lower() or "adjudicate" in prompt.lower():
            return "After weighing the evidence, the hypothesis is VERIFIED with moderate confidence."

        if "report" in prompt.lower() or "synthesize" in prompt.lower():
            return "# Research Report\n\nThis analysis examines the causal relationships...\n\n## Key Findings\n\n1. Primary causal path confirmed\n2. Secondary effects noted"

        return f"Mock response for prompt (call #{self._call_count}): {prompt[:100]}..."

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate mock structured response."""
        import asyncio
        if self._delay:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        # Handle CausalGraph schema
        if schema.__name__ == "CausalGraph":
            return CausalGraph(
                nodes=[
                    CausalNode(id="A", label="Factor A", description="Primary cause"),
                    CausalNode(id="B", label="Factor B", description="Mediator"),
                    CausalNode(id="C", label="Outcome C", description="Effect", node_type="OUTCOME"),
                ],
                edges=[
                    CausalEdge(source_id="A", target_id="B", hypothesis="increases"),
                    CausalEdge(source_id="B", target_id="C", hypothesis="leads to"),
                ],
            )

        if schema.__name__ == "CausalEdge":
            return CausalEdge(
                source_id="A",
                target_id="B",
                hypothesis="influences",
                status="VERIFIED",
                confidence=0.75,
                judge_reasoning="Evidence supports causal link",
            )
            
        import re
        
        # Helper to extract topic
        topic = "the topic"
        match = re.search(r"QUERY: (.*?)(\n|$)", prompt)
        if match:
            topic = match.group(1).strip()
        elif "startups" in prompt.lower():
            topic = "startup failure"
        elif "sky" in prompt.lower():
            topic = "sky color"
            
        if schema.__name__ == "PlannerOutput":
            # Create a mock planner output
            if "startup" in topic.lower():
                 return schema(
                    research_goal=f"To understand the causal mechanisms of {topic}.",
                    nodes=[
                        {"id": "NoPMF", "label": "Lack of Product-Market Fit", "description": "Product does not satisfy market demand", "node_type": "VARIABLE"},
                        {"id": "BurnRate", "label": "High Burn Rate", "description": "Spending capital too fast", "node_type": "MEDIATOR"},
                        {"id": "TeamConflict", "label": "Team Conflict", "description": "Internal disputes", "node_type": "VARIABLE"},
                        {"id": "Failure", "label": "Startup Failure", "description": "Business ceases operations", "node_type": "OUTCOME"},
                    ],
                    edges=[
                        {"source_id": "NoPMF", "target_id": "BurnRate", "hypothesis": "accelerates"},
                        {"source_id": "BurnRate", "target_id": "Failure", "hypothesis": "causes"},
                        {"source_id": "TeamConflict", "target_id": "Failure", "hypothesis": "contributes to"},
                    ],
                    reasoning=f"The causal chain involves factors like PMF and burn rate leading to {topic}.",
                )
            
            # Default generic DAG
            return schema(
                research_goal=f"To understand the causal mechanisms of {topic}.",
                nodes=[
                    {"id": "FactorA", "label": f"Factor A ({topic})", "description": "Primary driver", "node_type": "VARIABLE"},
                    {"id": "FactorB", "label": "Factor B", "description": "Mediating variable", "node_type": "MEDIATOR"},
                    {"id": "Outcome", "label": "Outcome", "description": "Final result", "node_type": "OUTCOME"},
                ],
                edges=[
                    {"source_id": "FactorA", "target_id": "FactorB", "hypothesis": "influences"},
                    {"source_id": "FactorB", "target_id": "Outcome", "hypothesis": "determines"},
                ],
                reasoning=f"Constructed a causal graph to analyze {topic}.",
            )

        if schema.__name__ == "ReportOutline":
            if "startup" in topic.lower():
                return schema(
                    summary=f"This report confirms that high burn rates and lack of PMF are primary drivers of {topic}.",
                    sections=[
                        {
                            "title": "Financial Factors", 
                            "content": "Financial mismanagement is a key cause.", 
                            "key_points": ["High Burn Rate causes Startup Failure", "NoPMF accelerates High Burn Rate"]
                        },
                        {
                            "title": "Team Dynamics", 
                            "content": "Internal conflict destabilizes the company.", 
                            "key_points": ["Team Conflict contributes to Startup Failure"]
                        }
                    ],
                    limitations=["This is a mock report."]
                )
            
            return schema(
                summary=f"This is a mock executive summary explaining the causal factors of {topic}.",
                sections=[
                    {
                        "title": "Introduction", 
                        "content": f"This report investigates {topic}.", 
                        "key_points": [f"{topic} is complex", "Multiple factors involved"]
                    },
                    {
                        "title": "Analysis", 
                        "content": f"Analysis shows significant relationships in {topic}.", 
                        "key_points": ["Verified causal links", "Evidence-based conclusion"]
                    }
                ],
                limitations=["This is a mock report."]
            )

        if schema.__name__ == "AttackQueries":
            return schema(
                queries=[f"contradicting evidence for {topic}", f"counter examples {topic}"],
                attack_strategy=f"Mock attack strategy focusing on counter-evidence for {topic}."
            )

        if schema.__name__ == "SupportQueries":
            return schema(
                queries=[f"supporting evidence for {topic}", f"proof of {topic}"],
                search_strategy=f"Mock search strategy focusing on supporting evidence for {topic}."
            )

        if schema.__name__ == "JudgmentOutput":
            # Randomize verdict for variety if needed, or stick to VERIFIED/UNCLEAR
            return schema(
                verdict="VERIFIED",
                confidence=0.85,
                reasoning="Mock judgment reasoning based on strong supporting evidence.",
                key_supporting_points=["Point 1", "Point 2"],
                key_contradicting_points=["Point 3"],
                methodological_concerns=["None"]
            )

        # Default: try to create instance with minimal data
        try:
            # Get schema fields and create minimal valid data
            fields = schema.model_fields
            data = {}
            for name, field in fields.items():
                if field.is_required():
                    if field.annotation == str:
                        data[name] = f"mock_{name}"
                    elif field.annotation == int:
                        data[name] = 1
                    elif field.annotation == float:
                        data[name] = 0.5
                    elif field.annotation == bool:
                        data[name] = True
                    elif field.annotation == list:
                        data[name] = []
            return schema(**data)
        except Exception:
            raise ValueError(f"Cannot create mock instance of {schema.__name__}")

    async def generate_list(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_items: int = 5,
    ) -> list[str]:
        """Generate mock list of strings."""
        import asyncio
        if self._delay:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        # Return relevant mock queries based on prompt
        if "disprove" in prompt.lower() or "contradict" in prompt.lower():
            return [
                "no correlation between X and Y study",
                "X does not cause Y evidence",
                "confounding variables in X Y relationship",
            ][:max_items]

        if "support" in prompt.lower() or "prove" in prompt.lower():
            return [
                "X causes Y research evidence",
                "causal mechanism X to Y",
                "controlled study X Y relationship",
            ][:max_items]

        return [f"mock query {i+1}" for i in range(max_items)]


class MockSearchAdapter(SearchPort):
    """
    Mock search adapter for testing.
    Returns deterministic search results.
    """

    def __init__(self, delay: float = 0.0):
        """
        Initialize mock search.

        Args:
            delay: Simulated network delay in seconds
        """
        self._delay = delay
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> list[Citation]:
        """Execute mock search."""
        import asyncio
        if self._delay:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        # Generate mock citations based on query patterns
        citations = []
        for i in range(min(max_results, 3)):
            # Vary credibility based on mock "source type"
            source_types = [
                ("arxiv.org", 0.9, "Academic paper"),
                ("nature.com", 0.95, "Journal article"),
                ("wikipedia.org", 0.7, "Encyclopedia entry"),
                ("medium.com", 0.5, "Blog post"),
                ("news.example.com", 0.6, "News article"),
            ]
            source = source_types[i % len(source_types)]

            citations.append(
                Citation(
                    url=f"https://{source[0]}/mock-{query.replace(' ', '-')[:20]}-{i}",
                    title=f"{source[2]}: {query[:30]}...",
                    snippet=f"This {source[2].lower()} discusses {query}. Key findings indicate "
                            f"relevant information about the topic. Mock result #{i+1}.",
                    credibility_score=source[1],
                    access_date=datetime.now(),
                )
            )

        return citations

    async def search_news(
        self,
        query: str,
        max_results: int = 5,
        days_back: int = 7,
    ) -> list[Citation]:
        """Mock news search."""
        import asyncio
        if self._delay:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        news_sources = ["reuters.com", "apnews.com", "bbc.com"]
        citations = []

        for i in range(min(max_results, 3)):
            source = news_sources[i % len(news_sources)]
            citations.append(
                Citation(
                    url=f"https://{source}/news/{query.replace(' ', '-')[:20]}-{i}",
                    title=f"Breaking: {query[:40]}...",
                    snippet=f"Recent developments in {query}. Experts report significant findings. "
                            f"Published within the last {days_back} days.",
                    credibility_score=0.8,
                    access_date=datetime.now(),
                )
            )

        return citations

    async def search_academic(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Citation]:
        """Mock academic search."""
        import asyncio
        if self._delay:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        academic_sources = ["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "semanticscholar.org"]
        citations = []

        for i in range(min(max_results, 3)):
            source = academic_sources[i % len(academic_sources)]
            citations.append(
                Citation(
                    url=f"https://{source}/paper/{query.replace(' ', '-')[:20]}-{i}",
                    title=f"Research Paper: {query[:40]}...",
                    snippet=f"Abstract: This paper investigates {query}. Our methodology includes "
                            f"rigorous experimental design. Results show significant p<0.05.",
                    credibility_score=0.9,
                    access_date=datetime.now(),
                )
            )

        return citations


class MockStorageAdapter(StoragePort):
    """
    Mock storage adapter for testing.
    Stores data in memory.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._reports: dict[str, ResearchReport] = {}
        self._graphs: dict[str, CausalGraph] = {}
        self._checkpoints: dict[str, dict] = {}

    @property
    def storage_type(self) -> str:
        return "mock"

    async def save_report(self, report: ResearchReport) -> str:
        """Save report to memory."""
        key = str(report.id)
        self._reports[key] = report
        return f"mock://reports/{key}"

    async def load_report(self, report_id: UUID) -> ResearchReport | None:
        """Load report from memory."""
        return self._reports.get(str(report_id))

    async def save_graph(self, graph: CausalGraph) -> str:
        """Save graph to memory."""
        key = str(graph.id)
        self._graphs[key] = graph
        return f"mock://graphs/{key}"

    async def load_graph(self, graph_id: UUID) -> CausalGraph | None:
        """Load graph from memory."""
        return self._graphs.get(str(graph_id))

    async def list_reports(self, limit: int = 10) -> list[dict]:
        """List reports from memory."""
        reports = []
        for key, report in list(self._reports.items())[:limit]:
            reports.append({
                "id": key,
                "topic": report.topic,
                "created_at": report.created_at.isoformat(),
                "verification_status": report.verification_status,
            })
        return reports

    async def delete_report(self, report_id: UUID) -> bool:
        """Delete report from memory."""
        key = str(report_id)
        if key in self._reports:
            del self._reports[key]
            return True
        return False

    async def save_checkpoint(self, session_id: str, state: dict) -> str:
        """Save checkpoint to memory."""
        self._checkpoints[session_id] = state
        return f"mock://checkpoints/{session_id}"

    async def load_checkpoint(self, session_id: str) -> dict | None:
        """Load checkpoint from memory."""
        return self._checkpoints.get(session_id)

    def clear_all(self):
        """Clear all stored data (for test cleanup)."""
        self._reports.clear()
        self._graphs.clear()
        self._checkpoints.clear()
