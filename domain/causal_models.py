"""Causal graph models for the CAG Research System."""
from typing import Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from domain.models import Evidence


class CausalNode(BaseModel):
    """
    A variable/concept in the causal system.
    Example: 'Interest Rates', 'Customer Churn', 'Inflation'
    """

    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Human-readable label")
    description: str = Field(default="", description="Detailed description of the variable")
    node_type: Literal["VARIABLE", "OUTCOME", "CONFOUNDER", "MEDIATOR"] = Field(
        default="VARIABLE"
    )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.id == other.id
        return False


class CausalEdge(BaseModel):
    """
    A proposed causal mechanism: Source -> Target.
    The core unit of hypothesis in the CAG system.
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: str = Field(..., description="ID of source node")
    target_id: str = Field(..., description="ID of target node")
    hypothesis: str = Field(
        ..., description="The proposed relationship, e.g., 'increases likelihood of'"
    )
    mechanism: str = Field(
        default="", description="Explanation of the causal mechanism"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score after verification"
    )
    status: Literal["PROPOSED", "INVESTIGATING", "VERIFIED", "FALSIFIED", "UNCLEAR"] = Field(
        default="PROPOSED"
    )

    # Evidence tracking
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)

    # Investigation metadata
    investigation_count: int = Field(default=0, description="Number of investigation attempts")
    judge_reasoning: str = Field(default="", description="Judge's reasoning for final status")

    @property
    def edge_label(self) -> str:
        """Human-readable edge description."""
        return f"{self.source_id} -> {self.target_id}: {self.hypothesis}"

    @property
    def evidence_ratio(self) -> float:
        """Ratio of supporting vs total evidence."""
        total = len(self.supporting_evidence) + len(self.contradicting_evidence)
        if total == 0:
            return 0.5  # Neutral when no evidence
        return len(self.supporting_evidence) / total

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the appropriate list based on support flag, avoiding duplicates."""
        target_list = self.supporting_evidence if evidence.supports_hypothesis else self.contradicting_evidence
        
        # Check for duplicates by ID or content
        for existing in target_list:
            if existing.id == evidence.id or existing.content == evidence.content:
                return

        target_list.append(evidence)

    def __hash__(self):
        return hash((self.source_id, self.target_id))

    def __eq__(self, other):
        if isinstance(other, CausalEdge):
            return self.source_id == other.source_id and self.target_id == other.target_id
        return False


class CausalGraph(BaseModel):
    """
    The Agent's World Model - a Directed Acyclic Graph of causal relationships.
    This is the central data structure for the CAG research methodology.
    """

    id: UUID = Field(default_factory=uuid4)
    nodes: list[CausalNode] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)
    root_query: str = Field(default="", description="The original research query")

    def get_node(self, node_id: str) -> CausalNode | None:
        """Get a node by ID."""
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_edge(self, source_id: str, target_id: str) -> CausalEdge | None:
        """Get an edge by source and target IDs."""
        return next(
            (e for e in self.edges if e.source_id == source_id and e.target_id == target_id),
            None,
        )

    def get_edge_by_id(self, edge_id: UUID) -> CausalEdge | None:
        """Get an edge by its UUID."""
        return next((e for e in self.edges if e.id == edge_id), None)

    def add_node(self, node: CausalNode) -> bool:
        """Add a node if it doesn't exist. Returns True if added."""
        if not self.get_node(node.id):
            self.nodes.append(node)
            return True
        return False

    def add_edge(self, edge: CausalEdge) -> bool:
        """Add an edge if it doesn't exist. Returns True if added."""
        if not self.get_edge(edge.source_id, edge.target_id):
            self.edges.append(edge)
            return True
        return False

    def update_edge(self, updated_edge: CausalEdge) -> bool:
        """Update an existing edge. Returns True if updated."""
        for i, edge in enumerate(self.edges):
            if edge.source_id == updated_edge.source_id and edge.target_id == updated_edge.target_id:
                self.edges[i] = updated_edge
                return True
        return False

    def get_unverified_edges(self) -> list[CausalEdge]:
        """Get all edges that haven't been verified yet."""
        return [e for e in self.edges if e.status in ("PROPOSED", "UNCLEAR")]

    def get_edges_by_status(self, status: str) -> list[CausalEdge]:
        """Get edges by their verification status."""
        return [e for e in self.edges if e.status == status]

    def get_outgoing_edges(self, node_id: str) -> list[CausalEdge]:
        """Get all edges where node_id is the source."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming_edges(self, node_id: str) -> list[CausalEdge]:
        """Get all edges where node_id is the target."""
        return [e for e in self.edges if e.target_id == node_id]

    def is_dag(self) -> bool:
        """
        Verify the graph is a Directed Acyclic Graph.
        Uses Kahn's algorithm for topological sort.
        """
        if not self.nodes:
            return True

        # Build adjacency list and in-degree count
        in_degree = {n.id: 0 for n in self.nodes}
        adj = {n.id: [] for n in self.nodes}

        for edge in self.edges:
            if edge.source_id in adj and edge.target_id in in_degree:
                adj[edge.source_id].append(edge.target_id)
                in_degree[edge.target_id] += 1

        # Find all nodes with no incoming edges
        queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(self.nodes)

    def get_verification_summary(self) -> dict:
        """Get a summary of edge verification statuses."""
        summary = {
            "total_edges": len(self.edges),
            "verified": len(self.get_edges_by_status("VERIFIED")),
            "falsified": len(self.get_edges_by_status("FALSIFIED")),
            "unclear": len(self.get_edges_by_status("UNCLEAR")),
            "proposed": len(self.get_edges_by_status("PROPOSED")),
            "investigating": len(self.get_edges_by_status("INVESTIGATING")),
        }
        summary["completion_rate"] = (
            (summary["verified"] + summary["falsified"]) / summary["total_edges"] * 100
            if summary["total_edges"] > 0
            else 0
        )
        return summary

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram syntax."""
        lines = ["graph TD"]

        # Add nodes
        for node in self.nodes:
            shape = {
                "OUTCOME": f"(({node.label}))",
                "CONFOUNDER": f"[/{node.label}/]",
                "MEDIATOR": f"{{{{{node.label}}}}}",
            }.get(node.node_type, f"[{node.label}]")
            lines.append(f"    {node.id}{shape}")

        # Add edges with status colors
        for edge in self.edges:
            style = {
                "VERIFIED": "-->",
                "FALSIFIED": "-.-x",
                "UNCLEAR": "-.->",
                "PROPOSED": "-->",
            }.get(edge.status, "-->")
            lines.append(f"    {edge.source_id} {style}|{edge.hypothesis}| {edge.target_id}")

        return "\n".join(lines)
