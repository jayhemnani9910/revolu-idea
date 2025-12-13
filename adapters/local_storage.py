"""Local file storage adapter implementation."""
import json
import os
from datetime import datetime
from uuid import UUID
from pathlib import Path

from ports.storage import StoragePort
from domain.models import ResearchReport
from domain.causal_models import CausalGraph
from domain.exceptions import AdapterError


class LocalStorageAdapter(StoragePort):
    """
    Adapter for local file system storage.
    Stores reports and graphs as JSON files.
    """

    def __init__(
        self,
        base_path: str = "output",
        reports_dir: str = "reports",
        graphs_dir: str = "graphs",
        checkpoints_dir: str = "checkpoints",
    ):
        """
        Initialize the local storage adapter.

        Args:
            base_path: Base directory for storage
            reports_dir: Subdirectory for reports
            graphs_dir: Subdirectory for graphs
            checkpoints_dir: Subdirectory for checkpoints
        """
        self.base_path = Path(base_path)
        self.reports_path = self.base_path / reports_dir
        self.graphs_path = self.base_path / graphs_dir
        self.checkpoints_path = self.base_path / checkpoints_dir

        # Create directories
        for path in [self.reports_path, self.graphs_path, self.checkpoints_path]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def storage_type(self) -> str:
        return "local"

    async def save_report(self, report: ResearchReport) -> str:
        """Persist the research report as JSON."""
        try:
            filename = f"{report.id}.json"
            filepath = self.reports_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

            # Also save markdown version
            md_filename = f"{report.id}.md"
            md_filepath = self.reports_path / md_filename
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())

            return str(filepath)

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "save_report", e)

    async def load_report(self, report_id: UUID) -> ResearchReport | None:
        """Retrieve a report by ID."""
        try:
            filename = f"{report_id}.json"
            filepath = self.reports_path / filename

            if not filepath.exists():
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ResearchReport(**data)

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "load_report", e)

    async def save_graph(self, graph: CausalGraph) -> str:
        """Persist a causal graph as JSON."""
        try:
            filename = f"{graph.id}.json"
            filepath = self.graphs_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(graph.model_dump_json(indent=2))

            # Also save mermaid diagram
            mermaid_filename = f"{graph.id}.mmd"
            mermaid_filepath = self.graphs_path / mermaid_filename
            with open(mermaid_filepath, "w", encoding="utf-8") as f:
                f.write(graph.to_mermaid())

            return str(filepath)

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "save_graph", e)

    async def load_graph(self, graph_id: UUID) -> CausalGraph | None:
        """Retrieve a causal graph by ID."""
        try:
            filename = f"{graph_id}.json"
            filepath = self.graphs_path / filename

            if not filepath.exists():
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CausalGraph(**data)

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "load_graph", e)

    async def list_reports(self, limit: int = 10) -> list[dict]:
        """List recent reports with metadata."""
        try:
            reports = []
            json_files = list(self.reports_path.glob("*.json"))

            # Sort by modification time, newest first
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for filepath in json_files[:limit]:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append({
                        "id": data.get("id"),
                        "topic": data.get("topic"),
                        "created_at": data.get("created_at"),
                        "verification_status": data.get("verification_status", "UNKNOWN"),
                        "filepath": str(filepath),
                    })

            return reports

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "list_reports", e)

    async def delete_report(self, report_id: UUID) -> bool:
        """Delete a report by ID."""
        try:
            json_filepath = self.reports_path / f"{report_id}.json"
            md_filepath = self.reports_path / f"{report_id}.md"

            deleted = False
            if json_filepath.exists():
                json_filepath.unlink()
                deleted = True
            if md_filepath.exists():
                md_filepath.unlink()

            return deleted

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "delete_report", e)

    async def save_checkpoint(self, session_id: str, state: dict) -> str:
        """Save a research session checkpoint."""
        try:
            filename = f"{session_id}.json"
            filepath = self.checkpoints_path / filename

            # Add timestamp to state
            state["_checkpoint_time"] = datetime.now().isoformat()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)

            return str(filepath)

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "save_checkpoint", e)

    async def load_checkpoint(self, session_id: str) -> dict | None:
        """Load a research session checkpoint."""
        try:
            filename = f"{session_id}.json"
            filepath = self.checkpoints_path / filename

            if not filepath.exists():
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "load_checkpoint", e)

    async def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        Remove checkpoints older than max_age_hours.
        Returns number of files deleted.
        """
        try:
            deleted_count = 0
            cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

            for filepath in self.checkpoints_path.glob("*.json"):
                if filepath.stat().st_mtime < cutoff:
                    filepath.unlink()
                    deleted_count += 1

            return deleted_count

        except Exception as e:
            raise AdapterError("LocalStorageAdapter", "cleanup_old_checkpoints", e)
