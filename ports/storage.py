"""Abstract interface for persistence operations."""
from abc import ABC, abstractmethod
from uuid import UUID
from domain.models import ResearchReport
from domain.causal_models import CausalGraph


class StoragePort(ABC):
    """
    Port for persistence.
    Abstracts away FileSystem, S3, Postgres, etc.

    Enables the research system to save and load research artifacts
    without coupling to a specific storage technology.
    """

    @abstractmethod
    async def save_report(self, report: ResearchReport) -> str:
        """
        Persist the research report.

        Args:
            report: The ResearchReport to save

        Returns:
            Storage path or ID
        """
        raise NotImplementedError

    @abstractmethod
    async def load_report(self, report_id: UUID) -> ResearchReport | None:
        """
        Retrieve a report by ID.

        Args:
            report_id: UUID of the report

        Returns:
            ResearchReport or None if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def save_graph(self, graph: CausalGraph) -> str:
        """
        Persist a causal graph.

        Args:
            graph: The CausalGraph to save

        Returns:
            Storage path or ID
        """
        raise NotImplementedError

    @abstractmethod
    async def load_graph(self, graph_id: UUID) -> CausalGraph | None:
        """
        Retrieve a causal graph by ID.

        Args:
            graph_id: UUID of the graph

        Returns:
            CausalGraph or None if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def list_reports(self, limit: int = 10) -> list[dict]:
        """
        List recent reports with metadata.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of report metadata dicts with id, topic, created_at
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_report(self, report_id: UUID) -> bool:
        """
        Delete a report by ID.

        Args:
            report_id: UUID of the report

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def save_checkpoint(self, session_id: str, state: dict) -> str:
        """
        Save a research session checkpoint for recovery.

        Args:
            session_id: Unique session identifier
            state: State dictionary to persist

        Returns:
            Checkpoint path or ID
        """
        raise NotImplementedError

    @abstractmethod
    async def load_checkpoint(self, session_id: str) -> dict | None:
        """
        Load a research session checkpoint.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def storage_type(self) -> str:
        """Return the storage type (e.g., 'local', 's3', 'postgres')."""
        raise NotImplementedError
