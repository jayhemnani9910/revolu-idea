import asyncio
from pathlib import Path
from threading import get_ident

from adapters import local_storage
from domain.causal_models import CausalGraph
from domain.models import ResearchReport


def test_storage_keeps_file_operations_off_event_loop(tmp_path, monkeypatch):
    storage = local_storage.LocalStorageAdapter(str(tmp_path))
    loop_thread = get_ident()
    blocking_calls = []

    def track(operation):
        def recorded(*args, **kwargs):
            if get_ident() == loop_thread:
                blocking_calls.append(operation.__name__)
            return operation(*args, **kwargs)

        return recorded

    async def round_trip():
        report = ResearchReport(topic="Storage")
        graph = CausalGraph(root_query=report.topic)
        with monkeypatch.context() as patch:
            patch.setattr(local_storage, "open", track(open), raising=False)
            for name in ("open", "exists", "glob", "stat", "unlink"):
                patch.setattr(Path, name, track(getattr(Path, name)))

            await storage.save_report(report)
            assert await storage.load_report(report.id) == report
            assert (await storage.list_reports())[0]["topic"] == report.topic
            await storage.save_graph(graph)
            assert await storage.load_graph(graph.id) == graph
            await storage.save_checkpoint("session", {"step": 2})
            assert (await storage.load_checkpoint("session"))["step"] == 2
            assert await storage.delete_report(report.id)
            assert await storage.cleanup_old_checkpoints() == 0

    asyncio.run(round_trip())
    assert not blocking_calls, f"File operations on the event loop: {blocking_calls}"
