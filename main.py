#!/usr/bin/env python3
"""CAG Deep Research System - Ollama + Tavily"""
import asyncio
import argparse
from datetime import datetime

from container import Container
from agents.state import create_initial_state


async def run_research(query: str, container: Container) -> dict:
    print("\n" + "=" * 60)
    print("CAG DEEP RESEARCH SYSTEM")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    initial_state = create_initial_state(query=query, max_depth=container.settings.max_recursion_depth)
    graph = container.get_graph()

    final_state = None
    async for event in graph.astream(initial_state):
        for node_name, updates in event.items():
            print(f"[{node_name}] completed")
            for msg in updates.get("audit_feedback", [])[-2:]:
                print(f"  -> {msg[:100]}")
            if updates.get("error"):
                print(f"  ERROR: {updates['error']}")
            final_state = updates

    return final_state


async def main():
    parser = argparse.ArgumentParser(description="CAG Deep Research System")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--model", default=None, help="Ollama model to use")
    args = parser.parse_args()

    if not args.query:
        args.query = input("Enter research query: ").strip()
        if not args.query:
            print("No query. Exiting.")
            return

    container = Container()
    if args.model:
        container.settings.ollama_model = args.model

    result = await run_research(args.query, container)

    report = result.get("final_report") if result else None
    if report:
        print("\n" + "=" * 60)
        print("RESEARCH COMPLETE")
        print("=" * 60)
        print(f"\nTopic: {report.topic}")
        print(f"Status: {report.verification_status}")
        print(f"Findings: {report.total_findings}")
        print("\n" + "-" * 60)
        print(report.to_markdown())

        path = await container.storage.save_report(report)
        print(f"\nSaved to: {path}")
    else:
        print("\nNo report generated.")


if __name__ == "__main__":
    asyncio.run(main())
