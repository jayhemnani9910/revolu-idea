[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-2ea44f?style=for-the-badge)](https://jayhemnani9910.github.io/revolu-idea/)

# CAG Deep Research System

A research CLI on LangGraph. It turns a question into a causal graph, has an adversary and a supporter agent search for evidence on each link, judges them, and writes a report. Ports/adapters layout. Experimental.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-purple)
![LangChain](https://img.shields.io/badge/LangChain-LLM-green)

## Live Demo (GitHub Pages)

- https://jayhemnani9910.github.io/revolu-idea/

Enable it via **Settings → Pages**:

- Source: **GitHub Actions**

## Overview

CAG (Causal Analysis Graph) Deep Research breaks a question into variables and cause-effect links. For each link, one agent looks for evidence that it holds and another looks for evidence that it does not. A judge weighs both and marks the link verified, falsified or unclear, and a writer turns the result into a report.

## Key Features

- **Causal planner**: builds a DAG of variables and edges from the question
- **Adversary + supporter**: search for evidence on each edge in parallel
- **Judge**: sets VERIFIED / FALSIFIED / UNCLEAR with a confidence
- **Auditor**: enforces depth and loop limits
- **One search provider per run**: Tavily, Exa, or DuckDuckGo (falls back to Wikipedia)
- **Mock LLM + search**: run the whole graph with no keys

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Orchestration** | LangGraph, LangChain Core |
| **Search** | Tavily, Exa, or DuckDuckGo (one per run) |
| **LLM** | OpenAI-compatible APIs: GitHub Models, Groq, DeepSeek; mock |
| **Architecture** | Ports & adapters |
| **Data** | Pydantic, httpx (async) |

## Quick Start

```bash
# Clone and install
git clone https://github.com/jayhemnani9910/revolu-idea.git
cd revolu-idea
pip install -r requirements.txt

# Configure: copy the template, then set LLM_API_KEY (and a search key, or SEARCH_PROVIDER=duckduckgo)
cp .env.example .env

# Run research
python main.py "What are the latest developments in quantum computing?"

# No keys? Run the whole graph on mock LLM + mock search
LLM_PROVIDER=mock SEARCH_PROVIDER=mock python main.py "test topic"
```

## Agent Workflow

```
User Query
    ↓
[Planner] → causal DAG of variables and edges
    ↓
[Auditor] → depth and loop limits
    ↓
[Selector] → picks the next unverified edge
    ↓
[Adversary + Supporter] → search for and against the edge, in parallel
    ↓
[Judge] → VERIFIED / FALSIFIED / UNCLEAR, then back to the Auditor
    ↓  (no edges left, or depth reached)
[Writer] → output/reports/*.md + .json
```

## Architecture

```
revolu-idea/
├── domain/           # models.py, causal_models.py, exceptions.py
├── ports/            # llm.py, search.py, storage.py
├── adapters/         # openai_compatible, fallback_llm, tavily, exa,
│                     # duckduckgo, mock adapters + local_storage.py
├── agents/
│   ├── state.py
│   └── nodes/        # causal_planner, edge_selector, adversary,
│                     # supporter, judge, auditor, writer
├── graph/            # cag_graph.py (workflow definition)
├── config/           # settings.py
├── container.py      # provider wiring
└── main.py           # CLI entrypoint
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# .env file

# LLM Provider (pick one)
# Option 1: GitHub Models (FREE with Copilot subscription)
LLM_PROVIDER=github
LLM_BASE_URL=https://models.inference.ai.azure.com
LLM_MODEL=gpt-4o-mini  # or "auto" for model pool
LLM_API_KEY=ghp_xxxxx  # GitHub token with models:read scope

# Option 2: Groq (fast, free tier available)
# LLM_PROVIDER=groq
# LLM_BASE_URL=https://api.groq.com/openai/v1
# LLM_MODEL=auto
# LLM_API_KEY=gsk_xxxxx

# Search (DuckDuckGo is free, no key needed)
SEARCH_PROVIDER=duckduckgo
# TAVILY_API_KEY=tvly-xxxxx  # for premium search

# Research parameters
MAX_RECURSION_DEPTH=5  # edge investigations before the writer runs (not graph depth)
MAX_INVESTIGATIONS_PER_EDGE=2
```

### GitHub Models Rate Limits

| Model | RPM | RPD | Tokens |
|-------|-----|-----|--------|
| gpt-4o-mini | 15 | 150 | 8k in, 4k out |
| gpt-4o | 10 | 50 | 8k in, 4k out |

## Output

Reports are saved to `output/reports/` as `.md` and `.json`:
- Summary
- Sections with inline citations and verdict-tagged findings
- Detailed findings per causal edge
- Verification status, methodology and limitations

## Use Cases

- Academic literature reviews
- Competitive intelligence
- Due diligence and fact-checking
- Technical content research
- Policy and market analysis

## License

[MIT License](LICENSE)
