[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-2ea44f?style=for-the-badge)](https://jayhemnani9910.github.io/revolu-idea/)

# CAG Deep Research System

AI-powered research automation using LangGraph orchestration, multiple search engines, and iterative verification. Built with hexagonal architecture for enterprise-grade research workflows.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-purple)
![LangChain](https://img.shields.io/badge/LangChain-LLM-green)

## Live Demo (GitHub Pages)

- https://jayhemnani9910.github.io/revolu-idea/

Enable it via **Settings → Pages**:

- Source: **GitHub Actions**

## Overview

CAG (Causal Analysis Graph) Deep Research is a sophisticated research automation platform that combines multiple AI agents, search engines, and verification loops to produce comprehensive, fact-checked research reports. Unlike simple Q&A systems, CAG performs iterative deep research with automatic quality assessment and knowledge graph construction.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for search, analysis, verification, and reporting
- **Iterative Deepening**: Automatically identifies knowledge gaps and performs follow-up research
- **Dual Search Integration**: Combines Tavily and Exa APIs for comprehensive coverage
- **Quality Assurance**: Built-in audit feedback and verification loops
- **Knowledge Graph**: Constructs domain entities and relationships during research
- **Hexagonal Architecture**: Clean separation between domain logic, ports, and adapters

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Orchestration** | LangGraph, LangChain Core |
| **Search** | Tavily API, Exa API, DuckDuckGo (free) |
| **LLM** | GitHub Models (free), Groq, DeepSeek, Ollama (local) |
| **Architecture** | Hexagonal/Ports & Adapters, DDD |
| **Data** | Pydantic, httpx (async) |

## Quick Start

```bash
# Set API keys
export TAVILY_API_KEY="your_key"
export EXA_API_KEY="your_key"

# Clone and install
git clone https://github.com/jayhemnani9910/revolu-idea.git
cd revolu-idea
pip install -r requirements.txt

# Run research
python main.py "What are the latest developments in quantum computing?"
```

## Agent Workflow

```
User Query
    ↓
[Search Planner] → Plans research strategy
    ↓
[Web Searcher] → Queries Tavily + Exa
    ↓
[Content Analyzer] → Extracts key information
    ↓
[Knowledge Builder] → Constructs entity graph
    ↓
[Report Generator] → Creates structured report
    ↓
[Audit Validator] → Verifies facts, checks gaps
    ↓
Final Report (Markdown)
```

## Architecture

```
revolu-idea/
├── domain/           # Core business logic
│   └── entities.py   # Research entities
├── ports/            # Interfaces
│   ├── llm_port.py
│   └── search_port.py
├── adapters/         # External integrations
│   ├── ollama_adapter.py
│   ├── tavily_adapter.py
│   └── exa_adapter.py
├── agents/           # LangGraph nodes
│   └── nodes/
├── graph/            # Workflow definition
├── config/           # Settings
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
MAX_RECURSION_DEPTH=5
MAX_INVESTIGATIONS_PER_EDGE=2
```

### GitHub Models Rate Limits

| Model | RPM | RPD | Tokens |
|-------|-----|-----|--------|
| gpt-4o-mini | 15 | 150 | 8k in, 4k out |
| gpt-4o | 10 | 50 | 8k in, 4k out |

## Output

Reports saved to `output/reports/` include:
- Executive summary
- Key findings with confidence scores
- Source citations
- Knowledge graph entities
- Verification status
- Follow-up questions

## Use Cases

- Academic literature reviews
- Competitive intelligence
- Due diligence and fact-checking
- Technical content research
- Policy and market analysis

## License

[MIT License](LICENSE)
