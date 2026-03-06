# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entrypoint that runs the CAG “deep research” workflow and prints progress.
- `container.py`: dependency injection container wiring LLM/Search/Storage adapters.
- `config/`: runtime settings (Pydantic Settings; loaded from `.env`).
- `graph/`: LangGraph construction/orchestration (`graph/cag_graph.py`).
- `agents/`: LangGraph nodes in `agents/nodes/` and shared state in `agents/state.py`.
- `domain/`: core models (causal graph + report structures/Markdown rendering).
- `ports/`: interfaces (`LLMPort`, `SearchPort`, `StoragePort`) implemented by `adapters/`.
- `output/`: generated artifacts (gitignored): `output/reports/`, `output/graphs/`, `output/checkpoints/`.

## Build, Test, and Development Commands
- Create a virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Configure env: `cp .env.example .env` then set `LLM_API_KEY` and optional search keys.
- Run locally: `python main.py "your topic"` (optional: `--model <model_name>`).
- No-key smoke run (uses mocks): `LLM_PROVIDER=mock SEARCH_PROVIDER=mock python main.py "test topic"`

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, prefer type hints on public APIs.
- Naming: `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- LangGraph nodes should be `async` callables: `async def __call__(self, state: ResearchState) -> dict`.
- Keep provider-specific logic in `adapters/`; depend on `ports/` from `graph/` and `agents/`.

## Testing Guidelines
- No automated test suite is currently committed.
- Before opening a PR, run an end-to-end query and confirm outputs are written under `output/`.
- If adding tests, use `pytest` in a `tests/` folder with `test_*.py` naming.

## Security & Configuration Tips
- Never commit API keys; `.env` is gitignored. If you add new settings, update `.env.example`.
- Keep generated artifacts out of commits; `output/` is gitignored by default.

## Commit & Pull Request Guidelines
- Git history is currently minimal (single “Initial commit…”); no established convention yet.
- Use clear, imperative commit subjects (recommended: Conventional Commits like `feat:`, `fix:`).
- PRs should include: what changed/why, how to run (exact command), and any sample output paths.
