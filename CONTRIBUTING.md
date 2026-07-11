# Contributing to AerialMind

This project follows a strict Git workflow designed for solo/small-team development with clear review checkpoints and traceable history.

## Branching Strategy

```
main (protected — every merge requires PR review and approval)
  │
  ├── doc/architecture       → documentation-only branches
  ├── feature/core-types     → one branch per feature
  ├── feature/vision-pipeline
  └── ...
```

**Rules**:

1. **Never commit directly to `main`** (except the one-time initial bootstrap commits). All work happens on a branch.
2. **One branch per feature.** Name it `feature/<short-name>` or `doc/<short-name>`.
3. **One commit per user story.** Each commit should be the smallest possible unit of *working* code — not a half-finished feature, but also not a giant bundle of unrelated changes.
4. **Every feature branch ends in a Pull Request into `main`.** The PR is reviewed and approved before merging.
5. **Commit messages** follow this format:
   ```
   Short imperative summary (max ~50 chars)

   Longer explanation of what changed and why, if needed.
   Wrapped at ~72 characters per line.
   ```
   Example: `Add YOLOv10 object detector`, not `added stuff for detection`.

## Feature Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the complete feature-by-feature, user-story-by-user-story breakdown of the entire project, including which branch each feature lives on and what each commit should contain.

## Development Setup

```bash
git clone https://github.com/omkarjoshi0304/aerialmind.git
cd aerialmind
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Running Tests

```bash
pytest                    # run all tests
pytest tests/unit         # unit tests only
pytest tests/integration  # integration tests only
pytest --cov=aerialmind   # with coverage report
```

## Code Style

- **Type hints are required** on all function signatures — this project uses `mypy --strict`.
- **Formatting and linting** via `ruff` (configured in `pyproject.toml`).
- **No comments explaining *what* code does** — code should be self-explanatory through naming. Comments are reserved for explaining *why* something non-obvious was done.
- **Dataclasses for data, Protocols for interfaces** — this project relies heavily on Python's structural typing (see [docs/architecture/08-api-contracts.md](docs/architecture/08-api-contracts.md)).

## Documentation Standards

- Every architectural decision lives in `docs/architecture/` as a Markdown file with Mermaid diagrams (GitHub renders these natively — no external tools needed).
- If you change an architectural decision during implementation, update the corresponding doc in the same PR. Docs must never go stale relative to the code.
- New modules should reference which architecture doc defines their contract.

## Pull Request Checklist

Before opening a PR:

- [ ] All commits in the branch represent complete, working user stories (no broken intermediate states)
- [ ] Tests pass locally (`pytest`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Relevant architecture docs updated if design changed during implementation
- [ ] PR description references which Feature/User Stories from [docs/ROADMAP.md](docs/ROADMAP.md) it implements
