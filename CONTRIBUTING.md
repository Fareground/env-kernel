# Contributing

## Development setup

The distribution name is `fg-env-kernel` and the import package is `fg_env_kernel`.
Do not rename either — downstream projects depend on them.

```bash
git clone https://github.com/Fareground/env-kernel
cd env-kernel
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the tests

The suite lives in `tests/` (many test modules). Run all of it with:

```bash
pytest
```

`testpaths` is set to `tests` in `pyproject.toml`, so a bare `pytest` from the
repo root discovers everything. Narrow a run with `pytest tests/<file>.py -k <expr>`.

## Lint and format

Tooling and rules are configured in `pyproject.toml` under `[tool.ruff]`
(target `py311`, line length 100):

```bash
ruff check .        # lint
ruff format .       # format
```

The kernel keeps a single runtime dependency (`pydantic`) — please keep it that way.
Optional features may add dev-only dependencies under `[project.optional-dependencies]`.

## Commits and pull requests

- Follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`.
- One logical change per PR; include tests for new behavior.
- Update `CHANGELOG.md` under `## [Unreleased]`.
- Run `pytest` and `ruff check .` locally before pushing.
- **Commits must not include AI or assistant co-author attribution** — no
  `Co-authored-by` trailers or generated-by notices for any AI tool.
