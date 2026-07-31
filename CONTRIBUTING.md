# Contributing to TurboGuard

## Development setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# .venv\Scripts\Activate.ps1    # PowerShell
pip install -r requirements.txt
pip install -e .
```

## Running tests

```bash
pytest -q
```

## Linting

```bash
ruff check src tests app scripts
```

## Style

- Python 3.11, type hints on public functions.
- No commented-out code, no debug prints left behind.
- Keep modules focused: one concern per file, matching `src/` layout in the
  README's Project Structure section.
- New features that change pipeline behavior should include a unit test
  under `tests/` and, if user-facing, a note in `CHANGELOG.md`.

## Commit style

Commits use a `type(scope): summary` prefix (`feat`, `fix`, `docs`, `chore`,
`test`, `refactor`). See `docs/ROADMAP.md` for the staged build history.
