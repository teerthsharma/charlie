# Contributing to OmniTopos

Thank you for your interest in contributing to OmniTopos.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/teerthsharma/omni-topos.git
cd omni-topos

# Install dependencies
pip install -e ".[dev]"

# Run the full quality gate
ruff check . && mypy src/ && pytest
```

## Quality Standards

Every contribution must pass these checks before merge:

| Check | Command | Required |
|-------|---------|----------|
| Lint | `ruff check .` | 0 violations |
| Type check | `mypy src/` | 0 errors |
| Tests | `pytest --cov=src/` | ≥ 90% coverage |
| Format | `ruff format .` | No diff |

## Branching Strategy

- `main` — stable, always deployable
- `feat/*` — feature development
- `fix/*` — bug fixes
- `doc/*` — documentation improvements

PRs must include:
1. Description of what changed and why
2. Evidence of all quality checks passing
3. For new features: a test demonstrating the expected behavior

## Reporting Issues

Open an issue at https://github.com/teerthsharma/omni-topos/issues with:
- Steps to reproduce
- Expected vs actual behavior
- Python version, platform, and version of OmniTopos

## Code of Conduct

Be respectful. Disagreements are fine; disrespect is not.