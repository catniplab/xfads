# Repository Guidelines

This repository contains the XFADS Python package for large‑scale variational Gaussian state‑space modeling. Follow these guidelines to contribute changes that are easy to review and maintain.

## Project Structure & Modules
- Source: `xfads/`
  - `ssm_modules/` (dynamics, likelihoods, encoders)
  - `smoothers/` (filters, Lightning trainers)
  - Utilities: `utils.py`, `linalg_utils.py`, `prob_utils.py`, `plot_utils.py`
- Examples: `examples/` (Hydra configs + runnable scripts)
- Experiments: `experiments/` (project‑specific runs)
- Config: `pyproject.toml`

## Build, Test, and Dev Commands
- Create env (Python 3.11): `conda create -n xfads python=3.11 && conda activate xfads`
- Editable install: `pip install -e .`
- Lint/format (Ruff): `ruff check . --fix` and `ruff format .`
- Run an example: `python examples/lds_example/inference_lightning.py`

## Coding Style & Naming
- Use NumPy-style docstrings (Parameters, Returns, Shapes) for public functions/classes. Keep examples concise.

```python
def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Scale features to unit norm.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape [T, D].

    Returns
    -------
    torch.Tensor
        Normalized tensor with shape [T, D].
    """
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)
```
- Python style with Ruff for linting/formatting; 4‑space indentation.
- Naming: modules/files `lower_snake_case.py`; functions/vars `lower_snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Prefer type hints and concise docstrings. Keep APIs in `xfads/` stable; place experiment‑specific code under `experiments/`.

## Testing Guidelines
- No formal unit test suite yet. Use example scripts as smoke tests and add focused tests for new logic.
- If adding tests, create `tests/` with `pytest` and name files `test_*.py`. Run via `pytest -q`.

## Commit & Pull Requests
- Commit style follows short, imperative messages (per current history): e.g., "fix rotation in utils".
- Keep subject ≤ 72 chars; add body for context and link issues (e.g., `#12`).
- PRs should include: purpose, summary of changes, usage notes (commands/paths), and screenshots/plots when relevant.

## Security & Configuration Tips
- Torch version is platform‑specific (see `pyproject.toml`); ensure GPU setup matches your environment.
- Hydra outputs default to local dirs (e.g., `logs/`, `ckpts/`); avoid committing large artifacts.
- Do not embed dataset paths or credentials; use configs/env vars.

## Architecture Overview
- Core workflow: define `ssm_modules` → assemble model in `smoothers` → train/evaluate via Lightning trainers → run via `examples/*` using Hydra configs.

## Change Log
- Record every repository modification in `CHANGELOG.md` grouped by UTC date using a top-level bullet `- YYYY-MM-DD`.
- List individual updates for that day as indented bullets beneath the date.
- Append new date groups to the bottom of the file so the log remains chronological.
