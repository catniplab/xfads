# Implementation progress

## 2026-11-09: Documented likelihood interface

- Work is isolated on `feature/likelihood-base-class` in a separate worktree.
- Added an abstract `Likelihood(nn.Module, ABC)` base class in
  `xfads/ssm_modules/likelihoods.py`.
- Documented input/output shapes, observation reduction, sample averaging,
  gradient preservation, and subclass initialization.
- Updated all three built-in likelihoods to inherit from the base class.
  Their constructors, parameter names, and likelihood calculations are unchanged.
- Updated the README to explain custom likelihoods and the additional
  requirements of specialized smoothers, including coordinated dropout.
- Documented that all latent- and trainable-parameter-dependent normalization
  terms must be retained; observation-only omissions must be explicit.
  Gaussian and Bernoulli are fully normalized; Poisson includes PyTorch's
  Stirling approximation to the observation-only log-factorial term.
- The first isolated test exposed a pre-existing circular import between
  `likelihoods.py` and `utils.py`.
  Moved the latter's likelihood import into `get_latent_rotation`, its only
  consumer, so the documented public interface can be imported directly.
- Validation completed with PyTorch 2.14.0 on CPU, using seed 20261109 for
  deterministic readout initialization:
  `uv run --no-project <shared .venv>/bin/python -m pytest -q tests`
  passed all 44 tests, including eight new likelihood tests.
- New tests cover abstract-class enforcement, module parameter registration,
  singleton and batched tensor shapes, values and gradients against
  `torch.distributions`, and the observation-only Poisson approximation offset.
- An AST comparison verified that all existing likelihood methods remain
  identical to the parent revision; `git diff --check` passed.
- Used the shared virtual environment read-only; all source imports resolve
  from this worktree.
- Preparing the PR: rebased only the likelihood commit onto current
  `origin/main`, excluding the separate residual-dynamics change from PR #33.
  Main now includes missing-observation support and per-neuron output for
  Gaussian likelihoods from PR #38.
  Updated the abstract signature, documentation, and contract tests to describe
  this existing behavior, without changing any concrete likelihood methods.
- After rebasing, all 13 tests on current main passed, including the eight new
  likelihood tests and existing missing-observation tests.
  Per-neuron outputs match independent distribution references and sum to the
  default outputs.
  An AST comparison against `origin/main` again confirmed unchanged concrete
  likelihood methods, and `git diff --check origin/main` passed.
