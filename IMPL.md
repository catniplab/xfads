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

## 2026-11-09: Optional observation-only normalization

- Memming approved exposing `full=True` on `get_ell` and emphasized the
  computation saved by skipping normalization work.
- Updated the describing documentation before changing the calculations.
  Default full-mode results are preserved.
  Gaussian drops only the constant half-log-two-pi term in non-full mode,
  Poisson skips its approximate log-factorial work, and Bernoulli is unchanged.
  All latent- and trainable-parameter-dependent terms remain present.
- Implemented the flag on the abstract method and all three concrete classes.
  Appended it after `reduce_neuron_dim` to preserve existing positional calls.
  Poisson forwards it to PyTorch; Gaussian branches before adding its constant.
- All 26 tests passed with the shared environment, including full/non-full
  reference values, latent/readout/variance gradient equivalence, missing
  entries and bins, per-neuron reductions, and fixed Gaussian variance.
  Existing default-mode integration and RNG regression tests also passed.
  `git diff --check` passed.
- Exploratory CPU kernel benchmark at the working changes following 54ad19c:
  `uv run --no-project <shared .venv>/bin/python work/benchmark_normalization.py`.
  PyTorch 2.14.0, float32, one thread, seed 20261109;
  log rates sampled from a standard normal with shape `(8, 8, 256, 128)`
  (samples, trials, time, neurons); counts drawn from Poisson(2) with shape
  `(8, 256, 128)` and broadcast over samples.
  Used `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.5)`
  with identical inputs, changing only `full`.
  Forward median (IQR): True 3.170 (0.060) ms, False 1.757 (0.027) ms.
  Forward/backward median (IQR): True 3.953 (0.130) ms,
  False 2.457 (0.018) ms.
  This isolates `poisson_nll_loss`; it does not measure full-model speedups.

## 2026-11-09: Default to unnormalized training likelihoods

- Memming requested `full=False` by default because optimization and reported
  BPS do not require observation-only normalization terms.
- Verified that `bits_per_spike` subtracts model and null NLLs evaluated on
  the same counts; their log-factorial terms cancel.
- Updated the describing documentation before switching defaults and
  disabling the redundant normalization computation in BPS.
  Raw losses shift by the omitted data-dependent constant; latent and
  parameter gradients, posterior samples, and BPS retain their meaning.
- Switched the abstract and three concrete methods to `full=False` and
  disabled normalization in both model and null BPS computations.
  `full=True` remains available for absolute-likelihood reporting, with the
  existing documented Stirling approximation for Poisson.
- All 28 tests passed, including float32/float64 BPS value and prediction
  gradient comparisons with the former full-mode calculation, with an
  always-silent neuron included.
- Updated the golden-loss test by subtracting only the omitted, trial-averaged
  observation constant from the original recorded baseline.
  The original RNG expectation and tolerance remain unchanged.
  `git diff --check` passed.
