# Implementation progress

Entries are chronological, and a later entry can reverse an earlier one;
reversed statements are marked [superseded].
Current state: every `get_ell` defaults to `full=False`; Poisson `full=True`
adds the exact `torch.lgamma(y + 1)` (from #39); bits per spike skips the
log-factorial terms, which cancel; a fixed Gaussian variance is the `R_diag`
buffer.

## 2026-11-09: Documented likelihood interface

- Work is isolated on `feature/likelihood-base-class` in a separate worktree.
- Added an abstract `Likelihood(nn.Module, ABC)` base class in
  `xfads/ssm_modules/likelihoods.py`.
- Documented input/output shapes, observation reduction, sample averaging,
  gradient preservation, and subclass initialization.
- Updated all three built-in likelihoods to inherit from the base class.
  Their constructors, parameter names, and likelihood calculations are unchanged.
  [superseded: later entries add `full` and change the defaults]
- Updated the README to explain custom likelihoods and the additional
  requirements of specialized smoothers, including coordinated dropout.
- Documented that all latent- and trainable-parameter-dependent normalization
  terms must be retained; observation-only omissions must be explicit.
  Gaussian and Bernoulli are fully normalized; Poisson includes PyTorch's
  Stirling approximation to the observation-only log-factorial term.
  [superseded: exact `torch.lgamma(y + 1)` since the rebase onto #39]
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
  [superseded: the default became `full=False` in the next entry]
  Gaussian drops only the constant half-log-two-pi term in non-full mode,
  Poisson skips its approximate log-factorial work, and Bernoulli is unchanged.
  All latent- and trainable-parameter-dependent terms remain present.
- Implemented the flag on the abstract method and all three concrete classes.
  Appended it after `reduce_neuron_dim` to preserve existing positional calls.
  Poisson forwards it to PyTorch; Gaussian branches before adding its constant.
  [superseded: Poisson adds #39's exact `torch.lgamma(y + 1)` instead]
- All 26 tests passed with the shared environment, including full/non-full
  reference values, latent/readout/variance gradient equivalence, missing
  entries and bins, per-neuron reductions, and fixed Gaussian variance.
  Existing default-mode integration and RNG regression tests also passed.
  `git diff --check` passed.
- Exploratory CPU kernel benchmark at the working changes following 54ad19c:
  `uv run --no-project <shared .venv>/bin/python work/benchmark_normalization.py`
  (an untracked scratch script: `work/` is excluded from git).
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
  [superseded: exact `torch.lgamma(y + 1)` since the rebase onto #39]
- All 28 tests passed, including float32/float64 BPS value and prediction
  gradient comparisons with the former full-mode calculation, with an
  always-silent neuron included.
- Updated the golden-loss test by subtracting only the omitted, trial-averaged
  observation constant from the original recorded baseline.
  The original RNG expectation and tolerance remain unchanged.
  `git diff --check` passed.

## 2026-11-09: Stack the interface PR on PR #39

- Memming approved making PR #41 build on PR #39, keeping the Poisson
  numerical implementation in #39 and the shared interface work in #41.
- Rebased the interface branch onto `origin/fix/poisson-drop-log-factorial`
  (87c2ea3), retaining its Poisson constructor and `get_ell` implementation,
  both original Poisson regression tests, and golden-loss adjustment.
- Updated the describing documentation and interface tests to use #39's
  exact `torch.lgamma(y + 1)` full-mode normalization rather than Stirling.
- Retained the base class, Gaussian/Bernoulli interface consistency, BPS
  optimization, and circular-import fix as the additional scope of #41.
- All 30 combined tests passed with the shared environment.
  AST comparisons confirmed that both Poisson methods and both inherited
  regression tests match #39 exactly; the golden-loss test has no diff.
  `git diff --check origin/fix/poisson-drop-log-factorial` passed.
- The earlier kernel benchmark is historical: it compared PyTorch's Stirling
  full mode against skipping that work, not #39's exact full-mode implementation.
  Removed that performance claim from the current stacked PR description.

## 2026-11-10: Review fixes

- Code review of #41 found no bug in the new code, but gaps in what it
  documents; Memming asked for these fixes plus the fixed-variance bug.
- `GaussianLikelihood(fix_R=True)` kept `log_R` as a plain tensor, so it
  ignored `.to()`, `.double()` and `load_state_dict`, and never read the
  registered `R_diag` buffer: a checkpoint with `R_diag=4` computed with
  variance 1. The fixed variance is now read from `R_diag` through a new
  `get_R()`, which `utils.get_latent_rotation` also uses. State-dict keys are
  unchanged, so old checkpoints load and now take effect.
  The extended fixed-variance test fails on the previous code.
- README: the custom-likelihood section now gives the full `get_ell`
  signature and points to the `Likelihood` docstring instead of restating it.
- The docstring now lists what specialized smoothers and trainers read
  beyond `get_ell`: `readout_fn[-1].weight`, `delta`, `readout_fn`.
- Marked the reversed statements above [superseded] and added a current-state
  summary at the top.
- Missing-data gaps found in the same review (`bits_per_spike` NaN, Bernoulli
  inf logit at a missing entry) go to a separate PR stacked on this one.
- All 30 tests passed with the shared environment.

## 2026-11-10: Missing counts in bits per spike and Bernoulli

- Branch `fix/missing-data-bps-bernoulli`, stacked on #41, which rewrites the
  same `bits_per_spike` lines.
- `bits_per_spike` summed NaN counts into both likelihoods, so any padded or
  missing bin made every logged BPS NaN. Missing counts now add no term to
  either likelihood or the spike count; each neuron's null rate is its mean
  over observed bins. Rates are replaced at missing entries before arithmetic,
  so an overflow there or a never-observed neuron (0 / 0) cannot produce NaN.
  On fully observed data the result is bitwise identical to before
  (float32 and float64, up to 256 x 35 x 137).
- `BernoulliLikelihood` masked after `binary_cross_entropy_with_logits`, so an
  infinite logit at a missing entry gave inf * 0 = NaN; logits are now
  replaced first, as Gaussian and Poisson already did.
- Two new tests: BPS with trailing padding, a never-observed neuron, and
  infinite rates at every missing entry equals BPS on the unpadded counts, with
  zero gradient at missing entries; an infinite readout at a missing neuron
  stays out of the ell and its gradient for all three likelihoods. Both fail
  on the previous code (Bernoulli only, for the second).
- All 34 tests passed with the shared environment.
