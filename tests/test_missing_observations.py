"""Missing observations marked as NaN (#37): padding, dropped bins, absent neurons."""
import math

import pytest
import torch
from omegaconf import OmegaConf

from xfads.ssm_modules.prebuilt_models import create_xfads_poisson_log_link
from xfads.ssm_modules.likelihoods import GaussianLikelihood, BernoulliLikelihood

SEED = 20261109
N, S = 6, 5  # neurons, samples
NAN = float('nan')
CFG = OmegaConf.create(dict(n_latents=4, n_latents_read=3, rank_local=2, rank_backward=1,
                            n_hidden_dynamics=8, n_hidden_local=8, n_hidden_backward=8,
                            bin_sz=0.02, device='cpu', p_local_dropout=0.0))


def make_model(model_type='n', **overrides):
    torch.manual_seed(SEED)
    return create_xfads_poisson_log_link(OmegaConf.merge(CFG, overrides), N, None, model_type=model_type)


def counts(n_trials, n_bins):
    return torch.poisson(torch.full((n_trials, n_bins, N), 2.0), generator=torch.Generator().manual_seed(SEED))


def seeded(fn, *args, **kwargs):
    torch.manual_seed(SEED)
    with torch.no_grad():
        return fn(*args, **kwargs)


@pytest.fixture
def no_dropout_draws(monkeypatch):
    # With p_mask_* = 0 the coordinated-dropout masks are all ones, but drawing them still
    # consumes RNG in proportion to n_time_bins. Skip the draw so that a padded and an
    # unpadded run share one RNG stream and the filter's samples align bin by bin.
    monkeypatch.setattr(torch, 'bernoulli', torch.ones_like)


def test_trailing_padding_matches_unpadded(no_dropout_draws):
    T_i, T = 7, 11
    y = counts(1, T_i)
    y_pad = torch.cat([y, torch.full((1, T - T_i, N), NAN)], dim=1)
    pad_kl = torch.zeros(1, T - T_i)

    # model types share the encoders and the likelihood, where missing entries are handled
    for model_type in ('n', 'c'):
        ssm = make_model(model_type)

        # smoother: ELBO and posterior over the real bins; padded bins add nothing
        loss, z_s, stats = seeded(ssm, y, S)
        loss_p, z_s_p, stats_p = seeded(ssm, y_pad, S)
        torch.testing.assert_close(loss_p, loss)
        torch.testing.assert_close(z_s_p[:, :, :T_i], z_s)
        torch.testing.assert_close(stats_p['m_f'][:, :T_i], stats['m_f'])
        torch.testing.assert_close(stats_p['kl'][:, T_i:], pad_kl, rtol=0, atol=1e-6)

        # causal filter
        _, stats = seeded(ssm.fast_filter_1_to_T, y, S, get_kl=True)
        _, stats_p = seeded(ssm.fast_filter_1_to_T, y_pad, S, get_kl=True)
        torch.testing.assert_close(stats_p['m_f'][:, :T_i], stats['m_f'])
        torch.testing.assert_close(stats_p['kl'][:, :T_i], stats['kl'])
        torch.testing.assert_close(stats_p['kl'][:, T_i:], pad_kl, rtol=0, atol=1e-6)

        if model_type == 'c':  # filter-only fitting
            torch.testing.assert_close(seeded(ssm.forward_filter, y_pad, S)[0], seeded(ssm.forward_filter, y, S)[0])


def test_interior_gap_carries_backward_state():
    ssm = make_model()
    T = 12
    y = counts(1, T)
    obs_bin = torch.ones(1, T, dtype=torch.bool)
    obs_bin[:, [0, 1, 5, 6, 7]] = False  # a leading and an interior gap

    # the backward GRU holds its state through missing bins: at observed bins its output
    # equals that of the sequence with the missing bins deleted
    k_y, K_y = seeded(ssm.local_encoder, y)
    k_b, K_b = seeded(ssm.backward_encoder, k_y, K_y, obs_bin)
    k_b_del, K_b_del = seeded(ssm.backward_encoder, k_y[obs_bin][None], K_y[obs_bin][None])
    torch.testing.assert_close(k_b[obs_bin][None], k_b_del)
    torch.testing.assert_close(K_b[obs_bin][None], K_b_del)
    torch.testing.assert_close(k_b[:, 5:8], k_b[:, 4:5].expand(-1, 3, -1))  # carried across the gap

    # the causal filter takes a predict-only step at a missing bin: KL(q_t || p_t) = 0
    y_gap = y.clone()
    y_gap[~obs_bin] = NAN
    _, stats = seeded(ssm.fast_filter_1_to_T, y_gap, S, get_kl=True)
    torch.testing.assert_close(stats['kl'][~obs_bin], torch.zeros(5), rtol=0, atol=1e-6)


def test_missing_neuron_drops_only_its_likelihood_term():
    ssm = make_model()
    y = counts(2, 9).clamp(max=1)  # binary, so the Bernoulli likelihood applies too
    t, n = 4, 2
    y_nan, y_zero = y.clone(), y.clone()
    y_nan[0, t, n], y_zero[0, t, n] = NAN, 0.

    readout = ssm.likelihood_pdf.readout_fn
    for lik in (ssm.likelihood_pdf, GaussianLikelihood(readout, N, torch.ones(N)), BernoulliLikelihood(readout, N)):
        ssm.likelihood_pdf = lik
        # the encoders see the same zero-filled input, so only one likelihood term differs
        loss_nan, z_s, _ = seeded(ssm, y_nan, S)
        loss_zero, z_s_zero, _ = seeded(ssm, y_zero, S)
        assert torch.equal(z_s, z_s_zero)
        ell_tn = lik.get_ell(y_zero, z_s, reduce_neuron_dim=False)[:, 0, t, n].mean()
        torch.testing.assert_close(loss_nan, loss_zero + ell_tn / y.shape[0])


def test_masked_entries_get_zero_gradient():
    ssm = make_model()
    y = counts(3, 10)
    y[0, 3:5] = NAN        # missing bins
    y[0, 7, [1, 4]] = NAN  # absent neurons
    y[1, 6:] = NAN         # trailing padding
    y[2] = NAN             # a trial with no data
    y[:, :, 5] = NAN       # a neuron never recorded, whose predicted rate overflows (exp(200) = inf)
    ssm.likelihood_pdf.readout_fn[-1].bias.data[5] = 200.
    missing = y.isnan()
    y.requires_grad_(True)

    torch.manual_seed(SEED)
    loss, _, _ = ssm(y, S, p_mask_y_in=0.2, p_mask_a=0.3, p_mask_b=0.2, p_mask_apb=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in ssm.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert (y.grad[missing] == 0).all()


# Recorded at d5ccf5d (main before #37) with the call below, torch 2.15.0 on CPU.
MAIN_LOSS, MAIN_NEXT_DRAW = 526.1984252929688, 1.0095324516296387


def test_fully_observed_matches_main():
    # coordinated dropout and local dropout on, so a change in RNG consumption shows up too
    ssm = make_model(p_local_dropout=0.2)
    y = counts(3, 10)
    torch.manual_seed(SEED)
    loss, _, _ = ssm(y, S, p_mask_y_in=0.2, p_mask_a=0.3, p_mask_b=0.2, p_mask_apb=0.1)
    next_draw = torch.randn(()).item()
    # d5ccf5d's loss included Stirling's log(y!) (Poisson full=True), a constant in every parameter
    # that the likelihood now omits; per trial, the loss sums over bins and neurons
    y_safe = torch.where(y > 1, y, 2.)
    stirling = torch.where(y > 1, y_safe * y_safe.log() - y_safe + 0.5 * (2 * math.pi * y_safe).log(), 0.)
    assert loss.item() == pytest.approx(MAIN_LOSS - stirling.sum((1, 2)).mean().item(), rel=1e-6)
    assert next_draw == MAIN_NEXT_DRAW
