import math
import warnings

import pytest
import torch
from omegaconf import OmegaConf

from xfads.utils import DynamicsLinear, DynamicsEye, lam_from_tau
from xfads.ssm_modules.prebuilt_models import build_dynamics_fn

L = 3
BIN_SZ = 20e-3


def test_lam_from_tau():
    assert lam_from_tau(1.0, BIN_SZ) == pytest.approx(math.exp(-BIN_SZ))


def test_lam_from_tau_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        lam_from_tau(0.0, BIN_SZ)


def rotation(theta, rho=1.0):
    c, s = math.cos(theta), math.sin(theta)
    return rho * torch.tensor([[c, -s], [s, c]])


def spectral_norm(F):
    return torch.linalg.matrix_norm(F, ord=2).item()


def spectral_radius(F):
    return torch.linalg.eigvals(F).abs().max().item()


# --- resolving lam ---

def test_default_tau_from_bin_sz():
    dyn = DynamicsLinear(L, bin_sz=BIN_SZ)
    assert dyn.lam.item() == pytest.approx(math.exp(-BIN_SZ / 10.0))


def test_explicit_tau():
    dyn = DynamicsLinear(L, tau=1.0, bin_sz=BIN_SZ)
    assert dyn.lam.item() == pytest.approx(math.exp(-BIN_SZ / 1.0))


def test_explicit_lam_without_bin_sz():
    dyn = DynamicsLinear(L, lam=0.9)
    assert dyn.lam.item() == pytest.approx(0.9)


def test_explicit_lam_with_bin_sz_keeps_lam():
    # bin_sz alone must not let the default tau override an explicit lam
    dyn = DynamicsLinear(L, lam=0.0, bin_sz=BIN_SZ)
    assert dyn.lam.item() == 0.0


def test_tau_overrides_lam_with_warning():
    with pytest.warns(UserWarning):
        dyn = DynamicsLinear(L, lam=0.5, tau=1.0, bin_sz=BIN_SZ)
    assert dyn.lam.item() == pytest.approx(math.exp(-BIN_SZ))


def test_tau_without_bin_sz_raises():
    with pytest.raises(ValueError):
        DynamicsLinear(L, tau=1.0)


def test_neither_lam_nor_bin_sz_raises():
    with pytest.raises(ValueError):
        DynamicsLinear(L)


@pytest.mark.parametrize('lam', [-0.1, 1.1])
def test_lam_out_of_range_raises(lam):
    with pytest.raises(ValueError):
        DynamicsLinear(L, lam=lam)


def test_nonpositive_tau_raises():
    with pytest.raises(ValueError):
        DynamicsLinear(L, tau=0.0, bin_sz=BIN_SZ)


# --- initialization ---

@pytest.mark.parametrize('lam', [1.0, 0.9])
def test_zero_init_is_scaled_identity(lam):
    dyn = DynamicsLinear(L, lam=lam)
    z = torch.randn(4, 7, L)
    torch.testing.assert_close(dyn(z), lam * z)


def test_lam_one_matches_dynamics_eye():
    dyn = DynamicsLinear(L, lam=1.0)
    z = torch.randn(5, L)
    torch.testing.assert_close(dyn(z), DynamicsEye()(z))


@pytest.mark.parametrize('latent_dim', [2, 10, 50])
def test_init_spectral_radius_equals_lam(latent_dim):
    dyn = DynamicsLinear(latent_dim, bin_sz=BIN_SZ)
    assert spectral_radius(dyn.transition_matrix) == pytest.approx(dyn.lam.item())


# --- the spectral-norm bound ---

def test_bound_caps_spectral_norm():
    dyn = DynamicsLinear(L, lam=0.9)
    with torch.no_grad():
        dyn.A.copy_(5.0 * torch.randn(L, L))
    assert spectral_norm(dyn.transition_matrix) <= 1.0 + 1e-6


def test_bound_respects_custom_rho_max():
    dyn = DynamicsLinear(L, lam=0.9, rho_max=0.5)
    with torch.no_grad():
        dyn.A.copy_(5.0 * torch.randn(L, L))
    assert spectral_norm(dyn.transition_matrix) == pytest.approx(0.5, abs=1e-6)


def test_bound_is_inactive_inside():
    dyn = DynamicsLinear(L, lam=0.9)
    with torch.no_grad():
        dyn.A.copy_(0.01 * torch.randn(L, L))
    torch.testing.assert_close(dyn.transition_matrix, 0.9 * torch.eye(L) + dyn.A)


def test_no_bound_when_rho_max_none():
    dyn = DynamicsLinear(L, lam=0.9, rho_max=None)
    with torch.no_grad():
        dyn.A.copy_(5.0 * torch.randn(L, L))
    torch.testing.assert_close(dyn.transition_matrix, 0.9 * torch.eye(L) + dyn.A)


@pytest.mark.parametrize('theta', [0.03, 0.6, 3.0])
def test_damped_rotation_passes_through_bound(theta):
    # regression test for the rejected ||A||_2 <= 1 - lam bound, which forbade
    # every damped rotation faster than 1/tau
    F = rotation(theta, rho=0.98)
    dyn = DynamicsLinear(2, tau=1.0, bin_sz=BIN_SZ)
    dyn.set_transition(F)
    torch.testing.assert_close(dyn.transition_matrix, F)


# --- set_transition ---

def test_set_transition_recovers_F():
    F = rotation(0.4, rho=0.95)
    dyn = DynamicsLinear(2, lam=0.0)
    dyn.set_transition(F)
    torch.testing.assert_close(dyn.transition_matrix, F)
    torch.testing.assert_close(dyn.A.detach(), F)  # lam = 0 -> A is F verbatim


def test_set_transition_rejects_unstable():
    dyn = DynamicsLinear(L, lam=0.0)
    with pytest.raises(ValueError):
        dyn.set_transition(1.01 * torch.eye(L))


NONNORMAL = torch.tensor([[0.9, 5.0], [0.0, 0.9]])  # spectral radius 0.9, spectral norm > 1


def test_set_transition_warns_when_bound_will_rescale():
    dyn = DynamicsLinear(2, lam=0.0)
    with pytest.warns(UserWarning):
        dyn.set_transition(NONNORMAL)
    assert spectral_norm(dyn.transition_matrix) <= 1.0 + 1e-6


def test_nonnormal_passes_through_unbounded():
    dyn = DynamicsLinear(2, lam=0.0, rho_max=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        dyn.set_transition(NONNORMAL)
    torch.testing.assert_close(dyn.transition_matrix, NONNORMAL)


# --- extension point for structured subclasses ---

def test_subclass_override_feeds_forward_and_bound():
    class FixedRotation(DynamicsLinear):
        def _unbounded_transition(self):
            return rotation(0.3, rho=2.0)

    dyn = FixedRotation(2, lam=1.0)
    z = torch.randn(3, 2)
    # the bound rescales the norm-2 rotation to norm 1 before forward uses it
    torch.testing.assert_close(dyn(z), z @ rotation(0.3, rho=1.0).mT)


# --- gradients and serialization ---

def test_gradient_reaches_A_through_bound():
    dyn = DynamicsLinear(L, lam=0.9)
    with torch.no_grad():
        dyn.A.copy_(5.0 * torch.randn(L, L))  # bound is active
    dyn(torch.randn(4, L)).pow(2).sum().backward()
    assert dyn.A.grad is not None and dyn.A.grad.abs().sum() > 0


def test_state_dict_round_trip_includes_lam():
    src = DynamicsLinear(L, lam=0.7)
    dst = DynamicsLinear(L, lam=0.1)
    dst.load_state_dict(src.state_dict())
    assert dst.lam.item() == pytest.approx(0.7)


# --- build_dynamics_fn ---

def make_cfg(**overrides):
    d = dict(n_latents=L, n_hidden_dynamics=8, device='cpu', bin_sz=BIN_SZ)
    d.update(overrides)
    return OmegaConf.create(d)


def test_factory_defaults():
    dyn = build_dynamics_fn(make_cfg(), 'linear')
    assert isinstance(dyn, DynamicsLinear)
    assert dyn.lam.item() == pytest.approx(math.exp(-BIN_SZ / 10.0))
    assert dyn.rho_max == 1.0


def test_factory_reads_keys():
    dyn = build_dynamics_fn(make_cfg(tau_dynamics=1.0, rho_max_dynamics=0.99), 'linear')
    assert dyn.lam.item() == pytest.approx(math.exp(-BIN_SZ))
    assert dyn.rho_max == 0.99


def test_factory_lam_key_is_not_overridden_by_default_tau():
    dyn = build_dynamics_fn(make_cfg(lam_dynamics=0.0), 'linear')
    assert dyn.lam.item() == 0.0


def test_factory_null_rho_max_disables_bound():
    dyn = build_dynamics_fn(make_cfg(rho_max_dynamics=None), 'linear')
    assert dyn.rho_max is None
