import math

import pytest
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal, Poisson

from xfads.prob_utils import bits_per_spike
# The public interface must import without preloading xfads.utils.
from xfads.ssm_modules.likelihoods import (
    BernoulliLikelihood,
    GaussianLikelihood,
    Likelihood,
    PoissonLikelihood,
)


def test_poisson_ell_omits_log_factorial_and_has_finite_data_gradient():
    # log(y!) is constant in every parameter, so it is omitted; torch's full=True would add
    # Stirling's approximation of it, whose backward is NaN w.r.t. y at y = 0
    lik = PoissonLikelihood(torch.nn.Identity(), n_neurons=4, delta=1.0)
    y = torch.tensor([0., 1., 2., 5.], requires_grad=True)
    eta = torch.tensor([[0.3, -0.2, 0.1, 1.0]])  # log rates, one sample
    ell = lik.get_ell(y, eta, reduce_neuron_dim=False)
    torch.testing.assert_close(ell, y.detach() * eta - eta.exp())
    ell.sum().backward()
    torch.testing.assert_close(y.grad, eta[0])


def test_poisson_ell_full_is_exact_log_pmf():
    # full=True adds the exact -log(y!), not torch's Stirling approximation
    lik = PoissonLikelihood(torch.nn.Identity(), n_neurons=4, delta=1.0)
    y = torch.tensor([0., 1., 2., 5.], requires_grad=True)
    eta = torch.tensor([[0.3, -0.2, 0.1, 1.0]])
    ell = lik.get_ell(y, eta, reduce_neuron_dim=False, full=True)
    torch.testing.assert_close(ell, torch.distributions.Poisson(eta.exp()).log_prob(y.detach()))
    ell.sum().backward()
    assert torch.isfinite(y.grad).all()


def test_likelihood_requires_get_ell():
    class MissingLikelihood(Likelihood):
        pass

    for likelihood_type in (Likelihood, MissingLikelihood):
        with pytest.raises(TypeError, match='get_ell'):
            likelihood_type()


@pytest.fixture(params=[
    GaussianLikelihood, PoissonLikelihood, BernoulliLikelihood,
])
def likelihood(request):
    torch.manual_seed(20261109)
    readout = nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        readout.weight.copy_(torch.linspace(-0.3, 0.4, 6).reshape(3, 2))
        readout.bias.copy_(torch.tensor([-0.2, 0.1, 0.3]))

    if request.param is GaussianLikelihood:
        variance = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64)
        return request.param(readout, 3, variance)
    if request.param is PoissonLikelihood:
        return request.param(readout, 3, delta=0.02)
    return request.param(readout, 3)


@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 2, 4)])
@pytest.mark.parametrize('full', [False, True])
def test_builtin_likelihood_contract(likelihood, shape, full):
    readout = likelihood.readout_fn
    assert isinstance(likelihood, Likelihood)
    assert isinstance(likelihood, nn.Module)
    assert dict(likelihood.named_parameters())['readout_fn.weight'] is readout.weight

    # Binary observations are shared across all three likelihood families;
    # the dedicated Poisson tests also cover larger counts.
    n_samples, n_trials, n_time_bins = shape
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    y = y.expand(n_trials, n_time_bins, 3)
    n_values = n_samples * n_trials * n_time_bins * 2
    z = torch.linspace(-0.8, 0.9, n_values, dtype=torch.float64)
    z = z.reshape(*shape, 2).requires_grad_()
    prediction = readout(z)
    if isinstance(likelihood, GaussianLikelihood):
        variance = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64)
        reference = Normal(prediction, variance.sqrt())
    elif isinstance(likelihood, PoissonLikelihood):
        reference = Poisson(likelihood.delta * prediction.exp())
    else:
        reference = Bernoulli(logits=prediction)

    actual = likelihood.get_ell(y, z, full=full)
    expected_per_neuron = reference.log_prob(y)
    if not full and isinstance(likelihood, GaussianLikelihood):
        expected_per_neuron = expected_per_neuron + 0.5 * math.log(2 * math.pi)
    expected = expected_per_neuron.sum(dim=-1)
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected)

    per_neuron = likelihood.get_ell(y, z, reduce_neuron_dim=False, full=full)
    assert per_neuron.shape == (*shape, 3)
    torch.testing.assert_close(per_neuron, expected_per_neuron)
    torch.testing.assert_close(per_neuron.sum(dim=-1), actual)

    parameters = (z, *likelihood.parameters())
    actual_gradients = torch.autograd.grad(actual.sum(), parameters)
    assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)
    assert all(gradient.abs().sum() > 0 for gradient in actual_gradients)

    # Compare shared gradients; the reference Gaussian variance is fixed.
    reference_parameters = (z, *readout.parameters())
    expected_gradients = torch.autograd.grad(expected.sum(), reference_parameters)
    parameter_gradients = dict(zip(parameters, actual_gradients))
    for parameter, expected_gradient in zip(reference_parameters, expected_gradients):
        torch.testing.assert_close(parameter_gradients[parameter], expected_gradient)


@pytest.mark.parametrize('reduce_neuron_dim', [False, True])
def test_full_flag_offsets_and_gradients_with_missing_data(likelihood, reduce_neuron_dim):
    # Include an observed count above one, a missing neuron, and a missing bin.
    values = [0.0, 1.0, 0.0] if isinstance(likelihood, BernoulliLikelihood) else [0.0, 2.0, 5.0]
    y = torch.tensor(values, dtype=torch.float64).repeat(1, 3, 1)
    y[:, 1, 1] = float('nan')
    y[:, 2] = float('nan')
    z = torch.linspace(-0.8, 0.9, 12, dtype=torch.float64).reshape(2, 1, 3, 2)
    z.requires_grad_()
    kwargs = dict(reduce_neuron_dim=reduce_neuron_dim)
    full = likelihood.get_ell(y, z, full=True, **kwargs)
    short = likelihood.get_ell(y, z, full=False, **kwargs)
    torch.testing.assert_close(likelihood.get_ell(y, z, **kwargs), short, rtol=0, atol=0)

    parameters = (z, *likelihood.parameters())
    full_gradients = torch.autograd.grad(full.sum(), parameters)
    short_gradients = torch.autograd.grad(short.sum(), parameters)
    for full_gradient, short_gradient in zip(full_gradients, short_gradients):
        assert torch.isfinite(short_gradient).all()
        torch.testing.assert_close(full_gradient, short_gradient)

    assert torch.isfinite(full).all() and torch.isfinite(short).all()
    torch.testing.assert_close(full[:, :, 2], torch.zeros_like(full[:, :, 2]), rtol=0, atol=0)
    torch.testing.assert_close(short[:, :, 2], torch.zeros_like(short[:, :, 2]), rtol=0, atol=0)
    if not reduce_neuron_dim:
        torch.testing.assert_close(short[:, :, 1, 1], torch.zeros_like(short[:, :, 1, 1]), rtol=0, atol=0)

    obs = ~y.isnan()
    if isinstance(likelihood, GaussianLikelihood):
        offset = torch.full_like(y, 0.5 * math.log(2 * math.pi)) * obs
    elif isinstance(likelihood, PoissonLikelihood):
        # Check the exact data-only offset; the separate reference test
        # checks the likelihood against torch.distributions.
        counts = torch.where(obs, y, 0.0)
        offset = torch.lgamma(counts + 1)
    else:
        offset = torch.zeros_like(y)
    if reduce_neuron_dim:
        offset = offset.sum(dim=-1)
    torch.testing.assert_close(short - full, offset.expand_as(short))


def test_gaussian_short_mode_retains_fixed_variance_normalizer():
    variance = torch.tensor([0.3, 1.4], dtype=torch.float64)
    likelihood = GaussianLikelihood(nn.Identity(), 2, variance, fix_R=True)
    z = torch.tensor([[[[0.2, -0.4]]]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[0.1, 0.8]]], dtype=torch.float64)
    expected = Normal(z, variance.sqrt()).log_prob(y) + 0.5 * math.log(2 * math.pi)
    actual = likelihood.get_ell(y, z, reduce_neuron_dim=False, full=False)
    torch.testing.assert_close(actual, expected)


def test_poisson_full_matches_distribution_at_multiple_latent_states():
    likelihood = PoissonLikelihood(nn.Identity(), 1, delta=0.02)
    y = torch.tensor([[[2.0]]], dtype=torch.float64)
    z = torch.tensor([-0.7, 1.3], dtype=torch.float64).reshape(2, 1, 1, 1)
    z.requires_grad_()

    actual = likelihood.get_ell(y, z, full=True)
    expected = Poisson(likelihood.delta * z.exp()).log_prob(y).sum(dim=-1)
    torch.testing.assert_close(actual, expected)
    actual_gradient, = torch.autograd.grad(actual.sum(), z)
    expected_gradient, = torch.autograd.grad(expected.sum(), z)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_bits_per_spike_matches_full_likelihood_difference(dtype):
    y = torch.tensor([[[0., 2., 0.], [1., 0., 0.]],
                      [[5., 1., 0.], [0., 3., 0.]]], dtype=dtype)
    log_rates = torch.linspace(-2.0, 0.7, y.numel(), dtype=dtype).reshape_as(y)
    log_rates.requires_grad_()

    # Reproduce the former metric, including its linear-rate null model.
    nll_model = torch.nn.functional.poisson_nll_loss(log_rates, y, full=True, reduction='sum')
    nll_null = torch.nn.functional.poisson_nll_loss(
        y.mean(dim=(0, 1), keepdim=True), y, log_input=False, full=True, reduction='sum',
    )
    expected = (nll_null - nll_model) / y.sum() / math.log(2)
    actual = bits_per_spike(log_rates, y)
    torch.testing.assert_close(actual, expected)

    actual_gradient, = torch.autograd.grad(actual, log_rates)
    expected_gradient, = torch.autograd.grad(expected, log_rates)
    torch.testing.assert_close(actual_gradient, expected_gradient)
