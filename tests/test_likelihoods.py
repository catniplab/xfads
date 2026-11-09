import torch

from xfads.ssm_modules.likelihoods import PoissonLikelihood


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
import pytest
from torch import nn
from torch.distributions import Bernoulli, Normal, Poisson

from xfads.ssm_modules.likelihoods import (
    BernoulliLikelihood,
    GaussianLikelihood,
    Likelihood,
    PoissonLikelihood,
)


def test_likelihood_requires_get_ell():
    class MissingLikelihood(Likelihood):
        pass

    for likelihood_type in (Likelihood, MissingLikelihood):
        with pytest.raises(TypeError, match='get_ell'):
            likelihood_type()


@pytest.mark.parametrize('likelihood_type', [
    GaussianLikelihood, PoissonLikelihood, BernoulliLikelihood,
])
@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 2, 4)])
def test_builtin_likelihood_contract(likelihood_type, shape):
    torch.manual_seed(20261109)
    readout = nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        readout.weight.copy_(torch.linspace(-0.3, 0.4, 6).reshape(3, 2))
        readout.bias.copy_(torch.tensor([-0.2, 0.1, 0.3]))

    if likelihood_type is GaussianLikelihood:
        variance = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64)
        likelihood = likelihood_type(readout, 3, variance)
    elif likelihood_type is PoissonLikelihood:
        likelihood = likelihood_type(readout, 3, delta=0.02)
    else:
        likelihood = likelihood_type(readout, 3)

    assert isinstance(likelihood, Likelihood)
    assert isinstance(likelihood, nn.Module)
    assert dict(likelihood.named_parameters())['readout_fn.weight'] is readout.weight

    # Zero/one counts make Poisson's factorial term exact, avoiding the
    # documented Stirling approximation for larger counts.
    n_samples, n_trials, n_time_bins = shape
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    y = y.expand(n_trials, n_time_bins, 3)
    n_values = n_samples * n_trials * n_time_bins * 2
    z = torch.linspace(-0.8, 0.9, n_values, dtype=torch.float64)
    z = z.reshape(*shape, 2).requires_grad_()
    prediction = readout(z)
    if likelihood_type is GaussianLikelihood:
        reference = Normal(prediction, variance.sqrt())
    elif likelihood_type is PoissonLikelihood:
        reference = Poisson(likelihood.delta * prediction.exp())
    else:
        reference = Bernoulli(logits=prediction)

    actual = likelihood.get_ell(y, z)
    expected = reference.log_prob(y).sum(dim=-1)
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected)

    per_neuron = likelihood.get_ell(y, z, reduce_neuron_dim=False)
    assert per_neuron.shape == (*shape, 3)
    torch.testing.assert_close(per_neuron, reference.log_prob(y))
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


def test_poisson_approximation_changes_only_observation_constant():
    likelihood = PoissonLikelihood(nn.Identity(), 1, delta=0.02)
    y = torch.tensor([[[2.0]]], dtype=torch.float64)
    z = torch.tensor([-0.7, 1.3], dtype=torch.float64).reshape(2, 1, 1, 1)
    z.requires_grad_()

    actual = likelihood.get_ell(y, z)
    expected = Poisson(likelihood.delta * z.exp()).log_prob(y).sum(dim=-1)
    offsets = actual - expected
    assert (offsets > 0).all()
    torch.testing.assert_close(offsets[0], offsets[1])
    actual_gradient, = torch.autograd.grad(actual.sum(), z)
    expected_gradient, = torch.autograd.grad(expected.sum(), z)
    torch.testing.assert_close(actual_gradient, expected_gradient)
