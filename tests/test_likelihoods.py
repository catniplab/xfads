import torch

import xfads.utils  # noqa: F401  (load before likelihoods: import cycle)
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
