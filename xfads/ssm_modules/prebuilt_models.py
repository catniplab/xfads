import warnings
import torch
import torch.nn as nn
import xfads.utils as utils
import xfads.prob_utils as prob_utils
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn, MaskedInputEncoder
from xfads.ssm_modules.likelihoods import PoissonLikelihood, BernoulliLikelihood

from xfads.decorators import *

from xfads.smoothers.nonlinear_smoother import (
    NonlinearFilter as NonlinearFilterN,
    LowRankNonlinearStateSpaceModel as LowRankNonlinearStateSpaceModelN,
)
from xfads.smoothers.nonlinear_smoother_causal import (
    NonlinearFilter as NonlinearFilterC,
    LowRankNonlinearStateSpaceModel as LowRankNonlinearStateSpaceModelC,
)

#from xfads.smoothers.nonlinear_smoother_causal_debug import (
#    NonlinearFilter as NonlinearFilterC,
#    LowRankNonlinearStateSpaceModel as LowRankNonlinearStateSpaceModelC,
#)

from xfads.smoothers.nonlinear_smoother_causal import (
    NonlinearFilterWithInput as NonlinearFilterCwInput,
    LowRankNonlinearStateSpaceModelWithInput as LowRankNonlinearStateSpaceModelCwInput,
)


def build_dynamics_fn(cfg, dynamics_type):
    """Build the prior dynamics module for a given dynamics_type.

    Likelihood-agnostic, so every create_xfads_* factory can share one dynamics
    selection: 'gru' (a nonlinear GRU flow), 'linear' (z_t = A z_{t-1}), or
    'diffusion' (fixed identity, z_t = z_{t-1} + noise -- a nonlinear-observation
    smoother).

    'nonlinear' is a deprecated alias for 'gru': it names a category rather than
    the specific flow it selects, so it is discouraged in favor of the explicit
    'gru' (leaving room for other nonlinear options, e.g. an MGU).
    """
    if dynamics_type == 'nonlinear':
        warnings.warn(
            "dynamics_type='nonlinear' is deprecated: it names a category, not the "
            "specific flow, and currently maps to the GRU. Use dynamics_type='gru' instead.",
            DeprecationWarning, stacklevel=2)
        dynamics_type = 'gru'

    if dynamics_type == 'gru':
        return utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics,
                                                 device=cfg.device,
                                                 use_layer_norm=getattr(cfg, 'use_layer_norm', False))
    elif dynamics_type == 'linear':
        return utils.DynamicsLinear(cfg.n_latents, device=cfg.device)
    elif dynamics_type == 'diffusion':
        return utils.DynamicsEye()
    else:
        raise ValueError(f"unsupported dynamics_type: {dynamics_type!r} "
                         f"(use 'gru', 'linear', or 'diffusion')")


@memory_cleanup
def create_xfads_poisson_log_link(cfg, n_neurons_obs, train_dataloader, model_type='n', dynamics_type='gru'):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))

    if train_dataloader is not None:
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)

    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = build_dynamics_fn(cfg, dynamics_type)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)

    """sequential vae"""
    if model_type == 'n':
        nl_filter = NonlinearFilterN(dynamics_mod, initial_condition_pdf, device=cfg.device)
        ssm = LowRankNonlinearStateSpaceModelN(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)
    elif model_type == 'c':
        nl_filter = NonlinearFilterC(dynamics_mod, initial_condition_pdf, device=cfg.device)
        ssm = LowRankNonlinearStateSpaceModelC(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

    return ssm


@memory_cleanup
def create_xfads_poisson_log_link_w_input(cfg, n_neurons_obs, n_inputs, train_dataloader, model_type='n', dynamics_type='gru'):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))

    if train_dataloader is not None:
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = build_dynamics_fn(cfg, dynamics_type)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)

    """input encoder"""
    input_encoder = MaskedInputEncoder(n_inputs, cfg.n_latents, cfg.n_latents_read, device=cfg.device)

    """sequential vae"""
    if model_type == 'c':
        nl_filter = NonlinearFilterCwInput(input_encoder, dynamics_mod, initial_condition_pdf, device=cfg.device)
        ssm = LowRankNonlinearStateSpaceModelCwInput(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                                     local_encoder, nl_filter, device=cfg.device)
    else:
        # the with-input factory only supports the causal filter (model_type='c')
        raise NotImplementedError(f"model_type={model_type!r} not supported by the with-input factory (use 'c')")

    return ssm


def create_xfads_bernoulli_log_link_w_input(cfg, n_neurons_obs, n_inputs, train_dataloader, model_type='n', dynamics_type='gru'):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs, device=cfg.device))
    readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz).to(cfg.device)
    likelihood_pdf = BernoulliLikelihood(readout_fn, n_neurons_obs, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = build_dynamics_fn(cfg, dynamics_type)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)

    """input encoder"""
    input_encoder = nn.Linear(n_inputs, cfg.n_latents, bias=False, device=cfg.device)

    """sequential vae"""
    if model_type == 'c':
        nl_filter = NonlinearFilterCwInput(input_encoder, dynamics_mod, initial_condition_pdf, device=cfg.device)
        ssm = LowRankNonlinearStateSpaceModelCwInput(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                                     local_encoder, nl_filter, device=cfg.device)
    else:
        # the with-input factory only supports the causal filter (model_type='c')
        raise NotImplementedError(f"model_type={model_type!r} not supported by the with-input factory (use 'c')")

    return ssm
