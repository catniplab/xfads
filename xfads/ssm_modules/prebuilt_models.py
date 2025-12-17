import sys
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


@memory_cleanup
def create_xfads_poisson_log_link(cfg, n_neurons_obs, train_dataloader, model_type='n'):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))

    if train_dataloader is not None:
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)

    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)

    if dynamics_type is 'nonlinear':
        dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)

    elif dynamics_type is 'diffusion':
        dynamics_fn = utils.DynamicsEye()

    else:
        sys.exit('not a supported "dynamics_type"')

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
def create_xfads_poisson_log_link_w_input(cfg, n_neurons_obs, n_inputs, train_dataloader, model_type='n', dynamics_type='nonlinear'):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))

    if train_dataloader is not None:
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)

    if dynamics_type is 'nonlinear':
        dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics,
                                                        device=cfg.device, use_layer_norm=cfg.use_layer_norm)
    elif dynamics_type is 'diffusion':
        dynamics_fn = utils.DynamicsEye()

    else:
        sys.exit('not a supported "dynamics_type"')

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
    if model_type == 'n':
        print('not supported')
        sys.exit()
        # nl_filter = NonlinearFilterN(dynamics_mod, initial_condition_pdf, device=cfg.device)
        # ssm = LowRankNonlinearStateSpaceModelN(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
        #                                   local_encoder, nl_filter, device=cfg.device)
    elif model_type == 'c':
        nl_filter = NonlinearFilterCwInput(input_encoder, dynamics_mod, initial_condition_pdf, device=cfg.device)
        ssm = LowRankNonlinearStateSpaceModelCwInput(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                                     local_encoder, nl_filter, device=cfg.device)
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

    return ssm


def create_xfads_bernoulli_log_link_w_input(cfg, n_neurons_obs, n_inputs, train_dataloader, model_type='n'):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs, device=cfg.device))
    readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz).to(cfg.device)
    likelihood_pdf = BernoulliLikelihood(readout_fn, n_neurons_obs, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
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
    if model_type == 'n':
        print('not supported')
        sys.exit()
        # nl_filter = NonlinearFilterN(dynamics_mod, initial_condition_pdf, device=cfg.device)
        # ssm = LowRankNonlinearStateSpaceModelN(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
        #                                   local_encoder, nl_filter, device=cfg.device)
    elif model_type == 'c':
        nl_filter = NonlinearFilterCwInput(input_encoder, dynamics_mod, initial_condition_pdf, device=cfg.device)
        ssm = LowRankNonlinearStateSpaceModelCwInput(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                                     local_encoder, nl_filter, device=cfg.device)
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

    return ssm
