import sys
import torch
import torch.nn as nn
from .. import utils, prob_utils
from .dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
from .encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from .likelihoods import PoissonLikelihood, BernoulliLikelihood


from .smoothers.nonlinear_smoother import (
    NonlinearFilter as NonlinearFilterN,
    LowRankNonlinearStateSpaceModel as LowRankNonlinearStateSpaceModelN,
)
from .smoothers.nonlinear_smoother_causal import (
    NonlinearFilter as NonlinearFilterC,
    LowRankNonlinearStateSpaceModel as LowRankNonlinearStateSpaceModelC,
)

# from xfads.smoothers.nonlinear_smoother_causal_debug import (
#    NonlinearFilter as NonlinearFilterC,
#    LowRankNonlinearStateSpaceModel as LowRankNonlinearStateSpaceModelC,
# )

from xfads.smoothers.nonlinear_smoother_causal import (
    NonlinearFilterWithInput as NonlinearFilterCwInput,
    LowRankNonlinearStateSpaceModelWithInput as LowRankNonlinearStateSpaceModelCwInput,
)


def create_xfads_poisson_log_link(cfg, n_neurons_obs, train_dataloader, model_type="n"):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))

    if train_dataloader is not None:
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(
            train_dataloader, cfg.bin_sz
        )

    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz)

    """dynamics module"""
    Q_diag = 1.0 * torch.ones(cfg.n_latents)
    dynamics_fn = utils.build_gru_dynamics_function(
        cfg.n_latents, cfg.n_hidden_dynamics
    )
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents)
    Q_0_diag = 1.0 * torch.ones(cfg.n_latents)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents,
        cfg.n_hidden_backward,
        cfg.n_latents,
        rank_local=cfg.rank_local,
        rank_backward=cfg.rank_backward,
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents,
        n_neurons_obs,
        cfg.n_hidden_local,
        cfg.n_latents,
        rank=cfg.rank_local,
        dropout=cfg.p_local_dropout,
    )

    """sequential vae"""
    if model_type == "n":
        nl_filter = NonlinearFilterN(dynamics_mod, initial_condition_pdf)
        ssm = LowRankNonlinearStateSpaceModelN(
            dynamics_mod,
            likelihood_pdf,
            initial_condition_pdf,
            backward_encoder,
            local_encoder,
            nl_filter,
        )
    elif model_type == "c":
        nl_filter = NonlinearFilterC(dynamics_mod, initial_condition_pdf)
        ssm = LowRankNonlinearStateSpaceModelC(
            dynamics_mod,
            likelihood_pdf,
            initial_condition_pdf,
            backward_encoder,
            local_encoder,
            nl_filter,
        )
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

    return ssm


def create_xfads_poisson_log_link_w_input(
    cfg, n_neurons_obs, n_inputs, train_dataloader, model_type="n"
):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))

    if train_dataloader is not None:
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(
            train_dataloader, cfg.bin_sz
        )
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz)

    """dynamics module"""
    Q_diag = 1.0 * torch.ones(cfg.n_latents)
    dynamics_fn = utils.build_gru_dynamics_function(
        cfg.n_latents,
        cfg.n_hidden_dynamics,
        use_layer_norm=cfg.use_layer_norm,
    )
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents)
    Q_0_diag = 1.0 * torch.ones(cfg.n_latents)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents,
        cfg.n_hidden_backward,
        cfg.n_latents,
        rank_local=cfg.rank_local,
        rank_backward=cfg.rank_backward,
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents,
        n_neurons_obs,
        cfg.n_hidden_local,
        cfg.n_latents,
        rank=cfg.rank_local,
        dropout=cfg.p_local_dropout,
    )

    """input encoder"""
    input_encoder = nn.Sequential(
        nn.Linear(n_inputs, cfg.n_latents, bias=False),
        nn.LayerNorm(cfg.n_latents, bias=False),
    )

    """sequential vae"""
    if model_type == "n":
        print("not supported")
        sys.exit()
        # nl_filter = NonlinearFilterN(dynamics_mod, initial_condition_pdf)
        # ssm = LowRankNonlinearStateSpaceModelN(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
        #                                   local_encoder, nl_filter)
    elif model_type == "c":
        nl_filter = NonlinearFilterCwInput(
            input_encoder, dynamics_mod, initial_condition_pdf
        )
        ssm = LowRankNonlinearStateSpaceModelCwInput(
            dynamics_mod,
            likelihood_pdf,
            initial_condition_pdf,
            backward_encoder,
            local_encoder,
            nl_filter,
        )
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

    return ssm


def create_xfads_bernoulli_log_link_w_input(
    cfg, n_neurons_obs, n_inputs, train_dataloader, model_type="n"
):
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
    readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(
        train_dataloader, cfg.bin_sz
    )
    likelihood_pdf = BernoulliLikelihood(readout_fn, n_neurons_obs)

    """dynamics module"""
    Q_diag = 1.0 * torch.ones(cfg.n_latents)
    dynamics_fn = utils.build_gru_dynamics_function(
        cfg.n_latents, cfg.n_hidden_dynamics
    )
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents)
    Q_0_diag = 1.0 * torch.ones(cfg.n_latents)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents,
        cfg.n_hidden_backward,
        cfg.n_latents,
        rank_local=cfg.rank_local,
        rank_backward=cfg.rank_backward,
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents,
        n_neurons_obs,
        cfg.n_hidden_local,
        cfg.n_latents,
        rank=cfg.rank_local,
        dropout=cfg.p_local_dropout,
    )

    """input encoder"""
    input_encoder = nn.Linear(n_inputs, cfg.n_latents, bias=False)

    """sequential vae"""
    if model_type == "n":
        print("not supported")
        sys.exit()
        # nl_filter = NonlinearFilterN(dynamics_mod, initial_condition_pdf)
        # ssm = LowRankNonlinearStateSpaceModelN(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
        #                                   local_encoder, nl_filter)
    elif model_type == "c":
        nl_filter = NonlinearFilterCwInput(
            input_encoder, dynamics_mod, initial_condition_pdf
        )
        ssm = LowRankNonlinearStateSpaceModelCwInput(
            dynamics_mod,
            likelihood_pdf,
            initial_condition_pdf,
            backward_encoder,
            local_encoder,
            nl_filter,
        )
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

    return ssm
