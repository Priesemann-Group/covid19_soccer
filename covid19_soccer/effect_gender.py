import logging
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt

from covid19_inference.model.model import modelcontext

log = logging.getLogger(__name__)


def R_t_soccer(alpha_prior, date_of_games, beta_prior=None, S=None, model=None, f_robust=1):
    """
    Constructs impact of soccer uefa games to the reproduction number.

    Parameters
    ----------
    alpha_priors : :class:`~theano.tensor.TensorVariable`
        Priors expectation for the effect of each uefa game. Since the effect onto the base
        reproduction number is additive a prior values of zero corresponds to
        no effect.
        Shape: (game)

    date_of_games : dateteime64, array-like
        For each of the games contains the corresponding date as datetime object.

    beta_prior : np.array, optional
        Prior location matrix, encodes where a game took place and eventual
        priors that encodes whether effective hygiene concepts were implemented
        during the game.
        Shape: (game)

    stadium_size : number, optional
        Stadium size/pop size i.e. weighting factor.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    -------
    R_t_soccer : :class:`~theano.tensor.TensorVariable`
        Soccer specific reproduction number
        shape: (time)
    """
    log.info("R_t_soccer with deltas")
    model = modelcontext(model)

    # Effect for each game
    alpha_raw = alpha(alpha_prior, f_robust=f_robust)
    alpha_raw = pm.Deterministic("alpha_R", alpha_raw)
    eff = alpha_raw

    if beta_prior is not None:
        beta_raw = beta(beta_prior, S)
        beta_raw = pm.Deterministic("beta_R", beta_raw)
        eff += beta_raw

    # Construct d = δ(t_g−t)
    t = np.arange(model.sim_len)
    t_g = [(game - model.sim_begin).days for game in date_of_games]
    d = _delta(np.subtract.outer(t, t_g))

    # Sum over all games
    R_soccer = tt.dot(d, eff)

    R_soccer = tt.clip(R_soccer, -50, 50)  # to avoid nans

    return R_soccer


def alpha(alpha_prior, f_robust):
    r"""
    Model of the effectiveness parameter alpha. The subscript :math:`g`
    is the is the game.

    .. math::

        \alpha_{g} =\alpha_\text{prior,g} ( \alpha_\text{mean} + \Delta \alpha_{g})
        \alpha_\text{mean} \sim \mathcal{N}\left(0, 1\right)
        \Delta \alpha_g \sim \mathcal{N}\left(0, \sigma_{\alpha, \text{game}}\right)\quad \forall g
        \sigma_{\alpha, \text{game}} &\sim HalfNormal\left(0.1\right)

    Parameters
    ----------
    alpha_prior : np.array
        Priors expectation for the effect of each uefa game. Since the effect onto the base
        reproduction number is additive a prior values of zero corresponds to
        no effect. Should have shape (game)

    Returns
    -------
    effect alpha
        shape: (game)
    """
    Δα_g = tt.as_tensor_variable(alpha_prior)

    # Same across all games
    α_mean = pm.Normal(name="alpha_mean", mu=0, sigma=5*f_robust)

    # Per game priors
    # - generated depending on a alpha_prior
    Δα_g_sparse = pm.Normal(
        "Delta_alpha_g_sparse", mu=0, sigma=1, shape=len(alpha_prior[alpha_prior > 0])
    )
    σ_g = pm.HalfNormal(name="sigma_alpha_g", sigma=5*f_robust)

    # Set the entries for the played games
    Δα_g = tt.set_subtensor(Δα_g[alpha_prior > 0], Δα_g_sparse)
    Δα_g = Δα_g * σ_g

    return alpha_prior * (α_mean + Δα_g)


def beta(beta_prior, S):
    """
    Model the effectivness parameter beta. The subsript :math:`g` is the game.

    .. math::

        \beta_{g} = \beta_\text{prior,g} S (\alpha_\text{mean} + \Delta \alpha_g)
        \beta_\text{mean} &\sim \mathcal{N}\left(0, 1\right)
        \Delta \beta_g &\sim \mathcal{N}\left(0, \sigma_{\beta, \text{game}}\right) \quad \forall g
        \sigma_{\beta, \text{game}} &\sim HalfNormal\left(0.1\right)

    Parameters
    ----------
    beta_prior : np.array
        Prior location matrix, encodes where a game took place and eventual
        priors that encodes whether effective hygiene concepts were implemented
        during the game. Should have shape: (game)

    S : number
        Stadium size/pop. size i.e. weighting factor

    Returns
    -------
    effect beta
        Shape (game)
    """
    Δβ_g = tt.as_tensor_variable(beta_prior)

    # Same across all games
    β_mean = pm.Normal(name="beta_mean", mu=0, sigma=5)

    # Offset per game
    # - generated depending on a alpha_prior
    Δβ_g_sparse = pm.Normal(
        "Delta_beta_g_sparse", mu=0, sigma=1, shape=len(beta_prior[beta_prior > 0])
    )
    σ_g = pm.HalfNormal(name="sigma_beta_g", sigma=5)

    # Set the entries for the played games
    Δβ_g = tt.set_subtensor(Δβ_g[beta_prior > 0], Δβ_g_sparse)
    Δβ_g = Δβ_g * σ_g

    return beta_prior * S * (β_mean + Δβ_g)


def _delta(x, a=1, normalized=True):
    r"""
    Dirac delta peak function, normalized.

        .. math::

        \delta_a(t) = e^{-(x/a)^2}

    Parameters
    ----------
    x : tensor
        Input tensor containing the values for the delta function.
    a : number
        Shape parameter of the delta peak function.
    """
    a = tt.exp(-((x / a) ** 2))
    if normalized:
        a = a / tt.sum(a, axis=0)
    return a
