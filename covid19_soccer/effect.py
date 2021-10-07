from pymc3 import Normal, Deterministic, HalfNormal, Gamma
import numpy as np
import theano.tensor as tt

try:
    from covid19_inference.model.model import modelcontext
    from covid19_inference.model import utility as ut

except:
    import sys

    sys.path.append("../covid19_inference_repo/")
    import covid19_inference
    from covid19_inference.model.model import modelcontext
    from covid19_inference.model import utility as ut


def alpha(nRegions, nPhases, alpha_prior, game2phase, name="alpha", factor_female=None):
    """
    Model of the effectivness parameter alpha. The subsript :math:`p`
    is the tournament phase, :math:`g` is the game and :math:`c`
    is the country.

    .. math::

        \alpha_{c,g} = \alpha_\text{prior,c,g} (\alpha_\text{mean}+ \Delta \alpha_c + \Delta \alpha_{p(g)})

        \alpha_mean \sim \mathcal{N}\left(0, 1\right)
        \Delta \alpha_c \sim \mathcal{N}\left(0, \sigma_{\alpha, \text{country}}\right) \quad \forall c
        \Delta \alpha_p \sim \mathcal{N}\left(0, \sigma_{\alpha, \text{phase}}\right)\quad \forall p
        \sigma_{\alpha, \text{country}}  &\sim HalfNormal\left(0.1\right)
        \sigma_{\alpha, \text{phase}} &\sim HalfNormal\left(0.1\right)

    Parameters
    ----------
    nPhases : int
        Number of phases in the tournament
    nRegions : int
        Number of regions/countries
    alpha_prior : tensor
        Prior expectation matrix, encodes the effect of a game. If a country didn't
        participate in a game, the entry is put to 0.
        shape: country, game
    game2phase : int array-like
        Mapping for each game to the corresponding phase.
        shape: game
    factor_female : pymc3 variable, optional
        Increase dimensions of output to account for different genders

    Returns
    -------
    effect alpha
        shape: ([gender,] country, game)
    """
    α_prior_c_g = alpha_prior

    # Same across all games & countries
    α_mean = Normal(name="alpha_mean", mu=0, sigma=1)

    # Offset for each country/team
    Δα_c = Normal(name="Delta_alpha_c", mu=0, sigma=1, shape=nRegions)

    σ_c = HalfNormal(name="sigma_alpha_c", sigma=0.1)

    Δα_c = Δα_c * σ_c

    # Offset for each tournament phase
    Δα_p = Normal(name="Delta_alpha_p", mu=0, sigma=1, shape=nPhases)

    σ_p = HalfNormal(name="sigma_alpha_p", sigma=0.1)

    Δα_p = Δα_p * σ_p

    # Stack Δα_p to match up with the game by the game2phase mapping
    Δα_g = tt.dot(game2phase, Δα_p)  # Δα_p(g) in manuscript

    # Reshape Δα_c to allow for easier addition
    Δα_c = tt.stack([Δα_c] * α_prior_c_g.shape[1], axis=1)

    # Calculate effect with previous defined priors
    alpha = α_prior_c_g * (α_mean + Δα_c + Δα_g)

    if factor_female is not None:
        alpha = tt.stack([alpha, alpha * factor_female])

    # Add to trace
    Deterministic(name, alpha)
    return alpha


def beta(
    nRegions, nPhases, beta_prior, S_c, game2phase, name="beta", factor_female=None
):
    """
    Model the effectivness parameter beta. The subsript :math:`p`
    is the tournament phase, :math:`g` is the game and :math:`c`
    is the country.

    .. math::

        \beta_{c,g} = \beta_\text{prior,c,g} S_c (\alpha_\text{mean}+ \Delta \alpha_c + \Delta \alpha_{p(g)})

        \beta_\text{mean} &\sim \mathcal{N}\left(0, 1\right)
        \Delta \beta_c &\sim \mathcal{N}\left(0, \sigma_{\beta, \text{country}}\right) \quad \forall c
        \Delta \beta_p &\sim \mathcal{N}\left(0, \sigma_{\beta, \text{phase}}\right)\quad \forall p
        \sigma_{\beta, \text{country}}  &\sim HalfNormal\left(0.1\right)
        \sigma_{\beta, \text{phase}} &\sim HalfNormal\left(0.1\right)

    Parameters
    ----------
    nPhases : int
        Number of phases in the tournament
    nRegions : int
        Number of regions/countries
    beta_prior : tensor
        Prior location matrix, encodes where a game took place and eventual
        priors that encodes whether effective hygiene concepts were implemented
        during the game. shape: regions, games
    S_c : tensor
        Weighting factor
        shape: country
    game2phase : int array-like
        Mapping for each game to the corresponding phase.
        shape: game
    factor_female : pymc3 variable, optional
        Increase dimensions of output to account for different genders

    Returns
    -------
    effect beta
        Shape ([gender,] country, game)
    """
    β_prior_c_g = beta_prior

    β_mean = Normal(name="beta_mean", mu=0, sigma=1)

    # Offset for each country/team
    Δβ_c = Normal(name="Delta_beta_c", mu=0, sigma=1, shape=nRegions)

    σ_c = HalfNormal(name="sigma_beta_c", sigma=0.1)

    Δβ_c = Δβ_c * σ_c

    # Offset for each tournament phase
    Δβ_p = Normal(name="Delta_beta_p", mu=0, sigma=1, shape=nPhases)

    σ_p = HalfNormal(name="sigma_beta_p", sigma=0.1)

    Δβ_p = Δβ_p * σ_p

    # Stack Δα_p to match up with the game by the game2phase mapping
    Δβ_g = tt.dot(game2phase, Δβ_p)  # Δα_p(g) in manuscript

    # Reshape Δβ_c to allow for easier addition
    Δβ_c = tt.stack([Δβ_c] * len(game2phase), axis=1)

    # Calculate effect with previous defined priors
    beta = S_c.dot(β_prior_c_g) * (β_mean + Δβ_c + Δβ_g)

    if factor_female is not None:
        beta = tt.stack([beta, beta * factor_female], axis=0)

    # Add to trace
    Deterministic(name, beta)
    return beta


def gamma(T_c, model=None):
    """
    Models the effect of local temperature on covid cases.

    .. math::

        \gamma_c(t) &= \eta\cdot\epsilon\cdot\ln \left(\exp\left(\frac{1}{\epsilon}\left(T_c\left(t\right) - T_\text{crit}\right) \right) + 1\right)
        T_\text{crit} &\sim \mathcal{N}\left(20, 7\right)
        \eta &\sim \mathcal{N}\left(0, 0.1\right)
        \epsilon &\sim Gamma\left(k=5, \theta=0.2\right)

    Parameters
    ----------
    T_c : tensor
        Data tensor of mean temperature in each country over time.
        shape: (time, country)

    ToDo
    ----
    Hierachy?!

    """

    model = modelcontext(model)

    # Priors

    # 𝜂 = Normal(name="eta", mu=0, sigma=0.1)

    𝜂, _ = ut.hierarchical_normal(
        name_L1=f"eta_hc_L1",
        name_L2=f"eta_hc_L2",
        name_sigma=f"sigma_eta_hc_L1",
        pr_mean=0,
        pr_sigma=0.1,
        error_cauchy=False,
    )

    𝜖 = (Gamma(name="epsilon", alpha=5, beta=5)) + 1e-5

    # T_crit = Normal(name="T_crit", mu=20, sigma=7)

    T_crit, _ = ut.hierarchical_normal(
        name_L1=f"T_crit_hc_L1",
        name_L2=f"T_crit_hc_L2",
        name_sigma=f"sigma_T_crit_hc_L1",
        pr_mean=20,
        pr_sigma=5,
        error_cauchy=False,
    )

    # Calculate
    gamma_c = 𝜂 * 𝜖 * tt.log(tt.exp(1 / 𝜖 * (T_c - T_crit)) + 1.0)

    # Mapping to full model timescale with matrix
    transfer = np.zeros((model.sim_len, model.data_len + model.diff_data_sim))
    for i in range(model.data_len + model.diff_data_sim):
        transfer[i, i] = 1

    # Convert to theano
    transfer = tt.as_tensor_variable(transfer)
    # Apply
    gamma_c = transfer.dot(gamma_c)

    gamma_c = tt.clip(gamma_c, -0.5, 0.5)

    # Add to trace
    Deterministic("gamma_c", gamma_c)

    return gamma_c


def _delta(x, a=1, normalized=True):
    """
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
