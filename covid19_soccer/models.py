import logging

import pymc3 as pm
import theano.tensor as tt
import datetime
import pandas as pd
import numpy as np

from .dataloader import Dataloader_gender, Dataloader
from . import effect_gender
from . import effect
from . import delay_by_weekday
from .utils import get_cps
from .compartmental_models import kernelized_spread_soccer

from covid19_inference.model import (
    lambda_t_with_sigmoids,
    uncorrelated_prior_E,
    week_modulation,
    student_t_likelihood,
    delay_cases,
    Cov19Model,
    uncorrelated_prior_I,
    kernelized_spread,
    SIR,
)

log = logging.getLogger(__name__)


def create_model(
    use_alpha=True,
    use_beta=True,
    use_gamma=False,
    last_year=False,
    hierarchical_spreading_rate=False,
    dataloader=None,
    spread_method="SIR",
):
    """
    Constructs a model using the defined model features. The default
    model is the main one described in our publication without any
    additionally modeled effect such as seasonality or temperature
    dependency.

    Parameters
    ----------
    alpha : bool, optional
        ToDo
        default: True
    beta : bool, optional
        ToDo
        default: True
    gamma : bool, optional
        Add the proposed effect of temperature to our
        model.
        default: False
    spread_method: str, optional
        Can be SIR or kernelized

    Returns
    -------
    pymc3.Model
    """

    """ # Building the model
    """
    if dataloader is None:
        dl = Dataloader(countries=["Germany", "FR", "Italy"])
    else:
        dl = dataloader
    # print(dl.countries)
    # Define changepoints (we define all vars to surpress the automatic prints)

    if spread_method == "SIR":
        pr_sigma_lambda_cp = 0.2
        pr_median_lambda = 0.125
    elif spread_method == "kernelized":
        pr_sigma_lambda_cp = 0.3
        pr_median_lambda = 1.0
    else:
        raise RuntimeError("spread_method has to be SIR or kernelized")

    change_points = []
    for day in pd.date_range(
        start=dl.data_begin - datetime.timedelta(days=10), end=dl.data_end
    ):
        if day.weekday() == 6:
            # Add cp
            change_points.append(
                dict(  # one possible change point every sunday
                    pr_mean_date_transient=day,
                    pr_sigma_date_transient=1,
                    pr_sigma_lambda=pr_sigma_lambda_cp,  # wiggle compared to previous point
                    relative_to_previous=True,
                    pr_factor_to_previous=1.0,
                    pr_sigma_transient_len=1,
                    pr_median_transient_len=4,
                    pr_median_lambda=pr_median_lambda,
                )
            )
    pr_delay = 10

    if last_year:
        new_cases_obs = dl.new_cases_obs_last_year
        temperature = dl.temperature_last_year
    else:
        new_cases_obs = dl.new_cases_obs
        temperature = dl.temperature

    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": dl.data_begin,
        "fcast_len": 0,
        "diff_data_sim": int((dl.data_begin - dl.sim_begin).days),
        "N_population": dl.population,
    }

    with Cov19Model(**params) as this_model:
        """
        First part of the basic spreading dynamics modeled alike to our publication:
            "Inferring change points in the spread of COVID-19
            reveals the effectiveness of interventions"
        see https://science.sciencemag.org/content/369/6500/eabb9789.full
        """
        # Get base reproduction number/spreading rate
        if not hierarchical_spreading_rate:
            sigma_lambda_cp = (
                pm.HalfStudentT(
                    name="sigma_lambda_cp",
                    nu=4,
                    sigma=1,
                    transform=pm.transforms.log_exp_m1,
                )
            ) * 0.1
            sigma_lambda_week_cp = None
        else:
            sigma_lambda_cp = (
                pm.HalfStudentT(
                    name="sigma_lambda_cp",
                    nu=4,
                    sigma=1,
                    transform=pm.transforms.log_exp_m1,
                )
            ) * 0.05
            sigma_lambda_week_cp = (
                pm.HalfStudentT(
                    name="sigma_lambda_week_cp",
                    nu=4,
                    sigma=1,
                    transform=pm.transforms.log_exp_m1,
                )
            ) * 0.1
        base_lambda_t_log = lambda_t_with_sigmoids(
            change_points_list=change_points,
            pr_median_lambda_0=pr_median_lambda,
            hierarchical=hierarchical_spreading_rate,
            name_lambda_t="base_lambda_t",
            sigma_lambda_cp=sigma_lambda_cp,
            sigma_lambda_week_cp=sigma_lambda_week_cp,
        )

        # Adds the recovery rate mu to the model as a random variable
        if spread_method == "SIR":
            mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)

        if spread_method == "SIR":
            pm.Deterministic("base_eff_spreading_rate", tt.exp(base_lambda_t_log) - mu)
        else:
            pm.Deterministic("base_R_t", tt.exp(base_lambda_t_log))

        # This builds a decorrelated prior for I_begin for faster inference. It is not
        # necessary to use it, one can simply remove it and use the default argument for
        # pr_I_begin in cov19.SIR
        if spread_method == "SIR":
            prior_I = uncorrelated_prior_I(
                lambda_t_log=base_lambda_t_log, mu=mu, pr_median_delay=pr_delay
            )
        else:
            new_E_begin = uncorrelated_prior_E()

        """
        Begin of new parts from the publication:
            "Inference of EURO 2020 match-induced effect in
            COVID-19 cases across Europe"
            see TODO
        """

        # Part to multiply with delta function
        eff = tt.zeros((dl.nRegions, dl.nGames))
        if use_alpha:
            eff += effect.alpha(
                nRegions=dl.nRegions,
                nPhases=dl.nPhases,
                alpha_prior=dl.alpha_prior,
                game2phase=dl.game2phase,
            )
        if use_beta:
            eff += effect.beta(
                nRegions=dl.nRegions,
                nPhases=dl.nPhases,
                beta_prior=dl.beta_prior,
                S_c=dl.stadium_size / dl.population,
                game2phase=dl.game2phase,
            )

        # Additive factor without delta function
        add = tt.zeros(dl.nRegions)
        if use_gamma:
            add += effect.gamma(T_c=temperature)

        #
        t = np.arange(this_model.sim_len)
        t_g = [(game - this_model.sim_begin).days for game in dl.date_of_games]

        d = effect._delta(np.subtract.outer(t, t_g))

        # Calc effect on R
        lambda_t_log = base_lambda_t_log + add + d.dot((eff).T)
        lambda_t = tt.exp(lambda_t_log)

        if spread_method == "SIR":
            pm.Deterministic("lambda_t", lambda_t)
            pm.Deterministic("R_t", lambda_t / mu)
            pm.Deterministic("eff_spreading_rate", lambda_t - mu)
        else:
            pm.Deterministic("R_t", lambda_t)

        """
        Second part of the basic spreading dynamics modeled alike to our publication:
            "Inferring change points in the spread of COVID-19
            reveals the effectiveness of interventions"
        see https://science.sciencemag.org/content/369/6500/eabb9789.full
        """
        # Use lambda_t_log and mu as parameters for the SIR model.
        # The SIR model generates the inferred new daily cases.
        if spread_method == "kernelized":
            new_cases = kernelized_spread(
                lambda_t_log=lambda_t_log, pr_new_E_begin=new_E_begin
            )
        else:
            new_cases = SIR(lambda_t_log=lambda_t_log, mu=mu, pr_I_begin=prior_I)

        # Delay the cases by a lognormal reporting delay and add them as a trace variable
        new_cases = delay_cases(
            cases=new_cases,
            name_cases="delayed_cases",
            pr_mean_of_median=pr_delay,
            pr_median_of_width=0.3,
        )

        # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
        # Also adds the "new_cases" variable to the trace that has all model features.
        new_cases = week_modulation(cases=new_cases, name_cases="new_cases")
        # Define the likelihood, uses the new_cases_obs set as model parameter
        student_t_likelihood(cases=new_cases)

    return this_model


def create_model_gender(
    dataloader=None,
    beta=True,
    use_gamma=False,
    draw_width_delay=True,
    use_weighted_alpha_prior=False,
    prior_delay=-1,
    width_delay_prior=0.1,
    sigma_incubation=-1,
    median_width_delay=1.0,
    interval_cps=10,
):
    """
    High level function to create an abstract pymc3 model using different defined
    model features. The default model is the main one described in our publication
    without any additionally modelled effect such as seasonality or temperature
    dependency.

    Parameters
    ----------
    dataloader : :class:`Dataloader`, optional
        Dataloader class for making the interaction with the data a lot
        easier. See `dataloader.py` for details. Even though the dataloader supports
        multiple countries you should only use a single one!
        Defaults to Scotland
    beta: bool
        Use beta model compartment
    use_gamma: bool
        Use gamma delay kernel

    Returns
    -------
    pymc3.Model

    """

    if dataloader is None:
        dl = Dataloader_gender(countries=["Scotland"])
    else:
        dl = dataloader

    if prior_delay == -1:
        if dl.countries[0] in ["Germany"]:
            prior_delay = 7
        elif dl.countries[0] in ["Scotland", "France", "England", "Netherlands"]:
            prior_delay = 4
        # elif dl.countries[0] in ["Portugal"]:
        # prior_delay = 5
        else:
            prior_delay = 5
            width_delay_prior = 0.15
            # raise RuntimeError("Country not known")

    # Median of the prior for the delay in case reporting, we assume 10 days
    default_interval = 10
    ratio_interval = interval_cps / default_interval
    cps_dict = dict(  # one possible change point every sunday
        relative_to_previous=True,
        pr_factor_to_previous=1.0,
        pr_sigma_transient_len=1 * ratio_interval,
        pr_median_transient_len=4 * ratio_interval,
        pr_sigma_date_transient=3.5 * ratio_interval,
    )

    # Change points every 10 days
    change_points = get_cps(
        dl.data_begin - datetime.timedelta(days=10),
        dl.data_end,
        interval=interval_cps,
        offset=5,
        **cps_dict,
    )

    if use_weighted_alpha_prior == 1:
        alpha_prior = dl.weighted_alpha_prior[0, :]
    elif use_weighted_alpha_prior == 0:
        alpha_prior = dl.alpha_prior[0, :]  # only select first country
    elif use_weighted_alpha_prior == -1:
        alpha_prior = 0.0
    else:
        raise RuntimeError(
            f"Unknown use_weighted_alpha_prior: {use_weighted_alpha_prior}"
        )

    if beta:
        beta_prior = dl.beta_prior[0, :]
        beta_weight = 1
        if (
            len(beta_prior[beta_prior > 0]) == 0
        ):  # No stadiums in home country -> don't use beta
            beta_prior = None
            stadium_size = None
    else:
        beta_prior = None
        beta_weight = None
        stadium_size = None

    # Construct model params dict
    params = {
        "new_cases_obs": dl.new_cases_obs[:, :, 0],  # only select first country
        "data_begin": dl.data_begin,
        "fcast_len": 16,
        "diff_data_sim": int((dl.data_begin - dl.sim_begin).days),
        "N_population": dl.population[:, 0],  # only select first country
    }

    with Cov19Model(**params) as this_model:

        # We construct a typical slowly changing base reproduction number
        # using sigmoidals. Even though the function is called lambda_t
        # the specific values are controlled by the priors and are around one.
        # https://science.sciencemag.org/content/369/6500/eabb9789.full
        sigma_lambda_cp = pm.HalfCauchy(
            name="sigma_lambda_cp", beta=0.5, transform=pm.transforms.log_exp_m1,
        )
        sigma_lambda_week_cp = None
        R_t_base_log = lambda_t_with_sigmoids(
            change_points_list=change_points,
            pr_median_lambda_0=1.0,
            hierarchical=False,
            sigma_lambda_cp=sigma_lambda_cp,
            sigma_lambda_week_cp=sigma_lambda_week_cp,
            shape=1,
        )
        # Let's also add that to the trace since we may want to plot this variable
        R_t_base = pm.Deterministic("R_t_base", tt.exp(R_t_base_log[..., 0]))

        # We model the effect of the soccer games with a per game
        # delta peak. The priors for each game are defined beforehand
        R_t_add = effect_gender.R_t_soccer(
            alpha_prior=alpha_prior,
            date_of_games=dl.date_of_games,
            beta_prior=beta_prior,
            S=beta_weight,
        )
        pm.Deterministic("R_t_soccer", R_t_add)

        sigma_lambda_cp_noise = pm.HalfCauchy(
            name="sigma_lambda_cp_noise", beta=0.2, transform=pm.transforms.log_exp_m1,
        )
        R_t_add_noise = lambda_t_with_sigmoids(
            change_points_list=change_points,
            pr_median_lambda_0=1.0,
            pr_sigma_lambda_0=0.1,
            sigma_lambda_cp=sigma_lambda_cp_noise,
            name_lambda_t="R_t_add_noise_fact",
            prefix_lambdas="R_t_add_noise_fact_",
            shape=1,
            hierarchical=False,
        )[:, 0]

        R_t_add += R_t_add_noise
        pm.Deterministic("R_t_add_fact", R_t_add)

        # Default gender interconnection matrix
        c_off = pm.Beta("c_off", alpha=8, beta=8)
        C_0 = tt.stack([1.0 - c_off, c_off])
        C_1 = tt.stack([c_off, 1.0 - c_off])
        C_base = tt.stack([C_0, C_1])
        pm.Deterministic("C_base", C_base)

        # Soccer gender interconnection matrix (i.e. for soccer matches)
        f_female = pm.Beta("factor_female", alpha=15, beta=60)
        # Set theano tensor (maybe there is a better way to do that)
        C_0 = tt.stack([1.0 - f_female, f_female])
        C_1 = tt.stack([f_female, f_female * f_female])
        C_soccer = tt.stack([C_0, C_1])
        # Let's also add that to the trace since we may want to plot this variable
        pm.Deterministic("C_soccer", C_soccer)

        # This builds a decorrelated prior for E_begin for faster inference. It is not
        # necessary to use it, one can simply remove it and use the default argument for
        # pr_E_begin in cov19.kernelized_spread gender
        new_E_begin = uncorrelated_prior_E()

        # Compartmental model
        new_cases = kernelized_spread_soccer(
            R_t_base=R_t_base,
            R_t_soccer=R_t_add,
            C_base=C_base,
            C_soccer=C_soccer,
            pr_new_E_begin=new_E_begin,
            use_gamma=use_gamma,
            pr_sigma_median_incubation=sigma_incubation
            if sigma_incubation > 0
            else None,
        )

        # Delay the cases by a log-normal reporting delay and add them as a trace variable
        new_cases = delay_cases(
            cases=new_cases,
            name_cases="delayed_cases",
            pr_mean_of_median=prior_delay,
            pr_sigma_of_median=width_delay_prior,
            pr_median_of_width=median_width_delay / 5 * prior_delay,
            pr_sigma_of_width=0.4 / 5 * prior_delay if draw_width_delay else None,
            seperate_on_axes=False,
            num_seperated_axes=2,  # num genders
            # num_variants=dl.nGenders,
            use_gamma=use_gamma,
            diff_input_output=0,
        )

        new_cases = delay_by_weekday.delay_cases_weekday(new_cases)

        # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
        # Also adds the "new_cases" variable to the trace that has all model features.
        weekend_factor_log = pm.Normal(
            name="weekend_factor_log", mu=tt.log(0.3), sigma=0.5,
        )
        weekend_factor = tt.exp(weekend_factor_log)
        new_cases = week_modulation(
            cases=new_cases,
            name_cases="new_cases",
            pr_mean_weekend_factor=weekend_factor,
        )

        # Define the likelihood, uses the new_cases_obs set as model parameter
        student_t_likelihood(cases=new_cases, sigma_shape=1)

    return this_model


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    sys.path.append("../covid19_inference/")
    import covid19_soccer

    model = covid19_soccer.create_model_gender()
