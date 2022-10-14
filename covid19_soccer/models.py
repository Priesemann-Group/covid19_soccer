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

def create_model_gender(
    dataloader=None,
    beta=True,
    prior_delay=-1,
    median_width_delay=1.0,
    interval_cps=10,
    f_female="0.33",
    use_abs_sine_weekly_modulation=False,
    force_alpha_prior=None,
    f_robust=1,
    generation_interval=4,
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
        if dl.countries[0] in ["Germany", "Germany_alt", "Spain"]:
            prior_delay = 7
            width_delay_prior = 0.1
        elif dl.countries[0] in ["Scotland", "France", "England", "Netherlands"]:
            prior_delay = 4
            width_delay_prior = 0.1
        else:
            prior_delay = 5
            width_delay_prior = 0.15

    log.info(f"Country: {dl.countries[0]}, prior_delay = {prior_delay}")
    # Median of the prior for the delay in case reporting, we assume 10 days
    default_interval = 10
    ratio_interval = interval_cps / default_interval
    cps_dict = dict(
        relative_to_previous=True,
        pr_factor_to_previous=1.0,
        pr_sigma_transient_len=1 * ratio_interval,
        pr_median_transient_len=4 * ratio_interval,
        pr_sigma_date_transient=2.5 * ratio_interval,
    )

    # Change points every 10 days
    change_points = get_cps(
        dl.data_begin - datetime.timedelta(days=10),
        dl.data_end,
        interval=interval_cps,
        offset=5,
        allow_uefa_cps=True,
        **cps_dict,
    )
    if force_alpha_prior is None:
        alpha_prior = dl.alpha_prior[0, :]  # only select first country
    else:
        alpha_prior = force_alpha_prior

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
    fcast_len = 16
    params = {
        "new_cases_obs": dl.new_cases_obs[:, :, 0],  # only select first country
        "data_begin": dl.data_begin,
        "fcast_len": fcast_len,
        "diff_data_sim": int((dl.data_begin - dl.sim_begin).days),
        "N_population": dl.population[:, 0],  # only select first country
    }

    with Cov19Model(**params) as this_model:

        # We construct a typical slowly changing base reproduction number
        # using sigmoidals. Even though the function is called lambda_t
        # the specific values are controlled by the priors and are around one.
        # https://science.sciencemag.org/content/369/6500/eabb9789.full
        sigma_lambda_cp = pm.HalfCauchy(
            name="sigma_lambda_cp", beta=0.5*f_robust, transform=pm.transforms.log_exp_m1,
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
            f_robust=f_robust
        )
        pm.Deterministic("R_t_soccer", R_t_add)

        sigma_lambda_cp_noise = pm.HalfCauchy(
            name="sigma_lambda_cp_noise", beta=0.2*f_robust, transform=pm.transforms.log_exp_m1,
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

        pm.Deterministic("R_t_noise", R_t_add_noise)

        # Default gender interconnection matrix
        c_off = pm.Beta("c_off", alpha=8*f_robust, beta=8*f_robust)
        C_0 = tt.stack([1.0 - c_off, c_off])
        C_1 = tt.stack([c_off, 1.0 - c_off])
        C_base = tt.stack([C_0, C_1])
        pm.Deterministic("C_base", C_base)

        # Soccer gender interconnection matrix (i.e. for soccer matches)
        if f_female == "0.2":
            f_female = pm.Beta("factor_female", alpha=6, beta=24)
        elif f_female == "0.33":
            f_female = pm.Beta("factor_female", alpha=10, beta=20)
        elif f_female == "0.5":
            f_female = pm.Beta("factor_female", alpha=3, beta=3)
        elif f_female == "bounded":
            f_female = (pm.Beta("factor_female", alpha=7, beta=4)) / 2
        else:
            raise RuntimeError("argument value not known")
        # to avoid nans, make sure that value isn't 0 nor 1
        f_female = f_female * 0.999 + 0.0005

        # Define interaction matrix between genders
        C_0 = tt.stack([(1.0 - f_female) ** 2, f_female * (1.0 - f_female)])
        C_1 = tt.stack([f_female * (1.0 - f_female), f_female * f_female])
        C_soccer = tt.stack([C_0, C_1])
        # Normalize such that balanced, [0.5,0.5], case numbers will lead to an unitary
        # increase of total case numbers.
        C_soccer = C_soccer / tt.sqrt(
            tt.sum((tt.dot(C_soccer, np.array([0.5, 0.5]) ** 2)))
        )

        C_0_g = tt.stack([1, 0])
        C_1_g = tt.stack([0, -1])
        C_gender_noise = tt.stack([C_0_g, C_1_g])

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
            R_t_noise=R_t_add_noise,
            C_base=C_base,
            C_soccer=C_soccer,
            C_noise=C_gender_noise,
            pr_new_E_begin=new_E_begin,
            use_gamma=True,
            pr_sigma_median_incubation=None,
            pr_mean_median_incubation=generation_interval,
        )

        # Delay the cases by a log-normal reporting delay and add them as a trace variable
        new_cases = delay_cases(
            cases=new_cases,
            name_cases="delayed_cases",
            pr_mean_of_median=prior_delay,
            pr_sigma_of_median=width_delay_prior,
            pr_median_of_width=median_width_delay / 5 * prior_delay,
            pr_sigma_of_width=0.4 / 5 * prior_delay,
            seperate_on_axes=False,
            num_seperated_axes=2,  # num genders
            # num_variants=dl.nGenders,
            use_gamma=True,
            diff_input_output=0,
        )

        new_cases = delay_by_weekday.delay_cases_weekday(new_cases, f_robust=f_robust)

        if use_abs_sine_weekly_modulation:
            # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
            # Also adds the "new_cases" variable to the trace that has all model features.
            weekend_factor = pm.Beta("weekend_factor_beta", alpha=1.5, beta=5)
            new_cases = week_modulation(
                cases=new_cases,
                name_cases="new_cases",
                pr_mean_weekend_factor=weekend_factor,
            )
        else:
            pm.Deterministic("new_cases", new_cases)

        # Define the likelihood, uses the new_cases_obs set as model parameter
        student_t_likelihood(cases=new_cases, sigma_shape=1)

    return this_model


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    sys.path.append("../covid19_inference/")
    import covid19_soccer

    model = covid19_soccer.create_model_gender()
