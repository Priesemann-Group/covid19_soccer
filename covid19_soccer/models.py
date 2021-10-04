import logging

import pymc3 as pm
import theano.tensor as tt
import datetime

from .dataloader import Dataloader_gender
from . import effect_gender
from .utils import get_cps
from .compartmental_models import kernelized_spread_soccer

from covid19_inference.model import (
    lambda_t_with_sigmoids,
    uncorrelated_prior_E,
    week_modulation,
    student_t_likelihood,
    delay_cases,
    Cov19Model,
)

log = logging.getLogger(__name__)


def create_model_gender(
    dataloader=None, beta=True, use_gamma=False,
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

    # Config for changepoints
    pr_sigma_lambda_cp = 0.3
    pr_median_lambda = 1.0

    # Median of the prior for the delay in case reporting, we assume 10 days
    pr_delay = 10

    cps_dict = dict(  # one possible change point every sunday
        pr_sigma_lambda=pr_sigma_lambda_cp,  # wiggle compared to previous point
        relative_to_previous=True,
        pr_factor_to_previous=1.0,
        pr_sigma_transient_len=1,
        pr_median_transient_len=4,
        pr_median_lambda=pr_median_lambda,
        pr_sigma_date_transient=3.5,
    )

    # Change points every 10 days
    change_points = get_cps(
        dl.data_begin - datetime.timedelta(days=10),
        dl.data_end,
        interval=10,
        offset=5,
        **cps_dict
    )

    alpha_prior = dl.alpha_prior[0, :]  # only select first country

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
        sigma_lambda_cp = (
            pm.HalfStudentT(
                name="sigma_lambda_cp",
                nu=4,
                sigma=1,
                transform=pm.transforms.log_exp_m1,
            )
        ) * 0.1
        sigma_lambda_week_cp = None
        R_t_base_log = lambda_t_with_sigmoids(
            change_points_list=change_points,
            pr_median_lambda_0=pr_median_lambda,
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

        R_t_add_noise = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=20,
                pr_median_transient_len=12,
                pr_sigma_transient_len=4,
                pr_sigma_date_transient=3,
            ),
            pr_median_lambda_0=1.0,
            pr_sigma_lambda_0=0.1,
            name_lambda_t="R_t_add_noise_fact",
            prefix_lambdas="R_t_add_noise_fact_",
            shape=1,
            hierarchical=False,
        )[:, 0]

        R_t_add += R_t_add_noise
        pm.Deterministic("R_t_add_fact", R_t_add)

        # Default gender interconnection matrix
        c_off = pm.Beta("c_off", alpha=4, beta=4)
        C_0 = tt.stack([1.0 - c_off, c_off])
        C_1 = tt.stack([c_off, 1.0 - c_off])
        C_base = tt.stack([C_0, C_1])
        pm.Deterministic("C_base", C_base)

        # Soccer gender interconnection matrix (i.e. for soccer matches)
        f_female = pm.Gamma("factor_female", alpha=4, beta=40)
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
        )

        # Delay the cases by a log-normal reporting delay and add them as a trace variable
        new_cases = delay_cases(
            cases=new_cases,
            name_cases="delayed_cases",
            pr_mean_of_median=pr_delay,
            pr_median_of_width=3.0,
            seperate_on_axes=False,
            num_seperated_axes=2,  # num genders
            # num_variants=dl.nGenders,
            use_gamma=use_gamma,
        )

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
        student_t_likelihood(cases=new_cases)

    return this_model


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    import covid19_uefa

    model = covid19_uefa.create_model_gender()
