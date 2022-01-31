import sys
sys.path.append("../")
sys.path.append("../covid19_inference")

import theano.tensor as tt
from covid19_soccer.dataloader import Dataloader_gender, Dataloader
from covid19_soccer import delay_by_weekday
from covid19_soccer.utils import get_cps


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

def create_model_delay_only(
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
        new_cases = pm.Normal("new_E_t", 100, 1, shape=model.sim_shape)
        
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