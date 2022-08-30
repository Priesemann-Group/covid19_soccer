import logging
import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm
from covid19_inference.model.model import modelcontext

log = logging.getLogger(__name__)


def delay_cases_weekday(
    cases, model=None, f_robust=1,
):
    log.info("Delaying cases by weekday")
    model = modelcontext(model)

    r_base_high = pm.Normal("fraction_delayed_weekend_raw", mu=-3, sigma=2*f_robust)
    r_base_low = pm.Normal("fraction_delayed_week_raw", mu=-5, sigma=1*f_robust)
    sigma_r = pm.HalfNormal("sigma_r", sigma=1*f_robust)

    delta_r = (pm.Normal("delta_fraction_delayed", mu=0, sigma=1, shape=7)) * sigma_r
    e = pm.HalfCauchy("error_fraction", beta=0.2*f_robust)

    r_base = tt.stack(
        [
            r_base_high,
            r_base_low,
            r_base_low,
            r_base_low,
            r_base_high,
            r_base_high,
            r_base_high,
        ]  # Monday @ zero
    )
    r_week = r_base + delta_r

    r_transformed_week = tt.nnet.sigmoid(r_week)
    pm.Deterministic("fraction_delayed_by_weekday", r_week)

    t = np.arange(model.sim_shape[0]) + model.sim_begin.weekday()  # Monday @ zero

    week_matrix = np.zeros((model.sim_shape[0], 7), dtype="float")
    week_matrix[np.stack([t] * 7, axis=1) % 7 == np.arange(7)] = 1.0

    r_t = tt.dot(week_matrix, r_transformed_week)[:, None]

    fraction = pm.Beta(
        "fraction_delayed", alpha=r_t / e, beta=(1 - r_t) / e, shape=model.sim_shape
    )

    def loop_delay_by_weekday(cases_t, fraction_t, cases_before, fraction_before):
        new_cases = (1 - fraction_t) * (cases_t + fraction_before * cases_before)
        return new_cases, fraction_t

    (cases_delayed, _), _ = theano.scan(
        fn=loop_delay_by_weekday,
        sequences=[cases, fraction],
        outputs_info=[cases[0], fraction[0]],  # shape gender, countries
    )
    pm.Deterministic("delayed_cases_by_weekday", cases_delayed)
    return cases_delayed
