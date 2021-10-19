import logging

import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm

from covid19_inference.model.model import modelcontext
from covid19_inference.model import utility as ut

log = logging.getLogger(__name__)


def kernelized_spread_soccer(
    R_t_base,
    R_t_soccer,
    C_base,
    C_soccer,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_S_t="S_t",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_new_E_begin=50,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    model=None,
    return_all=False,
    use_gamma=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    log-normal distributed.

    Parameters
    ----------
    R_t_base : :class:`~theano.tensor.TensorVariable`
        Base reproduction number
        shape: (time)

    R_t_soccer : :class:`~theano.tensor.TensorVariable`
        Soccer reproduction number is applied additively 
        shape: (time)

    C_base : :class:`~theano.tensor.TensorVariable`
        Base gender interaction matrix i.e. while no soccer games are in progress.
        shape: (gender, gender)

    C_soccer : :class:`~theano.tensor.TensorVariable`
        Gender interaction for soccer games.
        shape: (gender, gender)

    pr_new_E_begin : :class:`~theano.tensor.TensorVariable`, float or array_like, optional
        If a float is given defaults to prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool, optional
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``. Default: False

    Returns
    -------
    name_new_I_t : :class:`~theano.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    name_new_E_t : :class:`~theano.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    name_S_t : :class:`~theano.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("kernelized spread soccer")
    model = modelcontext(model)

    # Total number of people in population by gender
    N = model.N_population  # shape: (gender)

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, tt.Variable):
        new_E_begin = pr_new_E_begin
    else:
        new_E_begin = pm.HalfCauchy(
            name=name_new_E_begin,
            beta=pr_new_E_begin,
            shape=(11, N.shape[0]),  # 11, nGenders
        )

    # Initial susceptible compartment shape: (genders)
    S_begin = N - pm.math.sum(new_E_begin, axis=0)

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )
    # Choose transition rates (E to I) according to incubation period distribution
    x = np.arange(1, 11)
    if use_gamma:
        beta = ut.tt_gamma(x, median_incubation, np.exp(sigma_incubation))
    else:
        beta = ut.tt_lognormal(x, tt.log(median_incubation), sigma_incubation)

    # Define kernelized spread model:
    def next_day(
        R_base,
        R_soccer,
        S_t,
        nE1_m,
        nE2_m,
        nE3_m,
        nE4_m,
        nE5_m,
        nE6_m,
        nE7_m,
        nE8_m,
        nE9_m,
        nE10_m,
        nE1_f,
        nE2_f,
        nE3_f,
        nE4_f,
        nE5_f,
        nE6_f,
        nE7_f,
        nE8_f,
        nE9_f,
        nE10_f,
        _,
        beta,
        N,
        C_base,
        C_soccer,
    ):
        new_I_t_m = (
            beta[0] * nE1_m
            + beta[1] * nE2_m
            + beta[2] * nE3_m
            + beta[3] * nE4_m
            + beta[4] * nE5_m
            + beta[5] * nE6_m
            + beta[6] * nE7_m
            + beta[7] * nE8_m
            + beta[8] * nE9_m
            + beta[9] * nE10_m
        )

        new_I_t_f = (
            beta[0] * nE1_f
            + beta[1] * nE2_f
            + beta[2] * nE3_f
            + beta[3] * nE4_f
            + beta[4] * nE5_f
            + beta[5] * nE6_f
            + beta[6] * nE7_f
            + beta[7] * nE8_f
            + beta[8] * nE9_f
            + beta[9] * nE10_f
        )

        new_I_t = tt.stack([new_I_t_m, new_I_t_f])
        # shape gender
        new_E_t = (
            S_t
            / N
            * tt.tensordot(R_base * C_base + R_soccer * C_soccer, new_I_t, axes=1)
        )

        new_E_t = tt.clip(new_E_t, 0, N)

        # Update susceptible compartment
        S_t = S_t - new_E_t
        S_t = tt.clip(S_t, -1, N)
        return S_t, new_E_t[0], new_E_t[1], new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, new_I
    new_I_0 = tt.zeros(N.shape[0])
    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[R_t_base, R_t_soccer],
        outputs_info=[
            S_begin,  # shape: gender
            dict(
                initial=new_E_begin[..., 0],
                taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
            ),  # shape time, gender,
            dict(
                initial=new_E_begin[..., 1],
                taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
            ),  # shape time, gender,
            new_I_0,  # shape gender,
        ],
        non_sequences=[beta, N, C_base, C_soccer],
    )

    S_t, new_E_t_m, new_E_t_f, new_I_t = outputs
    pm.Deterministic(name_new_I_t, new_I_t)

    new_E_t = tt.stack((new_E_t_m, new_E_t_f), axis=-1)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t_m, new_E_t_f, S_t
    else:
        return new_E_t
