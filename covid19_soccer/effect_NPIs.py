import logging
import pymc3 as pm
import numpy as np
import theano.tensor as tt

from covid19_inference.model.delay import _delay_timeshift

log = logging.getLogger(__name__)


def oxford_stringency_modulation(stringency, diff_data_sim, fcast_len):
    log.info("Oxford stringency modulation")

    offset = pm.Normal("offset_stringency", mu=0, sigma=2)
    fact = (pm.Normal("factor_stringency", mu=1, sigma=2)) * 0.005
    #nonlinearity_stringency = pm.Normal("nonlinearity_stringency", mu=0, sigma=0.5)
    nonlinearity_stringency = 0
    padding = 10
    offset = tt.clip(offset, -padding+1, padding-1)
    pad_begin = padding + diff_data_sim
    pad_end = padding + fcast_len
    string = np.pad(stringency, (pad_begin, pad_end), mode="edge")
    string = _delay_timeshift(
        string, len(string), len(string) - padding, offset, padding
    )[:-padding]
    #nonlinearity_stringency = tt.clip(nonlinearity_stringency, -1.5, 1.5)
    #range_string = tt.max(stringency) - tt.min(stringency)
    #string = - (
    #    range_string
    #    * fact
    #    * ((string - tt.min(stringency)) / range_string)
    #    ** tt.exp(nonlinearity_stringency)
    #)
    string = -(string - np.mean(stringency))*fact
    string = tt.clip(string, -1.5, 1.5)
    string = pm.Deterministic("stringency_modulation", string)
    return string
