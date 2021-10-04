import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logging

from .rcParams import *

log = logging.getLogger(__name__)


def violins(model, trace, key, labels=None):
    """
    High level plotting function for violin plot.

    Parameters
    ----------
    model : Cov19Model
    trace: av.InferenceData
    key: str
    labels: array-like str, optional
        Labels for the single violin plots, has to be the same
        length as the last key dimension.
    """
    var = np.array(trace.posterior[key])
    var = var.reshape((var.shape[0] * var.shape[1],) + var.shape[2:])

    return _violins(x=var, labels=labels)


def _violins(x, labels=None, ax=None):
    # Create dataframe for easy seaborn plotting
    df = pd.DataFrame(x)

    if labels is not None:
        df.columns = labels
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 1 * x.shape[-1]))

    ax = sns.violinplot(data=df, orient="h", ax=ax)

    return ax


if __name__ == "__main__":
    # Load example trace
    import pickle

    def load():
        with open(f"../data/default.pickle", "rb") as f:
            return pickle.load(f)

    model, trace = load()
    ax = violin(model, trace, "Delta_alpha_c")
