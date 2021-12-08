import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import numpy as np

from covid19_inference.plot import _timeseries
from .utils import get_from_trace, format_date_axis

from .rcParams import *

log = logging.getLogger(__name__)


def incidence(
    ax, trace, model, dl, ylim=None, color=None, color_data=None,
):
    """
    Plots incidence: modelfit and data
    """
    new_cases = get_from_trace("new_cases", trace)
    # Plot model fit
    _timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=(new_cases[:, :, 0] + new_cases[:, :, 1])
        / (dl.population[0, 0] + dl.population[1, 0])
        * 1e6,  # incidence
        what="model",
        ax=ax,
        color=rcParams.color_model if color is None else color,
    )

    # Plot data
    data_points = (
        (dl.new_cases_obs[:, 0, 0] + dl.new_cases_obs[:, 1, 0])
        / (dl.population[0, 0] + dl.population[1, 0])
        * 1e6
    )
    _timeseries(
        x=pd.date_range(dl.data_begin, dl.data_end),
        y=data_points,
        what="data",
        ax=ax,
        color=rcParams.color_data if color_data is None else color_data,
        ms=1.5,
        alpha=0.8,
    )

    # Show time of uefa championship
    begin = datetime.datetime(2021, 6, 11)
    end = datetime.datetime(2021, 7, 11)
    ax.fill_betweenx(np.arange(0, 10000), begin, end, color=rcParams.color_championship_range)

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, data_points.max() + data_points.max() / 8)

    # Markup
    ax.set_ylabel("Incidence")
    format_date_axis(ax)

    return ax


def fraction_male_female(
    ax, trace, model, dl, ylim=None, color=None, color_data=None,
):
    """
    Plot fraction between male and female cases normalized by population size
    for the corresponding sex.
    """

    new_cases = get_from_trace("new_cases", trace)

    model_points = (new_cases[:, :, 0] / dl.population[0, 0]) / (
        new_cases[:, :, 1] / dl.population[1, 0]
    )
    # Plot model fit
    _timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=model_points,  # male/female
        what="model",
        ax=ax,
        color=rcParams.color_model if color is None else color,
    )

    # Plot data
    _timeseries(
        x=pd.date_range(dl.data_begin, dl.data_end),
        y=(dl.new_cases_obs[:, 0, 0] / dl.population[0, 0])
        / (dl.new_cases_obs[:, 1, 0] / dl.population[1, 0]),  # male/female
        what="data",
        ax=ax,
        color=rcParams.color_data if color_data is None else color_data,
        ms=1.5,
    )

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)

    # Markup
    ax.set_ylabel("Gender\nimbalance")
    format_date_axis(ax)

    return ax


def R_soccer(ax, trace, model, dl, ylim=None, color=None, add_noise=False,**kwargs):
    """
    Plots soccer related reproduction number
    """

    R_soccer = get_from_trace("R_t_soccer", trace)
    if add_noise:
        R_soccer = R_soccer + get_from_trace("R_t_add_noise_fact", trace)[:, :, 0]

    # Plot model fit
    _timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=R_soccer,
        ax=ax,
        what="model",
        color=rcParams.color_model if color is None else color,
        **kwargs
    )

    # Plot baseline
    if add_noise:
        ax.axhline(1, ls="--", color="tab:gray", zorder=-100)
    else:
        ax.axhline(0, ls="--", color="tab:gray", zorder=-100)

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)

    # Markup
    ylabel = "$R_{soccer}"
    if add_noise:
        ylabel += "+R_{noise}"
    ylabel += "$"
    ax.set_ylabel(ylabel)
    format_date_axis(ax)

    return ax


def R_base(ax, trace, model, dl, ylim=None, color=None,**kwargs):
    R_base = get_from_trace("R_t_base", trace)

    # Plot model fit
    _timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=R_base,
        ax=ax,
        what="model",
        color=rcParams.color_model if color is None else color,
        **kwargs
    )

    # Plot baseline
    ax.axhline(1, ls="--", color="tab:gray", zorder=-100)

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)

    # Markup
    ax.set_ylabel("$R_{base}$")
    format_date_axis(ax)

    return ax


def R_noise(ax, trace, model, dl, ylim=None, color=None):
    R_noise = get_from_trace("R_t_add_noise_fact", trace)[:, :, 0]

    # Plot model fit
    _timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=R_noise,
        ax=ax,
        what="model",
        color=rcParams.color_model if color is None else color,
    )

    # Plot baseline
    ax.axhline(1, ls="--", color="tab:gray", zorder=-100)

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)

    # Markup
    ax.set_ylabel("$R_{noise}$")
    format_date_axis(ax)

    return ax


def what_if(ax,traces,model,dl,ylim=None,colors=None,**kwargs):
    
    for i,trace in enumerate(traces):
        # Check if it a real trace or one sampled with fast posterior predictive
        if "posterior" in trace: 
            new_cases = get_from_trace("new_cases",trace)
        else:
            new_cases = np.array(trace["new_cases"])
            new_cases = new_cases.reshape((new_cases.shape[0] * new_cases.shape[1],) + new_cases.shape[2:])
        color = None if colors is None else colors[i] 
        # Plot model fit
        _timeseries(
            x=pd.date_range(model.sim_begin, model.sim_end),
            y=(new_cases[:, :, 0] + new_cases[:, :, 1])
            / (dl.population[0, 0] + dl.population[1, 0])
            * 1e6,  # incidence
            ax=ax,
            what="model",
            color=rcParams.color_model if color is None else color,
            **kwargs
        )
    format_date_axis(ax)
    return ax
    