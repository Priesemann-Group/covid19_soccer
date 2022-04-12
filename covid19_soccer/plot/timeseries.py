from re import X
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import numpy as np
import os
import matplotlib.transforms as transforms

from covid19_inference.plot import _timeseries, _format_date_xticks
from .utils import get_from_trace, format_date_axis, lighten_color

from .rcParams import *

log = logging.getLogger(__name__)


def _uefa_range(ax):
    # Show time of uefa championship
    begin = datetime.datetime(2021, 6, 11)
    end = datetime.datetime(2021, 7, 11)
    ylim = ax.get_ylim()
    ax.fill_betweenx(
        (-1000000, 1000000),
        begin,
        end,
        edgecolor=rcParams.color_championship_range,
        facecolor=rcParams.color_championship_range,
        zorder=-5,
        alpha=0.2,
    )
    ax.set_ylim(ylim)


def incidence(
    ax,
    trace,
    model,
    dl,
    ylim=None,
    color=None,
    color_data=None,
    data_forecast=False,
    lw=2,
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
        lw=lw,
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

    if data_forecast:
        dates = dl._cases.loc[
            model.data_end :, "male", "total",
        ].index.get_level_values("date")
        cases = np.stack(
            (
                dl._cases.loc[model.data_end :, "male", "total",].to_numpy(),
                dl._cases.loc[model.data_end :, "female", "total",].to_numpy(),
            ),
            axis=1,
        )
        incidence = (
            cases.sum(axis=(1, 2)) / (dl.population[0, 0] + dl.population[1, 0]) * 1e6
        )
        _timeseries(
            x=dates,
            y=incidence,
            what="data",
            ax=ax,
            color="tab:red",
            ms=1.5,
            alpha=0.8,
        )

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, data_points.max() + data_points.max() / 8)

    # Plot shaded uefa
    _uefa_range(ax)

    # Markup
    ax.set_ylabel("Incidence")
    format_date_axis(ax)

    return ax


def fraction_male_female(
    ax, trace, model, dl, ylim=None, color=None, color_data=None, data_forecast=True
):
    """
    Plot fraction between male and female cases normalized by population size
    for the corresponding sex.
    """

    new_cases = get_from_trace("new_cases", trace)

    model_points = (
        new_cases[:, :, 0] / dl.population[0, 0]
        - new_cases[:, :, 1] / dl.population[1, 0]
    ) / (
        new_cases[:, :, 1] / dl.population[1, 0]
        + new_cases[:, :, 0] / dl.population[0, 0]
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
        y=(
            dl.new_cases_obs[:, 0, 0] / dl.population[0, 0]
            - dl.new_cases_obs[:, 1, 0] / dl.population[1, 0]
        )
        / (
            dl.new_cases_obs[:, 1, 0] / dl.population[1, 0]
            + dl.new_cases_obs[:, 0, 0] / dl.population[0, 0]
        ),
        what="data",
        ax=ax,
        color=rcParams.color_data if color_data is None else color_data,
        ms=1.5,
    )

    if data_forecast:
        dates = dl._cases.loc[
            model.data_end :, "male", "total",
        ].index.get_level_values("date")
        cases = np.stack(
            (
                dl._cases.loc[model.data_end :, "male", "total",].to_numpy(),
                dl._cases.loc[model.data_end :, "female", "total",].to_numpy(),
            ),
            axis=1,
        )
        imbalance = (
            cases[:, 0, 0] / dl.population[0, 0] - cases[:, 1, 0] / dl.population[1, 0]
        ) / (
            cases[:, 0, 0] / dl.population[0, 0] + cases[:, 1, 0] / dl.population[1, 0]
        )
        _timeseries(
            x=dates,
            y=imbalance,
            what="data",
            ax=ax,
            color="tab:red",
            ms=1.5,
            alpha=0.8,
        )

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)

    # Plot shaded uefa
    _uefa_range(ax)

    # Dotted line at 0
    ax.axhline(0, ls="--", color="tab:gray", zorder=-100)

    # Markup
    ax.set_ylabel("Gender\nimbalance")
    format_date_axis(ax)

    return ax

def stringency(ax,trace,model,dl,ylim=None, color=None,legend=True,**kwargs):
    # Plot data
    _timeseries(
        x=dl._stringencyOxCGRT[0][model.sim_begin:model.sim_end].index,
        y=dl._stringencyOxCGRT[0][model.sim_begin:model.sim_end],
        ax=ax,
        what="model",
        color="black" if color is None else color,
        label="OxCGRT",
        **kwargs
    )       
    _timeseries(
        x=dl._stringencyPHSM[0][model.sim_begin:model.sim_end].index,
        y=dl._stringencyPHSM[0][model.sim_begin:model.sim_end],
        ax=ax,
        what="model",
        color="black" if color is None else color,
        ls="--",
        label="PHSM",
        **kwargs
    )  
    
    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)
        
    # Markup
    ylabel = "Stringency\nindex"
    ax.set_ylabel(ylabel)
    format_date_axis(ax)
    ax.legend()
    return ax
    
def R_soccer(ax, trace, model, dl, ylim=None, color=None, add_noise=False, **kwargs):
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
    ylabel = r"$\Delta R_{\mathrm{soccer}}"
    if add_noise:
        ylabel += "+R_{noise}"
    ylabel += r"$"
    ax.set_ylabel(ylabel)
    format_date_axis(ax)

    return ax


def R_base(ax, trace, model, dl, ylim=None, color=None, **kwargs):
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
    ax.set_ylabel("$R_\mathrm{base}$")
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
    ax.set_ylabel("$\Delta R_\mathrm{noise}$")
    format_date_axis(ax)

    return ax


def what_if(ax, traces, model, dl, ylim=None, colors=None, **kwargs):

    for i, trace in enumerate(traces):
        # Check if it a real trace or one sampled with fast posterior predictive
        if "posterior" in trace:
            new_cases = get_from_trace("new_cases", trace)
        else:
            new_cases = np.array(trace["new_cases"])
            new_cases = new_cases.reshape(
                (new_cases.shape[0] * new_cases.shape[1],) + new_cases.shape[2:]
            )
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


def stacked_filled(x, y, ax=None, colors=None, date_format=True, **kwargs):
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3))

    if "linewidth" in kwargs:
        del kwargs["linewidth"]
    if "marker" in kwargs:
        del kwargs["marker"]
    kwargs["lw"] = 0

    y = np.array(y)
    for i, y_i in enumerate(y):
        # Fill area between two bars
        ax.fill_between(
            x,
            np.zeros(y_i.shape) if i == 0 else np.sum(y[:i], axis=0),
            np.sum(y[:i], axis=0) + y[i],
            color=colors[i] if colors is not None else None,
            **kwargs
        )

    if date_format:
        format_date_axis(ax)
    return ax


def stacked_bars(x, y, ax=None, colors=None, date_format=True, **kwargs):
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3))

    for i, y_i in enumerate(y):
        ax.bar(
            x,
            y_i,
            bottom=np.sum(y[:i], axis=0),
            color=colors[i] if colors is not None else None,
            **kwargs
        )

    if date_format:
        format_date_axis(ax)
    return ax


def mark_days(ax, traces, model, dl, date_format=True, hosted=False, **kwargs):
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3))
    
    iso2 = dl.countries_iso2[0]
    temp = dl.timetable[~dl.timetable["id"].str.contains("a")]
    
    # location
    locations = pd.read_csv(
        os.path.join(dl.data_folder, "em_locations.csv"), header=7
    )
    stad_loc = locations[locations["country"] == iso2]["name"]
    if len(stad_loc) > 0:
        stad_loc = stad_loc.values[0]
    else:
        stad_loc = "well no stadium location found"
    
    selector_played = np.any([temp["team1"]==iso2,temp["team2"]==iso2],axis=0)
    selector_hosted = temp["location"]==stad_loc
    selector_union = np.any([selector_played,selector_hosted],axis=0)
    if hosted:
        games_played = temp[np.all([selector_played,~selector_hosted],axis=0)]
        games_hosted = temp[np.all([~selector_played,selector_hosted],axis=0)]
        games_both = temp[np.all([selector_played,selector_hosted],axis=0)]
    else:
        games_played = temp[selector_played]
        games_hosted = pd.DataFrame()
        games_both = pd.DataFrame()
    
    # Plot only played
    color_played = "tab:gray"
    color_hosted = "tab:black"
    
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)


    y = ax.get_ylim()[0]
    for i, g in games_played.iterrows():
         ax.plot(
            g["date"],
            0.06,
            fillstyle="full",
            color=color_played,
            marker="v",
            markeredgecolor="none",
            transform=trans,
            markersize=5,
         )
            
    for i, g in games_hosted.iterrows():
         ax.plot(
            g["date"],
            0.06,
            fillstyle="full",
            color=color_hosted,
            marker="v",
            markeredgecolor="none",
            markersize=5,
            transform=trans,
         )     

    for i, g in games_both.iterrows():
         ax.plot(
            g["date"],
            0.06,
            fillstyle="right",
            color=color_played,
            markerfacecoloralt=color_hosted,
            markeredgecolor="none",
            marker="v",
            markersize=5,
            transform=trans,
         )    
            
    if date_format:
        format_date_axis(ax)
        
    return ax
