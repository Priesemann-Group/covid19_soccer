import datetime
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pymc3 as pm
import string

from .timeseries import *
from .distributions import distribution, _distribution
from .other import *
from .utils import get_from_trace, sigmoid


def single(
    trace,
    model,
    dl,
    xlim=None,
    outer_grid=None,
    plot_delay=False,
    plot_beta=False,
    verbose=False,
    type_game_effects="violin",
    type_soccer_related_cases="violin",
):
    """
    Create a single simple overview plot for one model run / country.
    This plot contains Incidence, fraction_male_female the single game effects
    and the soccer related percentage of infections.

    Adjust colors with rcParams

    Parameters
    ----------
    outer_grid : mpl grid
        If you want to plot the overview plot inside another grid,
        usefull for comparisons
    plot_delay : bool
        Plot delay distribution into overview, default: False

    Returns
    -------
    axes

    """
    if outer_grid is None:
        fig = plt.figure(figsize=(7, 5))
        grid = fig.add_gridspec(3, 2, hspace=0.25, width_ratios=[1, 0.3])
    else:
        if type_soccer_related_cases == "skip":
            grid = outer_grid.subgridspec(3, 1, hspace=0.25,)
        else:
            grid = outer_grid.subgridspec(
                3, 2, wspace=0.2, hspace=0.25, width_ratios=[1, 0.3]
            )
        fig = outer_grid.get_gridspec().figure

    axes_ts = []
    """ Timeseries plots
    """
    # Cases
    ax = fig.add_subplot(grid[0, 0])
    incidence(ax, trace, model, dl)
    axes_ts.append(ax)

    # Gender imbalance
    ax = fig.add_subplot(grid[1, 0])
    fraction_male_female(ax, trace, model, dl)
    axes_ts.append(ax)

    # Single game effects
    ax = fig.add_subplot(grid[2, 0])
    game_effects(ax, trace, model, dl, type=type_game_effects)
    axes_ts.append(ax)

    """ Distribution(s)
    """
    if type_soccer_related_cases != "skip":
        if plot_delay:
            ax = fig.add_subplot(grid[0:-1, -1])
        else:
            ax = fig.add_subplot(grid[0:, -1])
        soccer_related_cases(
            ax,
            trace,
            model,
            dl,
            verbose=verbose,
            add_beta=plot_beta,
            type=type_soccer_related_cases,
        )

    if plot_delay:
        ax_delay = fig.add_subplot(grid[-1, -1])
        distribution(
            model,
            trace,
            "delay",
            nSamples_prior=5000,
            title="",
            dist_math="D",
            ax=ax_delay,
        )

    """ Markup
    """
    if xlim is None:
        xlim = [model.sim_begin, model.sim_end]

    for ax in axes_ts:
        ax.set_xlim(xlim)
        format_date_axis(ax)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # remove labels for first and second timeseries
    for ax in axes_ts[:2]:
        ax.set(xticklabels=[])

    axes_ts.append(ax)
    if plot_delay:
        axes_ts.append(ax_delay)

    return axes_ts


def single_extended(trace, model, dl, xlim=None):
    """
    Create an extended overview plot for a single model run.
    This includes incidence, gender imbalance, R_base, R_soccer+R_noise,
    delay, delay-width, factor_female, c_off, sigma_obs
    and the weekend factors.

    Adjust colors with rcParams
    """

    fig = plt.figure(figsize=(7, 1.35 * 4))
    axes_ts = []

    grid = fig.add_gridspec(
        4, 3, wspace=0.15, hspace=0.25, width_ratios=[1, 0.25, 0.25]
    )

    """ timeseries plots
    """
    # Cases
    ax = fig.add_subplot(grid[0, 0])
    incidence(ax, trace, model, dl)
    axes_ts.append(ax)

    # Gender imbalance
    ax = fig.add_subplot(grid[1, 0])
    fraction_male_female(ax, trace, model, dl)
    axes_ts.append(ax)

    # R_base
    ax = fig.add_subplot(grid[2, 0])
    R_base(ax, trace, model, dl)
    axes_ts.append(ax)

    # R_soccer
    ax = fig.add_subplot(grid[3, 0])
    R_soccer(ax, trace, model, dl, add_noise=True)
    axes_ts.append(ax)

    # R_noise
    # ax = fig.add_subplot(grid[4, 0])
    # R_noise(ax, trace, model, dl)
    # axes_ts.append(ax)

    """ distributions
    """
    axes_dist = []

    # delay
    ax = fig.add_subplot(grid[0, 1])
    if dl.countries[0] == "Germany":
        ax.set_xlim(4.1,8)
    else:
        ax.set_xlim(3.1,7)
    distribution(
        model, trace, "delay", nSamples_prior=5000, title="", dist_math="D", ax=ax,
    )
    axes_dist.append(ax)
    ax = fig.add_subplot(grid[0, 2])
    distribution(
        model,
        trace,
        "delay-width",
        nSamples_prior=5000,
        title="",
        dist_math="\sigma_{D}",
        ax=ax,
    )
    axes_dist.append(ax)

    # gender interaction factors
    ax = fig.add_subplot(grid[1, 1])
    ax.set_xlim(0.01,0.5)
    distribution(
        model,
        trace,
        "factor_female",
        nSamples_prior=5000,
        title="",
        dist_math="\omega_{fem}",
        ax=ax,
    )
    axes_dist.append(ax)    
    ax = fig.add_subplot(grid[1, 2])
    distribution(
        model,
        trace,
        "c_off",
        nSamples_prior=5000,
        title="",
        dist_math="c_{off}",
        ax=ax,
    )
    axes_dist.append(ax)


    # likelihood and week modulation
    ax = fig.add_subplot(grid[2, 1])
    posterior = get_from_trace("fraction_delayed_by_weekday",trace)
    prior = pm.sample_prior_predictive(
        samples=5000, model=model, var_names=["fraction_delayed_by_weekday"]
    )["fraction_delayed_by_weekday"]
    ax.set_xlim(0.1,1)
    _distribution(
        array_posterior=sigmoid(posterior[:,5]),
        array_prior=sigmoid(prior[:,5]),
        dist_name="",
        dist_math="r_{\mathrm{Sat}}",
        ax=ax,
    )
    axes_dist.append(ax)

    ax = fig.add_subplot(grid[2, 2])
    _distribution(
        array_posterior=sigmoid(posterior[:,6]),
        array_prior=sigmoid(prior[:,6]),
        dist_name="",
        dist_math="r_{\mathrm{Sun}}",
        ax=ax,
    )
    axes_dist.append(ax)

    ax = fig.add_subplot(grid[3, 1])    
    distribution(
        model,
        trace,
        "weekend_factor",
        nSamples_prior=5000,
        title="",
        dist_math="h_{w}",
        ax=ax,
    )
    axes_dist.append(ax)

    """ Legend
    """
    custom_lines = [
        Line2D(
            [0],
            [0],
            marker="d",
            color=rcParams.color_data,
            label="Scatter",
            markersize=4,
            lw=0,
        ),
        Line2D([0], [0], color=rcParams.color_model, lw=2),
        Line2D([0], [0], color=rcParams.color_prior, lw=2),
        Patch([0], [0], color=rcParams.color_posterior, lw=2.5,),
    ]
    ax = fig.add_subplot(grid[3, 2])
    ax.legend(
        custom_lines, ["Data", "Model", "Prior", "Posterior",], loc="center",
    )
    ax.axis("off")

    # Adjust xlim for timeseries plots
    for ax in axes_ts:
        if xlim is None:
            ax.set_xlim(model.sim_begin, model.sim_end)
        else:
            ax.set_xlim(xlim)

        # Hack: Disable every second ticklabel
        #for label in ax.xaxis.get_ticklabels()[::2]:
        #    label.set_visible(False)

            
    #Add axes annotations
    alphabet_string = list(string.ascii_uppercase)
    for i, ax in enumerate(axes_ts):
        letter = alphabet_string[i]
        ax.text(-0.05, 1.1, letter, transform=ax.transAxes,
              fontsize=8, fontweight='bold', va='top', ha='right')
        
    for i, ax in enumerate(axes_dist):
        letter = alphabet_string[i + len(axes_ts)]
        ax.text(-0.05, 1.1, letter, transform=ax.transAxes,
              fontsize=8, fontweight='bold', va='top', ha='right')
    
    return fig


def multi(
    traces,
    models,
    dls,
    nColumns=2,
    xlim=None,
    verbose=False,
    plot_delay=False,
    plot_beta=False,
    type_game_effects="violin",
    type_soccer_related_cases="violin",
):
    """
    Creates a overview plot for multiple model runs e.g. different countries.

    Order of runs goes left to right and up to down.

    Parameters
    ----------
    traces: list 1d

    models: list 1d

    dls: list 1d

    nColumns: number
        Number of columns in the plot, default: 2
    """

    nRows = ceil(len(traces) / nColumns)

    fig = plt.figure(figsize=(3.5 * nColumns, 2.5 * nRows))

    outer_grid = fig.add_gridspec(nRows, nColumns, wspace=0.4, hspace=0.35,)
    axes = []
    for i, (trace, model, dl) in enumerate(zip(traces, models, dls)):
        # Mapping to 2d index
        x = i % nColumns
        y = i // nColumns

        axes_t = single(
            trace,
            model,
            dl,
            outer_grid=outer_grid[y, x],
            xlim=xlim,
            verbose=verbose,
            plot_delay=plot_delay,
            plot_beta=plot_beta,
            type_game_effects=type_game_effects,
            type_soccer_related_cases=type_soccer_related_cases,
        )

        axes_t[0].set_title(dl.countries[0])
        axes.append(axes_t)

    # Kinda dirty fix to align y labels, does not work incredible well
    fig.align_ylabels()

    return axes


def multi_v2(
    traces,
    models,
    dls,
    selected_index=[0, 1, 2, 3],
    nColumns=2,
    xlim=None,
    verbose=False,
    plot_delay=False,
    plot_beta=False,
    type_game_effects="violin",
    type_soccer_related_cases="skip",
    fig=None,
    ypos_flags=-20,
):
    """ Create outer layout
    """
    nRows = ceil(len(selected_index) / nColumns)

    if fig is None:
        fig = plt.figure(figsize=(3.5 * nColumns, 2.5 * (nRows + 1)))

    outer_outer_grid = fig.add_gridspec(2, 1, hspace=0.15, height_ratios=(nRows,0.8))
    """ Create single overview plots for all selected countries
    """
    outer_grid = outer_outer_grid[0].subgridspec(
        nRows, nColumns, wspace=0.25, hspace=0.25,
    )
    axes = []
    sel_traces = [traces[i] for i in selected_index]
    sel_models = [models[i] for i in selected_index]
    sel_dls = [dls[i] for i in selected_index]
    for i, (trace, model, dl) in enumerate(zip(sel_traces, sel_models, sel_dls)):
        # Mapping to 2d index
        x = i % nColumns
        y = i // nColumns

        axes_t = single(
            trace,
            model,
            dl,
            outer_grid=outer_grid[y, x],
            xlim=xlim,
            verbose=verbose,
            plot_delay=plot_delay,
            plot_beta=plot_beta,
            type_game_effects=type_game_effects,
            type_soccer_related_cases=type_soccer_related_cases,
        )
        if x > 0:
            for ax in axes_t:
                ax.set_ylabel("")
        axes_t[0].set_title(dl.countries[0])
        axes.append(axes_t)

    """ Last row: overview plot of all countries:
    """
    inner_grid = outer_outer_grid[-1].subgridspec(1, 2, width_ratios=(12.0, 0.5))
    ax_row = []

    # Plot percentage of soccer
    ax = fig.add_subplot(inner_grid[0, 0])
    soccer_related_cases_overview(ax, traces, models, dls, plot_flags=True,ypos_flags=ypos_flags,remove_outliers=True,bw=0.5)
    ax_row.append(ax)

    # Plot legend into corner
    ax = fig.add_subplot(inner_grid[0, 1])
    legend(ax=ax, posterior=False, prior=False,championship_range=True)
    ax_row.append(ax)
    ax_row.append(None)
    axes.append(ax_row)

    return axes
