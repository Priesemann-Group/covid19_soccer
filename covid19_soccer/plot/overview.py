import datetime
import matplotlib.pyplot as plt
from math import ceil
from .timeseries import *
from .distributions import distribution
from .other import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
    game_effects(ax, trace, model, dl,type=type_game_effects)
    axes_ts.append(ax)

    """ Distribution(s)
    """
    if plot_delay:
        ax = fig.add_subplot(grid[0:-1, -1])
    else:
        ax = fig.add_subplot(grid[0:, -1])

    soccer_related_cases(ax, trace, model, dl, verbose=verbose, add_beta=plot_beta,type=type_soccer_related_cases)

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

    # delay
    ax = fig.add_subplot(grid[0, 1])
    distribution(
        model, trace, "delay", nSamples_prior=5000, title="", dist_math="D", ax=ax,
    )
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

    # gender interaction factors
    ax = fig.add_subplot(grid[1, 1])
    distribution(
        model,
        trace,
        "factor_female",
        nSamples_prior=5000,
        title="",
        dist_math="\omega_{fem}",
        ax=ax,
    )
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

    # likelihood and week modulation
    ax = fig.add_subplot(grid[2, 1])
    distribution(
        model,
        trace,
        "sigma_obs",
        nSamples_prior=5000,
        title="",
        dist_math="\sigma_{obs}",
        ax=ax,
    )
    ax = fig.add_subplot(grid[2, 2])
    distribution(
        model,
        trace,
        "weekend_factor",
        nSamples_prior=5000,
        title="",
        dist_math="h_{w}",
        ax=ax,
    )

    ax = fig.add_subplot(grid[3, 1])
    distribution(
        model,
        trace,
        "offset_modulation",
        nSamples_prior=5000,
        title="",
        dist_math="\chi_{w}",
        ax=ax,
    )

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
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

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
            type_soccer_related_cases=type_soccer_related_cases
        )

        axes_t[0].set_title(dl.countries[0])
        axes.append(axes_t)

    # Kinda dirty fix to align y labels, does not work incredible well
    fig.align_ylabels()

    return axes
