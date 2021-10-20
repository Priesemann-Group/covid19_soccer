import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import get_from_trace, lighten_color
from .rcParams import colors
import covid19_inference as cov19
from .distributions import distribution


def plot_overview_quad(traces, models, dls):
    id2_country = np.array([["England", "Scotland"], ["Germany", "France"]])

    fig = plt.figure(figsize=(6, 5))
    # Two columns/rows
    outer_grid = fig.add_gridspec(
        2,
        2,
        wspace=0.35,
        hspace=0.25,
    )
    # out = fig.add_subplot(outer_grid[0:,-1])
    # out.set_ylabel("Percentage of soccer related infections\nduring the duration of the Championship")
    # Two rows
    plot_beta = False
    axes = []
    for a in range(2):
        for b in range(2):

            # gridspec inside gridspec
            if plot_beta:
                inner_grid = outer_grid[a, b].subgridspec(
                    3, 3, width_ratios=[1, 0.3, 0.3], wspace=0.5
                )
            else:
                inner_grid = outer_grid[a, b].subgridspec(3, 2, width_ratios=[1, 0.3])

            # Create three subplots
            # - a1: fraction
            # - a2: R_soccer
            # - a3: alpha mean
            # - a4: beta mean
            country = id2_country[a, b]

            a0 = fig.add_subplot(inner_grid[0, 0])
            plot_cases(
                a0, traces[country], models[country], dls[country], ylims_cases[country]
            )

            a1 = fig.add_subplot(inner_grid[1, 0])
            plot_fraction(
                a1,
                traces[country],
                models[country],
                dls[country],
                ylims_fraction[country],
            )

            a2 = fig.add_subplot(inner_grid[2, 0])
            # plot_rsoccer(a2, traces[country], models[country], dls[country])
            plot_reproductionViolin(a2, traces[country], models[country], dls[country])

            a3 = fig.add_subplot(inner_grid[0:, -1])
            plot_relative_from_soccer(
                a3, traces[country], models[country], dls[country]
            )

            if plot_beta and id2_country[a, b] != "France":
                a4 = fig.add_subplot(inner_grid[0:, -1])
                plot_relative_from_soccer(
                    a4, traces[country], models[country], dls[country]
                )

            # Markup
            a0.set_title(country)
            for ax in [a0, a1, a2]:
                ax.set_xlim(xlim_ts)
                # Locator
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
                # Hide the right and top spines
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
            # a2.set_ylim(-1,15)
            # a2.set_xlim((xlim_ts[0]-model.sim_begin).days,(xlim_ts[1]-model.sim_begin).days)
            # a2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

            # remove labels for first and second timeseries
            for ax in [a0, a1]:
                ax.set(xticklabels=[])

            # Restrain y axis of violin plots
            a3.set_yticks([-20, 0, 20])
            if country in ["England", "Scotland"]:
                a3.set_ylim(-5, 30)
            elif country in ["Germany"]:
                a3.set_ylim(-10, 30)
            elif country in ["France"]:
                a3.set_ylim(-30, 30)

            if plot_beta:
                a4.spines["top"].set_visible(False)
                a4.spines["bottom"].set_visible(False)
                a4.spines["left"].set_visible(False)
                a4.tick_params(bottom=False)
                a4.set_xlabel("Stadium")
                a3.set_xlabel("Public\nviewing")
                a3.set_ylabel("")

                if a == 1 and b == 1:
                    a4.set_ylim(-2, 12)
                    a4.set(yticks=[0, 4, 8])
                else:
                    a4.set_ylim(-1, 6)
                    a4.set(yticks=[0, 2, 4])
            if a == 1:
                a2.set_ylim(-1.5, 1.5)

    fig.align_ylabels()
    # Save figure as pdf and png
    kwargs = {"transparent": True, "dpi": 300, "bbox_inches": "tight"}
    fig.savefig(f"{fig_path}/fig_1.pdf", **kwargs)
    fig.savefig(f"{fig_path}/fig_1.png", **kwargs)

    plt.show()
    plt.close(fig=fig)


def plot_overview_single(
    trace,
    model,
    dl,
    ylim_cases=[0, 1000],
    ylim_fraction=[0.6, 1.5],
    ylim_relative=None,
    xlim_ts=[datetime.datetime(2021, 5, 30), datetime.datetime(2021, 7, 23)],
    title="",
    verbose=True,
    violin=False,
):

    fig = plt.figure(figsize=(6, 5))

    grid = fig.add_gridspec(3, 2, wspace=0.35, hspace=0.25, width_ratios=[1, 0.3])

    a0 = fig.add_subplot(grid[0, 0])
    plot_cases(a0, trace, model, dl, ylim_cases)

    a1 = fig.add_subplot(grid[1, 0])
    plot_fraction(a1, trace, model, dl, ylim_fraction)

    a2 = fig.add_subplot(grid[2, 0])
    # plot_rsoccer(a2, traces[country], models[country], dls[country])
    plot_reproductionViolin(a2, trace, model, dl)

    a3 = fig.add_subplot(grid[0:-1, -1])
    plot_relative_from_soccer(
        a3, trace, model, dl, ylim_relative, verbose=verbose, violin=violin
    )

    a4 = fig.add_subplot(grid[-1, -1])
    distribution(
        model,
        trace,
        "delay",
        nSamples_prior=1000,
        title="",
        dist_math="D",
        ax=a4,
    )

    # Set right and bottom axis
    a4.yaxis.tick_right()
    a4.spines["right"].set_visible(True)
    a4.spines["top"].set_visible(False)
    a4.spines["left"].set_visible(False)

    # Markup
    for ax in [a0, a1, a2]:
        ax.set_xlim(xlim_ts)
        # Locator
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    # a2.set_ylim(-1,15)
    # a2.set_xlim((xlim_ts[0]-model.sim_begin).days,(xlim_ts[1]-model.sim_begin).days)
    # a2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

    # remove labels for first and second timeseries
    for ax in [a0, a1]:
        ax.set(xticklabels=[])

    a0.set_title(title)
    return fig


# Functions
def plot_cases(ax, trace, model, dl, ylim=None, color=None, color_data=None):
    """
    Plots number of cases
    """
    new_cases = get_from_trace("new_cases", trace)

    cov19.plot._timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=(new_cases[:, :, 0] + new_cases[:, :, 1])
        / (dl.population[0, 0] + dl.population[1, 0])
        * 1e6,  # incidence
        what="model",
        ax=ax,
        color=colors["cases"] if color is None else color,
    )
    cov19.plot._timeseries(
        x=pd.date_range(model.data_begin, model.data_end),
        y=(dl.new_cases_obs[:, 0, 0] + dl.new_cases_obs[:, 1, 0])
        / (dl.population[0, 0] + dl.population[1, 0])
        * 1e6,  # male/female
        what="data",
        ax=ax,
        color=colors["data"],
        ms=1.5,
        alpha=0.8,
    )

    # Show time of uefa championship
    begin = datetime.datetime(2021, 6, 11)
    end = datetime.datetime(2021, 7, 11)
    ax.fill_betweenx(np.arange(0, 10000), begin, end, alpha=0.1)

    # Adjust ylim
    if ylim is not None:
        ax.set_ylim(ylim)

    # Markup
    ax.set_ylabel("Incidence")

    return ax


def plot_fraction(ax, trace, model, dl, ylim_fraction=None):

    new_cases = get_from_trace("new_cases", trace)

    ## Fraction male/female
    cov19.plot._timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=(new_cases[:, :, 0] / dl.population[0, 0])
        / (new_cases[:, :, 1] / dl.population[1, 0]),  # male/female
        what="model",
        ax=ax,
        color=colors["fraction"],
        alpha=1,
        alpha_ci=0.3,
    )
    cov19.plot._timeseries(
        x=pd.date_range(model.data_begin, model.data_end),
        y=(dl.new_cases_obs[:, 0, 0] / dl.population[0, 0])
        / (dl.new_cases_obs[:, 1, 0] / dl.population[1, 0]),  # male/female
        what="data",
        ax=ax,
        color=colors["data"],
        ms=1.5,
    )

    ax.set_ylabel("Gender\nimbalance")

    if ylim_fraction is not None:
        ax.set_ylim(ylim_fraction)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    return ax


def plot_rsoccer(ax, trace, model, dl):
    """
    Plots the base and soccer reproduction number

    Parameters
    ----------
    trace:
        arviz trace of model run
    model:
        corresponding model
    dl:
        dataloader
    """
    R_soccer = get_from_trace("R_t_soccer", trace)
    C_soccer = get_from_trace("C_soccer", trace)

    # Plot base and soccer Reproduction number
    cov19.plot._timeseries(
        x=pd.date_range(model.sim_begin, model.sim_end),
        y=R_soccer,
        what="model",
        ax=ax,
        color=colors["Repr"],
    )
    ax.axhline(0, color="tab:gray", ls="--", zorder=-5, lw=0.5)
    ax.set_ylabel("Additive\nreproduct. number")
    ax.set_ylim(-0.85, 3.5)

    return ax


def plot_alphaMean(ax, trace, model, dl, beta=False):

    if not beta:
        alpha = get_from_trace(f"alpha_mean", trace)
        R_soccer = np.exp(alpha) - 1
    else:
        try:
            alpha = get_from_trace(f"beta_mean", trace)
            R_soccer = np.exp(alpha) - 1
        except:
            return ax

    import seaborn.categorical

    seaborn.categorical._Old_Violin = seaborn.categorical._ViolinPlotter

    sns.violinplot(
        data=R_soccer,
        scale="width",
        inner="quartile",
        orient="v",
        ax=ax,
        color=colors["Repr"],
        linewidth=1,
        saturation=1,
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.collections[0].set_edgecolor(colors["Repr"])  # Set outline colors
    ax.set_xticklabels([])
    ax.set_ylabel("Mean effect of all \n played soccer games")
    ax.axhline(0, color="tab:gray", ls="--", zorder=-5)
    return ax


def plot_reproductionViolin(ax, trace, model, dl, plot_dates=True, color=None):
    """
    Violin plot for the additional R values for each game and country.

    """
    key = "alpha"

    α_mean = get_from_trace(f"{key}_mean", trace)
    σ_g = get_from_trace(f"sigma_{key}_g", trace)
    Δα_g_sparse = get_from_trace(f"Delta_{key}_g_sparse", trace)
    alpha = α_mean[:, None] + np.einsum("dg,d->dg", Δα_g_sparse, σ_g)

    R_soccer = np.exp(alpha) - 1

    nGames = alpha.shape[-1]

    # Get date of game and participants
    games = dl.timetable.loc[(dl.alpha_prior > 0)[0]]
    ticks = [(vals["date"] - model.sim_begin).days for i, vals in games.iterrows()]
    if plot_dates:
        ticks = np.array([vals["date"] for i, vals in games.iterrows()])
    else:
        ticks = np.array(ticks) - np.min(ticks)

    # Create pandas dataframe for easy plotting
    R_soccer = pd.DataFrame(R_soccer, columns=ticks)

    if not plot_dates:
        ticks = np.arange((dl.alpha_prior > 0).sum()) * 4

    p = np.percentile(R_soccer, [2.5, 50, 97.5], axis=0)
    ax.errorbar(
        x=ticks,
        y=p[1],
        yerr=[p[1] - p[0], p[2] - p[0]],
        # width=2,
        # ecolor="tab:gray",
        color=colors["Repr"] if color is None else color,
        ls="",
        marker="_",
        ms=4,
        # color = 'k'
        capsize=1.5,
        # error_kw= {"alpha":1,"lw":0.8,"ecolor":colors[1]}
    )
    ax.axhline(0, color="tab:gray", lw=0.5, alpha=0.5, ls="--", zorder=-5)

    R_t_soccer = get_from_trace("R_t_soccer", trace)

    ax.set_ylabel("Additive rep.\nnumber")

    return ax


def plot_relative_from_soccer(
    ax,
    trace,
    model,
    dl,
    ylim_relative=None,
    begin=None,
    end=None,
    verbose=True,
    violin=False,
):
    if begin is None:
        begin = datetime.datetime(2021, 6, 11)
    if end is None:
        end = datetime.datetime(2021, 7, 11)

    # Get params from trace and dataloader
    new_E_t = get_from_trace("new_E_t", trace)
    S_t = get_from_trace("S_t", trace)
    new_I_t = get_from_trace("new_I_t", trace)
    R_t_base = (
        get_from_trace("R_t_base", trace)
        + get_from_trace("R_t_add_noise_fact", trace)[..., 0]
    )
    C_base = get_from_trace("C_base", trace)
    C_soccer = get_from_trace("C_soccer", trace)
    R_t_soccer = get_from_trace("R_t_soccer", trace)
    pop = model.N_population
    i_begin = (begin - model.sim_begin).days
    i_end = (end - model.sim_begin).days + 1  # inclusiv last day

    """ Calculate cases in agegroup because of soccer and without soccer
    """
    # d is draws
    # t is time
    # i,j is gender
    R_t_ij_base = np.einsum("dt,dij->dtij", R_t_base, C_base)
    infections_base = S_t / pop * np.einsum("dti,dtij->dti", new_I_t, R_t_ij_base)

    R_t_ij_soccer = np.einsum("dt,dij->dtij", R_t_soccer, C_soccer)
    infections_soccer = S_t / pop * np.einsum("dti,dtij->dtj", new_I_t, R_t_ij_soccer)

    # Sum over the choosen range (i.e. month of uefa championship) male and femal
    num_infections_base = np.sum(infections_base[..., i_begin:i_end, :], axis=-2)
    num_infections_soccer = np.sum(infections_soccer[..., i_begin:i_end, :], axis=-2)

    # Create pandas dataframe for easy violin plot
    ratio_soccer = num_infections_soccer / (num_infections_base + num_infections_soccer)
    male = np.stack((ratio_soccer[:, 0], np.zeros(ratio_soccer[:, 0].shape)), axis=1)
    female = np.stack((ratio_soccer[:, 1], np.ones(ratio_soccer[:, 1].shape)), axis=1)

    percentage = pd.DataFrame(
        np.concatenate((male, female)), columns=["percentage_soccer", "gender"]
    )
    percentage["gender"] = pd.cut(
        percentage["gender"], bins=[-1, 0.5, 1], labels=["male", "female"]
    )
    percentage["percentage_soccer"] = percentage["percentage_soccer"] * 100
    if violin:
        percentage["percentage_soccer"] = percentage[
            "percentage_soccer"
        ] + np.random.normal(
            size=len(percentage["percentage_soccer"]), loc=0, scale=0.0001
        )
    percentage["dummy"] = 0

    if violin:
        g = sns.violinplot(
            data=percentage,
            y="percentage_soccer",
            x="dummy",
            hue="gender",
            scale="width",
            inner="quartile",
            orient="v",
            ax=ax,
            split=True,
            palette={"male": colors["male"], "female": colors["female"]},
            linewidth=1,
            saturation=1,
        )
        ax.collections[0].set_edgecolor(colors["male"])  # Set outline colors
        ax.collections[1].set_edgecolor(colors["female"])  # Set outline colors
    else:
        g = sns.stripplot(
            data=percentage,
            y="percentage_soccer",
            x="dummy",
            hue="gender",
            dodge=True,
            palette={"male": colors["male"], "female": colors["female"]},
            alpha=0.25,
            zorder=1,
            size=2,
        )
        sns.pointplot(
            y="percentage_soccer",
            x="dummy",
            hue="gender",
            data=percentage,
            dodge=0.8 - 0.8 / 3,
            join=False,
            palette={
                "male": lighten_color(colors["male"], 1.3),
                "female": lighten_color(colors["female"], 1.3),
            },
            markers="d",
            ci=95,
            scale=0.6,
        )
    ax.legend([], [], frameon=False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xticklabels([])

    if ylim_relative is not None:
        ax.set_ylim(ylim_relative)

    # Set y tick format
    if (
        1.0 < np.mean(percentage["percentage_soccer"])
        or np.mean(percentage["percentage_soccer"]) < -1.0
    ):
        fmt = "%.0f%%"  # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(xticks)

    # Set labels
    ax.set_ylabel("Percentage of soccer related\ninfections during the Championship")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.axhline(0, color="tab:gray", ls="--", zorder=-5)

    # Remove spines
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if verbose:
        print(f"CI [50,2.5,97.5] {dl.countries}:")
        print(f"\tmale {np.percentile(ratio_soccer[:,0], [50,2.5,97.5])}")
        print(f"\tfemale {np.percentile(ratio_soccer[:,1], [50,2.5,97.5])}")

    return ax


from .timeseries import *
from .distributions import distribution
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def single_extended(trace, model, dl, xlim=None):
    """
    Create an extended overview plot for a single model run. Adjust colors with rcParams
    """

    fig = plt.figure(figsize=(7, 1.7 * 4))
    axes_ts = []

    grid = fig.add_gridspec(
        5, 3, wspace=0.15, hspace=0.25, width_ratios=[1, 0.25, 0.25]
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

    # R_soccer + R_noise
    ax = fig.add_subplot(grid[3, 0])
    R_soccer(ax, trace, model, dl, add_noise=True)
    axes_ts.append(ax)

    """ distributions
    """
    # delay
    ax = fig.add_subplot(grid[0, 1])
    distribution(
        model,
        trace,
        "delay",
        nSamples_prior=5000,
        title="",
        dist_math="D",
        ax=ax,
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
        dist_math="h_{w}",  # What was
        ax=ax,
    )

    ax = fig.add_subplot(grid[3, 1])
    distribution(
        model,
        trace,
        "offset_modulation",
        nSamples_prior=5000,
        title="",
        dist_math="\chi_{w}",  # What was
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
        Patch(
            [0],
            [0],
            color=rcParams.color_posterior,
            lw=2.5,
        ),
    ]
    ax = fig.add_subplot(grid[3, 2])
    ax.legend(
        custom_lines,
        [
            "Data",
            "Model",
            "Prior",
            "Posterior",
        ],
        loc="center",
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
