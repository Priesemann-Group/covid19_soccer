import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import logging
import datetime

from .utils import get_from_trace, lighten_color, format_date_axis
from .rcParams import *


log = logging.getLogger(__name__)


def game_effects(
    ax, trace, model, dl, color=None, type="violin", key="alpha", xlim=None
):
    """
    Plots the additional effect for each game onto a timeaxis, it is possible to
    plot as bar and violins.

    Parameters
    ----------
    type : str
            Can be "violin" or "boxplot"
    key : str
            Can be "alpha" or "beta"
    """
    eff = get_from_trace(f"{key}_R", trace)

    # Extract games
    selector = eff.sum(axis=0) != 0
    eff = eff[:, selector]
    dates = pd.to_datetime(dl.date_of_games[selector].values)
    df = pd.DataFrame(data=eff.T, index=dates)

    # Positions for violin plots
    if xlim is None:
        xlim = (
            dates[0] - datetime.timedelta(days=1),
            dates[-1] + datetime.timedelta(days=1),
        )
    pos = (dates - xlim[0]).days

    # Set color of plot
    color = rcParams.color_model if color is None else color

    # Calc quantiles
    quartile1, medians, quartile3 = np.percentile(df, [2.5, 50, 97.5], axis=1)

    # Plot
    if type == "violin":
        violin_parts = ax.violinplot(
            dataset=df.T,
            positions=mpl.dates.date2num(dates),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            # quantiles=[0.025,0.975],
            points=100,
            widths=1.5,
        )
        # Color violinstatistics
        for partname in violin_parts:
            if partname == "bodies":
                continue
            vp = violin_parts[partname]
            vp.set_edgecolor(color)
            vp.set_linewidth(1)

        for pc in violin_parts["bodies"]:
            pc.set_facecolor(lighten_color(color, 0.8))
            pc.set_edgecolor(lighten_color(color, 0.8))

        ax.scatter(dates, medians, marker="x", color=color, s=15, zorder=3, lw=1)
        ax.vlines(dates, quartile1, quartile3, color=color, linestyle="-", lw=1)
        ax.scatter(dates, quartile1, color=color, marker="_", s=10, zorder=3, lw=1)
        ax.scatter(dates, quartile3, color=color, marker="_", s=10, zorder=3, lw=1)

    elif type == "boxplot":
        box_parts = ax.boxplot(
            df.T,
            positions=mpl.dates.date2num(dates),
            sym="",
            vert=True,  # vertical box alignment
            patch_artist=True,  # fill with color
            widths=1.5,
            conf_intervals=[[0.025, 0.975]] * len(dates),
        )

        for box in box_parts["boxes"]:
            box.set_facecolor(lighten_color(color, 0.2))
            box.set_edgecolor(color)
        for parts in ["medians", "whiskers", "caps"]:
            for bp in box_parts[parts]:
                bp.set_color(color)
    else:
        log.error("Type not possible!")

    ax.axhline(0, color="tab:gray", ls="--", lw=1, zorder=-100)

    """ Markup
    """
    ax.set_ylabel("Game effects")

    # Format x axis
    ax.set_xlim(xlim)
    format_date_axis(ax)

    return ax


def soccer_related_cases(
    ax,
    trace,
    model,
    dl,
    begin=None,
    end=None,
    colors=None,
    type="violin",
    xlim=None,
    verbose=True,
):
    """
    Plots the soccer related casenumbers by male and female.

    Parameters
    ----------
    type : str
            Can be "violin" or "scatter"
    begin : datetime
        Begin date for relative cases calculation
    end : datetime
        End date for relative cases calculation
    colors : list, len = 2
        Colors of male and female

    """
    if begin is None:
        begin = datetime.datetime(2021, 6, 11)
    if end is None:
        end = datetime.datetime(2021, 7, 11)

    # Get params from trace and dataloader
    new_E_t = get_from_trace("new_E_t", trace)
    S_t = get_from_trace("S_t", trace)
    new_I_t = get_from_trace("new_I_t", trace)
    R_t_base = get_from_trace("R_t_base", trace)
    C_base = get_from_trace("C_base", trace)
    C_soccer = get_from_trace("C_soccer", trace)
    R_t_soccer = get_from_trace("R_t_add_fact", trace)

    pop = model.N_population
    i_begin = (begin - model.sim_begin).days
    i_end = (end - model.sim_begin).days + 1  # inclusiv last day

    # Calculate cases per gender contact
    # d is draws
    # t is time
    # i,j is gender
    R_t_ij_base = np.einsum("dt,dij->dtij", R_t_base, C_base)
    infections_base = S_t / pop * np.einsum("dti,dtij->dti", new_I_t, R_t_ij_base)
    R_t_ij_soccer = np.einsum("dt,dij->dtij", R_t_soccer, C_soccer)
    infections_soccer = S_t / pop * np.einsum("dti,dtij->dtj", new_I_t, R_t_ij_soccer)

    # Sum over the choosen range (i.e. month of uefa championship)
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
    percentage["pos"] = 0
    # Colors
    color_male = rcParams.color_male if colors is None else colors[0]
    color_female = rcParams.color_female if colors is None else colors[1]

    if type == "violin":
        g = sns.violinplot(
            data=percentage,
            y="percentage_soccer",
            x="pos",
            hue="gender",
            scale="count",
            inner="quartile",
            orient="v",
            ax=ax,
            split=True,
            palette={"male": color_male, "female": color_female},
            linewidth=1,
            saturation=1,
        )
        ax.collections[0].set_edgecolor(color_male)  # Set outline colors
        ax.collections[1].set_edgecolor(color_female)  # Set outline colors
    elif type == "scatter":
        g = sns.stripplot(
            data=percentage,
            y="percentage_soccer",
            x="pos",
            hue="gender",
            dodge=True,
            palette={"male": color_male, "female": color_female},
            alpha=0.25,
            zorder=1,
            size=2,
        )
        sns.pointplot(
            y="percentage_soccer",
            x="pos",
            hue="gender",
            data=percentage,
            dodge=0.8 - 0.8 / 3,
            join=False,
            palette={
                "male": lighten_color(color_male, 1.3),
                "female": lighten_color(color_female, 1.3),
            },
            markers="d",
            ci=95,
            scale=0.6,
        )
    else:
        log.error("Type not possible!")

    ax.axhline(0, color="tab:gray", ls="--", zorder=-5, lw=1)

    """ Markup
    """
    ax.set_ylabel("Percentage of soccer related\ninfections during the Championship")
    ax.set_xlabel("")

    # Remove legend
    ax.legend([], [], frameon=False)

    # Set axis labels to the right
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    # Remove x tick
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Set y tick formats
    fmt = "%.0f%%"  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(xticks)

    # Remove spines
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if verbose:
        log.info(f"CI [50,2.5,97.5] {dl.countries}:")
        log.info(f"\tmale {np.percentile(ratio_soccer[:,0], [50,2.5,97.5])}")
        log.info(f"\tfemale {np.percentile(ratio_soccer[:,1], [50,2.5,97.5])}")

    return ax
