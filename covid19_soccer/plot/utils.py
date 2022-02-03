import numpy as np
import matplotlib.dates as mdates
from cairosvg import svg2png
import urllib

from .rcParams import *
from ..effect_gender import _delta


def get_from_trace(var, trace):
    """Reshapes and returns an numpy array from an arviz trace"""
    key = var

    if key in ["alpha", "beta"] and key not in trace.posterior:
        mean = get_from_trace(f"{key}_mean", trace)
        sparse = get_from_trace(f"Delta_{key}_g_sparse", trace)
        sigma = get_from_trace(f"sigma_{key}_g", trace)
        var = mean[:, None] + np.einsum("dg,d->dg", sparse, sigma)
    else:
        var = np.array(trace.posterior[var])
        var = var.reshape((var.shape[0] * var.shape[1],) + var.shape[2:])

    return var


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def format_date_axis(ax):
    """
    Formats axis with dates
    """
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4, byweekday=mdates.SU))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1, byweekday=mdates.SU))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(rcParams.date_format))


def _apply_delta(eff, model, dl):
    t = np.arange(model.sim_len)
    t_g = [(game - model.sim_begin).days for game in dl.date_of_games]
    d = _delta(np.subtract.outer(t, t_g)).eval()

    return np.dot(d, eff)


def get_flag(iso2):
    if iso2 == "DE2":
        iso2 = "DE"
    try:
        svg2png(
            url=f"https://hatscripts.github.io/circle-flags/flags/{iso2}.svg",
            write_to=f"./figures/{iso2}.png",
        )
        return f"./figures/{iso2}.png"
    except urllib.error.HTTPError:
        return f"./figures/united_nations.png"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
