import numpy as np
import matplotlib.dates as mdates
from cairosvg import svg2png
import urllib
import os

from .rcParams import *
from ..effect_gender import _delta

 
def get_from_trace(var, trace, from_type="posterior"):
    """Reshapes and returns an numpy array from an arviz trace"""
    key = var

    if key in ["alpha", "beta"] and key not in getattr(trace, from_type):
        mean = get_from_trace(f"{key}_mean", trace)
        sparse = get_from_trace(f"Delta_{key}_g_sparse", trace)
        sigma = get_from_trace(f"sigma_{key}_g", trace)
        var = mean[:, None] + np.einsum("dg,d->dg", sparse, sigma)
    else:
        var = np.array(getattr(trace, from_type)[var])
        if from_type == "predictions":
            var = var.reshape((var.shape[0] * var.shape[1],) + var.shape[2:])
        var = var.reshape((var.shape[0] * var.shape[1],) + var.shape[2:])

    # Remove nans (normally there are 0 nans but can happen if you use where operations)
    var = var[~np.isnan(var).any(tuple(range(1, var.ndim))), ...]
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
    try:
        t_g = [(game - model.sim_begin).days for game in dl.date_of_games]
        d = _delta(np.subtract.outer(t, t_g)).eval()

        return np.dot(d, eff)
    except:
        try:
            t_g = [
                (game - model.sim_begin).days
                for game in dl.timetable[~dl.timetable["id"].str.contains("b")]["date"]
            ]
            d = _delta(np.subtract.outer(t, t_g)).eval()

            return np.dot(d, eff)
        except:
            t_g = [
                (game - model.sim_begin).days
                for game in dl.timetable[~dl.timetable["id"].str.contains("a|b",
                                                                          regex=True)]["date"]
            ]
            d = _delta(np.subtract.outer(t, t_g)).eval()

            return np.dot(d, eff)


def get_flag(iso2, path="./figures/iso2/"):
    if iso2 == "DE2":
        iso2 = "DE"
    if iso2.lower() in ["eng", "sct"]:
        iso2 = f"gb-{iso2}"
    try:
        # Check if png exists:
        if not os.path.exists(f"{path}{iso2}.png"):
            os.makedirs(path, exist_ok=True)
            png = svg2png(
                url=f"https://hatscripts.github.io/circle-flags/flags/{iso2}.svg",
            )
            with open(f"{path}{iso2}.png", "wb") as bin_file:
                bin_file.write(png)
        return f"{path}{iso2}.png"
    except urllib.error.HTTPError:
        return f"{path}football.png"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def k_formatter(x, pos):
    # converts tick to k notation\n",
    if x >= 1e3 or x <= -1e3:
        return "{:.0f}k".format(x / 1e3)
    else:
        return "{:.0f}".format(x)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 5
            else:
                ndivs = 4
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a " "%s type." % type(self)
        )
