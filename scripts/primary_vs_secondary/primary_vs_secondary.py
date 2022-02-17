# This scripts generates a traces for primary and no soccer effect

import os
import sys
from multiprocessing import Process
import pickle
from tqdm.auto import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../notebooks/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils import getPrimary, getNoSoccer, ThreadWithResult
import covid19_soccer

# config
countries = [
    "England",
    "Scotland",
    "Germany",
    "France",
    "Spain",
    "Slovakia",
    "Portugal",
    "Netherlands",
    "Italy",
    "Czechia",
    "Belgium",
    "Austria",
]


def load(fstr):
    with open(fstr, "rb") as f:
        return pickle.load(f)


def computeTraces(country, save_fpath="primary_vs_secondary.pkl"):
    """Computes the traces for noSoccer effect and primary effect

    Parameters
    ----------
    fpath: str
        Path to the pickle file containing the trace and mode

    """

    # Load trace
    model = None
    fstr = lambda tune, draws, max_treedepth: (
        f"/data.nst/smohr/covid19_soccer_publication_traces/"
        + f"-beta=False"
        + f"-country={country}"
        + f"-offset_data=0"
        + f"-prior_delay=-1"
        + f"-width_delay_prior=0.1"
        + f"-sigma_incubation=-1.0"
        + f"-median_width_delay=1.0"
        + f"-interval_cps=10.0"
        + f"-f_fem=0.2"
        + f"-uc=True"
        + f"-len=normal"
        + f"-t={tune}"
        + f"-d={draws}"
        + f"-max_treedepth={max_treedepth}.pkl"
    )
    if os.path.exists(fstr(4000, 8000, 12)):
        try:
            model, initial_trace = load(fstr(4000, 8000, 12))
            print(f"Use 8000 sample runs for {country}")
        except:
            pass
    if model is None and os.path.exists(fstr(2000, 4000, 12)):
        try:
            model, initial_trace = load(fstr(2000, 4000, 12))
            print(f"Use 4000 sample runs for {country}")
        except:
            pass
    if model is None and os.path.exists(fstr(1000, 1500, 12)):
        try:
            model, initial_trace = load(fstr(1000, 1500, 12))
            print(f"Use 1500 sample runs for {country}")
        except:
            pass
    if model is None:
        print(fstr(tune, draws, max_treedepth), " not found")
        return
    # Remove chains with likelihood larger than -200, should only be the case for 2 chains in France
    mask = np.mean(initial_trace.sample_stats.lp, axis=1) > -200
    initial_trace.posterior = initial_trace.posterior.sel(chain=~mask.to_numpy())

    initial_trace.posterior = initial_trace.posterior.assign_coords(
        {"chain": list(range((~mask.to_numpy()).sum()))}
    )

    dl = covid19_soccer.dataloader.Dataloader_gender(countries=[country])
    dl = covid19_soccer.dataloader.Dataloader_gender(countries=[country])
    print(f"{country} loaded")
    noSoccer = getNoSoccer(initial_trace, model)
    primary = getPrimary(initial_trace, noSoccer, model, dl)

    with open(save_fpath, "wb") as f:
        pickle.dump((noSoccer, primary), f)


if __name__ == "__main__":

    processes = []
    for country in countries:
        save_fstr = f"/data.nst/smohr/covid19_soccer_publication_traces/primary_vs_secondary/{country}.pkl"
        t = Process(target=computeTraces, args=(country, save_fstr))
        processes.append(t)

    # Start processes
    for t in processes:
        t.start()

    # Join processes
    for t in processes:
        t.join()
