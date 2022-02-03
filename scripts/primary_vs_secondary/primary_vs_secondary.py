# This scripts generates a traces for primary and no soccer effect

import os
import sys
from multiprocessing import Process
import pickle

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


def computeTraces(fpath, country, save_fpath="primary_vs_secondary.pkl"):
    """Computes the traces for noSoccer effect and primary effect

    Parameters
    ----------
    fpath: str
        Path to the pickle file containing the trace and mode

    """
    model, initial_trace = load(fpath)
    dl = covid19_soccer.dataloader.Dataloader_gender(countries=[country])
    print(f"{country} loaded")
    noSoccer = getNoSoccer(initial_trace, model)
    primary = getPrimary(initial_trace, noSoccer, model, dl)

    with open(save_fpath, "wb") as f:
        pickle.dump((noSoccer, primary), f)


if __name__ == "__main__":

    processes = []
    for country in countries:
        # Create thread
        fstr = (
            f"/data.nst/jdehning/covid_uefa_traces11/UEFA"
            + f"-beta=False"
            + f"-country={country}"
            + f"-offset_data=0"
            + f"-draw_delay=True"
            + f"-weighted_alpha_prior=0"
            + f"-prior_delay=-1"
            + f"-width_delay_prior=0.1"
            + f"-sigma_incubation=-1.0"
            + f"-median_width_delay=1.0"
            + f"-interval_cps=10.0"
            + f"-f_fem=0.2"
            + f"-len=normal"
            + f"-tune={1000}"
            + f"-draws={1500}"
            + f"-max_treedepth={12}.pickled"
        )

        save_fstr = (
            f"/data.nst/smohr/covid_uefa_traces/primary_vs_secondary/{country}.pkl"
        )

        t = Process(target=computeTraces, args=(fstr, country, save_fstr))
        processes.append(t)

    # Start processes
    for t in processes:
        t.start()

    # Join processes
    for t in processes:
        t.join()
