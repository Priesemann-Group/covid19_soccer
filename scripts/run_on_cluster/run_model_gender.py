# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-07-05 12:19:04
# @Last Modified: 2021-10-07 12:14:32
# ------------------------------------------------------------------------------ #
# Run the soccer model with different input parameters
#
#
#

import logging

log = logging.getLogger(__name__)
import argparse
import datetime
import sys
import pickle
import os


sys.path.append("../../covid19_inference/")
import covid19_inference as cov19

sys.path.append("../../")
import covid19_soccer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Run model")

parser.add_argument(
    "-b",
    "--beta",
    type=str2bool,
    help="Use beta compartment of our soccer model.",
    default=False,
)

parser.add_argument(
    "-c",
    "--country",
    type=str,
    help="Country data to use for our model",
    default="Scotland",
)

parser.add_argument(
    "--offset_data", type=int, help="Offset of the data in days", default=0
)

parser.add_argument(
    "--prior_delay", type=int, help="prior_delay", default=5,
)


parser.add_argument(
    "--median_width_delay",
    type=float,
    help="prior width of the testing delay",
    default=1.0,
)

parser.add_argument(
    "--interval_cps",
    type=float,
    help="number of days between change points",
    default=10.0,
)

parser.add_argument(
    "--f_fem",
    type=str,
    help="factor less participation of women at soccer reltated gatherings",
    default="0.33",
)


parser.add_argument(
    "--len", type=str, help="duration of the model", default="normal",
)

parser.add_argument(
    "--abs_sine",
    type=str2bool,
    help="Whether to use the absolute sine weekly modulation",
    default=False,
)

parser.add_argument(
    "--f_robust",
    type=float,
    help="A parameter to explore the robustness of the model",
    default=1.0,
)


parser.add_argument(
    "--t", type=int, help="How many tuning steps?", default=1000,
)

parser.add_argument(
    "--d", type=int, help="How many draws?", default=1000,
)

parser.add_argument(
    "--max_treedepth", type=int, help="Which maximal tree depth?", default=10,
)

parser.add_argument(
    "--log", type=str, help="Directory for saving log(s)", default="./log"
)
parser.add_argument(
    "--dir", type=str, help="Directory for saving traces", default="./pickled"
)


""" Basic logger setup
We want each job to print to a different file, for easier debuging once
run on a cluster.
"""


def dict_2_string(dictonary):
    """
    Creates a string from a dictornary
    """

    f_str = ""
    for arg in dictonary:
        if arg in ["log", "dir"]:
            continue
        f_str += f"-{arg}={args.__dict__[arg]}"
    return f_str


def log_to_file(log_dir, fstring):
    """
    Writes all logs to a file
    
    Parameters
    ----------
    log_dir: string
        Directory for logging
    fstring: string
        Filename for the logging
    """

    # Write all logs to file
    fh = logging.FileHandler(log_dir + "/" + fstring + ".log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s::%(levelname)-4s [%(name)s] %(message)s")
    )
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    cov19.log.addHandler(fh)

    # Redirect all errors to file
    # sys.stderr = open(args.log + "/" + f_str + ".stderr", "w")
    # sys.stdout = open(args.log + "/" + f_str + ".stdout", "w")

    # Redirect pymc3 output
    logPymc3 = logging.getLogger("pymc3")
    logPymc3.addHandler(fh)


# Print some basic infos

if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()
    input_args_dict = dict(**args.__dict__)

    # Setup logging to file
    log_to_file(args.log, dict_2_string(input_args_dict))

    # Log starting time and simulation parameters
    log.info(f"Script started: {datetime.datetime.now()}")
    log.info(f"Args: {args.__dict__}")

    """ Create our model
    We also create a default dataloader here, and print the countries which are used!
    """
    cov19.data_retrieval.set_data_dir(fname="./data_covid19_inference")

    if args.len == "normal":
        data_begin = datetime.datetime(2021, 6, 1)
        data_end = datetime.datetime(2021, 8, 15)
        sim_begin = data_begin - datetime.timedelta(days=16)
    elif args.len == "short":
        data_begin = datetime.datetime(2021, 6, 4)
        data_end = datetime.datetime(2021, 7, 18)
        sim_begin = data_begin - datetime.timedelta(days=16)
    elif args.len == "long":
        data_begin = datetime.datetime(2021, 5, 1)
        data_end = datetime.datetime(2021, 8, 15)
        sim_begin = data_begin - datetime.timedelta(days=16)
    else:
        raise RuntimeError("argument value not known")

    dl = covid19_soccer.dataloader.Dataloader_gender(
        data_folder="../../data/",
        countries=[args.country],
        offset_data=args.offset_data,
        data_begin=data_begin,
        data_end=data_end,
        sim_begin=sim_begin,
    )
    log.info(f"Data loaded for {dl.countries}")

    i = 0
    while i < 50:
        # seems to be some multiprocessing error with c-backend, try again until success
        try:
            model = covid19_soccer.models.create_model_gender(
                dataloader=dl,
                beta=args.beta,
                prior_delay=args.prior_delay,
                median_width_delay=args.median_width_delay,
                interval_cps=args.interval_cps,
                f_female=args.f_fem,
                use_abs_sine_weekly_modulation=args.abs_sine,
                f_robust=args.f_robust,
            )
        except AssertionError as error:
            if i < 10:
                i += 1
                continue
            else:
                raise error
        i = 1000

    """ MCMC sampling
    """
    save_name = f"run{dict_2_string(input_args_dict)}"
    save_file = os.path.join(args.dir, save_name)
    callback = cov19.sampling.Callback(path=args.dir, name="backup" + save_name, n=500)

    multitrace, trace = cov19.robust_sample(
        model,
        tune=args.t,
        draws=args.d,
        burnin_draws=args.d // 6,
        burnin_draws_2nd=args.d // 3,
        burnin_chains=30,
        burnin_chains_2nd=15,
        final_chains=8,
        sample_kwargs={"cores": 11},
        max_treedepth=args.max_treedepth,
        target_accept=0.95,
        callback=callback,
    )

    # Save trace/model so we dont have to rerun sampling every time we change some plotting routines
    with open(save_file + ".pkl", "wb") as f:
        pickle.dump((model, trace), f)

    log.info(f"Script finished: {datetime.datetime.now()}")
