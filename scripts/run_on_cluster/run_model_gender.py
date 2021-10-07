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
    "--offset_games", type=int, help="Offset of the soccer games in days", default=0
)
parser.add_argument(
    "--draw_delay",
    type=str2bool,
    help="Use distribution to draw width of delay",
    default=False,
)
parser.add_argument(
    "--weighted_alpha_prior", type=int, help="Use weighted alpha prior", default=0,
)

parser.add_argument(
    "--prior_delay", type=int, help="prior_delay", default=5,
)

parser.add_argument(
    "--width_delay_prior", type=float, help="width of the delay prior", default=0.2,
)

parser.add_argument(
    "--sigma_incubation",
    type=float,
    help="prior width of the mean latent period",
    default=1.0,
)

parser.add_argument(
    "--tune", type=int, help="How many tuning steps?", default=1000,
)

parser.add_argument(
    "--draws", type=int, help="How many draws?", default=1000,
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

    f_str = "UEFA"
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
    input_args_dict = args.__dict__

    # Setup logging to file
    log_to_file(args.log, dict_2_string(input_args_dict))

    # Log starting time and simulation parameters
    log.info(f"Script started: {datetime.datetime.now()}")
    log.info(f"Args: {args.__dict__}")

    """ Create our model
    We also create a default dataloader here, and print the countries which are used!
    """
    cov19.data_retrieval.set_data_dir(fname="./data_covid19_inference")

    dl = covid19_soccer.dataloader.Dataloader_gender(
        data_folder="../../data/",
        countries=[args.country],
        offset_games=args.offset_games,
    )
    log.info(f"Data loaded for {dl.countries}")

    model = covid19_soccer.models.create_model_gender(
        dataloader=dl,
        beta=args.beta,
        use_gamma=True,
        draw_width_delay=args.draw_delay,
        use_weighted_alpha_prior=args.weighted_alpha_prior,
        prior_delay=args.prior_delay,
        width_delay_prior=args.width_delay_prior,
        sigma_incubation=args.sigma_incubation,
    )

    """ MCMC sampling
    """

    multitrace, trace, multitrace_tuning, trace_tuning = cov19.robust_sample(
        model,
        tune=args.tune,
        draws=args.draws,
        tuning_chains=30,
        final_chains=8,
        cores=32,
        return_tuning=True,
        max_treedepth=args.max_treedepth,
        target_accept=0.95,
    )

    # Save trace/model so we dont have to rerun sampling every time we change some plotting routines
    with open(
        os.path.join(args.dir, f"{dict_2_string(input_args_dict)}.pickled"), "wb"
    ) as f:
        pickle.dump((model, trace), f)

    log.info(f"Script finished: {datetime.datetime.now()}")
