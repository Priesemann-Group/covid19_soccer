# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-07-05 12:19:04
# @Last Modified: 2021-07-05 13:37:55
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


sys.path.append("../../covid19_inference_repo/")
import covid19_inference as cov19

sys.path.append("../../")
import covid19_uefa


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

args = parser.parse_args()

""" Basic logger setup
We want each job to print to a different file, for easier debuging once
run on a cluster.
"""

f_str = "UEFA"
for arg in args.__dict__:
    if arg in ["log", "dir"]:
        continue
    f_str += f"-{arg}={args.__dict__[arg]}"

# Write all logs to file
fh = logging.FileHandler(args.log + "/" + f_str + ".log")
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
log.info(f"Script started: {datetime.datetime.now()}")
log.info(f"Args: {args.__dict__}")


""" Create our model
We also create a default dataloader here, and print the countries which are used!
"""
cov19.data_retrieval.set_data_dir(fname="./data_covid19_inference")

dl = covid19_uefa.dataloader.Dataloader_gender(
    data_folder="../../data/", countries=[args.country]
)
log.info(f"Data loaded for {dl.countries}")

model = covid19_uefa.models.create_model_gender(
    dataloader=dl, beta=args.beta, use_gamma=True
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
)


# Save trace/model so we dont have to rerun sampling every time we change some plotting routines
with open(os.path.join(args.dir, f"{f_str}.pickled"), "wb") as f:
    pickle.dump((model, trace), f)
