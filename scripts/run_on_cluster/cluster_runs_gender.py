# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 14:52:21
# @Last Modified: 2021-10-07 12:19:21
# ------------------------------------------------------------------------------ #
# This script should be called be the submit.sh script, it takes the ids and
# maps them to our model combinations

import argparse
import logging
import os
import itertools

log = logging.getLogger("ClusterRunner")

parser = argparse.ArgumentParser(description="Run soccer script")
parser.add_argument(
    "-i", "--id", type=int, help="ID", required=True,
)

args = parser.parse_args()
args.id = args.id - 1
log.info(f"ID: {args.id}")


dir_traces = "/data.nst/jdehning/covid_uefa_traces"

""" Create possible different combinations
"""


# Countries with gender data
countries = ["Scotland", "Germany", "France", "England"]

# [tune,draw,treedepth]
sampling = [
    [200, 300, 10],
    # [500, 1000, 12],
    [1000, 1500, 12],
]

# True or false
beta = [0, 1]

# Games offset i.e. effect if soccer games would be x days later
offset = [0, -28, -10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10, 28]

# draw delay width i.e. true false
draw_delay = [1]

# Use weighted alpha prior
weighted_alpha = [0, -1]

prior_delay = [-1, 2, 4, 5, 6, 7, 8, 10, 12]

# prior width of the mean latent period
sigma_incubation = [-1]

width_delay_prior = [0.1]

mapping = []


for b in beta:
    for country in countries:
        for draw_args in sampling:
            for delay in draw_delay:
                for off in offset:
                    for wa in weighted_alpha:
                        for pd in prior_delay:
                            for wdp in width_delay_prior:
                                for si in sigma_incubation:
                                    default_vals = True if b == 0 else False
                                    if not off == 0:
                                        if not default_vals:
                                            continue
                                        else:
                                            default_vals = False
                                    if not wa == 0:
                                        if not default_vals:
                                            continue
                                        else:
                                            default_vals = False
                                    if not pd == -1:
                                        if not default_vals:
                                            continue
                                        else:
                                            default_vals = False

                                    ma = []
                                    ma.append(b)
                                    ma.append(country)
                                    ma += draw_args
                                    ma.append(off)
                                    ma.append(delay)
                                    ma.append(wa)
                                    ma.append(pd)
                                    ma.append(wdp)
                                    ma.append(si)
                                    mapping.append(tuple(ma))


def exec(
    beta,
    country,
    tune,
    draws,
    max_treedepth,
    offset,
    draw_delay,
    weighted_alpha,
    prior_delay,
    width_delay_prior,
    sigma_incubation,
):
    """
    Executes python script
    """
    os.system(
        f"python run_model_gender.py "
        f"-b {beta} -c {country} "
        f"--dir {dir_traces} "
        f"--tune {tune} --draws {draws} --max_treedepth {max_treedepth} "
        f"--log ./log/ "
        f"--offset_games {offset} "
        f"--draw_delay {draw_delay} "
        f"--weighted_alpha {weighted_alpha} "
        f"--prior_delay {prior_delay} "
        f"--width_delay_prior {width_delay_prior} "
        f"--sigma_incubation {sigma_incubation} "
    )


exec(*mapping[args.id])
