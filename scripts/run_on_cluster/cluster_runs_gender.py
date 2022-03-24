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
from multiprocessing import Pool

log = logging.getLogger("ClusterRunner")

parser = argparse.ArgumentParser(description="Run soccer script")
parser.add_argument(
    "-i", "--id", type=int, help="ID", required=True,
)

args = parser.parse_args()
args.id = args.id - 1
log.info(f"ID: {args.id}")


dir_traces = "/data.nst/jdehning/covid_uefa_traces11"

""" Create possible different combinations
"""


# Countries with gender data
# countries = ["Scotland", "Germany", "France", "England", "Spain", "Czechia", "Italy"]

countries = [
    "Scotland",
    "Germany",
    "France",
    "England",
    "Spain",
    "Czechia",
    "Italy",
    "Belgium",
    "Netherlands",
    "Portugal",
    "Slovakia",
    "Austria",
]
# countries = ["England", "Portugal", "France"]


# [tune,draw,treedepth]
sampling = [
    #    [200, 300, 10],
    #    [500, 1000, 12],
    #    [1000, 1500, 12],
    #    [1000, 1000, 12],
    #     [1500, 3000, 12]
    [2000, 4000, 12],
    # [4000, 8000, 12],
]

# True or false
# beta = [0, 1]
beta = [1]  # , 1]


# Games offset i.e. effect if soccer games would be x days later
# important offsets = [0, -35, -21, -14, -10, -7, -4, -2, 2, 4, 7, 10, 14, 21, 35]
# offset = [0, -35, -15, -10, -8, -6, -5, -4, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10, 15, 35]
# offset = [-35, -28, -10, -8, -6, -4, -2, -1, 35]
# offset = [0]
# offset = [0, -5, -4, -2, -1, 1, 2, 3, 4, 5]
# offset = [0, -3, -2, -1, 1, 2, 3, 4, 5]
offset = [0]
# offset = [0, -35, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 35]
# offset = [-35, -3, 3, 35]
# offset = [-21, -14, -10, -7, 7, 10, 14, 21]
# offset = [-21, -14, -10, -7, -4, -2, 2, 4, 7, 10, 14, 21]
# offset = [0, -35, -21, -14, -10, -7, -4, -2, 2, 4, 7, 10, 14, 21, 35]
# offset = [0, -14, -10, -7, -4, -2, 2, 4, 7, 10, 14]

# draw delay width i.e. true false
draw_delay = [1]

# Use weighted alpha prior
# weighted_alpha = [0, -1]
weighted_alpha = [0]

# prior_delay = [-1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
prior_delay = [-1]

# prior width of the mean latent period
sigma_incubation = [-1]

width_delay_prior = [0.1]

# median_width_delay = [0.5, 1.0, 2.0]
median_width_delay = [1.0]

# interval_cps = [10.0, 6.0, 20.0]
interval_cps = [10.0]

# f_fem_list = [0.2, 0.5]
f_fem_list = [0.2]

allow_uefa_cps_list = [True]

# len_model_list = ["normal", "short"]
# len_model_list = ["normal"]
len_model_list = ["normal"]


mapping = []

for draw_args in sampling:
    for b in beta:
        for country in countries:
            for delay in draw_delay:
                for off in offset:
                    for wa in weighted_alpha:
                        for pd in prior_delay:
                            for wdp in width_delay_prior:
                                for si in sigma_incubation:
                                    for mwd in median_width_delay:
                                        for inter in interval_cps:
                                            for f_fem in f_fem_list:
                                                for len_mod in len_model_list:
                                                    for (
                                                        allow_uefa_cps
                                                    ) in allow_uefa_cps_list:
                                                        default_vals = (
                                                            True if b == 0 else False
                                                        )
                                                        if country in [
                                                            "Belgium",
                                                            "Netherlands",
                                                            "Portugal",
                                                            "Slovakia",
                                                            "Austria",
                                                        ]:
                                                            if not default_vals:
                                                                continue
                                                            else:
                                                                default_vals = False
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
                                                        if not mwd == 1.0:
                                                            if not default_vals:
                                                                continue
                                                            else:
                                                                default_vals = False
                                                        if not inter == 10.0:
                                                            if not default_vals:
                                                                continue
                                                            else:
                                                                default_vals = False
                                                        if not f_fem == 0.2:
                                                            if not default_vals:
                                                                continue
                                                            else:
                                                                default_vals = False
                                                        if not len_mod == "normal":
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
                                                        ma.append(mwd)
                                                        ma.append(inter)
                                                        ma.append(f_fem)
                                                        ma.append(len_mod)
                                                        ma.append(allow_uefa_cps)
                                                        mapping.append(tuple(ma))


num_jobs_per_node = 3
mapping_clustered = []
ended = False
for i in range(len(mapping)):
    if not num_jobs_per_node * i >= len(mapping):
        mapping_clustered.append([])
    for j in range(num_jobs_per_node):
        i_mapping = num_jobs_per_node * i + j
        if i_mapping < len(mapping):
            mapping_clustered[-1].append(mapping[i_mapping])
        else:
            ended = True
            break
    if ended:
        break


def exec(args_list):
    """
    Executes python script
    """
    (
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
        median_width_delay,
        interval_cps,
        f_fem,
        len_mod,
        allow_uefa_cps,
    ) = args_list
    os.system(
        f"python run_model_gender.py "
        f"-b {beta} -c {country} "
        f"--dir {dir_traces} "
        f"--t {tune} --d {draws} --max_treedepth {max_treedepth} "
        f"--log ./log/ "
        f"--offset_data {offset} "
        f"--draw_delay {draw_delay} "
        f"--weighted_alpha {weighted_alpha} "
        f"--prior_delay {prior_delay} "
        f"--width_delay_prior {width_delay_prior} "
        f"--sigma_incubation {sigma_incubation} "
        f"--median_width_delay {median_width_delay} "
        f"--interval_cps {interval_cps} "
        f"--f_fem {f_fem} "
        f"--len {len_mod} "
        f"--uc {allow_uefa_cps} "
    )


if __name__ == "__main__":
    with Pool(num_jobs_per_node) as p:
        p.map(exec, mapping_clustered[args.id])
