# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 14:52:21
# @Last Modified: 2021-07-05 15:35:37
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


""" Get possible different combinations
The first 5 parameters are of boolean type, the last one is a string with 3 different combinations.
"""

countries = ["Scotland", "Germany", "France", "England"]


mapping = []

first = list(itertools.combinations_with_replacement([1, 0], 1))
for i in first:
    for country in countries:
        for draw_args in [[500, 700, 10], [1000, 1500, 12]]:
            ma = []
            ma.append(i[0])
            ma.append(country)
            ma += draw_args
            mapping.append(tuple(ma))


def exec(
    beta, country, tune, draws, max_treedepth,
):
    """
    Executes python script
    """
    os.system(
        f"python run_model_gender.py "
        f"-b {beta} -c {country} "
        f"--tune {tune} --draws {draws} --max_treedepth {max_treedepth} "
        f"--log ./log/"
    )


exec(*mapping[args.id])
