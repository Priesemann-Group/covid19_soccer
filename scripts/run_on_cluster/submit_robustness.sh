#!/bin/bash [could also be /bin/tcsh]
#$ -S /bin/bash
#$ -N COVID19-soccer-robustness2
#$ -pe mvapich2-sam 32
#$ -cwd
#$ -o $HOME/repositories/covid19_soccer/scripts/run_on_cluster/log/output-soccer-robustness2
#$ -e $HOME/repositories/covid19_soccer/scripts/run_on_cluster/log/errors-soccer-robustness2
#$ -t 1-31

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/tmp/theano_jdehning/theano2_2/$SGE_TASK_ID"

# >>>  conda initialize >>>
source /data.nst/jdehning/anaconda3/bin/activate
conda activate pymc3_new
# >>>  conda initialize >>>

cd $HOME/repositories/covid19_soccer/scripts/run_on_cluster/

python3 -u ./cluster_runs_gender_robustness.py -i $SGE_TASK_ID


