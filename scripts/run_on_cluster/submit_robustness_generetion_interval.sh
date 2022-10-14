#!/bin/bash [could also be /bin/tcsh]
#$ -S /bin/bash
#$ -N COVID19-soccer-revision-robustness
#$ -pe mvapich2-zal 32
#$ -cwd
#$ -o $HOME/repositories/covid19_soccer/scripts/run_on_cluster/log/output-soccer-revision_robustness_generation_interval
#$ -e $HOME/repositories/covid19_soccer/scripts/run_on_cluster/log/errors-soccer-revision_robustness_generation_interval
#$ -t 1-24

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/tmp/theano_jdehning/theano2_1/$SGE_TASK_ID"

# >>>  conda initialize >>>
source /data.nst/jdehning/anaconda3/bin/activate
conda activate pymc3_new
# >>>  conda initialize >>>

cd $HOME/repositories/covid19_soccer/scripts/run_on_cluster/

python3 -u ./cluster_runs_gender_robustness_generation_interval.py -i $SGE_TASK_ID


