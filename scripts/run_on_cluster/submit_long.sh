#!/bin/bash [could also be /bin/tcsh]
#$ -S /bin/bash
#$ -N COVID19-soccer-long2
#$ -pe mvapich2-sam 32
#$ -cwd
#$ -o $HOME/repositories/covid19_soccer/scripts/run_on_cluster/log/output-soccer-long2
#$ -e $HOME/repositories/covid19_soccer/scripts/run_on_cluster/log/errors-soccer-long2
#$ -t 1-6

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/tmp/theano_jdehning/theano2_3/$SGE_TASK_ID"

# >>>  conda initialize >>>
source /data.nst/jdehning/anaconda3/bin/activate
conda activate pymc3_new
# >>>  conda initialize >>>

cd $HOME/repositories/covid19_soccer/scripts/run_on_cluster/

python3 -u ./cluster_runs_gender_long.py -i $SGE_TASK_ID


