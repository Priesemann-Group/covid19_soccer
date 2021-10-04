#!/bin/bash [could also be /bin/tcsh]
#$ -S /bin/bash
#$ -N COVID19-soccer
#$ -pe mvapich2-sam 32
#$ -cwd
#$ -o $HOME/Repositories/covid19_uefaeuro2020/scripts/run_on_cluster/log/output-soccer
#$ -o $HOME/Repositories/covid19_uefaeuro2020/scripts/run_on_cluster/log/errors-soccer
#$ -t 1:20:1

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/tmp/theano/"

# >>>  conda initialize >>>
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate uefa
# >>>  conda initialize >>>

cd $HOME/Repositories/covid19_uefaeuro2020/scripts/run_on_cluster/

python3 -u ./cluster_runs.py -i $SGE_TASK_ID
