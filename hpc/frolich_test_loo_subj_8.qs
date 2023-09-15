#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --job-name=frolich_loo_subj_8
#SBATCH --partition=standard --gres=gpu
# The below is maximum time for the job.
#SBATCH --time=0-48:00:00
#SBATCH --time-min=0-01:00:00
#SBATCH --mail-user='ajmeek@udel.edu'
# this could be --mail-type=END, FAIL, TIME_LIMIT_90. but I thought it was too many emails
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --export=NONE
#SBATCH --mem=64G
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"

vpkg_devrequire intel-python/2022u1:python3
source activate /work/cniel/ajmeek/BOWaves/venv/
pip install seaborn

# Run bash / python script below

python ../tests/frolich_replication_v2.py --loo_subj=8
