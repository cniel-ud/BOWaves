#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --job-name=eeg_preprocessing
#SBATCH --partition=standard --gres=gpu
# The below is maximum time for the job.
#SBATCH --time=0-10:00:00
#SBATCH --time-min=0-01:00:00
#SBATCH --mail-user='ajmeek@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --export=NONE
#SBATCH --mem=64G
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"

vpkg_devrequire intel-python/2022u1:python3
source activate /work/cniel/ajmeek/BOWaves/venv/

# Run bash / python script below

python ../tests/frolich_replication_v2.py
