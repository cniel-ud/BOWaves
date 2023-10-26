#!/bin/bash -l
# SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=resample_emotion_codebooks
# SBATCH --partition=standard --gres=gpu
# The below is maximum time for the job.
#SBATCH --time=7-00:00:00
#SBATCH --time-min=0-01:00:00
#SBATCH --mail-user='ajmeek@udel.edu'
# this could be --mail-type=END, FAIL, TIME_LIMIT_90. but I thought it was too many emails
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --export=NONE
#SBATCH --mem=64G
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"
# SBATCH --array=0-11

vpkg_devrequire intel-python/2022u1:python3
source activate /work/cniel/ajmeek/BOWaves/venv/
pip install seaborn

# Run bash / python script below

python ../tests/resample_emotion_to_cue.py