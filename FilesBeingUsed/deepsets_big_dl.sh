#!/bin/bash

## SLURM JOB COMMANDS ###
#SBATCH --partition=cms-uhh   ## or allgpu / cms / cms-uhh / maxwell
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --job-name Deepsets_big_dl
#SBATCH --output ./deepsets_big_dl3.out   # terminal output
#SBATCH --error ./deepsets_big_dl3.err
#SBATCH --mail-type END
#SBATCH --mail-user stuart.nicholls@desy.de
#SBATCH --constraint="P100"


export PATH=/software/anaconda3/5.2/bin:$PATH
module load maxwell
module load cuda

python DeepSets_big_dl.py
