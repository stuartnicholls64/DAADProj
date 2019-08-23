#!/bin/bash

## SLURM JOB COMMANDS ###
#SBATCH --partition=allgpu   ## or allgpu / cms / cms-uhh / maxwell
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --job-name Deepsets_big_radi
#SBATCH --output ./deepsets_big6.out   # terminal output
#SBATCH --error ./deepsets_big6.err
#SBATCH --mail-type END
#SBATCH --mail-user stuart.nicholls@desy.de
#SBATCH --constraint="V100"


export PATH=/software/anaconda3/5.2/bin:$PATH
module load maxwell
module load cuda

python DeepSets_big.py
