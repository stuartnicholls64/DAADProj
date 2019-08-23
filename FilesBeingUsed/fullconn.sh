#!/bin/bash

## SLURM JOB COMMANDS ###
#SBATCH --partition=cms-uhh   ## or allgpu / cms / cms-uhh / maxwell
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --job-name fullconn
#SBATCH --output ./fullconn2.out   # terminal output
#SBATCH --error ./fullconn2.err
#SBATCH --mail-type END
#SBATCH --mail-user stuart.nicholls@desy.de
#SBATCH --constraint="P100"


export PATH=/software/anaconda3/5.2/bin:$PATH
module load maxwell
module load cuda

python FullConn.py
