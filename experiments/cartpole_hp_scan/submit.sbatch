#!/bin/bash
#SBATCH --job-name=CPScan
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --output=/home/btimar/code/dag-gen/experiments/cartpole_hp_scan/slurm.out
#SBATCH --time=5:00:00

pwd
date
module load python
python run.py 40
date
