#!/bin/bash
#SBATCH --job-name=CPScan
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-core=1
#SBATCH --output=/home/btimar/code/dag-gen/experiments/cartpole_hp_scan/slurm.out
#SBATCH --time=10:00:00

pwd
date
echo "Now starting hp scans... (DUMMY RUN)"
echo "Job id: ${SLURM_JOB_ID}"
echo "Number of tasks: ${SLURM_NTASKS}"
module load python
python run.py 40
echo "All done!"
date
