#!/bin/bash

#SBATCH --job-name=n-s{seed}
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --mail-user=shikhark@mit.edu
#SBATCH --time=12:00:00

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec -n 2 --machinefile hostfile python3 run_openmc_capi.py 18 2d-beavrs
