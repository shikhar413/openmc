#!/bin/bash

#SBATCH --job-name=test3-3d-beavrs
#SBATCH --partition=sched_mit_nse
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=32
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=12:00:00

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq | while read line; do echo ${line}":2"; done > hostfile
mpiexec -n 10 --machinefile hostfile python3 run_openmc.py 3d-beavrs
