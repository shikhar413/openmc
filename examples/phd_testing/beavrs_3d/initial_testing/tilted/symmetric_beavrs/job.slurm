#!/bin/bash
#SBATCH --job-name=3db-nocmfd
#SBATCH --partition=sched_mit_nse
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --time=12:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec -machinefile hostfile -n 4 python3 run_openmc.py 16 3d-beavrs nocmfd
