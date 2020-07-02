#!/bin/bash
#SBATCH --job-name=n-s{seed}
#SBATCH --partition=sched_mit_nse
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec -n 2 python3 run_openmc_capi.py 16 2d-beavrs
