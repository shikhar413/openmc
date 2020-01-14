#!/bin/bash
#SBATCH --job-name=2db-t{n_threads}
#SBATCH --partition=sched_mit_nse
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec -n 2 python3 run_openmc_cmfd.py {n_threads} {prob}
