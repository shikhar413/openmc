#!/bin/bash

#SBATCH -N {num_nodes}
#SBATCH -n {num_procs}
#SBATCH --time=12:00:00
#SBATCH -p sched_mit_nse
#SBATCH --mem-per-cpu=4000 #in MB, max is 4000
#SBATCH --job-name={filename}
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec --machinefile hostfile -n {num_mpi_procs} python3 run_openmc_capi.py {num_omp_threads} {prob_type}
