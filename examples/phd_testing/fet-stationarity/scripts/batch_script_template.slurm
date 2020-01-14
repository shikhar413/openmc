#!/bin/bash

#SBATCH -N {num_nodes}
#SBATCH -n {num_procs}
#SBATCH --time=12:00:00
#SBATCH -p sched_mit_nse
#SBATCH --mem-per-cpu=4000 #in MB, max is 4000
#SBATCH --job-name={filename}
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

#load modules your job depends on.
#better here than in your $HOME/.bashrc to make debugging and requirements easier to track.
#here we are using gcc under MPI mpich, you may also use intel icc with intel impi
#module load mpich/ge/gcc/64/3.1

#I like to echo the running environment
#env

#Finally, the command to execute.
#The job starts in the directory it was submitted from.
#Note that mpirun knows from SLURM how many processor we have
#In this case, we use all processes.

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec --machinefile hostfile -n {num_mpi_procs} python3 run_openmc_capi.py {num_omp_threads} {prob_type}
