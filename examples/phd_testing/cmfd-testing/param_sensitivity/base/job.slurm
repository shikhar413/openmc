#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=sched_mit_nse
#SBATCH -N {nodes}
#SBATCH -n {tasks}
#SBATCH --time={walltime}
#SBATCH -p sched_mit_nse
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

control show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec -machinefile hostfile -n {nprocs} python3 run_openmc.py {nthreads} {prob_type}
