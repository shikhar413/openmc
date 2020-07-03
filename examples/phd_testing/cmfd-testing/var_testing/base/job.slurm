#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ppn}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time={walltime}

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq > hostfile
mpiexec -n 2 --machinefile hostfile python3 run_openmc.py {openmc_args}
