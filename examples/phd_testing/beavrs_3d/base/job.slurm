#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time={walltime}

scontrol show hostnames $SLURM_JOB_NODELIST | sort | uniq | while read line; do echo ${line}":{procs_per_node}"; done > hostfile
mpiexec -n {nproc} --machinefile hostfile python3 run_openmc.py {openmc_args}
