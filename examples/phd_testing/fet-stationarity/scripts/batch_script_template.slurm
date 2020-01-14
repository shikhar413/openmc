#!/bin/bash
# submit with sbatch <filename>.slurm
# commandline arguments may instead by supplied with #SBATCH <flag> <value>
# commandline arguments override these values

# Number of nodes
#SBATCH -N {num_nodes}
# Number of processor core (num_nodes*32=1024, psfc, mit and emiliob nodes have 32 cores per node)
#SBATCH -n {num_procs}
# specify how long your job needs. Be HONEST, it affects how long the job may wait for its turn.
#SBATCH --time=12:00:00
# which partition or queue the jobs runs in
#SBATCH -p sched_mit_nse

#SBATCH --mem-per-cpu=4000 #in MB, max is 4000
#customize the name of the stderr/stdout file. %j is the job number
#SBATCH -o {filename}-%j.out

#

#load default system modules
. /etc/profile.d/modules.sh

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

mpiexec -n {num_mpi_procs} python3 run_openmc_capi.py {num_omp_threads} {prob_type}
touch done.out
