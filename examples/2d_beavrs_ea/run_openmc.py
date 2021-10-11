import logging
import os
import sys
import time

from mpi4py import MPI

import openmc.lib as capi

def create_logger(rank):
    seed = capi.settings.seed
    log_dir = 'logs/'
    log_fname = log_dir + 'seed{}process{}.log'.format(seed, rank)
    log_dfmt = '%m/%d/%Y %I:%M:%S %p'
    os.system('mkdir -p {}'.format(log_dir))

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt=log_dfmt,
                        filename=log_fname, filemode='w',  level=logging.INFO)
    logging.info('Log file created, CMFD initialization complete')

def log_event(log_msg):
    current_time = time.time()
    logging.info(log_msg + ", time={}".format(current_time))

if __name__ == "__main__":
    # Define MPI communicator
    comm_global = MPI.COMM_WORLD

    # Get number of seeds as command line arg
    n_seeds = int(sys.argv[1])

    # Get number of processes per seed as command line arg
    n_procs_per_seed = int(sys.argv[2])

    # Get OpenMP threads per process as command line arg
    n_threads = sys.argv[3]

    # Check correct number of total processes
    available_procs = comm_global.Get_size()
    expected_procs = n_seeds * n_procs_per_seed
    assert_condition = available_procs == expected_procs
    assert_statement = ("Incorrect number of MPI processes; Expected {} "
                        "processes, available {} processes")
    assert assert_condition, assert_statement.format(expected_procs,
                                                     available_procs)

    color = int(comm_global.Get_rank()/n_procs_per_seed)
    seed = color + 1
    comm_split = MPI.Comm.Split(comm_global, color=color)

    args=['-s', n_threads]
    
    with capi.run_in_memory(intracomm=comm_split, args=args):
        capi.settings.seed = seed
        if capi.master():
            create_logger(comm_split.Get_rank())
        capi.simulation_init()
        for _ in capi.iter_batches():
            curr_gen = capi.current_batch()
            
            if comm_split.Get_rank() == 0:
                log_event('Finished next batch, current batch {}'.format(curr_gen))

        capi.simulation_finalize()
