import logging
import os
import sys
import time

from mpi4py import MPI

import openmc.lib as capi

def create_logger():
    rank = MPI.COMM_WORLD.Get_rank()
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
    comm = MPI.COMM_WORLD

    # Get number of OpenMP threads as command line arg
    n_threads = sys.argv[1]

    # Get seed number as command line arg
    seed = int(sys.argv[2])

    args=['-s', n_threads]

    with capi.run_in_memory(intracomm=comm, args=args):
        capi.settings.seed = seed
        if capi.master():
            create_logger()
        capi.simulation_init()
        for _ in capi.iter_batches():
            curr_gen = capi.current_batch()
            
            if comm.Get_rank() == 0:
                log_event('Finished next batch, current batch {}'.format(curr_gen))

        capi.simulation_finalize()
