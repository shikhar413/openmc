import time
import sys
from mpi4py import MPI
import openmc.lib as capi


def get_conv_tally_id(tallies, problem_type):
    problem_filter_types = {
        '1d-homog': ['spatiallegendre'],
        '2d-beavrs': ['zernike'],
        '3d-exasmr': ['zernike', 'spatiallegendre'],
        '3d-homog': ['spatiallegendre', 'spatiallegendre', 'spatiallegendre'],
    }
    problem_score_types = {
        '1d-homog': ['nu-fission'],
        '2d-beavrs': ['nu-fission'],
        '3d-exasmr': ['nu-fission'],
        '3d-homog': ['nu-fission'],
    }

    for tid in tallies:
        filter_types = [filter.filter_type for filter in tallies[tid].filters]
        score_types = tallies[tid].scores
        if (filter_types == problem_filter_types[problem_type] and
                score_types == problem_score_types[problem_type]):
            return tid

    return -1


if __name__ == "__main__":
    # Define MPI communicator
    comm = MPI.COMM_WORLD

    problem_type = sys.argv[1]

    use_tallies = (len(sys.argv) == 3 and sys.argv[2] == "-t")

    with capi.run_in_memory(args=['-s', '8']):
        # Get id of convergence tally and set to active tally
        if use_tallies:
            t = capi.tallies
            conv_tally_id = get_conv_tally_id(t, problem_type)
            t[conv_tally_id].active = True

        capi.simulation_init()
        time_start = time.time()

        for _ in capi.iter_batches():
            curr_gen = capi.current_batch()
            if curr_gen % 10 == 0 and capi.master():
                time_elapsed = time.time() - time_start
                print("Batch {}: {}".format(curr_gen, time_elapsed))

        capi.simulation_finalize()
