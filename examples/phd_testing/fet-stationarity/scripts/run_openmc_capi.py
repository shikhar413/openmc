import glob
import os
import sys
from mpi4py import MPI
import openmc.lib as capi
from openmc import StatePoint
import numpy as np


def init_prob_params(problem_type, n_axial=10, n_radial=10):
    if problem_type == '1d-homog':
        labels = np.array(['P{}'.format(str(i)) for i in range(n_axial+1)])
        coeffs = np.zeros((len(labels), 1),dtype=float)
        for j in range(len(labels)):
            coeffs[j,0] = (2*int(labels[j].split("P")[-1])+1)/2

    elif problem_type == '2d-beavrs':
        n_coeffs = int((n_radial+1)*(n_radial+2)/2)
        labels = np.array([])
        coeffs = np.zeros((n_coeffs, 1),dtype=float)
        j = 0
        for n in range(n_radial+1):
            for m in range(-1*n, n+2, 2):
                labels = np.append(labels, 'Z{},{}'.format(str(n), str(m)))
                if m == 0:
                    coeffs[j,0] = n+1
                else:
                    coeffs[j,0] = 2*n + 2
                j += 1

    elif problem_type == '3d-exasmr':
        n_coeffs = int((n_radial+1)*(n_radial+2)/2)*(n_axial+1)
        labels = np.array([])
        coeffs = np.zeros((n_coeffs, 2),dtype=float)
        j = 0
        for n in range(n_radial+1):
            for m in range(-1*n, n+2, 2):
                for l in range(n_axial+1):
                    labels = np.append(labels, 'Z{},{}-P{}'.format(str(n), str(m), str(l)))
                    if m == 0:
                        coeffs[j,0] = (n+1)
                        coeffs[j,1] = (2*l+1)/2
                    else:
                        coeffs[j,0] = (2*n+2)
                        coeffs[j,1] = (2*l+1)/2
                    j += 1
    else:
        err_msg = 'Logic for problem type {} has not been defined yet'
        print(err_msg.format(problem_type))
        sys.exit()

    return labels, coeffs

def get_conv_tally_params(tallies, problem_type):
    problem_filter_types = {
        '1d-homog': ['spatiallegendre'],
        '2d-beavrs': ['zernike'],
        '3d-exasmr': ['zernike', 'spatiallegendre'],
    }
    problem_score_types = {
        '1d-homog': ['nu-fission'],
        '2d-beavrs': ['nu-fission'],
        '3d-exasmr': ['nu-fission'],
    }

    conv_tid = -1
    n_axial = -1
    n_radial = -1
    for tid in tallies:
        filter_types = [filter.filter_type for filter in tallies[tid].filters]
        score_types = tallies[tid].scores
        if (filter_types == problem_filter_types[problem_type] and
                score_types == problem_score_types[problem_type]):
            conv_tid = tid
            for f in tallies[tid].filters:
                if f.filter_type == 'spatiallegendre':
                    n_axial = f.order
                if f.filter_type == 'zernike':
                    n_radial = f.order

    return conv_tid, n_axial, n_radial

if __name__ == "__main__":
    # Define MPI communicator
    comm = MPI.COMM_WORLD

    # Get number of OMP threads as command line arg
    n_threads = sys.argv[1]

    # Get problem_type as command line arg
    prob_type = sys.argv[2]

    # Find restart file
    statepoints = glob.glob(os.path.join("statepoint.*.h5"))

    # Restart file exists
    if len(statepoints) == 1:
        res_file = statepoints[0]
        args=["-s",num_threads,"-r",res_file]

        # Load all simulation variables from disk
        prev_sp = res_file
        fet_data = np.load('fet_data.npy')
        entropy_data = np.load('entropy_data.npy')

    # Restart file doesn't exist
    elif len(statepoints) == 0:
        args=['-s', n_threads]

    else:
        print("Multiple statepoint files found. Please retry")
        sys.exit()

    statepoint_interval = 10

    with capi.run_in_memory(args=args):
        capi.simulation_init()

        # Get id of convergence tally and set to active tally
        t = capi.tallies
        conv_tally_id, n_axial, n_radial = get_conv_tally_params(t, prob_type)
        t[conv_tally_id].active = True

        # Initialize tally label id's and coefficients
        labels, coeffs = init_prob_params(prob_type, n_axial, n_radial)

        for _ in capi.iter_batches():
            curr_gen = capi.current_batch()

            # Get tally and entropy data from C API
            tally = t[conv_tally_id]
            tally_data = tally.mean.ravel()

            entropy_p = capi.entropy_p()
            # TODO: get all other entropies! Probably have to define base entropy
            entropy = np.sum(np.log(entropy_p[entropy_p>0])/(np.log(2)) * entropy_p[entropy_p>0])*-1

            # Reset tally
            tally.reset()

            # Define empty arrays to store FET and entropy data
            if curr_gen == 1:
                fet_data = np.empty((0,1+len(labels)), float)
                entropy_data = np.empty((0,2+len(entropy_p)), float)

            # Compute scaled FET coefficients
            a_n = np.product(coeffs, axis=1) * tally_data

            # Store a_n, curr_gen, and entropy to numpy array
            fet_data = np.vstack((fet_data, [curr_gen] + list(a_n)))
            entropy_data = np.vstack((entropy_data, [curr_gen] + [entropy] + list(entropy_p.flatten())))

            # Create new statepoint, remove previous one and save numpy arrays
            if curr_gen % statepoint_interval == 0:
                if comm.Get_rank() == 0:
                    np.save("entropy_data", entropy_data)
                    np.save("fet_data", fet_data)

                statepoint = 'statepoint.{}.h5'.format(str(curr_gen))
                capi.statepoint_write(filename=statepoint)

                # Remove previous statepoint if more than one exists
                if curr_gen != statepoint_interval and comm.Get_rank() == 0:
                    os.system('rm {}'.format(prev_sp))

                # Update previous statepoint
                prev_sp = statepoint

        # End of simulation, save numpy arrays
        if comm.Get_rank() == 0:
            np.save("entropy_data", entropy_data)
            np.save("fet_data", fet_data)
        capi.simulation_finalize()
