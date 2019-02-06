import glob
import os
import sys
from mpi4py import MPI
import openmc.capi as capi
from openmc import StatePoint
import numpy as np


def get_coeffs_from_dataframe(problem_type, dataframe=None, labels=None):
    if problem_type == '1d-homog':
        if labels is None:
            labels = dataframe['spatiallegendre'].values
        coeffs = np.zeros((len(labels), 1),dtype=float)
        for j in range(len(labels)):
            coeffs[j,0] = (2*int(labels[j].split("P")[-1])+1)/2

    elif problem_type == '2d-beavrs':
        if labels is None:
            labels = dataframe['zernike'].values
        coeffs = np.zeros((len(labels), 1),dtype=float)
        for j in range(len(labels)):
            m = int(labels[j].split(",")[-1])
            n = int(labels[j].split(",")[0].split("Z")[-1])
            if m == 0:
                coeffs[j,0] = n+1
            else:
                coeffs[j,0] = 2*n + 2

    elif problem_type == '3d-exasmr':
        if labels is not None:
            define_labels = False
            leg_bins = [label.split("-")[1] for label in labels]
            zern_bins = [label.split("-")[0] for label in labels]
        else:
            define_labels = True
            leg_bins = dataframe['spatiallegendre'].values
            zern_bins = dataframe['zernike'].values
            labels = np.zeros((0,),dtype=str)

        num_bins = len(zern_bins)
        coeffs = np.zeros((num_bins, 2),dtype=float)
        counter = 0

        for j in range(num_bins):
          zern = zern_bins[j]
          leg = leg_bins[j]
          m = int(zern.split(",")[-1])
          n = int(zern.split(",")[0].split("Z")[-1])
          if m == 0:
            coeffs[j,0] = n+1
          else:
            coeffs[j,0] = 2*n + 2
          coeffs[j, 1] = (2*int(leg.split("P")[-1])+1)/2
          if define_labels:
              labels = np.append(labels, zern + "-" + leg)

    elif problem_type == '3d-homog':
        if labels is None:
            vals = df['spatiallegendre'].values
            labels = vals[:,0] + "-" + vals[:,1] + "-" + vals[:,2]
            # Strip character "P" from values
            strip = np.char.lstrip(vals.astype(str), 'P')
        else:
            strip = np.zeros((len(labels), 3))
            for j in range(len(labels)):
                vals = labels[j].split('-')
                strip[j,:] = np.char.lstrip(vals, 'P').astype(float)
        coeffs = (2.0 * strip.astype(float) + 1.0) / 2.0

    else:
        err_msg = 'Logic for problem type {} has not been defined yet'
        print(err_msg.format(problem_type))
        sys.exit()

    return labels, coeffs



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

    # Get number of OMP threads as command line arg
    num_omp_threads = sys.argv[1]

    # Get problem_type as command line arg
    problem_type = sys.argv[2]

    if problem_type not in ['1d-homog', '2d-beavrs', '3d-exasmr', '3d-homog']:
        print('Unrecognized problem type ', problem_type)
        sys.exit()

    # Find restart file
    statepoints = glob.glob(os.path.join("statepoint.*.h5"))

    # Restart file exists
    if len(statepoints) == 1:
        res_file = statepoints[0]
        capi.init(args=["-s",num_omp_threads,"-r",res_file])
        capi.simulation_init()

        # Load all simulation variables from disk
        prev_sp = res_file
        labels = np.load('labels.npy')
        # TODO UPDATE
        coeffs = get_coeffs_from_dataframe(problem_type, labels=labels)[1]
        fet_data = np.load('fet_data.npy')
        entropy_data = np.load('entropy_data.npy')

    # Restart file doesn't exist
    elif len(statepoints) == 0:
        capi.init(args=['-s',num_omp_threads])
        capi.simulation_init()

    else:
        print("Multiple statepoint files found. Please retry")
        sys.exit()

    statepoint_interval = 10

    # Get id of convergence tally and set to active tally
    t = capi.tallies
    conv_tally_id = get_conv_tally_id(t, problem_type)
    t[conv_tally_id].active = True


    status = 0
    while status == 0:
        status = capi.next_batch()
        curr_gen = capi.current_batch()

        # Get tally and entropy data from C API
        tally = t[conv_tally_id]
        tally_data = tally.mean.ravel()

        entropy_p = capi.entropy_p()
        # TODO: get all other entropies! Probably have to define base entropy
        entropy = np.sum(np.log(entropy_p[entropy_p>0])/(np.log(2)) * entropy_p[entropy_p>0])*-1

        # Reset tally
        tally.reset()

        # If first batch, get labels and coeff data from statepoint file
        if curr_gen == 1:
            # Create statepoint
            statepoint = 'statepoint.1.h5'
            capi.statepoint_write(filename=statepoint)

            # Extract tally labels from statepoint and compute coeffs
            with StatePoint(statepoint) as sp:
                tally = sp.tallies[conv_tally_id]
                df = tally.get_pandas_dataframe()
                labels, coeffs = get_coeffs_from_dataframe(problem_type,
                                                           dataframe=df)
                np.save("labels", labels)

            # Remove statepoint file
            if comm.Get_rank() == 0:
                os.system('rm {}'.format(statepoint))

            # Define empty arrays to store FET and entropy data
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

        # End of loop, save numpy arrays
        if status != 0:
            if comm.Get_rank() == 0:
                np.save("entropy_data", entropy_data)
                np.save("fet_data", fet_data)
