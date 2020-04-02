import openmc.lib
from openmc.ensemble_averaging import EnsAvgCMFDRun
from openmc import cmfd
import openmc.lib as capi
import glob
import os
import sys
import numpy as np

def init_prob_params(problem_type):
    if problem_type == '1d-homog':
        n_modes = 10
        labels = np.array(['P{}'.format(str(i)) for i in range(n_modes+1)])
        coeffs = np.zeros((len(labels), 1),dtype=float)
        for j in range(len(labels)):
            coeffs[j,0] = (2*int(labels[j].split("P")[-1])+1)/2

    elif problem_type == '2d-beavrs':
        n_radial_modes = 10
        n_coeffs = int((n_radial_modes+1)*(n_radial_modes+2)/2)
        labels = np.array([])
        coeffs = np.zeros((n_coeffs, 1),dtype=float)
        j = 0
        for n in range(n_radial_modes+1):
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

if __name__ == "__main__":

    # IGNORE STATEPOINTS FOR NOW
    '''
    # Find restart file
    statepoints = glob.glob(os.path.join("statepoint.*.h5"))

    # Restart file exists
    if len(statepoints) == 1:
        res_file = statepoints[0]
        args=["-s",n_threads,"-r",res_file]

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
    '''
    # Get problem type as command line argument
    prob_type = sys.argv[1]
    labels, coeffs = init_prob_params(prob_type)
    statepoint_interval = 10

    # Initialize and run CMFDRun object
    ea_cmfd_run = EnsAvgCMFDRun()
    ea_cmfd_run.cfg_file = "params.cfg"
    with ea_cmfd_run.run_in_memory():
        seed_num = capi.settings.seed
        for _ in ea_cmfd_run.iter_batches():
            if ea_cmfd_run.node_type == "OpenMC":
                curr_gen = capi.current_batch()
                if capi.master():
                    entropy_p = capi.entropy_p()
                    entropy = np.sum(np.log(entropy_p[entropy_p>0])/(np.log(2)) * entropy_p[entropy_p>0])*-1
                    fet_tallies = capi.convergence_tally()

                    if curr_gen == 1:
                        fet_data = np.empty((0,1+len(labels)), float)
                        entropy_data = np.empty((0,2+len(entropy_p)), float)

                    # Compute scaled FET coefficients
                    a_n = np.product(coeffs, axis=1) * fet_tallies

                    # Store a_n, curr_gen, and entropy to numpy array
                    fet_data = np.vstack((fet_data, [curr_gen] + list(a_n)))
                    entropy_data = np.vstack((entropy_data, [curr_gen] + [entropy] + list(entropy_p.flatten())))

                # Create new statepoint, remove previous one and save numpy arrays
                if curr_gen % statepoint_interval == 0:
                    #cmfd_run.statepoint_write()
                    if capi.master():
                        # TODO communicate FET, Entropy data to CMFD node, aggregate data across all seeds 
                        np.save("entropy_data_seed{}".format(seed_num), entropy_data)
                        np.save("fet_data_seed{}".format(seed_num), fet_data)
                        '''
                        # Remove previous statepoint if more than one exists
                        if curr_gen != statepoint_interval:
                            os.system('rm {}'.format(prev_sp))
                        # Update previous statepoint
                        prev_sp = glob.glob(os.path.join("statepoint.*.h5"))[0]
                        '''

        # End of simulation, save fet and entropy data
        if capi.master() and ea_cmfd_run.node_type == "OpenMC":
            np.save("entropy_data_seed{}".format(seed_num), entropy_data)
            np.save("fet_data_seed{}".format(seed_num), fet_data)


