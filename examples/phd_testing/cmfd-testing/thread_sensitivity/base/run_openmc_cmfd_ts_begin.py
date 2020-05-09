from openmc import cmfd
import glob
import os
import sys
from mpi4py import MPI
import openmc.lib as capi
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
    # Define MPI communicator
    comm = MPI.COMM_WORLD

    # Get number of OpenMP threads as command line arg
    n_threads = sys.argv[1]

    # Get problem type as command line argument
    prob_type = sys.argv[2]
    labels, coeffs = init_prob_params(prob_type)

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

    statepoint_interval = 10

    # Define CMFD parameters
    cmfd_mesh = cmfd.CMFDMesh()
    cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
    cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
    cmfd_mesh.albedo = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    cmfd_mesh.dimension = [34, 34, 1]
    cmfd_mesh.map = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    cmfd_mesh.energy = [0.0, 0.625, 20000000]

    # Initialize CMFDRun object
    cmfd_run = cmfd.CMFDRun()

    # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc)
    # All error checking done under the hood when setter function called
    cmfd_run.mesh = cmfd_mesh
    cmfd_run.tally_begin = {tbeg}   # VARIED PARAMETER
    cmfd_run.solver_begin = {sbeg}  # VARIED PARAMETER
    cmfd_run.ref_d = [{ref_d}]      # VARIED PARAMETER
    cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': True, 'source': True}
    cmfd_run.feedback = True
    cmfd_run.downscatter = True
    cmfd_run.norm = 193
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]

    cmfd_run.window_type = 'expanding'

    with cmfd_run.run_in_memory(args=args):
        capi.settings.seed = {seed} # VARIED PARAMETER
        for _ in cmfd_run.iter_batches():
            curr_gen = capi.current_batch()
            
            if comm.Get_rank() == 0:
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
                cmfd_run.statepoint_write()
                if comm.Get_rank() == 0:
                    np.save("entropy_data", entropy_data)
                    np.save("fet_data", fet_data)
                    # Remove previous statepoint if more than one exists
                    if curr_gen != statepoint_interval:
                        os.system('rm {}'.format(prev_sp))
                    # Update previous statepoint
                    prev_sp = glob.glob(os.path.join("statepoint.*.h5"))[0]

    # End of simulation, save fet and entropy data
    if comm.Get_rank() == 0:
        np.save("entropy_data", entropy_data)
        np.save("fet_data", fet_data)
