from openmc import cmfd
import glob
import os
import sys
from mpi4py import MPI
import h5py
import openmc.lib as capi
import numpy as np

def init_cmfd_params(problem_type):
    # Define CMFD parameters
    cmfd_mesh = cmfd.CMFDMesh()
    if problem_type == '1d-homog':
        cmfd_mesh.lower_left = [-5., -5., -200.]
        cmfd_mesh.upper_right = [5., 5., 200.]
        cmfd_mesh.albedo = [1., 1., 1., 1., 0., 0.]
    else:
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.albedo = [0., 0., 0., 0., 1., 1.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

    cmfd_mesh.dimension = [1, 1, 1000]       # VARIED PARAMETER
    

    # Initialize CMFDRun object
    cmfd_run = cmfd.CMFDRun()

    # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc)
    # All error checking done under the hood when setter function called
    cmfd_run.mesh = cmfd_mesh
    if problem_type == '1d-homog':
        cmfd_run.ref_d = []
        cmfd_run.tally_begin = 1000
        cmfd_run.solver_begin = 2000
    else:
        cmfd_run.ref_d = [1.42669, 0.400433]
        cmfd_run.tally_begin = 2
        cmfd_run.solver_begin = 3
    
    cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': True, 'source': True}
    cmfd_run.feedback = False
    cmfd_run.downscatter = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    cmfd_run.window_type = 'expanding'      # VARIED PARAMETER

    return cmfd_run

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

def save_cmfd_data(cmfd_run):
    filename = "cmfd_data{}.h5".format(capi.current_batch())
    with h5py.File(filename, 'w') as f:
        print(' Writing CMFD data to {}...'.format(filename))
        sys.stdout.flush()
        cmfd_group = f.create_group("cmfd")
        cmfd_group.attrs['num_realizations'] = cmfd_run._flux_rate.shape[-1]
        cmfd_group.create_dataset('flux', data=cmfd_run._flux)
        cmfd_group.create_dataset('totalxs', data=cmfd_run._totalxs)
        cmfd_group.create_dataset('scattxs', data=cmfd_run._scattxs)
        cmfd_group.create_dataset('nfissxs', data=cmfd_run._nfissxs)
        cmfd_group.create_dataset('p1scattxs', data=cmfd_run._p1scattxs)
        cmfd_group.create_dataset('current', data=cmfd_run._current)
        cmfd_group.create_dataset('dtilde', data=cmfd_run._dtilde)
        cmfd_group.create_dataset('dhat', data=cmfd_run._dhat)
        cmfd_group.create_dataset('cmfd_src', data=cmfd_run._cmfd_src)
        cmfd_group.create_dataset('openmc_src', data=cmfd_run._openmc_src)
        cmfd_group.create_dataset('weightfactors', data=cmfd_run._weightfactors)
        cmfd_group.create_dataset('phi', data=cmfd_run._phi)
        cmfd_group.attrs['keff'] = cmfd_run._keff
        
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
    curr_window_size = 1
    n_tallies = 0

    cmfd_run = init_cmfd_params(prob_type)

    with cmfd_run.run_in_memory(args=args):
        for _ in cmfd_run.iter_batches():
            curr_gen = capi.current_batch()
            
            if capi.master():
                print(cmfd_run._window_size)
                print(cmfd_run._flux_rate.shape)
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
                    np.save("entropy_data", entropy_data)
                    np.save("fet_data", fet_data)
                    '''
                    # Remove previous statepoint if more than one exists
                    if curr_gen != statepoint_interval:
                        os.system('rm {}'.format(prev_sp))
                    # Update previous statepoint
                    prev_sp = glob.glob(os.path.join("statepoint.*.h5"))[0]
                    '''

            curr_window_size = cmfd_run._window_size
            '''if curr_gen >= cmfd_run._tally_begin:
                n_tallies += 1
            if capi.master() and curr_gen in [16, 24, 40, 72, 136, 264, 520, 1032, 2056]:
                if cmfd_run._window_size >= 16:
                    save_cmfd_data(cmfd_run)
                n_tallies = 0
            '''

    # End of simulation, save fet and entropy data
    if capi.master():
        np.save("entropy_data", entropy_data)
        np.save("fet_data", fet_data)
