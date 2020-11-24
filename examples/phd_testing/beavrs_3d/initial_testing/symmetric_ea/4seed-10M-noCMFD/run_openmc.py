from openmc.ensemble_averaging import EnsAvgCMFDRun
from openmc import cmfd
import openmc.lib as capi
import glob
import os
import sys
import numpy as np
from mpi4py import MPI


def init_prob_params(problem_type):
    if '1d-homog' in problem_type:
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

    elif problem_type == '3d-beavrs':
        n_axial = 10
        n_radial = 10
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

def get_beavrs_mesh_properties(mesh_type):
    n_assembly_x = 17
    n_assembly_y = 17
    mesh_properties = {
        'assembly': {
            'mesh_refinement_x': 1,
            'mesh_refinement_y': 1,
            'assembly_map_layout': [1]
        },
        'qassembly': {
            'mesh_refinement_x': 2,
            'mesh_refinement_y': 2,
            'assembly_map_layout': [1, 1, 1, 1]
        },
        'pincell': {
            'mesh_refinement_x': 17,
            'mesh_refinement_y': 17,
            'assembly_map_layout': [
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        },
    }
    assembly_map = np.array([
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]).reshape(n_assembly_x, n_assembly_y)

    if mesh_type == 'none':
        return [1, 1, 1], None
    if mesh_type not in mesh_properties:
        err_msg = 'Logic for mesh type {} has not been defined yet'
        print(err_msg.format(mesh_type))
        sys.exit()

    mesh_refinement_x = mesh_properties[mesh_type]['mesh_refinement_x']
    mesh_refinement_y = mesh_properties[mesh_type]['mesh_refinement_y']
    assembly_mesh_map = np.array(mesh_properties[mesh_type]['assembly_map_layout']).reshape(mesh_refinement_x, mesh_refinement_y)
    mesh_map = np.zeros((n_assembly_x*mesh_refinement_x, 
                         n_assembly_y*mesh_refinement_y),
                        dtype=int)
    mesh_dim = mesh_map.shape + (1,)

    for i in range(n_assembly_x):
        for j in range(n_assembly_y):
            if assembly_map[i, j]:
                mesh_map[i*mesh_refinement_x:(i+1)*mesh_refinement_y, j*mesh_refinement_x:(j+1)*mesh_refinement_y] = assembly_mesh_map

    return mesh_dim, mesh_map.flatten()

if __name__ == "__main__":
    # Define global MPI communicator
    global_comm = MPI.COMM_WORLD

    # Get number of seeds and procs per seed  as command line args
    n_seeds = int(sys.argv[1])

    # Define local MPI communicator
    available_procs = global_comm.Get_size()
    n_procs_per_seed = int(available_procs/n_seeds)
    color = int(global_comm.Get_rank()/n_procs_per_seed)
    seed = color + 1
    local_comm =  MPI.Comm.Split(global_comm, color=color)

    # Get number of OpenMP threads as command line arg
    n_threads = sys.argv[2]

    # Get problem type as command line argument
    prob_type = sys.argv[3]

    labels, coeffs = init_prob_params(prob_type)
    args=['-s', n_threads]
    statepoint_interval = 10

    with capi.run_in_memory(args=args, intracomm=local_comm):
        capi.settings.seed = seed
        capi.simulation_init()
        for _ in capi.iter_batches():
            curr_gen = capi.current_batch()

            if local_comm.Get_rank() == 0:
                entropy_p = capi.entropy_p()
                entropy = np.sum(np.log(entropy_p[entropy_p>0])/(np.log(2)) * entropy_p[entropy_p>0])*-1
                fet_tallies = capi.convergence_tally()

                if curr_gen == 1:
                    fet_data = np.empty((0,1+len(labels)), float)
                    entropy_data = np.empty((0,2+len(entropy_p)), float)
                    os.system('mkdir -p npy_files')

                # Compute scaled FET coefficients
                a_n = np.product(coeffs, axis=1) * fet_tallies

                # Store a_n, curr_gen, and entropy to numpy array
                fet_data = np.vstack((fet_data, [curr_gen] + list(a_n)))
                entropy_data = np.vstack((entropy_data, [curr_gen] + [entropy] + list(entropy_p.flatten())))

            # Create new statepoint, remove previous one and save numpy arrays
            if curr_gen % statepoint_interval == 0:
                #openmc_run.statepoint_write()
                if local_comm.Get_rank() == 0:
                    np.save("npy_files/entropy_data_seed{}".format(seed), entropy_data)
                    np.save("npy_files/fet_data_seed{}".format(seed), fet_data)

        if local_comm.Get_rank() == 0:
            np.save("npy_files/entropy_data_seed{}".format(seed), entropy_data)
            np.save("npy_files/fet_data_seed{}".format(seed), fet_data)
        capi.simulation_finalize()
