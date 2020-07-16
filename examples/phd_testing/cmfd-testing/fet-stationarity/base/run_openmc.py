from openmc import cmfd
import glob
import os
import sys
from mpi4py import MPI
import openmc.lib as capi
import numpy as np

def init_openmc_run(problem_type, test_num, mesh_type, use_ref_d):
    test_num_dict = {
        '0': {
            'runtype': 'nocmfd',
            'cmfd-attrs': {
                'window_type': 'expanding'
            }
        },
        '1': {
            'runtype': 'cmfd',
            'cmfd-attrs': {
                'window_type': 'expanding'
            },
            'solver_end': True
        },
        '2': {
            'runtype': 'cmfd',
            'cmfd-attrs': {
                'window_type': 'expanding',
                'max_window_size': 64
            }
        },
        '3': {
            'runtype': 'cmfd',
            'cmfd-attrs': {
                'window_type': 'expanding',
            }
        }
    }
    solver_end_dict = {
        '1d-homog': 150,
        '1d-homog-offset': 150,
        '2d-beavrs': 150,
    }

    if test_num not in test_num_dict:
        print('Logic for test number {} not defined')
        sys.exit()

    params = test_num_dict[test_num]

    if 'solver_end' in params:
        params['cmfd-attrs']['solver_end'] = solver_end_dict[problem_type]

    # Define CMFD parameters
    cmfd_mesh = cmfd.CMFDMesh()
    if '1d-homog' in problem_type:
        cmfd_mesh.lower_left = [-5., -5., -200.]
        cmfd_mesh.upper_right = [5., 5., 200.]
        cmfd_mesh.albedo = [1., 1., 1., 1., 0., 0.]
        mesh_dim, mesh_map = get_mesh_properties(mesh_type)
        cmfd_mesh.dimension = mesh_dim

    elif problem_type == '2d-beavrs':
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.albedo = [0., 0., 0., 0., 1., 1.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

        mesh_dim, mesh_map = get_mesh_properties(mesh_type, is_1d=False)
        cmfd_mesh.dimension = mesh_dim
        cmfd_mesh.map = mesh_map
    else:
        err_msg = 'Logic for problem type {} has not been defined yet'
        print(err_msg.format(problem_type))
        sys.exit()

    # Initialize CMFDRun object
    cmfd_run = cmfd.CMFDRun()

    # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc)
    # All error checking done under the hood when setter function called
    cmfd_run.mesh = cmfd_mesh
    if problem_type == '1d-homog':
        cmfd_run.ref_d = []
        if params['runtype'] == 'nocmfd':
            cmfd_run.tally_begin = 2000
            cmfd_run.solver_begin = 2000
        else:
            cmfd_run.tally_begin = 2
            cmfd_run.solver_begin = 3
    elif problem_type == '1d-homog-offset':
        cmfd_run.ref_d = []
        if params['runtype'] == 'nocmfd':
            cmfd_run.tally_begin = 2000
            cmfd_run.solver_begin = 2000
        else:
            cmfd_run.tally_begin = 10
            cmfd_run.solver_begin = 20
    else:
        if use_ref_d:
            cmfd_run.ref_d = [1.42669, 0.400433]
        if params['runtype'] == 'nocmfd':
            cmfd_run.tally_begin = 2000
            cmfd_run.solver_begin = 2000
        else:
            cmfd_run.tally_begin = 2
            cmfd_run.solver_begin = 3

    cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': False, 'source': True}
    cmfd_run.feedback = True
    cmfd_run.downscatter = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    if mesh_type in ['pincell', '0p4cm']:
        cmfd_run.use_all_threads = True

    for attr in params['cmfd-attrs']:
        setattr(cmfd_run, attr, params['cmfd-attrs'][attr])

    return cmfd_run

def init_fet_params(problem_type):
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

def get_mesh_properties(mesh_type, is_1d=True):
    if is_1d:
        mesh_dims = {
            'none': [1, 1, 1],
            '4cm': [1, 1, 100],
            '1cm': [1, 1, 400],
            '20cm': [1, 1, 20],
            '0p4cm': [1, 1, 1000],
        }
        if mesh_type not in mesh_dims:
            err_msg = 'Logic for 1D mesh type {} has not been defined yet'
            print(err_msg.format(mesh_type))
            sys.exit()
        return mesh_dims[mesh_type], None
            
    else:
        n_assembly_x = 17
        n_assembly_y = 17
        mesh_properties = {
            'none': {
                'mesh_refinement_x': 1,
                'mesh_refinement_y': 1,
                'assembly_map_layout': [1]
            },
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
        if mesh_type not in mesh_properties:
            err_msg = 'Logic for 2D mesh type {} has not been defined yet'
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
    # Define MPI communicator
    comm = MPI.COMM_WORLD

    # Get number of OpenMP threads as command line arg
    n_threads = sys.argv[1]

    # Get problem type as command line argument
    prob_type = sys.argv[2]

    # Get test number as command line argument
    test_num = sys.argv[3]

    # Get mesh type for CMFD as command line argument
    mesh_type = sys.argv[4]

    # Get flag for using ref_d as command line argument
    use_ref_d = len(sys.argv) == 6 and sys.argv[5] == '--refd'

    labels, coeffs = init_fet_params(prob_type)

    # Find restart file
    statepoints = glob.glob(os.path.join("statepoint.*.h5")) 

    # Restart file exists
    if len(statepoints) == 1:
        res_file = statepoints[0]
        args=["-s",n_threads,"-r",res_file]

        # Load all simulation variables from disk
        prev_sp = res_file
        fet_data = np.load('fet_data.npy')
        #entropy_data = np.load('entropy_data.npy')

    # Restart file doesn't exist
    elif len(statepoints) == 0:
        args=['-s', n_threads]

    else:
        print("Multiple statepoint files found. Please retry")
        sys.exit()

    statepoint_interval = 10

    # TODO Get OpenMC instance to run in memory
    openmc_run = init_openmc_run(prob_type, test_num, mesh_type, use_ref_d)

    with openmc_run.run_in_memory(args=args):
        for _ in openmc_run.iter_batches():
            curr_gen = capi.current_batch()
            
            if comm.Get_rank() == 0:
                #entropy_p = capi.entropy_p()
                #entropy = np.sum(np.log(entropy_p[entropy_p>0])/(np.log(2)) * entropy_p[entropy_p>0])*-1
                fet_tallies = capi.convergence_tally()

                if curr_gen == 1:
                    fet_data = np.empty((0,1+len(labels)), float)
                    #entropy_data = np.empty((0,2+len(entropy_p)), float)

                # Compute scaled FET coefficients
                a_n = np.product(coeffs, axis=1) * fet_tallies

                # Store a_n, curr_gen, and entropy to numpy array
                fet_data = np.vstack((fet_data, [curr_gen] + list(a_n)))
                #entropy_data = np.vstack((entropy_data, [curr_gen] + [entropy] + list(entropy_p.flatten())))

            # Create new statepoint, remove previous one and save numpy arrays
            if curr_gen % statepoint_interval == 0:
                openmc_run.statepoint_write()
                if comm.Get_rank() == 0:
                    #np.save("entropy_data", entropy_data)
                    np.save("fet_data", fet_data)
                    # Remove previous statepoint if more than one exists
                    if curr_gen != statepoint_interval:
                        os.system('rm {}'.format(prev_sp))
                    # Update previous statepoint
                    prev_sp = glob.glob(os.path.join("statepoint.*.h5"))[0]

    # End of simulation, save fet and entropy data
    if comm.Get_rank() == 0:
        #np.save("entropy_data", entropy_data)
        np.save("fet_data", fet_data)
