from openmc import cmfd
import glob
import os
import sys
from mpi4py import MPI
import openmc.lib as capi
import numpy as np

def init_openmc_run(problem_type, test_num, max_window_size, use_logger):
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

    if test_num in ['1', '2']:
        if max_window_size is not None:
            params['cmfd-attrs']['max_window_size'] = max_window_size
        if '1d-homog' in problem_type:
            mesh_type = '20cm'
        else:
            mesh_type = 'qassembly'
    else:
        mesh_type = 'none'

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
            cmfd_run.tally_begin = 1
            cmfd_run.solver_begin = 1
    elif problem_type == '1d-homog-offset':
        cmfd_run.ref_d = []
        if params['runtype'] == 'nocmfd':
            cmfd_run.tally_begin = 2000
            cmfd_run.solver_begin = 2000
        else:
            cmfd_run.tally_begin = 1
            cmfd_run.solver_begin = 1
    else:
        cmfd_run.ref_d = [1.42669, 0.400433]
        if params['runtype'] == 'nocmfd':
            cmfd_run.tally_begin = 2000
            cmfd_run.solver_begin = 2000
        else:
            cmfd_run.tally_begin = 1
            cmfd_run.solver_begin = 1

    cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': False, 'source': True}
    cmfd_run.feedback = True
    cmfd_run.downscatter = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    if use_logger:
        cmfd_run.use_logger = True

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

    # Get test number as command line argument
    test_num = sys.argv[4]

    # Get use_logger as command line argument
    use_logger = sys.argv[5] == '--log'

    # Get max window size as command line argument if provided
    max_window_size = int(sys.argv[6]) if len(sys.argv) == 7 else None

    labels, coeffs = init_fet_params(prob_type)

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
        #entropy_data = np.load('entropy_data.npy')

    # Restart file doesn't exist
    elif len(statepoints) == 0:
        args=['-s', n_threads]

    else:
        print("Multiple statepoint files found. Please retry")
        sys.exit()
    '''

    args=['-s', n_threads]
    statepoint_interval = 10

    # TODO Get OpenMC instance to run in memory
    openmc_run = init_openmc_run(prob_type, test_num, max_window_size, use_logger)

    with openmc_run.run_in_memory(args=args, intracomm=local_comm, seed=seed):
        global_comm.Barrier()
        for _ in openmc_run.iter_batches():
            pass
