import openmc.lib
from openmc.ensemble_averaging import EnsAvgCMFDRun
from openmc import cmfd
import openmc.lib as capi
import glob
import os
import sys
import numpy as np
from mpi4py import MPI

def init_prob_params(problem_type, mesh_type):
    cmfd_mesh = cmfd.CMFDMesh()
    if problem_type == '1d-homog':
        cmfd_mesh.lower_left = [-5., -5., -200.]
        cmfd_mesh.upper_right = [5., 5., 200.]
        cmfd_mesh.albedo = [1., 1., 1., 1., 0., 0.]

        invalid_mesh = not mesh_type.endswith("cm")
        if not invalid_mesh:
            try:
                mesh_sz = float(mesh_type[:-2])
            except:
                invalid_mesh = True
        if invalid_mesh:
            err_msg = 'Logic for 1D mesh type {} has not been defined yet'
            print(err_msg.format(mesh_type))
            sys.exit()
        cmfd_mesh.dimension = [1, 1, int(400./mesh_sz)]

    elif problem_type == '2d-beavrs':
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.albedo = [0., 0., 0., 0., 1., 1.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

        mesh_dim, mesh_map = get_2db_mesh_properties(mesh_type)
        cmfd_mesh.dimension = mesh_dim
        cmfd_mesh.map = mesh_map

    else:
        err_msg = 'Logic for problem type {} has not been defined yet'
        print(err_msg.format(problem_type))
        sys.exit()

    return cmfd_mesh

def get_2db_mesh_properties(mesh_type):
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
    mesh_type = sys.argv[2]
    cmfd_mesh = init_prob_params(prob_type, mesh_type)
    statepoint_interval = 10

    # Initialize and run CMFDRun object
    ea_cmfd_run = EnsAvgCMFDRun()
    ea_cmfd_run.cfg_file = "params.cfg"
    ea_cmfd_run.mesh = cmfd_mesh
    with ea_cmfd_run.run_in_memory():
        for _ in ea_cmfd_run.iter_batches():
            pass
