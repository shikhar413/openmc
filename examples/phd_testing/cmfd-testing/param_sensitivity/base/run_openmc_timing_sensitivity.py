from openmc import cmfd
import glob
import os
import sys
from mpi4py import MPI
import openmc.lib as capi
import numpy as np

def init_cmfd_params(problem_type, mesh_type):
    # Define CMFD parameters
    cmfd_mesh = cmfd.CMFDMesh()
    if problem_type == '1d-homog':
        cmfd_mesh.lower_left = [-5., -5., -200.]
        cmfd_mesh.upper_right = [5., 5., 200.]
        cmfd_mesh.albedo = [1., 1., 1., 1., 0., 0.]
        if mesh_type == '0p4cm':
            cmfd_mesh.dimension = [1, 1, 1000]
        elif mesh_type == '20cm':
            cmfd_mesh.dimension = [1, 1, 20]
        else:
            cmfd_mesh.dimension = [1, 1, 1]

    else:
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.albedo = [0., 0., 0., 0., 1., 1.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

        mesh_dim, mesh_map = get_2db_mesh_properties(mesh_type)
        cmfd_mesh.dimension = mesh_dim
        if mesh_map is not None:
            cmfd_mesh.map = mesh_map

    # Initialize CMFDRun object
    cmfd_run = cmfd.CMFDRun()

    # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc)
    # All error checking done under the hood when setter function called
    cmfd_run.mesh = cmfd_mesh
    if mesh_type != 'nocmfd':
        cmfd_run.tally_begin = 2
        cmfd_run.solver_begin = 3
    else:
        cmfd_run.tally_begin = sys.maxsize
        cmfd_run.solver_begin = sys.maxsize
    if problem_type == '1d-homog':
        cmfd_run.ref_d = []
    else:
        cmfd_run.ref_d = [1.42669, 0.400433]

    cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': False, 'source': True}
    cmfd_run.feedback = True
    cmfd_run.downscatter = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    cmfd_run.window_type = 'expanding'
    if mesh_type == 'pincell' or mesh_type == '0p4cm':
        cmfd_mesh.use_all_threads = True
    return cmfd_run

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
    if mesh_type == 'nocmfd':
        return [1, 1, 1], None
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

    # Get mesh type as command line argument
    mesh_type = sys.argv[3]

    cmfd_run = init_cmfd_params(prob_type, mesh_type)

    with cmfd_run.run_in_memory(args=['-s', n_threads]):
        for _ in cmfd_run.iter_batches():
            pass
    os.system('rm *.h5')
