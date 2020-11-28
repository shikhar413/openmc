import openmc.lib
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

def get_3db_mesh():
    dz = 20.
    baf = 36.748
    taf = 402.508
    spacer_regions = [( 36.748,  40.520), ( 98.025, 103.740), (150.222, 155.937),
                      (202.419, 208.134), (254.616, 260.331), (306.813, 312.528),
                      (359.010, 364.725)]
    n_spacers = len(spacer_regions)

    lower_left = (-182.78094, -182.78094, baf - dz)
    upper_right = (182.78094, 182.78094, taf + dz)
    dimension = (17, 17) # Z-dimenion will be calculated once z_grid calculated

    x_grid = np.linspace(lower_left[0], upper_right[0], dimension[0]+1)
    y_grid = np.linspace(lower_left[1], upper_right[1], dimension[1]+1)
    # Add 10cm buffer cell before baf and baf to z grid
    z_grid = np.array([lower_left[2], baf])

    # Add spacer grid and fuel mesh cells. Fuel mesh cell heights are
    # distributed evenly between spacer grids with minimum number of cells
    # such that the height of each region is no larger than 10cm
    for (idx, spacer_region) in enumerate(spacer_regions):
        spacer_top = spacer_region[1]
        next_spacer_bottom = spacer_regions[idx+1][0] if idx + 1 < n_spacers else taf
        n_cells = int(np.ceil((next_spacer_bottom - spacer_top) / dz))
        z_grid = np.append(z_grid, np.linspace(spacer_top, next_spacer_bottom, n_cells+1))

    # Add 10cm buffer cell after taf
    z_grid = np.append(z_grid, upper_right[2])

    # Update dimension with z dimension
    dimension += (len(z_grid)-1,)

    cmfd_mesh = cmfd.CMFDMesh()
    cmfd_mesh.mesh_type = 'rectilinear'
    cmfd_mesh.grid = [x_grid, y_grid, z_grid]
    # Define map from 2D BEAVRS qassembly map
    _, accel_map = get_mesh_properties('assembly', is_1d=False)
    n_2d_cells = len(accel_map)
    noaccel_map = np.zeros((n_2d_cells,), dtype=int)

    # Iterate over axial range and build mesh map
    mesh_map = np.empty((0,), dtype=int)
    for i in range(dimension[2]):
        if i == 0 or i == dimension[2] - 1:
            mesh_map = np.append(mesh_map, noaccel_map)
        else:
            mesh_map = np.append(mesh_map, accel_map)
    cmfd_mesh.map = mesh_map

    cmfd_mesh.albedo = [0., 0., 0., 0., 0., 0.]
    cmfd_mesh.energy = [0.0, 0.625, 20000000]

    return cmfd_mesh

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

    # Hard code quarter assembly mesh if 2D BEAVRS problem type
    if prob_type == '2d-beavrs':
        cmfd_mesh = cmfd.CMFDMesh()
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.albedo = [0., 0., 0., 0., 1., 1.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]
        mesh_dim, mesh_map = get_mesh_properties('qassembly', is_1d=False)
        cmfd_mesh.dimension = mesh_dim
        cmfd_mesh.map = mesh_map
        ea_cmfd_run.mesh = cmfd_mesh

    elif prob_type == '3d-beavrs':
        ea_cmfd_run.mesh = get_3db_mesh()

    with ea_cmfd_run.run_in_memory():
        seed_num = capi.settings.seed
        n_seeds = ea_cmfd_run.n_seeds
        n_procs_per_seed = ea_cmfd_run.n_procs_per_seed
        # Get entropy mesh id and dimension size from settings.xml and CAPI
        for _ in ea_cmfd_run.iter_batches():
            curr_gen = ea_cmfd_run.current_batch
            if ea_cmfd_run.node_type == "OpenMC":
                if capi.master():
                    entropy_p = capi.entropy_p()
                    entropy = np.sum(np.log(entropy_p[entropy_p>0])/(np.log(2)) * entropy_p[entropy_p>0])*-1
                    fet_tallies = capi.convergence_tally()

                    # Define local FET / entropy data arrays
                    if curr_gen == 1:
                        fet_data = np.empty((0, 1+len(labels)), dtype=np.float)
                        entropy_data = np.empty((0, 2+len(entropy_p)), dtype=np.float)
                        os.system('mkdir -p npy_files')

                    # Compute scaled FET coefficients
                    a_n = np.product(coeffs, axis=1) * fet_tallies

                    # Store a_n, curr_gen, and entropy to numpy array
                    fet = [curr_gen] + list(a_n)
                    ent = [curr_gen] + [entropy] + list(entropy_p.flatten())
                    fet_data = np.vstack((fet_data, fet))
                    entropy_data = np.vstack((entropy_data, ent))

                    # Create new statepoint, remove previous one and save numpy arrays
                    if curr_gen % statepoint_interval == 0:
                        #cmfd_run.statepoint_write()
                        seed_num = int(ea_cmfd_run.global_comm.Get_rank() / 2)
                        np.save("npy_files/entropy_data_seed{}".format(seed_num), entropy_data)
                        np.save("npy_files/fet_data_seed{}".format(seed_num), fet_data)
                        '''
                        # Remove previous statepoint if more than one exists
                        if curr_gen != statepoint_interval:
                            os.system('rm {}'.format(prev_sp))
                        # Update previous statepoint
                        prev_sp = glob.glob(os.path.join("statepoint.*.h5"))[0]
                        '''

        # End of simulation, save fet and entropy data
        if capi.master() and ea_cmfd_run.node_type == "OpenMC":
            np.save("npy_files/entropy_data_seed{}".format(seed_num), entropy_data)
            np.save("npy_files/fet_data_seed{}".format(seed_num), fet_data)
