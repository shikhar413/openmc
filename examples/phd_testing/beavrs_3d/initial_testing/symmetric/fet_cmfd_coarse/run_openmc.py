from openmc import cmfd
import glob
import os
import sys
from mpi4py import MPI
import openmc.lib as capi
import numpy as np
import h5py

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

def init_openmc_run(problem_type, run_cmfd, window_size):
    # Define CMFD parameters
    cmfd_mesh = cmfd.CMFDMesh()
    if problem_type == '1d-homog':
        cmfd_mesh.lower_left = [-5., -5., -200.]
        cmfd_mesh.upper_right = [5., 5., 200.]
        cmfd_mesh.albedo = [1., 1., 1., 1., 0., 0.]
        if run_cmfd:
            cmfd_mesh.dimension = [1, 1, 20]
        else:
            cmfd_mesh.dimension = [1, 1, 1]

    elif problem_type == '2d-beavrs':
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.albedo = [0., 0., 0., 0., 1., 1.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

        if run_cmfd:
            mesh_dim, mesh_map = get_beavrs_mesh_properties('qassembly')
            cmfd_mesh.dimension = mesh_dim
            cmfd_mesh.map = mesh_map
        else:
            cmfd_mesh.dimension = [1, 1, 1,]

    elif problem_type == '3d-beavrs':
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

        if run_cmfd:
            cmfd_mesh.mesh_type = 'rectilinear'
            cmfd_mesh.grid = [x_grid, y_grid, z_grid]
            # Define map from 2D BEAVRS qassembly map
            _, accel_map = get_beavrs_mesh_properties('assembly')
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

        else:
            cmfd_mesh.lower_left = [lower_left[0], lower_left[1], baf-dz]
            cmfd_mesh.upper_right = [upper_right[0], upper_right[1], taf+dz]
            cmfd_mesh.dimension = [1, 1, 1]

        cmfd_mesh.albedo = [0., 0., 0., 0., 0., 0.]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

    else:
        err_msg = 'CMFD logic for problem type {} has not been defined yet'
        print(err_msg.format(problem_type))
        sys.exit()

    # Initialize CMFDRun object
    cmfd_run = cmfd.CMFDRun()

    # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc)
    # All error checking done under the hood when setter function called
    cmfd_run.mesh = cmfd_mesh
    if run_cmfd:
        cmfd_run.tally_begin = 1
        cmfd_run.solver_begin = 1
    else:
        cmfd_run.tally_begin = sys.maxsize
        cmfd_run.solver_begin = sys.maxsize
        
    if problem_type == '1d-homog':
        cmfd_run.ref_d = []
    elif problem_type == '2d-beavrs':
        cmfd_run.ref_d = [1.42669, 0.400433]
    elif problem_type == '3d-beavrs':
        cmfd_run.ref_d = [1.42669, 0.400433]
    
    cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': False, 'source': True}
    cmfd_run.feedback = True
    cmfd_run.window_type = 'expanding'
    cmfd_run.downscatter = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    cmfd_run.use_all_threads = True
    #cmfd_run.linprolong_axis = 'z'
    if window_size:
        cmfd_run.max_window_size = window_size

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
        err_msg = 'FET logic for problem type {} has not been defined yet'
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
    # Define MPI communicator
    comm = MPI.COMM_WORLD

    # Get number of OpenMP threads as command line arg
    n_threads = sys.argv[1]

    # Get problem type as command line argument
    prob_type = sys.argv[2]

    # Get mesh type as command line argument
    run_cmfd = sys.argv[3] == 'cmfd'

    # Get mesh type as command line argument
    window_size = int(sys.argv[4]) if len(sys.argv) == 5 and sys.argv[4] != 'nolim' else None

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

    # TODO Get OpenMC instance to run in memory
    openmc_run = init_openmc_run(prob_type, run_cmfd, window_size)

    with openmc_run.run_in_memory(args=args):
        for _ in openmc_run.iter_batches():
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
                ###openmc_run.statepoint_write()
                if comm.Get_rank() == 0:
                    np.save("entropy_data", entropy_data)
                    np.save("fet_data", fet_data)
                    # Remove previous statepoint if more than one exists
                    ###if curr_gen != statepoint_interval:
                    ###    os.system('rm {}'.format(prev_sp))
                    # Update previous statepoint
                    ###prev_sp = glob.glob(os.path.join("statepoint.*.h5"))[0]

    # End of simulation, save fet and entropy data
    if comm.Get_rank() == 0:
        np.save("entropy_data", entropy_data)
        np.save("fet_data", fet_data)
