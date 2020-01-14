import time
import sys
from mpi4py import MPI
import openmc.lib as capi
from openmc import cmfd


def get_conv_tally_id(tallies, problem_type='1d-homog'):
    problem_filter_types = {
        '1d-homog': ['spatiallegendre'],
        '2d-beavrs': ['zernike'],
        '3d-exasmr': ['zernike', 'spatiallegendre'],
        '3d-homog': ['spatiallegendre', 'spatiallegendre', 'spatiallegendre'],
    }
    problem_score_types = {
        '1d-homog': ['nu-fission'],
        '2d-beavrs': ['nu-fission'],
        '3d-exasmr': ['nu-fission'],
        '3d-homog': ['nu-fission'],
    }

    for tid in tallies:
        filter_types = [filter.filter_type for filter in tallies[tid].filters]
        score_types = tallies[tid].scores
        if (filter_types == problem_filter_types[problem_type] and
                score_types == problem_score_types[problem_type]):
            return tid

    return -1


if __name__ == "__main__":
    # Define MPI communicator
    comm = MPI.COMM_WORLD

    problem_type = sys.argv[1]

    use_tallies = (len(sys.argv) == 3 and sys.argv[2] == "-t")


    # Initialize CMFDRun object
    cmfd_run = cmfd.CMFDRun()

    # Define CMFD parameters
    if problem_type == "1d-homog":
        # Initialize CMFD Mesh
        cmfd_mesh= cmfd.CMFDMesh()
        cmfd_mesh.lower_left = [-5., -5., -200.]
        cmfd_mesh.upper_right = [5., 5., 200.]
        cmfd_mesh.dimension = [1, 1, 200]
        cmfd_mesh.albedo = [1., 1., 1., 1., 0., 0.]

        # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc.)
        # All error checking done under the hood when setter function called
        cmfd_run.mesh = cmfd_mesh
        cmfd_run.tally_begin = 2
        cmfd_run.solver_begin = 100
        cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': True, 'source': True}
        cmfd_run.feedback = True
        cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
        cmfd_run.window_type = 'expanding'
        cmfd_run.ref_d = []
        cmfd_run.reset = []
    elif problem_type == "2d-beavrs":
        # Initialize CMFD Mesh
        cmfd_mesh = cmfd.CMFDMesh()
        cmfd_mesh.lower_left = [-182.78094, -182.78094, 220.0]
        cmfd_mesh.upper_right = [182.78094, 182.78094, 240.0]
        cmfd_mesh.dimension = [17, 17, 1]
        cmfd_mesh.albedo = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        cmfd_mesh.map = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cmfd_mesh.energy = [0.0, 0.625, 20000000]

        # Set all runtime parameters (cmfd_mesh, tolerances, tally_resets, etc)
        # All error checking done under the hood when setter function called
        cmfd_run.mesh = cmfd_mesh
        cmfd_run.tally_begin = 2
        cmfd_run.solver_begin = 100
        cmfd_run.display = {'balance': True, 'dominance': True, 'entropy': True, 'source': True}
        cmfd_run.feedback = True
        cmfd_run.downscatter = True
        cmfd_run.norm = 193
        cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
        cmfd_run.ref_d = [1.42669, 0.400433]
        cmfd_run.window_type = 'rolling'

    with cmfd_run.run_in_memory(args=['-s', '8']):
        # Get id of convergence tally and set to active tally
        if use_tallies:
            t = capi.tallies
            conv_tally_id = get_conv_tally_id(t, problem_type)
            t[conv_tally_id].active = True

        time_start = time.time()
        for _ in cmfd_run.iter_batches():
            curr_gen = capi.current_batch()
            if curr_gen % 10 == 0 and capi.master():
                time_elapsed = time.time() - time_start
                print("Batch {}: {}".format(curr_gen, time_elapsed))
