import openmc.lib
from openmc import cmfd

if __name__ == "__main__":
    cmfd_mesh = cmfd.CMFDMesh()
    cmfd_mesh.lower_left = (-10.0, -1.0, -1.0)
    cmfd_mesh.upper_right = (10.0, 1.0, 1.0)
    cmfd_mesh.dimension = (10, 1, 1)
    cmfd_mesh.albedo = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    # Initialize and run CMFDRun object
    cmfd_run = cmfd.CMFDRun()
    cmfd_run.mesh = cmfd_mesh
    cmfd_run.solver_begin = 5
    cmfd_run.feedback_begin = 5
    cmfd_run.display = {'dominance': True}
    cmfd_run.feedback = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    with cmfd_run.run_in_memory(args=['-s','2']):
        for _ in cmfd_run.iter_batches():
            pass
