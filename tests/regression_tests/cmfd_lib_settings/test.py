from tests.testing_harness import CMFDTestHarness
from openmc import cmfd
import numpy as np
import scipy.sparse


def test_cmfd_lib_settings():
    """Test 1 group CMFD solver by setting openmc.lib.settings
       variables with settings_args"""
    # Initialize and set CMFD mesh
    cmfd_mesh = cmfd.CMFDMesh()
    cmfd_mesh.lower_left = (-10.0, -1.0, -1.0)
    cmfd_mesh.upper_right = (10.0, 1.0, 1.0)
    cmfd_mesh.dimension = (10, 1, 1)
    cmfd_mesh.albedo = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    # Initialize and run CMFDRun object
    cmfd_run = cmfd.CMFDRun()
    cmfd_run.mesh = cmfd_mesh
    cmfd_run.tally_begin = 1
    cmfd_run.solver_begin = 5
    cmfd_run.display = {'dominance': True}
    cmfd_run.feedback = True
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    settings_args = {"seed": 5,
                     "inactive": 5,
                     "batches": 15,
                     "particles": 500}
    cmfd_run.run(settings_args=settings_args)

    # Initialize and run CMFD test harness
    harness = CMFDTestHarness('statepoint.15.h5', cmfd_run)
    harness.main()
