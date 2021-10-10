from tests.testing_harness import CMFDTestHarness
from openmc import cmfd
import numpy as np
import pytest


def test_cmfd_nofeed():
    """Test 1 group CMFD solver without CMFD feedback"""
    # Initialize and set CMFD mesh
    cmfd_mesh = cmfd.CMFDMesh()
    cmfd_mesh.lower_left = (-10.0, -1.0, -1.0)
    cmfd_mesh.upper_right = (10.0, 1.0, 1.0)
    cmfd_mesh.dimension = (10, 1, 1)
    cmfd_mesh.albedo = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    # Initialize and run CMFDRun object
    cmfd_run = cmfd.CMFDRun()
    cmfd_run.mesh = cmfd_mesh
    cmfd_run.solver_begin = 5
    cmfd_run.display = {'dominance': True}
    cmfd_run.feedback = False
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]
    cmfd_run.run()

    # Initialize and run CMFD test harness
    harness = CMFDTestHarness('statepoint.20.h5', cmfd_run)
    harness.main()

def test_cmfd_nofeed_tally_begin_err():
    """Test 1 group CMFD solver without CMFD feedback"""
    # Initialize and set CMFD mesh
    cmfd_mesh = cmfd.CMFDMesh()
    cmfd_mesh.lower_left = (-10.0, -1.0, -1.0)
    cmfd_mesh.upper_right = (10.0, 1.0, 1.0)
    cmfd_mesh.dimension = (10, 1, 1)
    cmfd_mesh.albedo = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    # Initialize and run CMFDRun object
    cmfd_run = cmfd.CMFDRun()
    cmfd_run.mesh = cmfd_mesh
    cmfd_run.solver_begin = 5
    cmfd_run.tally_begin = 10
    cmfd_run.display = {'dominance': True}
    cmfd_run.feedback = False
    cmfd_run.gauss_seidel_tolerance = [1.e-15, 1.e-20]

    # Should raise exception since tally_begin > solver_begin
    with pytest.raises(ValueError):
        cmfd_run.run()
