"""TODO update
This module can be used to specify parameters used for coarse mesh finite
difference (CMFD) acceleration in OpenMC. CMFD was first proposed by [Smith]_
and is widely used in accelerating neutron transport problems.

References
----------

.. [Smith] K. Smith, "Nodal method storage reduction by non-linear
   iteration", *Trans. Am. Nucl. Soc.*, **44**, 265 (1983).

"""

from contextlib import contextmanager
from collections.abc import Mapping
from numbers import Integral
import sys
import time
import configparser
import json

import numpy as np
import h5py

import openmc.lib
from openmc import cmfd
from openmc.checkvalue import (check_type, check_length, check_value,
                               check_greater_than, check_less_than)
from openmc.exceptions import OpenMCError
from openmc.ensemble_averaging.openmc_node import OpenMCNode
from openmc.ensemble_averaging.cmfd_node import CMFDNode

# See if mpi4py module can be imported, define have_mpi global variable
try:
    from mpi4py import MPI
    have_mpi = True
except ImportError:
    have_mpi = False


class EnsAvgCMFDRun(object):
    # TODO add documentation for all methods
    # TODO display output
    # TODO error checking so program will exit if parameters not configured properly in self._node.run_in_memory()
    r"""Class for running CMFD with ensemble averaging.

    """

    def __init__(self):
        """Constructor for EnseAvgCMFDRun class. Default values for instance
        variables set in this method.

        """
        # Variables that users can modify
        self._cfg_file = None
        self._n_seeds = 1
        self._n_procs_per_seed = 1
        self._seed_begin = 1
        self._verbosity = 1
        self._openmc_verbosity = 1
        self._n_particles = 1000
        self._n_inactive = 10
        self._n_batches = 20
        self._tally_begin = 1
        self._window_type = 'none'
        self._solver_begin = 1
        self._display = {'balance': False, 'dominance': False,
                         'entropy': False, 'source': False}
        self._mesh = None

        # External variables used during runtime but users cannot control
        self._local_comm = None
        self._global_comm = None
        self._node = None
        self._global_args = {}
        self._openmc_args = {}
        self._cmfd_args = {}

    @property
    def cfg_file(self):
        return self._cfg_file

    @property
    def n_seeds(self):
        return self._n_seeds

    @property
    def n_procs_per_seed(self):
        return self._n_procs_per_seed

    @property
    def seed_begin(self):
        return self._seed_begin

    @property
    def verbosity(self):
        return self._verbosity

    @property
    def openmc_verbosity(self):
        return self._openmc_verbosity

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def n_inactive(self):
        return self._n_inactive

    @property
    def n_batches(self):
        return self._n_batches

    @property
    def tally_begin(self):
        return self._tally_begin

    @property
    def solver_begin(self):
        return self._solver_begin

    @property
    def window_type(self):
        return self._window_type

    @property
    def display(self):
        return self._display

    @property
    def mesh(self):
        return self._mesh

    @property
    def local_comm(self):
        return self._local_comm

    @property
    def global_comm(self):
        return self._global_comm

    @cfg_file.setter
    def cfg_file(self, cfg_file):
        check_type('Ensemble averaging config file', cfg_file, str)
        self._cfg_file = cfg_file

    @n_seeds.setter
    def n_seeds(self, n_seeds):
        check_type('Number of seeds', n_seeds, Integral)
        check_greater_than('Number of seeds', n_seeds, 0)
        self._n_seeds = n_seeds

    @n_procs_per_seed.setter
    def n_procs_per_seed(self, n_procs_per_seed):
        check_type('Number of processes per seed', n_procs_per_seed, Integral)
        check_greater_than('Number of processes per seed', n_procs_per_seed, 0)
        self._n_procs_per_seed = n_procs_per_seed

    @seed_begin.setter
    def seed_begin(self, begin):
        check_type('Seed begin', begin, Integral)
        check_greater_than('Seed begin', begin, 0)
        self._seed_begin = begin

    @verbosity.setter
    def verbosity(self, verbosity):
        check_type('Verbosity', verbosity, Integral)
        check_greater_than('Verbosity', verbosity, 0)
        self._verbosity = verbosity

    @openmc_verbosity.setter
    def openmc_verbosity(self, verbosity):
        check_type('OpenMC verbosity', verbosity, Integral)
        check_greater_than('OpenMC verbosity', verbosity, 0)
        self._openmc_verbosity = verbosity

        self._display = {'balance': False, 'dominance': False,
                         'entropy': False, 'source': False}
        self._mesh = None

    @n_particles.setter
    def n_particles(self, n_particles):
        check_type('Number of particles', n_particles, Integral)
        check_greater_than('Number of particles', n_particles, 0)
        self._n_particles = n_particles

    @n_inactive.setter
    def n_inactive(self, n_inactive):
        check_type('Number of inactive batches', n_inactive, Integral)
        check_greater_than('Number of inactive batches', n_inactive, 0)
        self._n_inactive = n_inactive

    @n_batches.setter
    def n_batches(self, n_batches):
        check_type('Number of batches', n_batches, Integral)
        check_greater_than('Number of batches', n_batches, 0)
        self._n_batches = n_batches

    @tally_begin.setter
    def tally_begin(self, begin):
        check_type('CMFD tally begin batch', begin, Integral)
        check_greater_than('CMFD tally begin batch', begin, 0)
        self._tally_begin = begin

    @solver_begin.setter
    def solver_begin(self, begin):
        check_type('CMFD solver begin batch', begin, Integral)
        check_greater_than('CMFD solver begin batch', begin, 0)
        self._solver_begin = begin

    @window_type.setter
    def window_type(self, window_type):
        check_type('CMFD window type', window_type, str)
        check_value('CMFD window type', window_type,
                    ['none', 'rolling', 'expanding'])
        self._window_type = window_type

    @display.setter
    def display(self, display):
        check_type('display', display, Mapping)
        for key, value in display.items():
            check_value('display key', key,
                        ('balance', 'entropy', 'dominance', 'source'))
            check_type("display['{}']".format(key), value, bool)
            self._display[key] = value

    @mesh.setter
    def mesh(self, cmfd_mesh):
        check_type('CMFD mesh', cmfd_mesh, cmfd.CMFDMesh)

        # Check dimension defined
        if cmfd_mesh.dimension is None:
            raise ValueError('CMFD mesh requires spatial '
                             'dimensions to be specified')

        # Check lower left defined
        if cmfd_mesh.lower_left is None:
            raise ValueError('CMFD mesh requires lower left coordinates '
                             'to be specified')

        # Check that both upper right and width both not defined
        if cmfd_mesh.upper_right is not None and cmfd_mesh.width is not None:
            raise ValueError('Both upper right coordinates and width '
                             'cannot be specified for CMFD mesh')

        # Check that at least one of width or upper right is defined
        if cmfd_mesh.upper_right is None and cmfd_mesh.width is None:
            raise ValueError('CMFD mesh requires either upper right '
                             'coordinates or width to be specified')

        # Check width and lower length are same dimension and define
        # upper_right
        if cmfd_mesh.width is not None:
            check_length('CMFD mesh width', cmfd_mesh.width,
                         len(cmfd_mesh.lower_left))
            cmfd_mesh.upper_right = np.array(cmfd_mesh.lower_left) + \
                np.array(cmfd_mesh.width) * np.array(cmfd_mesh.dimension)

        # Check upper_right and lower length are same dimension and define
        # width
        elif cmfd_mesh.upper_right is not None:
            check_length('CMFD mesh upper right', cmfd_mesh.upper_right,
                         len(cmfd_mesh.lower_left))
            # Check upper right coordinates are greater than lower left
            if np.any(np.array(cmfd_mesh.upper_right) <=
                      np.array(cmfd_mesh.lower_left)):
                raise ValueError('CMFD mesh requires upper right '
                                 'coordinates to be greater than lower '
                                 'left coordinates')
            cmfd_mesh.width = np.true_divide((np.array(cmfd_mesh.upper_right) -
                                             np.array(cmfd_mesh.lower_left)),
                                             np.array(cmfd_mesh.dimension))
        self._mesh = cmfd_mesh

    def run(self, **kwargs):
        # TODO documentation
        with self.run_in_memory(**kwargs):
            for _ in self.iter_batches():
                pass

    @contextmanager
    def run_in_memory(self):
        # TODO documentation
        self.init()
        kwargs = {
            'global_args': self._global_args,
            'openmc_args': self._openmc_args,
            'cmfd_args': self._cmfd_args
        }
        with self._node.run_in_memory(**kwargs):
            yield
        self.finalize()

    def init(self):
        # TODO documentation
        if not have_mpi:
            raise OpenMCError('mpi4py Python module is required to'
                              'run CMFD with ensemble averaging')

        # Read config file
        if self._cfg_file is not None:
            self._read_cfg_file()

        # Define local and global communicators
        self._define_comms()

        # Set node parameters
        self._set_node_params()

    def finalize(self):
        # TODO documentation
        pass
        # TODO print out timing stats

    def iter_batches(self):
        status = 0
        while status == 0:
            status = self._node.next_batch()
            # Put barrier to synchronize processes
            self._node.global_comm.Barrier()
            yield

    def _read_cfg_file(self):
        mesh_params = ['lower_left', 'upper_right', 'dimension', 'width',
                       'energy', 'albedo', 'map']
        ea_params = ['n_seeds', 'n_procs_per_seed', 'seed_begin', 'verbosity',
                     'openmc_verbosity', 'n_particles', 'n_inactive',
                     'n_batches', 'tally_begin', 'solver_begin', 'display',
                     'window_type']
        openmc_params = ['n_threads']
        cmfd_params = ['ref_d', 'downscatter', 'cmfd_ktol', 'norm', 'w_shift',
                       'stol', 'spectral', 'gauss_seidel_tolerance',
                       'n_threads', 'window_size']

        config = configparser.ConfigParser()
        config.read(self._cfg_file)

        section = 'Ensemble Averaging'
        if section in config.sections():
            for param in ea_params:
                if param in config[section]:
                    value = json.loads(config.get(section, param))
                    setattr(self, param, value)

        section = 'CMFD Mesh'
        if section in config.sections():
            cmfd_mesh = cmfd.CMFDMesh()
            for param in mesh_params:
                if param in config[section]:
                    value = json.loads(config.get(section, param))
                    setattr(cmfd_mesh, param, value)
        self.mesh = cmfd_mesh

        section = 'OpenMC Node'
        if section in config.sections():
            for param in openmc_params:
                if param in config[section]:
                    value = json.loads(config.get(section, param))
                    self._openmc_args[param] = value

        section = 'CMFD Node'
        if section in config.sections():
            for param in cmfd_params:
                if param in config[section]:
                    value = json.loads(config.get(section, param))
                    self._cmfd_args[param] = value

    def _define_comms(self):
        # Define global communicator
        self._global_comm = MPI.COMM_WORLD

        # Check correct number of total processes
        available_procs = self._global_comm.Get_size()
        expected_procs = (self._n_seeds+1) * self._n_procs_per_seed
        assert_condition = available_procs == expected_procs
        assert_statement = ("Incorrect number of MPI processes; Expected {} "
                            "processes, available {} processes")
        assert assert_condition, assert_statement.format(expected_procs,
                                                         available_procs)

        color = int(self._global_comm.Get_rank()/self._n_procs_per_seed)
        self._local_comm =  MPI.Comm.Split(self._global_comm, color=color)

    def _set_node_params(self):
        # TODO print statement to say what node is set to
        # Define node type
        if self._global_comm.Get_rank() < self._n_procs_per_seed:
            self._node = CMFDNode()
        else:
            self._node = OpenMCNode()

        # Define global args to pass to node
        global_params = ['local_comm', 'global_comm', 'n_seeds', 'verbosity',
                         'openmc_verbosity', 'n_procs_per_seed', 'mesh',
                         'tally_begin', 'seed_begin', 'solver_begin',
                         'n_particles', 'n_inactive', 'n_batches', 'window_type',
                         'display']

        for param in global_params:
            self._global_args[param] = getattr(self, param)
