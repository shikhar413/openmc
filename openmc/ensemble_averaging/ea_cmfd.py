"""
This module can be used to specify parameters used for ensemble-avegaging of
coarse mesh finite difference (CMFD) acceleration in OpenMC. CMFD was first
proposed by [Smith]_ and is widely used in accelerating neutron transport
problems. Ensemble averaging is the method of aggregating CMFD tallies
over multiple independent seeds and using a global CMFD operator to update
particle weights for each of these seeds

References
----------

.. [Smith] K. Smith, "Nodal method storage reduction by non-linear
   iteration", *Trans. Am. Nucl. Soc.*, **44**, 265 (1983).
TODO Put my paper

"""

from contextlib import contextmanager
from numbers import Integral
import configparser
import json
import sys

import numpy as np

from openmc import cmfd
from openmc.checkvalue import (check_type, check_length, check_value,
                               check_greater_than)
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
    # TODO display output
    # TODO error checking so program will exit if parameters not configured properly in self._node.run_in_memory()
    # TODO add any relevant timing stats
    r"""Class for running CMFD with ensemble averaging.

    Attributes
    ----------
    current_batch: int
        Current batch of the simulation
    cfg_file : str
        Config file to set ensemble-averaging parameters
    node_type: {'openmc', 'cmfd'}
        Specifies the type of node running on MPI process
    tally_begin : int
        Batch number at which CMFD tallies should begin accummulating
    solver_begin: int
        Batch number at which CMFD solver should start executing
    mesh : openmc.cmfd.CMFDMesh
        Structured mesh to be used for acceleration
    ref_d : list of floats
        List of reference diffusion coefficients to fix CMFD parameters to
    window_type : {'expanding', 'rolling', 'none'}
        Specifies type of tally window scheme to use to accumulate CMFD
        tallies. Options are:

          * "expanding" - Have an expanding window that doubles in size
            to give more weight to more recent tallies as more generations are
            simulated
          * "rolling" - Have a fixed window size that aggregates tallies from
            the same number of previous generations tallied
          * "none" - Don't use a windowing scheme so that all tallies from last
            time they were reset are used for the CMFD algorithm.

    n_procs_per_seed : int
        Number of MPI processes used for each seed
    n_seeds : int
        Total number of seeds used for ensemble averaging
    openmc_verbosity : int
        Verbosity to set OpenMC instance to
    verbosity : int
        Verbosity for CMFDNode class
    n_batches : int
        Total number of batches simulated
    global_comm : mpi4py.MPI.Intracomm
        MPI intercommunicator to comunicate between CMFD and OpenMC nodes
    local_comm : mpi4py.MPI.Intracomm
        MPI intercommunicator to communicate locally between CMFD nodes

    """

    def __init__(self):
        """Constructor for EnseAvgCMFDRun class. Default values for instance
        variables set in this method.

        """
        # Variables that users can modify
        self._cfg_file = None
        self._n_seeds = 1
        self._n_procs_per_seed = 1
        self._verbosity = 1
        self._openmc_verbosity = 7
        self._n_batches = 20
        self._tally_begin = 1
        self._window_type = 'none'
        self._ref_d = np.array([])
        self._solver_begin = 1
        self._mesh = None

        # External variables used during runtime but users cannot control
        self._local_comm = None
        self._global_comm = None
        self._node = None
        self._node_type = None
        self._global_params = None
        self._current_batch = None
        self._global_args = {}
        self._openmc_args = {}
        self._cmfd_args = {}

    @property
    def current_batch(self):
        return self._node._current_batch

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
    def verbosity(self):
        return self._verbosity

    @property
    def openmc_verbosity(self):
        return self._openmc_verbosity

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
    def ref_d(self):
        return self._ref_d

    @property
    def mesh(self):
        return self._mesh

    @property
    def local_comm(self):
        return self._local_comm

    @property
    def global_comm(self):
        return self._global_comm

    @property
    def node_type(self):
        return self._node_type

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

    @ref_d.setter
    def ref_d(self, diff_params):
        check_type('Reference diffusion params', diff_params,
                   Iterable, Real)
        self._ref_d = np.array(diff_params)

    @window_type.setter
    def window_type(self, window_type):
        check_type('CMFD window type', window_type, str)
        check_value('CMFD window type', window_type,
                    ['none', 'rolling', 'expanding'])
        self._window_type = window_type

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

    def run(self):
        """ Run OpenMC with ensemble-averaged CMFD """
        with self.run_in_memory():
            for _ in self.iter_batches():
                pass

    @contextmanager
    def run_in_memory(self):
        """ Context manager for EnsAvgCMFDNode

        This function can be used with a 'with' statement to ensure the
        EnseAvgCMFDRun class is properly initialized/finalized. For example::

            from openmc.ensemble_averaging import EnsAvgCMFDRun
            ea_run = EnsAvgCMFDRun()
            with ea_run.run_in_memory():
                do_stuff_before_simulation_start()
                for _ in ea_run.iter_batches():
                    do_stuff_between_batches()

        """
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
        """ Initialize EnsAvgRunCMFD instance by setting all necessary
        parameters

        """
        if not have_mpi:
            raise OpenMCError('mpi4py Python module is required to'
                              'run CMFD with ensemble averaging')
        # Read config file
        if self._cfg_file:
            self._read_cfg_file()

        # Define local and global communicators
        self._define_comms()

        # Set node parameters
        self._set_node_params()

        # Write summary of ensemble averaging parameters
        if self._verbosity >= 1 and self._global_comm.Get_rank() == 0:
            self._write_summary()
        self.global_comm.Barrier()

    def finalize(self):
        """ Finalize ensemble averaging simulation """
        pass
        # TODO print out timing stats

    def iter_batches(self):
        """ Iterator over batches.

        This function returns a generator-iterator that allows Python code to
        be run between batches when running ensemble averaging. It should
        be used in conjunction with
        :func:`openmc.ensemble_averaging.EnsAvgCMFDRun.run_in_memory` to ensure
        proper initialization/finalization

        """
        status = 0
        while status == 0:
            status = self._node.next_batch()
            if self._verbosity >= 2:
                rank = self.global_comm.Get_rank()
                print("{:>11s}Process {} finished batch".format('', rank))
                sys.stdout.flush()
            # Put barrier to synchronize processes
            self.global_comm.Barrier()
            yield

    def _read_cfg_file(self):
        """ Read config file and set all global, CMFD, and OpenMC parameters """
        openmc_params = ['n_threads', 'seed_begin', 'n_particles',
                         'n_inactive']
        cmfd_params = ['downscatter', 'cmfd_ktol', 'norm',
                       'w_shift', 'stol', 'spectral', 'window_size',
                       'gauss_seidel_tolerance', 'display', 'n_threads']
        mesh_params = ['lower_left', 'upper_right', 'dimension', 'width',
                       'energy', 'albedo', 'map']
        self._global_params = ['n_seeds', 'n_procs_per_seed', 'verbosity',
                               'openmc_verbosity', 'n_batches', 'tally_begin',
                               'solver_begin', 'window_type', 'ref_d']

        config = configparser.ConfigParser()
        config.read(self._cfg_file)

        section = 'Ensemble Averaging'
        if section in config.sections():
            for param in self._global_params:
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
        self._global_params += 'local_comm', 'global_comm', 'mesh'

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

    def _write_summary(self):
        """ Write summary of global ensemble averaging parameters """
        outstr = "**** SUMMARY OF ENSEMBLE AVERAGING PARAMETERS ****\n"
        for param in self._global_params:
            if 'comm' in param:
                param_repr = (str(getattr(self, param).Get_size()) +
                              " total procs")
            else:
                param_repr = str(getattr(self, param))
            outstr += "     {}: {}\n".format(param, param_repr)
        outstr += "**************************************************\n"
        print(outstr)
        sys.stdout.flush()

    def _define_comms(self):
        """ Define intercommunicators for ensemble averaging """
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
        """ Set node properties based on MPI process rank """
        # Define node type
        if self._global_comm.Get_rank() < self._n_procs_per_seed:
            self._node = CMFDNode()
            self._node_type = "CMFD"
        else:
            self._node = OpenMCNode()
            self._node_type = "OpenMC"

        # Define global args to pass to node

        for param in self._global_params:
            self._global_args[param] = getattr(self, param)
