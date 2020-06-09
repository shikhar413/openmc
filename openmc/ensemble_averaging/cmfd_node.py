"""
This module runs all CMFD-related functions when running ensemble averaging.
Specifically, it receives CMFD tallies from OpenMC, banks them to compute CMFD
cross sections, runs the CMFD solver, and sends updated weightfactors
to each OpenMC process

"""

from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from numbers import Real, Integral
import sys
import time
import warnings

import numpy as np
from scipy import sparse
import h5py
from mpi4py import MPI

import openmc.lib
from openmc.checkvalue import (check_type, check_length, check_value,
                               check_greater_than, check_less_than)
from openmc.exceptions import OpenMCError

# Maximum/minimum neutron energies
_ENERGY_MAX_NEUTRON = np.inf
_ENERGY_MIN_NEUTRON = 0.

# Tolerance for detecting zero flux values
_TINY_BIT = 1.e-8

# For non-accelerated regions on coarse mesh overlay
_CMFD_NOACCEL = -1

# Constant to represent a zero flux "albedo"
_ZERO_FLUX = 999.0

# Map that returns index of current direction in numpy current matrix
_CURRENTS = {
    'out_left':   0, 'in_left':   1, 'out_right': 2, 'in_right': 3,
    'out_back':   4, 'in_back':   5, 'out_front': 6, 'in_front': 7,
    'out_bottom': 8, 'in_bottom': 9, 'out_top':  10, 'in_top':  11
}


class CMFDNode(object):
    # TODO Will likely need to use openmc.settings.cmfd_run for statepoint stuff
    # TODO what is needed for statepoint write?
    # TODO add timing stats for ensemble averaging communication
    # TODO broadcast error to all procs if error happens
    r"""Class for running CMFD node when running CMFD with ensemble averaging.

    Attributes
    ----------
    tally_begin: int
        Batch number at which CMFD tallies should begin accummulating
    solver_begin: int
        Batch number at which CMFD solver should start executing
    display : dict
        Dictionary indicating which CMFD results to output. Note that CMFD
        k-effective will always be outputted. Acceptable keys are:

        * "balance" - Whether to output RMS [%] of the resdiual from the
          neutron balance equation on CMFD tallies (bool)
        * "dominance" - Whether to output the estimated dominance ratio from
          the CMFD iterations (bool)
        * "entropy" - Whether to output the *entropy* of the CMFD predicted
          fission source (bool)
        * "source" - Whether to ouput the RMS [%] between the OpenMC fission
          source and CMFD fission source (bool)

    downscatter : bool
        Indicate whether an effective downscatter cross section should be used
        when using 2-group CMFD.
    cmfd_ktol : float
        Tolerance on the eigenvalue when performing CMFD power iteration
    mesh : openmc.cmfd.CMFDMesh
        Structured mesh to be used for acceleration
    norm : float
        Normalization factor applied to the CMFD fission source distribution
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

    window_size : int
        Size of window to use for tally window scheme. Only relevant when
        window_type is set to "rolling"
    max_window_size : int
        Maximum size of window to use for an expanding tally window scheme
    w_shift : float
        Optional Wielandt shift parameter for accelerating power iterations. By
        default, it is very large so there is effectively no impact.
    stol : float
        Tolerance on the fission source when performing CMFD power iteration
    spectral : float
        Optional spectral radius that can be used to accelerate the convergence
        of Gauss-Seidel iterations during CMFD power iteration.
    gauss_seidel_tolerance : Iterable of float
        Two parameters specifying the absolute inner tolerance and the relative
        inner tolerance for Gauss-Seidel iterations when performing CMFD.
    damping_factor : float
        Damping factor to control weight of OpenMC vs. CMFD source to apply
        towards particle source weights during CMFD feedback. A value of 0.0
        corresponds to no CMFD feedback while a value of 1.0 corresponds to
        a full weightage of CMFD weight factors. #TODO reword
    indices : numpy.ndarray
        Stores spatial and group dimensions as [nx, ny, nz, ng]
    cmfd_src : numpy.ndarray
        CMFD source distribution calculated from solving CMFD equations
    entropy : list of floats
        "Shannon entropy" from CMFD fission source, stored for each generation
        that CMFD is invoked
    balance : list of floats
        RMS of neutron balance equations, stored for each generation that CMFD
        is invoked
    src_cmp : list of floats
        RMS deviation of OpenMC and CMFD normalized source, stored for each
        generation that CMFD is invoked
    dom : list of floats
        Dominance ratio from solving CMFD matrix equations, stored for each
        generation that CMFD is invoked
    k_cmfd : list of floats
        List of CMFD k-effectives, stored for each generation that CMFD is
        invoked
    time_cmfdnode : float
        Time in CMFD node, in seconds
    time_cmfd : float
        Time in CMFD routine excluding communication costs, in seconds
    time_cmfdbuild : float
        Time for building CMFD matrices, in seconds
    time_cmfdsolve : float
        Time for solving CMFD matrix equations, in seconds
    time_waittallies : float
        Time waiting for tallies from OpenMC tallies, in seconds
    time_waitsrccnts : float
        Time waiting for source counts from OpenMC tallies, in seconds
    time_sendweightfactors : float
        Time sending weightfactors to OpenMC nodes
    n_threads : int
        Number of threads allocated to OpenMC for CMFD solver
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
        """ Constructor for CMFDNode class. Default values for instance variables
        set in this method.

        """
        # Variables that users can modify
        self._display = {'balance': False, 'dominance': False,
                         'entropy': False, 'source': False}
        self._downscatter = False
        self._cmfd_ktol = 1.e-8
        self._norm = 1.
        self._window_size = 10
        self._max_window_size = sys.maxsize
        self._w_shift = 1.e6
        self._stol = 1.e-8
        self._spectral = 0.0
        self._gauss_seidel_tolerance = [1.e-10, 1.e-5]
        self._damping_factor = 1.0
        self._n_threads = 1

        # Variables defined by EnsAvgCMFDRun class
        self._mesh = None
        self._window_type = None
        self._ref_d = None
        self._tally_begin = None
        self._solver_begin = None
        self._n_procs_per_seed = None
        self._n_seeds = None
        self._openmc_verbosty = None
        self._verbosity = None
        self._n_batches = None
        self._global_comm = None
        self._local_comm = None

        # External variables used during runtime but users cannot control
        self._set_reference_params = False
        self._indices = np.zeros(4, dtype=np.int32)
        self._egrid = None
        self._albedo = None
        self._coremap = None
        self._mesh_id = None
        self._mat_dim = _CMFD_NOACCEL
        self._keff_bal = None
        self._keff = None
        self._openmc_keff = None
        self._phi = None
        self._openmc_src_rate = None
        self._flux_rate = None
        self._total_rate = None
        self._p1scatt_rate = None
        self._scatt_rate = None
        self._nfiss_rate = None
        self._current_rate = None
        self._flux = None
        self._totalxs = None
        self._p1scattxs = None
        self._scattxs = None
        self._nfissxs = None
        self._diffcof = None
        self._dtilde = None
        self._dhat = None
        self._hxyz = None
        self._current = None
        self._cmfd_src = None
        self._openmc_src = None
        self._sourcecounts = None
        self._weightfactors = None
        self._entropy = []
        self._balance = []
        self._src_cmp = []
        self._dom = []
        self._k_cmfd = []
        self._resnb = None
        self._time_cmfdnode = None
        self._time_cmfd = None
        self._time_cmfdbuild = None
        self._time_cmfdsolve = None
        self._time_waittallies = None
        self._time_waitsrccnts = None
        self._time_sendweightfactors = None
        self._current_batch = 0

        # All index-related variables, for numpy vectorization
        self._first_x_accel = None
        self._last_x_accel = None
        self._first_y_accel = None
        self._last_y_accel = None
        self._first_z_accel = None
        self._last_z_accel = None
        self._notfirst_x_accel = None
        self._notlast_x_accel = None
        self._notfirst_y_accel = None
        self._notlast_y_accel = None
        self._notfirst_z_accel = None
        self._notlast_z_accel = None
        self._is_adj_ref_left = None
        self._is_adj_ref_right = None
        self._is_adj_ref_back = None
        self._is_adj_ref_front = None
        self._is_adj_ref_bottom = None
        self._is_adj_ref_top = None
        self._accel_idxs = None
        self._accel_neig_left_idxs = None
        self._accel_neig_right_idxs = None
        self._accel_neig_back_idxs = None
        self._accel_neig_front_idxs = None
        self._accel_neig_bot_idxs = None
        self._accel_neig_top_idxs = None
        self._loss_row = None
        self._loss_col = None
        self._prod_row = None
        self._prod_col = None
        self._flux_slice = None
        self._total_slice = None
        self._scatt_slice = None
        self._nfiss_slice = None
        self._current_slice = None
        self._p1scatt_slice = None
        self._tally_data_size = None

    @property
    def display(self):
        return self._display

    @property
    def downscatter(self):
        return self._downscatter

    @property
    def cmfd_ktol(self):
        return self._cmfd_ktol

    @property
    def norm(self):
        return self._norm

    @property
    def window_size(self):
        return self._window_size

    @property
    def w_shift(self):
        return self._w_shift

    @property
    def max_window_size(self):
        return self._max_window_size

    @property
    def stol(self):
        return self._stol

    @property
    def spectral(self):
        return self._spectral

    @property
    def gauss_seidel_tolerance(self):
        return self._gauss_seidel_tolerance

    @property
    def damping_factor(self):
        return self._damping_factor

    @property
    def n_threads(self):
        return self._n_threads

    @property
    def mesh(self):
        return self._mesh

    @property
    def window_type(self):
        return self._window_type

    @property
    def ref_d(self):
        return self._ref_d

    @property
    def tally_begin(self):
        return self._tally_begin

    @property
    def solver_begin(self):
        return self._solver_begin

    @property
    def n_procs_per_seed(self):
        return self._n_procs_per_seed

    @property
    def n_seeds(self):
        return self._n_seeds

    @property
    def openmc_verbosity(self):
        return self._openmc_verbosity

    @property
    def verbosity(self):
        return self._verbosity

    @property
    def n_batches(self):
        return self._n_batches

    @property
    def global_comm(self):
        return self._global_comm

    @property
    def local_comm(self):
        return self._local_comm

    @property
    def indices(self):
        return self._indices

    @property
    def cmfd_src(self):
        return self._cmfd_src

    @property
    def dom(self):
        return self._dom

    @property
    def src_cmp(self):
        return self._src_cmp

    @property
    def balance(self):
        return self._balance

    @property
    def entropy(self):
        return self._entropy

    @property
    def k_cmfd(self):
        return self._k_cmfd

    @display.setter
    def display(self, display):
        check_type('display', display, Mapping)
        for key, value in display.items():
            check_value('display key', key,
                        ('balance', 'entropy', 'dominance', 'source'))
            check_type("display['{}']".format(key), value, bool)
            self._display[key] = value

    @downscatter.setter
    def downscatter(self, downscatter):
        check_type('CMFD downscatter', downscatter, bool)
        self._downscatter = downscatter

    @cmfd_ktol.setter
    def cmfd_ktol(self, cmfd_ktol):
        check_type('CMFD eigenvalue tolerance', cmfd_ktol, Real)
        self._cmfd_ktol = cmfd_ktol

    @norm.setter
    def norm(self, norm):
        check_type('CMFD norm', norm, Real)
        self._norm = norm

    @window_size.setter
    def window_size(self, window_size):
        check_type('CMFD window size', window_size, Integral)
        check_greater_than('CMFD window size', window_size, 0)
        if self._window_type != 'rolling':
            warn_msg = 'Window size will have no effect on CMFD simulation ' \
                       'unless window type is set to "rolling".'
            warnings.warn(warn_msg, RuntimeWarning)
        self._window_size = window_size

    @max_window_size.setter
    def max_window_size(self, window_size):
        check_type('CMFD max window size', window_size, Integral)
        check_greater_than('CMFD max window size', window_size, 0)
        if self._window_type != 'expanding':
            warn_msg = 'Window size will have no effect on CMFD simulation ' \
                       'unless window type is set to "expanding".'
            warnings.warn(warn_msg, RuntimeWarning)
        self._max_window_size = window_size

    @w_shift.setter
    def w_shift(self, w_shift):
        check_type('CMFD Wielandt shift', w_shift, Real)
        self._w_shift = w_shift

    @stol.setter
    def stol(self, stol):
        check_type('CMFD fission source tolerance', stol, Real)
        self._stol = stol

    @spectral.setter
    def spectral(self, spectral):
        check_type('CMFD spectral radius', spectral, Real)
        self._spectral = spectral

    @gauss_seidel_tolerance.setter
    def gauss_seidel_tolerance(self, gauss_seidel_tolerance):
        check_type('CMFD Gauss-Seidel tolerance', gauss_seidel_tolerance,
                   Iterable, Real)
        check_length('Gauss-Seidel tolerance', gauss_seidel_tolerance, 2)
        self._gauss_seidel_tolerance = gauss_seidel_tolerance
        
    @damping_factor.setter
    def damping_factor(self, damping_factor):
        check_type('CMFD damping factor', damping_factor, Real)
        check_greater_than('CMFD damping factor', damping_factor, 0., True)
        check_less_than('CMFD damping factor', damping_factor, 1., True)
        self._damping_factor = damping_factor

    @n_threads.setter
    def n_threads(self, threads):
        check_type('CMFD threads', threads, Integral)
        check_greater_than('CMFD threads', threads, 0)
        self._n_threads = threads

    # All error checking for following methods done in EnsAvgCMFDRun class
    @global_comm.setter
    def global_comm(self, comm):
        self._global_comm = comm

    @local_comm.setter
    def local_comm(self, comm):
        self._local_comm = comm

    @n_seeds.setter
    def n_seeds(self, n_seeds):
        self._n_seeds = n_seeds

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

    @openmc_verbosity.setter
    def openmc_verbosity(self, verbosity):
        self._openmc_verbosity = verbosity

    @n_procs_per_seed.setter
    def n_procs_per_seed(self, procs):
        self._n_procs_per_seed = procs

    @mesh.setter
    def mesh(self, cmfd_mesh):
        self._mesh = cmfd_mesh

    @window_type.setter
    def window_type(self, window_type):
        self._window_type = window_type

    @ref_d.setter
    def ref_d(self, ref_d):
        self._ref_d = ref_d

    @tally_begin.setter
    def tally_begin(self, begin):
        self._tally_begin = begin

    @solver_begin.setter
    def solver_begin(self, begin):
        self._solver_begin = begin

    @n_batches.setter
    def n_batches(self, n_batches):
        self._n_batches = n_batches

    @contextmanager
    def run_in_memory(self, **kwargs):
        """ Context manager for running CMFDNode functions.

        This function can be used with a 'with' statement to ensure the
        CMFDNode class is properly initialized/finalized. For example::

            from openmc.ensemble_averaging.cmfd_node import CMFDNode
            node = CMFDNode()
            with node.run_in_memory():
                status = 0
                do_stuff_before_simulation_start()
                while status == 0:
                    status = node.next_batch()
                    do_stuff_between_batches()

        Parameters
        ----------
        **kwargs
            All keyword arguments used to initialize CMFDNode class.

        """
        # Extract arguments passed from EnsAvgCMFDRun class
        global_args = kwargs['global_args']
        cmfd_args = kwargs['cmfd_args']

        # Initialize ensemble averaging parameters
        self._initialize_ea_params(global_args, cmfd_args)
        if self._verbosity >= 1:
            self._write_summary()

        # Run and pass arguments to C API run_in_memory function 
        args = ['-s', str(self._n_threads)]
        with openmc.lib.run_in_memory(args=args, intracomm=self._local_comm):
            if openmc.lib.master():
                self.init()
                yield
                self.finalize()
            else:
                yield

    def init(self):
        """ Initialize CMFDNode instance by setting up CMFD parameters """
        # Configure OpenMC parameters
        self._configure_openmc()

        # Configure CMFD parameters
        self._configure_cmfd()

        # Compute and store array indices used to build cross section
        # arrays
        self._precompute_array_indices()

        # Compute and store row and column indices used to build CMFD
        # matrices
        self._precompute_matrix_indices()

        # Initialize all variables used for linear solver in C++
        self._initialize_linsolver()

        # Set cmfd_run variable to True through C API
        openmc.lib.settings.cmfd_run = True

    def next_batch(self):
        """ Run next batch for CMFDNode.

        Returns
        -------
        int
            Status after running a batch (0=normal, 1=reached maximum number of
            batches)

        """
        # Add 1 to current batch
        self._current_batch += 1
        if openmc.lib.master():
            print("Current batch", self._current_batch)
            sys.stdout.flush()

        # Receive tally data from all OpenMCNode objects before CMFD execution
        if openmc.lib.master():
            self._execute_cmfd()

        status = 1 if self._current_batch == self._n_batches else 0

        return status

    def finalize(self):
        """ Finalize simulation by printing Ensemble Averaging timing stats. """
        openmc_timing_stats = self._recv_openmc_timing_stats()
        # Print out CMFD timing statistics
        self._write_timing_stats(openmc_timing_stats)

    def _initialize_ea_params(self, global_args, cmfd_args):
        """ Initialize global parameters inherited from EnsAvgCMFDRun class """
        # Initialize global parameters inherited from EnsAvgCMFDRun class
        for param in global_args:
            setattr(self, param, global_args[param])

        # Initialize CMFD parameters inherited from EnsAvgCMFDRun class
        for param in cmfd_args:
            setattr(self, param, cmfd_args[param])

    def _write_summary(self):
        """ Write summary of CMFD node parameters """
        cmfd_params = ['downscatter', 'cmfd_ktol', 'norm',
                       'w_shift', 'stol', 'spectral', 'window_size',
                       'gauss_seidel_tolerance', 'display', 'n_threads',
                       'damping_dactor', 'max_window_size']
        rank = self._global_comm.Get_rank()
        if self._global_comm.Get_rank() == 0:
            outstr = "********** PROCESS {}: CMFD NODE, ACTIVE **********\n"
            outstr = outstr.format(rank)
            for param in cmfd_params:
                param_repr = str(getattr(self, param))
                outstr += "     {}: {}\n".format(param, param_repr)
        else:
            outstr = "********** PROCESS {}: CMFD NODE, UNUSED **********\n"
            outstr = outstr.format(rank)
        outstr += "**************************************************\n"
        print(outstr)
        sys.stdout.flush()

    def _initialize_linsolver(self):
        """ Initialize linear solver used to run CMFD

        Returns
        -------
        int
            Status from calling `openmc.lib._dll.openmc_initialize_linsolver`

        """
        # Determine number of rows in CMFD matrix
        ng = self._indices[3]
        n = self._mat_dim*ng
        use_all_threads = True

        # Create temp loss matrix to pass row/col indices to C++ linear solver
        loss_row = self._loss_row
        loss_col = self._loss_col
        temp_data = np.ones(len(loss_row))
        temp_loss = sparse.csr_matrix((temp_data, (loss_row, loss_col)),
                                      shape=(n, n))

        # Pass coremap as 1-d array of 32-bit integers
        coremap = np.swapaxes(self._coremap, 0, 2).flatten().astype(np.int32)

        args = temp_loss.indptr, len(temp_loss.indptr), \
            temp_loss.indices, len(temp_loss.indices), n, \
            self._spectral, self._indices, coremap, use_all_threads
        return openmc.lib._dll.openmc_initialize_linsolver(*args)

    def _write_cmfd_output(self):
        """ Write CMFD output to buffer at the end of each batch """
        # Display CMFD k-effective
        outstr = '\n{:>11s}CMFD k:    {:0.5f}'.format('', self._k_cmfd[-1])
        # Display value of additional fields based on display dict
        outstr += '\n'
        if self._display['dominance']:
            outstr += ('{:>11s}Dom Rat:   {:0.5f}\n'
                       .format('', self._dom[-1]))
        if self._display['entropy']:
            outstr += ('{:>11s}CMFD Ent:  {:0.5f}\n'
                       .format('', self._entropy[-1]))
        if self._display['source']:
            outstr += ('{:>11s}RMS Src:   {:0.5f}\n'
                       .format('', self._src_cmp[-1]))
        if self._display['balance']:
            outstr += ('{:>11s}RMS Bal:   {:0.5f}\n'
                       .format('', self._balance[-1]))

        print(outstr)
        sys.stdout.flush()

    def _configure_openmc(self):
        """ Configure OpenMC parameters through OpenMC lib """
        # Define all variables necessary for running CMFD
        openmc.lib.verbosity = self._openmc_verbosity

    def _configure_cmfd(self):
        """ Initialize CMFD parameters and set CMFD input variables """
        # Define all variables necessary for running CMFD
        self._initialize_cmfd()

    def _write_timing_stats(self, openmc_times):
        """ Write ensemble averaging timing stats to buffer after finalizing simulation """
        self._time_cmfd = (self._time_cmfdnode - self._time_waittallies -
                           self._time_waitsrccnts -
                           self._time_sendweightfactors)
        outstr = ("==================>     "
                  "CMFD NODE TIMING STATISTICS     <==================\n\n"
                  "   Time in CMFD Node               =  {:.5e} seconds\n"
                  "     CMFD routines                 =  {:.5e} seconds\n"
                  "     Building matrices             =  {:.5e} seconds\n"
                  "     Solving matrices              =  {:.5e} seconds\n"
                  "     Waiting for tallies           =  {:.5e} seconds\n"
                  "     Waiting for sourcecounts      =  {:.5e} seconds\n"
                  "     Sending weightfactors         =  {:.5e} seconds\n")
        print(outstr.format(self._time_cmfdnode, self._time_cmfd,
                            self._time_cmfdbuild, self._time_cmfdsolve,
                            self._time_waittallies, self._time_waitsrccnts,
                            self._time_sendweightfactors))

        outstr = ("\n=================>     "
                  "OPENMC NODE TIMING STATISTICS     <=================\n\n")
        time_openmc = openmc_times[:, 0] - np.sum(openmc_times[:, 1:], axis=1)
        for i in range(self._n_seeds):
            outstr += ("SEED {}\n"
                       "   Time in OpenMC Node             =  {:.5e} seconds\n"
                       "     OpenMC routines               =  {:.5e} seconds\n"
                       "     Sending tallies               =  {:.5e} seconds\n"
                       "     Sending sourcecounts          =  {:.5e} seconds\n"
                       "     Waiting for weightfactors     =  {:.5e} seconds\n")
            print(outstr.format(i+1, openmc_times[i, 0], time_openmc[i], 
                                openmc_times[i, 1], openmc_times[i, 2],
                                openmc_times[i, 3]))
        sys.stdout.flush()

    def _initialize_cmfd(self):
        """ Sets values of CMFD instance variables based on user input """
        # Print message to user and flush output to stdout
        if self._verbosity >= 7 and openmc.lib.master():
            print(' Configuring CMFD parameters for simulation')
            sys.stdout.flush()

        # Check if CMFD mesh is defined
        if self._mesh is None:
            raise ValueError('No CMFD mesh has been specified for '
                             'simulation')

        # Set spatial dimensions of CMFD object
        for i, n in enumerate(self._mesh.dimension):
            self._indices[i] = n

        # Set number of energy groups
        if self._mesh.energy is not None:
            ng = len(self._mesh.energy)
            self._egrid = np.array(self._mesh.energy)
            self._indices[3] = ng - 1
        else:
            self._egrid = np.array([_ENERGY_MIN_NEUTRON, _ENERGY_MAX_NEUTRON])
            self._indices[3] = 1

        # Get acceleration map, otherwise set all regions to be accelerated
        if self._mesh.map is not None:
            check_length('CMFD coremap', self._mesh.map,
                         np.product(self._indices[0:3]))
            self._coremap = np.array(self._mesh.map)
        else:
            self._coremap = np.ones((np.product(self._indices[0:3])),
                                    dtype=int)

        # Initialize parameters for CMFD tally windows
        self._set_tally_window()

        # Set reference diffusion parameters
        if self._ref_d.size > 0:
            self._set_reference_params = True
            # Check length of reference diffusion parameters equal to number of
            # energy groups
            if self._ref_d.size != self._indices[3]:
                raise OpenMCError('Number of reference diffusion parameters '
                                  'must equal number of CMFD energy groups')

        # Define all variables that will exist only on master process
        # Set global albedo
        if self._mesh.albedo is not None:
            self._albedo = np.array(self._mesh.albedo)
        else:
            self._albedo = np.array([1., 1., 1., 1., 1., 1.])

        # Set up CMFD coremap
        self._set_coremap()

        # Extract spatial and energy indices
        nx, ny, nz, ng = self._indices

        # Allocate parameters that need to be stored for tally window
        self._openmc_src_rate = np.zeros((nx, ny, nz, ng, 0))
        self._flux_rate = np.zeros((nx, ny, nz, ng, 0))
        self._total_rate = np.zeros((nx, ny, nz, ng, 0))
        self._p1scatt_rate = np.zeros((nx, ny, nz, ng, 0))
        self._scatt_rate = np.zeros((nx, ny, nz, ng, ng, 0))
        self._nfiss_rate = np.zeros((nx, ny, nz, ng, ng, 0))
        self._current_rate = np.zeros((nx, ny, nz, 12, ng, 0))

        # Initialize timers
        self._time_cmfdnode = 0.0
        self._time_cmfd = 0.0
        self._time_cmfdbuild = 0.0
        self._time_cmfdsolve = 0.0
        self._time_waittallies = 0.0
        self._time_waitsrccnts = 0.0
        self._time_sendweightfactors = 0.0

    def _recv_tallies_from_openmc(self):
        """ Receive tally data from all OpenMC seeds

        Returns
        -------
        numpy.ndarray
            Seed-averaged tally data

        """
        all_tally_data = np.empty([self._n_seeds, self._tally_data_size],
                                  dtype=np.float64)
        for i in range(self._n_seeds):
            tally_data = np.empty(self._tally_data_size,dtype=np.float64)
            status = MPI.Status()
            self._global_comm.Recv(tally_data, source=MPI.ANY_SOURCE, status=status, tag=0)
            source = status.Get_source()
            if self._verbosity >= 2:
                dest = self._global_comm.Get_rank()
                outstr = "{:>11s}Process {} received tally data from process {}"
                print(outstr.format('', dest, source))
                sys.stdout.flush()
            seed_idx = int((source-self._n_procs_per_seed)/self._n_procs_per_seed)
            all_tally_data[seed_idx,:] = tally_data
        return np.sum(all_tally_data, axis=0)/self._n_seeds

    def _recv_sourcecounts_from_openmc(self):
        """ Receive sourcecounts from all OpenMC seeds and bank seed-averaged
        data to `self._sourcecounts`

        """
        source_data_shape = np.prod(self._indices[0:3]), self._indices[3]
        all_sourcecount_data = np.empty((self._n_seeds,) + source_data_shape,
                                  dtype=np.float64)
        for i in range(self._n_seeds):
            source_data = np.empty(source_data_shape, dtype=np.float64)
            status = MPI.Status()
            self._global_comm.Recv(source_data, source=MPI.ANY_SOURCE, status=status, tag=1)
            source = status.Get_source()
            if self._verbosity >= 2:
                dest = self._global_comm.Get_rank()
                outstr = "{:>11s}Process {} received source data from process {}"
                print(outstr.format('', dest, source))
                sys.stdout.flush()
            seed_idx = int((source-self._n_procs_per_seed)/self._n_procs_per_seed)
            all_sourcecount_data[seed_idx,:] = source_data

        # Compute seed-averaged sourcecounts
        self._sourcecounts = np.sum(all_sourcecount_data, axis=0)/self._n_seeds

    def _recv_openmc_timing_stats(self):
        """ Receive timing stats from all OpenMC seeds

        Returns
        -------
        numpy.ndarray
            Timing data by seed

        """
        timing_data_size = 4  # Receiving 4 time entries from each OpenMC seed
        all_timing_data = np.empty([self._n_seeds, timing_data_size],
                                  dtype=np.float64)
        for i in range(self._n_seeds):
            timing_data = np.empty(timing_data_size, dtype=np.float64)
            status = MPI.Status()
            self._global_comm.Recv(timing_data, source=MPI.ANY_SOURCE, status=status, tag=2)
            source = status.Get_source()
            if self._verbosity >= 2:
                dest = self._global_comm.Get_rank()
                outstr = "{:>11s}Process {} received timing data from process {}"
                print(outstr.format('', dest, source))
                sys.stdout.flush()
            seed_idx = int((source-self._n_procs_per_seed)/self._n_procs_per_seed)
            all_timing_data[seed_idx,:] = timing_data
        return all_timing_data

    def _compute_xs(self, seed_avg_tally_data):
        """Takes seed-averaged CMFD tallies from OpenMC node and computes
        macroscopic cross sections, flux, and diffusion coefficients for each
        mesh cell using a tally window scheme

        """
        # Update window size for expanding window if necessary
        num_cmfd_batches = self._current_batch - self._tally_begin + 1
        if (self._window_type == 'expanding' and
                num_cmfd_batches == self._window_size * 2 and
                self._window_size * 2 <= self._max_window_size):
            self._window_size *= 2

        # Discard tallies from oldest batch if window limit reached
        tally_windows = self._flux_rate.shape[-1] + 1
        if tally_windows > self._window_size:
            self._flux_rate = self._flux_rate[...,1:]
            self._total_rate = self._total_rate[...,1:]
            self._p1scatt_rate = self._p1scatt_rate[...,1:]
            self._scatt_rate = self._scatt_rate[...,1:]
            self._nfiss_rate = self._nfiss_rate[...,1:]
            self._current_rate = self._current_rate[...,1:]
            self._openmc_src_rate = self._openmc_src_rate[...,1:]
            tally_windows -= 1

        # Extract spatial and energy indices
        nx, ny, nz, ng = self._indices

        # Set conditional numpy array as boolean vector based on coremap
        is_accel = self._coremap != _CMFD_NOACCEL

        # Get flux from seed-averaged data
        flux = seed_avg_tally_data[self._flux_slice]

        # Define target tally reshape dimensions. This defines how openmc
        # tallies are ordered by dimension
        target_tally_shape = [nz, ny, nx, ng, 1]

        # Reshape flux array to target shape. Swap x and z axes so that
        # flux shape is now [nx, ny, nz, ng, 1]
        reshape_flux = np.swapaxes(flux.reshape(target_tally_shape), 0, 2)

        # Flip energy axis as tally results are given in reverse order of
        # energy group
        reshape_flux = np.flip(reshape_flux, axis=3)

        # Bank flux to flux_rate
        self._flux_rate = np.append(self._flux_rate, reshape_flux, axis=4)

        # Compute flux as aggregate of banked flux_rate over tally window
        self._flux = np.where(is_accel[..., np.newaxis],
                              np.sum(self._flux_rate, axis=4), 0.0)

        # Detect zero flux, abort if located and cmfd is on
        zero_flux = np.logical_and(self._flux < _TINY_BIT,
                                   is_accel[..., np.newaxis])
        if np.any(zero_flux) and self._current_batch >= self._solver_begin:
            # Get index of first zero flux in flux array
            idx = np.argwhere(zero_flux)[0]

            # Throw error message (one-based indexing)
            # Index of group is flipped
            err_message = 'Detected zero flux without coremap overlay' + \
                          ' at mesh: (' + \
                          ', '.join(str(i+1) for i in idx[:-1]) + \
                          ') in group ' + str(ng-idx[-1])
            raise OpenMCError(err_message)

        # Get total reaction rate from seed averaged data
        totalrr = seed_avg_tally_data[self._total_slice]

        # Reshape totalrr array to target shape. Swap x and z axes so that
        # shape is now [nx, ny, nz, ng, 1]
        reshape_totalrr = np.swapaxes(totalrr.reshape(target_tally_shape),
                                      0, 2)

        # Total rr is flipped in energy axis as tally results are given in
        # reverse order of energy group
        reshape_totalrr = np.flip(reshape_totalrr, axis=3)

        # Bank total rr to total_rate
        self._total_rate = np.append(self._total_rate, reshape_totalrr,
                                     axis=4)

        # Compute total xs as aggregate of banked total_rate over tally window
        # divided by flux
        self._totalxs = np.divide(np.sum(self._total_rate, axis=4),
                                  self._flux, where=self._flux > 0,
                                  out=np.zeros_like(self._totalxs))

        # Get scattering reaction rate from seed averaged data
        scattrr = seed_avg_tally_data[self._scatt_slice]

        # Define target tally reshape dimensions for xs with incoming
        # and outgoing energies
        target_tally_shape = [nz, ny, nx, ng, ng, 1]

        # Reshape scattrr array to target shape. Swap x and z axes so that
        # shape is now [nx, ny, nz, ng, ng, 1]
        reshape_scattrr = np.swapaxes(scattrr.reshape(target_tally_shape),
                                      0, 2)

        # Scattering rr is flipped in both incoming and outgoing energy axes
        # as tally results are given in reverse order of energy group
        reshape_scattrr = np.flip(reshape_scattrr, axis=3)
        reshape_scattrr = np.flip(reshape_scattrr, axis=4)

        # Bank scattering rr to scatt_rate
        self._scatt_rate = np.append(self._scatt_rate, reshape_scattrr,
                                     axis=5)

        # Compute scattering xs as aggregate of banked scatt_rate over tally
        # window divided by flux. Flux dimensionality increased to account for
        # extra dimensionality of scattering xs
        extended_flux = self._flux[:,:,:,:,np.newaxis]
        self._scattxs = np.divide(np.sum(self._scatt_rate, axis=5),
                                  extended_flux, where=extended_flux > 0,
                                  out=np.zeros_like(self._scattxs))

        # Get nu-fission reaction rate and num realizations from seed averaged
        # data
        nfissrr = seed_avg_tally_data[self._nfiss_slice]
        num_realizations = seed_avg_tally_data[-2]

        # Reshape nfissrr array to target shape. Swap x and z axes so that
        # shape is now [nx, ny, nz, ng, ng, 1]
        reshape_nfissrr = np.swapaxes(nfissrr.reshape(target_tally_shape),
                                      0, 2)

        # Nu-fission rr is flipped in both incoming and outgoing energy axes
        # as tally results are given in reverse order of energy group
        reshape_nfissrr = np.flip(reshape_nfissrr, axis=3)
        reshape_nfissrr = np.flip(reshape_nfissrr, axis=4)

        # Bank nu-fission rr to nfiss_rate
        self._nfiss_rate = np.append(self._nfiss_rate, reshape_nfissrr,
                                     axis=5)

        # Compute nu-fission xs as aggregate of banked nfiss_rate over tally
        # window divided by flux. Flux dimensionality increased to account for
        # extra dimensionality of nu-fission xs
        self._nfissxs = np.divide(np.sum(self._nfiss_rate, axis=5),
                                  extended_flux, where=extended_flux > 0,
                                  out=np.zeros_like(self._nfissxs))

        # Openmc source distribution is sum of nu-fission rr in incoming
        # energies
        openmc_src = np.sum(reshape_nfissrr, axis=3)

        # Bank OpenMC source distribution from current batch to
        # openmc_src_rate
        self._openmc_src_rate = np.append(self._openmc_src_rate, openmc_src,
                                          axis=4)

        # Compute source distribution over entire tally window
        self._openmc_src = np.sum(self._openmc_src_rate, axis=4)

        # Compute k_eff from source distribution
        self._keff_bal = (np.sum(self._openmc_src) / num_realizations /
                          tally_windows)

        # Normalize openmc source distribution
        self._openmc_src /= np.sum(self._openmc_src) * self._norm

        # Get surface currents from seed averaged data
        current = seed_avg_tally_data[self._current_slice]

        # Define target tally reshape dimensions for current
        target_tally_shape = [nz, ny, nx, 12, ng, 1]

        # Reshape current array to target shape. Swap x and z axes so that
        # shape is now [nx, ny, nz, 12, ng, 1]
        reshape_current = np.swapaxes(current.reshape(target_tally_shape),
                                      0, 2)

        # Current is flipped in energy axis as tally results are given in
        # reverse order of energy group
        reshape_current = np.flip(reshape_current, axis=4)

        # Bank current to current_rate
        self._current_rate = np.append(self._current_rate, reshape_current,
                                       axis=5)

        # Compute current as aggregate of banked current_rate over tally window
        self._current = np.where(is_accel[..., np.newaxis, np.newaxis],
                                 np.sum(self._current_rate, axis=5), 0.0)

        if self._set_reference_params:
            # Set diffusion coefficients based on reference value
            self._diffcof = np.where(self._flux > 0,
                                     self._ref_d[None, None, None, :], 0.0)
        else:
            # Get p1 scatter reaction rate from seed averaged data
            p1scattrr = seed_avg_tally_data[self._p1scatt_slice]

            # Define target tally reshape dimensions for p1 scatter tally
            target_tally_shape = [nz, ny, nx, 2, ng, 1]

            # Reshape and extract only p1 data from tally results as there is
            # no need for p0 data
            reshape_p1scattrr = np.swapaxes(p1scattrr.reshape(target_tally_shape),
                                            0, 2)[:,:,:,1,:,:]

            # p1-scatter rr is flipped in energy axis as tally results are given in
            # reverse order of energy group
            reshape_p1scattrr = np.flip(reshape_p1scattrr, axis=3)

            # Bank p1-scatter rr to p1scatt_rate
            self._p1scatt_rate = np.append(self._p1scatt_rate, reshape_p1scattrr,
                                           axis=4)

            # Compute p1-scatter xs as aggregate of banked p1scatt_rate over tally
            # window divided by flux
            self._p1scattxs = np.divide(np.sum(self._p1scatt_rate, axis=4),
                                        self._flux, where=self._flux > 0,
                                        out=np.zeros_like(self._p1scattxs))

            # Calculate and store diffusion coefficient
            with np.errstate(divide='ignore', invalid='ignore'):
                self._diffcof = np.where(self._flux > 0, 1.0 / (3.0 *
                                         (self._totalxs-self._p1scattxs)), 0.)

        # Store seed averaged k-effective
        self._openmc_keff = seed_avg_tally_data[-1]

    def _execute_cmfd(self):
        """ Runs CMFD calculation on master node """
        # Start CMFD timer
        time_start_cmfd = time.time()

        if self._current_batch >= self._tally_begin:
            openmc_tallies = self._recv_tallies_from_openmc()
            time_stop_wait_tallies = time.time()
            self._time_waittallies += time_stop_wait_tallies - time_start_cmfd
            self._compute_xs(openmc_tallies)

        if self._current_batch >= self._solver_begin:
            # Create CMFD data based on OpenMC tallies
            self._set_up_cmfd()

            # Call solver
            self._cmfd_solver_execute()

            # Store k-effective
            self._k_cmfd.append(self._keff)

            # Calculate fission source
            self._calc_fission_source()

            # Receive sourcecounts from OpenMC
            time_start_recv_sourcecounts = time.time()
            self._recv_sourcecounts_from_openmc()
            time_stop_recv_sourcecounts = time.time()
            self._time_waitsrccnts += (time_stop_recv_sourcecounts -
                                       time_start_recv_sourcecounts)

            # Update source weightfactors and broadcast to all OpenMC processes
            self._update_weightfactors()

        # Stop CMFD timer
        time_stop_cmfd = time.time()
        self._time_cmfdnode += time_stop_cmfd - time_start_cmfd
        if self._current_batch >= self._solver_begin:
            # Write CMFD output if CMFD on for current batch
            self._write_cmfd_output()

    def _set_up_cmfd(self):
        """Configures CMFD object for a CMFD eigenvalue calculation """
        # Compute effective downscatter cross section
        if self._downscatter:
            self._compute_effective_downscatter()

        # Check neutron balance
        self._neutron_balance()

        # Calculate dtilde
        self._compute_dtilde()

        # Calculate dhat
        self._compute_dhat()

    def _cmfd_solver_execute(self):
        """ Sets up and runs power iteration solver for CMFD """
        # Start timer for build
        time_start_buildcmfd = time.time()

        # Build the loss and production matrices
        loss = self._build_loss_matrix()
        prod = self._build_prod_matrix()

        # Stop timer for build
        time_stop_buildcmfd = time.time()
        self._time_cmfdbuild += time_stop_buildcmfd - time_start_buildcmfd

        # Begin power iteration
        time_start_solvecmfd = time.time()
        phi, keff, dom = self._execute_power_iter(loss, prod)
        time_stop_solvecmfd = time.time()
        self._time_cmfdsolve += time_stop_solvecmfd - time_start_solvecmfd

        # Save results, normalizing phi to sum to 1
        self._keff = keff
        self._phi = phi/np.sqrt(np.sum(phi*phi))

        self._dom.append(dom)

    def _calc_fission_source(self):
        """ Calculates CMFD fission source from CMFD flux. If a coremap is
        defined, there will be a discrepancy between the spatial indices in the
        variables ``phi`` and ``nfissxs``, so ``phi`` needs to be mapped to the
        spatial indices of the cross sections. This can be done in a vectorized
        numpy manner or with for loops

        """
        # Extract number of groups and number of accelerated regions
        nx, ny, nz, ng = self._indices
        n = self._mat_dim

        # Compute cmfd_src in a vecotorized manner by phi to the spatial
        # indices of the actual problem so that cmfd_flux can be multiplied by
        # nfissxs

        # Calculate volume
        vol = np.product(self._hxyz, axis=3)

        # Reshape phi by number of groups
        phi = self._phi.reshape((n, ng))

        # Extract indices of coremap that are accelerated
        idx = self._accel_idxs

        # Initialize CMFD flux map that maps phi to actual spatial and
        # group indices of problem
        cmfd_flux = np.zeros((nx, ny, nz, ng))

        # Loop over all groups and set CMFD flux based on indices of
        # coremap and values of phi
        for g in range(ng):
            phi_g = phi[:,g]
            cmfd_flux[idx + (g,)] = phi_g[self._coremap[idx]]

        # Compute fission source
        cmfd_src = (np.sum(self._nfissxs[:,:,:,:,:] *
                    cmfd_flux[:,:,:,:,np.newaxis], axis=3) *
                    vol[:,:,:,np.newaxis])

        # Normalize source such that it sums to 1.0
        self._cmfd_src = cmfd_src / np.sum(cmfd_src)

        # Compute entropy
        if openmc.lib.settings.entropy_on:
            # Compute source times log_2(source)
            source = self._cmfd_src[self._cmfd_src > 0] \
                * np.log(self._cmfd_src[self._cmfd_src > 0])/np.log(2)

            # Sum source and store
            self._entropy.append(-1.0 * np.sum(source))

        # Normalize source so average is 1.0
        self._cmfd_src = self._cmfd_src/np.sum(self._cmfd_src) * self._norm

        # Calculate differences between normalized sources
        self._src_cmp.append(np.sqrt(1.0 / self._norm
                             * np.sum((self._cmfd_src - self._openmc_src)**2)))

    def _update_weightfactors(self):
        """ Performs weighting of particles in source bank
        and broadcasts weights to all OpenMC processes

        """
        # Get spatial dimensions and energy groups
        nx, ny, nz, ng = self._indices

        # Compute normalization factor
        norm = np.sum(self._sourcecounts) / np.sum(self._cmfd_src)

        # Define target reshape dimensions for sourcecounts. This
        # defines how self._sourcecounts is ordered by dimension
        target_shape = [nz, ny, nx, ng]

        # Reshape sourcecounts to target shape. Swap x and z axes so
        # that the shape is now [nx, ny, nz, ng]
        sourcecounts = np.swapaxes(
                self._sourcecounts.reshape(target_shape), 0, 2)

        # Flip index of energy dimension
        sourcecounts = np.flip(sourcecounts, axis=3)

        # Compute weight factors
        div_condition = np.logical_and(sourcecounts > 0,
                                       self._cmfd_src > 0)
        self._weightfactors = (np.divide(self._cmfd_src * norm,
                               sourcecounts, where=div_condition,
                               out=np.ones_like(self._cmfd_src),
                               dtype=np.float32))

        # Apply damping factor
        self._weightfactors = (1.0 - (1.0 - self._weightfactors) *
                               self._damping_factor)

        # Broadcast weight factors to all procs
        time_start_sendweightfactors = time.time()
        for i in range(self._n_seeds*self._n_procs_per_seed):
            dest = i + self._n_procs_per_seed
            self._global_comm.Send(self._weightfactors, dest=dest)
            if self._verbosity >= 2:
                source = self._global_comm.Get_rank()
                outstr = "{:>11s}Sending weight factors from process {} to {}"
                print(outstr.format('', source, dest))
                sys.stdout.flush()
        time_stop_sendweightfactors = time.time()
        self._time_sendweightfactors += (time_stop_sendweightfactors -
                                         time_start_sendweightfactors)

    def _build_loss_matrix(self):
        """ Builds loss matrix in CMFD calculation

        Returns
        -------
        loss : numpy.ndarray
            Loss matrix

        """
        # Extract spatial and energy indices and define matrix dimension
        ng = self._indices[3]
        n = self._mat_dim*ng

        # Define data entries used to build csr matrix
        data = np.array([])

        dtilde_left = self._dtilde[:,:,:,:,0]
        dtilde_right = self._dtilde[:,:,:,:,1]
        dtilde_back = self._dtilde[:,:,:,:,2]
        dtilde_front = self._dtilde[:,:,:,:,3]
        dtilde_bottom = self._dtilde[:,:,:,:,4]
        dtilde_top = self._dtilde[:,:,:,:,5]
        dhat_left = self._dhat[:,:,:,:,0]
        dhat_right = self._dhat[:,:,:,:,1]
        dhat_back = self._dhat[:,:,:,:,2]
        dhat_front = self._dhat[:,:,:,:,3]
        dhat_bottom = self._dhat[:,:,:,:,4]
        dhat_top = self._dhat[:,:,:,:,5]

        dx = self._hxyz[:,:,:,np.newaxis,0]
        dy = self._hxyz[:,:,:,np.newaxis,1]
        dz = self._hxyz[:,:,:,np.newaxis,2]

        # Define net leakage coefficient for each surface in each matrix
        # element
        jnet = (((dtilde_right + dhat_right)-(-1.0 * dtilde_left + dhat_left))
                / dx +
                ((dtilde_front + dhat_front)-(-1.0 * dtilde_back + dhat_back))
                / dy +
                ((dtilde_top + dhat_top)-(-1.0 * dtilde_bottom + dhat_bottom))
                / dz)

        for g in range(ng):
            # Define leakage terms that relate terms to their neighbors to the
            # left
            dtilde = self._dtilde[:,:,:,g,0][self._accel_neig_left_idxs]
            dhat = self._dhat[:,:,:,g,0][self._accel_neig_left_idxs]
            dx = self._hxyz[:,:,:,0][self._accel_neig_left_idxs]
            vals = (-1.0 * dtilde - dhat) / dx
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define leakage terms that relate terms to their neighbors to the
            # right
            dtilde = self._dtilde[:,:,:,g,1][self._accel_neig_right_idxs]
            dhat = self._dhat[:,:,:,g,1][self._accel_neig_right_idxs]
            dx = self._hxyz[:,:,:,0][self._accel_neig_right_idxs]
            vals = (-1.0 * dtilde + dhat) / dx
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define leakage terms that relate terms to their neighbors in the
            # back
            dtilde = self._dtilde[:,:,:,g,2][self._accel_neig_back_idxs]
            dhat = self._dhat[:,:,:,g,2][self._accel_neig_back_idxs]
            dy = self._hxyz[:,:,:,1][self._accel_neig_back_idxs]
            vals = (-1.0 * dtilde - dhat) / dy
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define leakage terms that relate terms to their neighbors in the
            # front
            dtilde = self._dtilde[:,:,:,g,3][self._accel_neig_front_idxs]
            dhat = self._dhat[:,:,:,g,3][self._accel_neig_front_idxs]
            dy = self._hxyz[:,:,:,1][self._accel_neig_front_idxs]
            vals = (-1.0 * dtilde + dhat) / dy
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define leakage terms that relate terms to their neighbors to the
            # bottom
            dtilde = self._dtilde[:,:,:,g,4][self._accel_neig_bot_idxs]
            dhat = self._dhat[:,:,:,g,4][self._accel_neig_bot_idxs]
            dz = self._hxyz[:,:,:,2][self._accel_neig_bot_idxs]
            vals = (-1.0 * dtilde - dhat) / dz
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define leakage terms that relate terms to their neighbors to the
            # top
            dtilde = self._dtilde[:,:,:,g,5][self._accel_neig_top_idxs]
            dhat = self._dhat[:,:,:,g,5][self._accel_neig_top_idxs]
            dz = self._hxyz[:,:,:,2][self._accel_neig_top_idxs]
            vals = (-1.0 * dtilde + dhat) / dz
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define terms that relate to loss of neutrons in a cell. These
            # correspond to all the diagonal entries of the loss matrix
            jnet_g = jnet[:,:,:,g][self._accel_idxs]
            total_xs = self._totalxs[:,:,:,g][self._accel_idxs]
            scatt_xs = self._scattxs[:,:,:,g,g][self._accel_idxs]
            vals = jnet_g + total_xs - scatt_xs
            # Store data to add to CSR matrix
            data = np.append(data, vals)

            # Define terms that relate to in-scattering from group to group.
            # These terms relate a mesh index to all mesh indices with the same
            # spatial dimensions but belong to a different energy group
            for h in range(ng):
                if h != g:
                    scatt_xs = self._scattxs[:,:,:,h,g][self._accel_idxs]
                    vals = -1.0 * scatt_xs
                    # Store data to add to CSR matrix
                    data = np.append(data, vals)

        # Create csr matrix
        loss_row = self._loss_row
        loss_col = self._loss_col
        loss = sparse.csr_matrix((data, (loss_row, loss_col)), shape=(n, n))
        return loss

    def _build_prod_matrix(self):
        """ Builds production matrix in CMFD calculation

        Returns
        -------
        prod : numpy.ndarray
            Production matrix

        """
        # Extract spatial and energy indices and define matrix dimension
        ng = self._indices[3]
        n = self._mat_dim*ng

        # Define rows, columns, and data used to build csr matrix
        data = np.array([])

        # Define terms that relate to fission production from group to group.
        for g in range(ng):
            for h in range(ng):
                # Get nu-fission macro xs
                vals = (self._nfissxs[:, :, :, h, g])[self._accel_idxs]
                # Store rows, cols, and data to add to CSR matrix
                data = np.append(data, vals)

        # Create csr matrix
        prod_row = self._prod_row
        prod_col = self._prod_col
        prod = sparse.csr_matrix((data, (prod_row, prod_col)), shape=(n, n))
        return prod

    def _execute_power_iter(self, loss, prod):
        """ Main power iteration routine for the CMFD calculation

        Parameters
        ----------
        loss : scipy.sparse.spmatrix
            Sparse matrix storing elements of CMFD loss matrix
        prod : scipy.sparse.spmatrix
            Sparse matrix storing elements of CMFD production matrix

        Returns
        -------
        phi_n : numpy.ndarray
            Flux vector of CMFD problem
        k_n : float
            Eigenvalue of CMFD problem
        dom : float
            Dominance ratio of CMFD problem

        """
        # Get problem size
        n = loss.shape[0]

        # Set up tolerances for C++ solver
        atoli = self._gauss_seidel_tolerance[0]
        rtoli = self._gauss_seidel_tolerance[1]
        toli = rtoli * 100

        # Set up flux vectors, intital guess set to 1
        phi_n = np.ones((n,))
        phi_o = np.ones((n,))

        # Set up source vectors
        s_n = np.zeros((n,))
        s_o = np.zeros((n,))

        # Set initial guess
        k_n = self._openmc_keff
        k_o = k_n
        dw = self._w_shift
        k_s = k_o + dw
        k_ln = 1.0/(1.0/k_n - 1.0/k_s)
        k_lo = k_ln

        # Set norms to 0
        norm_n = 0.0
        norm_o = 0.0

        # Maximum number of power iterations
        maxits = 10000

        # Perform Wielandt shift
        loss -= 1.0/k_s*prod

        # Begin power iteration
        for i in range(maxits):
            # Check if reach max number of iterations
            if i == maxits - 1:
                raise OpenMCError('Reached maximum iterations in CMFD power '
                                  'iteration solver.')

            # Compute source vector
            s_o = prod.dot(phi_o)

            # Normalize source vector
            s_o /= k_lo

            # Compute new flux with C++ solver
            innerits = openmc.lib._dll.openmc_run_linsolver(loss.data, s_o,
                                                             phi_n, toli)

            # Compute new source vector
            s_n = prod.dot(phi_n)

            # Compute new shifted eigenvalue
            k_ln = np.sum(s_n) / np.sum(s_o)

            # Compute new eigenvalue
            k_n = 1.0/(1.0/k_ln + 1.0/k_s)

            # Renormalize the old source
            s_o *= k_lo

            # Check convergence
            iconv, norm_n = self._check_convergence(s_n, s_o, k_n, k_o, i+1,
                                                    innerits)

            # If converged, calculate dominance ratio and break from loop
            if iconv:
                dom = norm_n / norm_o
                return phi_n, k_n, dom

            # Record old values if not converged
            phi_o = phi_n
            k_o = k_n
            k_lo = k_ln
            norm_o = norm_n

            # Update tolerance for inner iterations
            toli = max(atoli, rtoli*norm_n)

    def _check_convergence(self, s_n, s_o, k_n, k_o, iter, innerits):
        """ Checks the convergence of the CMFD problem

        Parameters
        ----------
        s_n : numpy.ndarray
            Source vector from current iteration
        s_o : numpy.ndarray
            Source vector from previous iteration
        k_n : float
            K-effective  from current iteration
        k_o : float
            K-effective from previous iteration
        iter: int
            Iteration number
        innerits: int
            Number of iterations required for convergence in inner GS loop

        Returns
        -------
        iconv : bool
            Whether the power iteration has reached convergence
        serr : float
            Error in source from previous iteration to current iteration, used
            for dominance ratio calculations

        """
        # Calculate error in keff
        kerr = abs(k_o - k_n) / k_n

        # Calculate max error in source
        with np.errstate(divide='ignore', invalid='ignore'):
            serr = np.sqrt(np.sum(np.where(s_n > 0, ((s_n-s_o) / s_n)**2, 0))
                           / len(s_n))

        # Check for convergence
        iconv = kerr < self._cmfd_ktol and serr < self._stol

        return iconv, serr

    def _set_tally_window(self):
        """ Set parameters to handle different tally window options """
        # Set parameters for window_type equal to "expanding" or "none"
        if self._window_type != 'rolling':
            self._window_size = 1

    def _set_coremap(self):
        """ Sets the core mapping information. All regions marked with zero
        are set to CMFD_NOACCEL, while all regions marked with 1 are set to a
        unique index that maps each fuel region to a row number when building
        CMFD matrices

        """
        # Set number of accelerated regions in problem. This will be related to
        # the dimension of CMFD matrices
        self._mat_dim = np.sum(self._coremap)

        # Define coremap as cumulative sum over accelerated regions,
        # otherwise set value to _CMFD_NOACCEL
        self._coremap = np.where(self._coremap == 0, _CMFD_NOACCEL,
                                 np.cumsum(self._coremap)-1)

        # Reshape coremap to three dimensional array
        # Indices of coremap in user input switched in x and z axes
        nx, ny, nz = self._indices[:3]
        self._coremap = self._coremap.reshape(nz, ny, nx)
        self._coremap = np.swapaxes(self._coremap, 0, 2)

    def _compute_effective_downscatter(self):
        """ Changes downscatter rate for zero upscatter """
        # Extract energy index
        ng = self._indices[3]

        # Return if not two groups
        if ng != 2:
            return

        # Extract cross sections and flux for each group
        flux1 = self._flux[:,:,:,0]
        flux2 = self._flux[:,:,:,1]
        sigt1 = self._totalxs[:,:,:,0]
        sigt2 = self._totalxs[:,:,:,1]

        # First energy index is incoming energy, second is outgoing energy
        sigs11 = self._scattxs[:,:,:,0,0]
        sigs21 = self._scattxs[:,:,:,1,0]
        sigs12 = self._scattxs[:,:,:,0,1]
        sigs22 = self._scattxs[:,:,:,1,1]

        # Compute absorption xs
        siga1 = sigt1 - sigs11 - sigs12
        siga2 = sigt2 - sigs22 - sigs21

        # Compute effective downscatter XS
        sigs12_eff = sigs12 - sigs21 * np.divide(flux2, flux1,
                                                 where=flux1 > 0,
                                                 out=np.zeros_like(flux2))

        # Recompute total cross sections and record
        self._totalxs[:,:,:,0] = siga1 + sigs11 + sigs12_eff
        self._totalxs[:,:,:,1] = siga2 + sigs22

        # Record effective dowmscatter xs
        self._scattxs[:,:,:,0,1] = sigs12_eff

        # Zero out upscatter cross section
        self._scattxs[:,:,:,1,0] = 0.0

    def _neutron_balance(self):
        """ Computes the RMS neutron balance over the CMFD mesh """
        # Extract energy indices
        ng = self._indices[3]

        # Get number of accelerated regions
        num_accel = self._mat_dim

        # Get openmc k-effective
        keff = self._openmc_keff

        # Define leakage in each mesh cell and energy group
        leakage = (((self._current[:,:,:,_CURRENTS['out_right'],:] -
                   self._current[:,:,:,_CURRENTS['in_right'],:]) -
                   (self._current[:,:,:,_CURRENTS['in_left'],:] -
                   self._current[:,:,:,_CURRENTS['out_left'],:])) +
                   ((self._current[:,:,:,_CURRENTS['out_front'],:] -
                    self._current[:,:,:,_CURRENTS['in_front'],:]) -
                   (self._current[:,:,:,_CURRENTS['in_back'],:] -
                    self._current[:,:,:,_CURRENTS['out_back'],:])) +
                   ((self._current[:,:,:,_CURRENTS['out_top'],:] -
                    self._current[:,:,:,_CURRENTS['in_top'],:]) -
                   (self._current[:,:,:,_CURRENTS['in_bottom'],:] -
                    self._current[:,:,:,_CURRENTS['out_bottom'],:])))

        # Compute total rr
        interactions = self._totalxs * self._flux

        # Compute scattering rr by broadcasting flux in outgoing energy and
        # summing over incoming energy
        scattering = np.sum(self._scattxs * self._flux[:,:,:,:, np.newaxis],
                            axis=3)

        # Compute fission rr by broadcasting flux in outgoing energy and
        # summing over incoming energy
        fission = np.sum(self._nfissxs * self._flux[:,:,:,:, np.newaxis],
                         axis=3)

        # Compute residual
        res = leakage + interactions - scattering - (1.0 / keff) * fission

        # Normalize res by flux and bank res
        self._resnb = np.divide(res, self._flux, where=self._flux > 0,
                                out=np.zeros_like(self._flux))

        # Calculate RMS and record for this batch
        self._balance.append(np.sqrt(
            np.sum(np.multiply(self._resnb, self._resnb)) /
            (ng * num_accel)))

    def _precompute_array_indices(self):
        """ Initializes cross section arrays and computes the indices
        used to populate dtilde and dhat

        """
        # Extract spatial indices
        nx, ny, nz, ng = self._indices

        # Allocate dimensions for each mesh cell
        self._hxyz = np.zeros((nx, ny, nz, 3))
        self._hxyz[:] = self._mesh.width

        # Allocate flux, cross sections and diffusion coefficient
        self._flux = np.zeros((nx, ny, nz, ng))
        self._totalxs = np.zeros((nx, ny, nz, ng))
        self._p1scattxs = np.zeros((nx, ny, nz, ng))
        self._scattxs = np.zeros((nx, ny, nz, ng, ng))  # Incoming, outgoing
        self._nfissxs = np.zeros((nx, ny, nz, ng, ng))  # Incoming, outgoing
        self._diffcof = np.zeros((nx, ny, nz, ng))

        # Allocate dtilde and dhat
        self._dtilde = np.zeros((nx, ny, nz, ng, 6))
        self._dhat = np.zeros((nx, ny, nz, ng, 6))

        # Logical for determining whether region of interest is accelerated
        # region
        is_accel = self._coremap != _CMFD_NOACCEL
        # Logical for determining whether a zero flux "albedo" b.c. should be
        # applied
        is_zero_flux_alb = abs(self._albedo - _ZERO_FLUX) < _TINY_BIT
        x_inds, y_inds, z_inds = np.indices((nx, ny, nz))

        # Define slice equivalent to is_accel[0,:,:]
        slice_x = x_inds[:1,:,:]
        slice_y = y_inds[:1,:,:]
        slice_z = z_inds[:1,:,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._first_x_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                               slice_z[bndry_accel])

        # Define slice equivalent to is_accel[-1,:,:]
        slice_x = x_inds[-1:,:,:]
        slice_y = y_inds[-1:,:,:]
        slice_z = z_inds[-1:,:,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._last_x_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                              slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,0,:]
        slice_x = x_inds[:,:1,:]
        slice_y = y_inds[:,:1,:]
        slice_z = z_inds[:,:1,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._first_y_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                               slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,-1,:]
        slice_x = x_inds[:,-1:,:]
        slice_y = y_inds[:,-1:,:]
        slice_z = z_inds[:,-1:,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._last_y_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                              slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,:,0]
        slice_x = x_inds[:,:,:1]
        slice_y = y_inds[:,:,:1]
        slice_z = z_inds[:,:,:1]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._first_z_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                               slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,:,-1]
        slice_x = x_inds[:,:,-1:]
        slice_y = y_inds[:,:,-1:]
        slice_z = z_inds[:,:,-1:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._last_z_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                              slice_z[bndry_accel])

        # Define slice equivalent to is_accel[1:,:,:]
        slice_x = x_inds[1:,:,:]
        slice_y = y_inds[1:,:,:]
        slice_z = z_inds[1:,:,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._notfirst_x_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                                  slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:-1,:,:]
        slice_x = x_inds[:-1,:,:]
        slice_y = y_inds[:-1,:,:]
        slice_z = z_inds[:-1,:,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._notlast_x_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                                 slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,1:,:]
        slice_x = x_inds[:,1:,:]
        slice_y = y_inds[:,1:,:]
        slice_z = z_inds[:,1:,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._notfirst_y_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                                  slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,:-1,:]
        slice_x = x_inds[:,:-1,:]
        slice_y = y_inds[:,:-1,:]
        slice_z = z_inds[:,:-1,:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._notlast_y_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                                 slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,:,1:]
        slice_x = x_inds[:,:,1:]
        slice_y = y_inds[:,:,1:]
        slice_z = z_inds[:,:,1:]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._notfirst_z_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                                  slice_z[bndry_accel])

        # Define slice equivalent to is_accel[:,:,:-1]
        slice_x = x_inds[:,:,:-1]
        slice_y = y_inds[:,:,:-1]
        slice_z = z_inds[:,:,:-1]
        bndry_accel = is_accel[(slice_x, slice_y, slice_z)]
        self._notlast_z_accel = (slice_x[bndry_accel], slice_y[bndry_accel],
                                 slice_z[bndry_accel])

        # Store logical for whether neighboring cell is reflector region
        # in all directions
        adj_reflector_left = np.roll(self._coremap, 1, axis=0) == _CMFD_NOACCEL
        self._is_adj_ref_left = adj_reflector_left[
                self._notfirst_x_accel + (np.newaxis,)]

        adj_reflector_right = np.roll(self._coremap, -1, axis=0) == \
            _CMFD_NOACCEL
        self._is_adj_ref_right = adj_reflector_right[
                self._notlast_x_accel + (np.newaxis,)]

        adj_reflector_back = np.roll(self._coremap, 1, axis=1) == \
            _CMFD_NOACCEL
        self._is_adj_ref_back = adj_reflector_back[
                self._notfirst_y_accel + (np.newaxis,)]

        adj_reflector_front = np.roll(self._coremap, -1, axis=1) == \
            _CMFD_NOACCEL
        self._is_adj_ref_front = adj_reflector_front[
                self._notlast_y_accel + (np.newaxis,)]

        adj_reflector_bottom = np.roll(self._coremap, 1, axis=2) == \
            _CMFD_NOACCEL
        self._is_adj_ref_bottom = adj_reflector_bottom[
                self._notfirst_z_accel + (np.newaxis,)]

        adj_reflector_top = np.roll(self._coremap, -1, axis=2) == \
            _CMFD_NOACCEL
        self._is_adj_ref_top = adj_reflector_top[
                self._notlast_z_accel + (np.newaxis,)]

        # Precompute array slices to extract tallies received from OpenMC Node
        tally_idx = 0
        total_tallies = nx*ny*nz*ng
        self._flux_slice = slice(tally_idx, tally_idx+total_tallies)

        tally_idx += total_tallies
        self._total_slice = slice(tally_idx, tally_idx+total_tallies)

        tally_idx += total_tallies
        total_tallies = nx*ny*nz*ng*ng
        self._scatt_slice = slice(tally_idx, tally_idx+total_tallies)

        tally_idx += total_tallies
        self._nfiss_slice = slice(tally_idx, tally_idx+total_tallies)

        tally_idx += total_tallies
        total_tallies = nx*ny*nz*ng*12
        self._current_slice = slice(tally_idx, tally_idx+total_tallies)

        if not self._set_reference_params:
            tally_idx += total_tallies
            total_tallies = nx*ny*nz*ng*2
            self._p1scatt_slice = slice(tally_idx, tally_idx+total_tallies)

        # Add 2 to total tally data size to account for keff and
        # num_realizations
        self._tally_data_size = tally_idx + total_tallies + 2

    def _precompute_matrix_indices(self):
        """ Computes the indices and row/column data used to populate CMFD CSR
        matrices. These indices are used in _build_loss_matrix and
        _build_prod_matrix.

        """
        # Extract energy group indices
        ng = self._indices[3]

        # Shift coremap in all directions to determine whether leakage term
        # should be defined for particular cell in matrix
        coremap_shift_left = np.pad(self._coremap, ((1,0),(0,0),(0,0)),
                                    mode='constant',
                                    constant_values=_CMFD_NOACCEL)[:-1,:,:]

        coremap_shift_right = np.pad(self._coremap, ((0,1),(0,0),(0,0)),
                                     mode='constant',
                                     constant_values=_CMFD_NOACCEL)[1:,:,:]

        coremap_shift_back = np.pad(self._coremap, ((0,0),(1,0),(0,0)),
                                    mode='constant',
                                    constant_values=_CMFD_NOACCEL)[:,:-1,:]

        coremap_shift_front = np.pad(self._coremap, ((0,0),(0,1),(0,0)),
                                     mode='constant',
                                     constant_values=_CMFD_NOACCEL)[:,1:,:]

        coremap_shift_bottom = np.pad(self._coremap, ((0,0),(0,0),(1,0)),
                                      mode='constant',
                                      constant_values=_CMFD_NOACCEL)[:,:,:-1]

        coremap_shift_top = np.pad(self._coremap, ((0,0),(0,0),(0,1)),
                                   mode='constant',
                                   constant_values=_CMFD_NOACCEL)[:,:,1:]

        # Create empty row and column vectors to store for loss matrix
        row = np.array([])
        col = np.array([])

        # Store all indices used to populate production and loss matrix
        is_accel = self._coremap != _CMFD_NOACCEL
        self._accel_idxs = np.where(is_accel)
        self._accel_neig_left_idxs = (np.where(is_accel &
                                      (coremap_shift_left != _CMFD_NOACCEL)))
        self._accel_neig_right_idxs = (np.where(is_accel &
                                       (coremap_shift_right != _CMFD_NOACCEL)))
        self._accel_neig_back_idxs = (np.where(is_accel &
                                      (coremap_shift_back != _CMFD_NOACCEL)))
        self._accel_neig_front_idxs = (np.where(is_accel &
                                       (coremap_shift_front != _CMFD_NOACCEL)))
        self._accel_neig_bot_idxs = (np.where(is_accel &
                                     (coremap_shift_bottom != _CMFD_NOACCEL)))
        self._accel_neig_top_idxs = (np.where(is_accel &
                                     (coremap_shift_top != _CMFD_NOACCEL)))

        for g in range(ng):
            # Extract row and column data of regions where a cell and its
            # neighbor to the left are both fuel regions
            idx_x = ng * (self._coremap[self._accel_neig_left_idxs]) + g
            idx_y = ng * (coremap_shift_left[self._accel_neig_left_idxs]) + g
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            # Extract row and column data of regions where a cell and its
            # neighbor to the right are both fuel regions
            idx_x = ng * (self._coremap[self._accel_neig_right_idxs]) + g
            idx_y = ng * (coremap_shift_right[self._accel_neig_right_idxs]) + g
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            # Extract row and column data of regions where a cell and its
            # neighbor to the back are both fuel regions
            idx_x = ng * (self._coremap[self._accel_neig_back_idxs]) + g
            idx_y = ng * (coremap_shift_back[self._accel_neig_back_idxs]) + g
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            # Extract row and column data of regions where a cell and its
            # neighbor to the front are both fuel regions
            idx_x = ng * (self._coremap[self._accel_neig_front_idxs]) + g
            idx_y = ng * (coremap_shift_front[self._accel_neig_front_idxs]) + g
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            # Extract row and column data of regions where a cell and its
            # neighbor to the bottom are both fuel regions
            idx_x = ng * (self._coremap[self._accel_neig_bot_idxs]) + g
            idx_y = ng * (coremap_shift_bottom[self._accel_neig_bot_idxs]) \
                + g
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            # Extract row and column data of regions where a cell and its
            # neighbor to the top are both fuel regions
            idx_x = ng * (self._coremap[self._accel_neig_top_idxs]) + g
            idx_y = ng * (coremap_shift_top[self._accel_neig_top_idxs]) + g
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            # Extract all regions where a cell is a fuel region
            idx_x = ng * (self._coremap[self._accel_idxs]) + g
            idx_y = idx_x
            row = np.append(row, idx_x)
            col = np.append(col, idx_y)

            for h in range(ng):
                if h != g:
                    # Extract all regions where a cell is a fuel region
                    idx_x = ng * (self._coremap[self._accel_idxs]) + g
                    idx_y = ng * (self._coremap[self._accel_idxs]) + h
                    row = np.append(row, idx_x)
                    col = np.append(col, idx_y)

        # Store row and col as rows and columns of production matrix
        self._loss_row = row
        self._loss_col = col

        # Create empty row and column vectors to store for production matrix
        row = np.array([], dtype=int)
        col = np.array([], dtype=int)

        for g in range(ng):
            for h in range(ng):
                # Extract all regions where a cell is a fuel region
                idx_x = ng * (self._coremap[self._accel_idxs]) + g
                idx_y = ng * (self._coremap[self._accel_idxs]) + h
                # Store rows, cols, and data to add to CSR matrix
                row = np.append(row, idx_x)
                col = np.append(col, idx_y)

        # Store row and col as rows and columns of production matrix
        self._prod_row = row
        self._prod_col = col

    def _compute_dtilde(self):
        """ Computes the diffusion coupling coefficient using a vectorized
        numpy approach. Aggregate values for the dtilde multidimensional array
        are populated by first defining values on the problem boundary, and
        then for all other regions. For indices not lying on a boundary, dtilde
        values are distinguished between regions that neighbor a reflector
        region and regions that don't neighbor a reflector

        """
        # Logical for determining whether a zero flux "albedo" b.c. should be
        # applied
        is_zero_flux_alb = abs(self._albedo - _ZERO_FLUX) < _TINY_BIT

        # Define dtilde at left surface for all mesh cells on left boundary
        # Separate between zero flux b.c. and alebdo b.c.
        boundary = self._first_x_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dx = self._hxyz[boundary + (np.newaxis, 0)]
        if is_zero_flux_alb[0]:
            self._dtilde[boundary_grps + (0,)] = 2.0 * D / dx
        else:
            alb = self._albedo[0]
            self._dtilde[boundary_grps + (0,)] = ((2.0 * D * (1.0 - alb))
                                                  / (4.0 * D * (1.0 + alb) +
                                                  (1.0 - alb) * dx))

        # Define dtilde at right surface for all mesh cells on right boundary
        # Separate between zero flux b.c. and alebdo b.c.
        boundary = self._last_x_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dx = self._hxyz[boundary + (np.newaxis, 0)]
        if is_zero_flux_alb[1]:
            self._dtilde[boundary_grps + (1,)] = 2.0 * D / dx
        else:
            alb = self._albedo[1]
            self._dtilde[boundary_grps + (1,)] = ((2.0 * D * (1.0 - alb))
                                                  / (4.0 * D * (1.0 + alb) +
                                                  (1.0 - alb) * dx))

        # Define dtilde at back surface for all mesh cells on back boundary
        # Separate between zero flux b.c. and alebdo b.c.
        boundary = self._first_y_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dy = self._hxyz[boundary + (np.newaxis, 1)]
        if is_zero_flux_alb[2]:
            self._dtilde[boundary_grps + (2,)] = 2.0 * D / dy
        else:
            alb = self._albedo[2]
            self._dtilde[boundary_grps + (2,)] = ((2.0 * D * (1.0 - alb))
                                                  / (4.0 * D * (1.0 + alb) +
                                                  (1.0 - alb) * dy))

        # Define dtilde at front surface for all mesh cells on front boundary
        # Separate between zero flux b.c. and alebdo b.c.
        boundary = self._last_y_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dy = self._hxyz[boundary + (np.newaxis, 1)]
        if is_zero_flux_alb[3]:
            self._dtilde[boundary_grps + (3,)] = 2.0 * D / dy
        else:
            alb = self._albedo[3]
            self._dtilde[boundary_grps + (3,)] = ((2.0 * D * (1.0 - alb))
                                                  / (4.0 * D * (1.0 + alb) +
                                                  (1.0 - alb) * dy))

        # Define dtilde at bottom surface for all mesh cells on bottom boundary
        # Separate between zero flux b.c. and alebdo b.c.
        boundary = self._first_z_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dz = self._hxyz[boundary + (np.newaxis, 2)]
        if is_zero_flux_alb[4]:
            self._dtilde[boundary_grps + (4,)] = 2.0 * D / dz
        else:
            alb = self._albedo[4]
            self._dtilde[boundary_grps + (4,)] = ((2.0 * D * (1.0 - alb))
                                                  / (4.0 * D * (1.0 + alb) +
                                                  (1.0 - alb) * dz))

        # Define dtilde at top surface for all mesh cells on top boundary
        # Separate between zero flux b.c. and alebdo b.c.
        boundary = self._last_z_accel
        boundary_grps = boundary + (slice(None),)

        D = self._diffcof[boundary_grps]
        dz = self._hxyz[boundary + (np.newaxis, 2)]
        if is_zero_flux_alb[5]:
            self._dtilde[boundary_grps + (5,)] = 2.0 * D / dz
        else:
            alb = self._albedo[5]
            self._dtilde[boundary_grps + (5,)] = ((2.0 * D * (1 - alb))
                                                  / (4.0 * D * (1.0 + alb) +
                                                  (1.0 - alb) * dz))

        # Define reflector albedo for all cells on the left surface, in case
        # a cell borders a reflector region on the left
        current_in_left = self._current[:,:,:,_CURRENTS['in_left'],:]
        current_out_left = self._current[:,:,:,_CURRENTS['out_left'],:]
        ref_albedo = np.divide(current_in_left, current_out_left,
                               where=current_out_left > 1.0e-10,
                               out=np.ones_like(current_out_left))

        # Diffusion coefficient of neighbor to left
        neig_dc = np.roll(self._diffcof, 1, axis=0)
        # Cell dimensions of neighbor to left
        neig_hxyz = np.roll(self._hxyz, 1, axis=0)

        # Define dtilde at left surface for all mesh cells not on left boundary
        # Dtilde is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notfirst_x_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dx = self._hxyz[boundary + (np.newaxis, 0)]
        neig_D = neig_dc[boundary_grps]
        neig_dx = neig_hxyz[boundary + (np.newaxis, 0)]
        alb = ref_albedo[boundary_grps]
        is_adj_ref = self._is_adj_ref_left
        dtilde = np.where(is_adj_ref, (2.0 * D * (1.0 - alb)) /
                          (4.0 * D * (1.0 + alb) + (1.0 - alb) * dx),
                          (2.0 * D * neig_D) / (neig_dx * D + dx * neig_D))
        self._dtilde[boundary_grps + (0,)] = dtilde

        # Define reflector albedo for all cells on the right surface, in case
        # a cell borders a reflector region on the right
        current_in_right = self._current[:,:,:,_CURRENTS['in_right'],:]
        current_out_right = self._current[:,:,:,_CURRENTS['out_right'],:]
        ref_albedo = np.divide(current_in_right, current_out_right,
                               where=current_out_right > 1.0e-10,
                               out=np.ones_like(current_out_right))

        # Diffusion coefficient of neighbor to right
        neig_dc = np.roll(self._diffcof, -1, axis=0)
        # Cell dimensions of neighbor to right
        neig_hxyz = np.roll(self._hxyz, -1, axis=0)

        # Define dtilde at right surface for all mesh cells not on right
        # boundary. Dtilde is defined differently for regions that do and don't
        # neighbor reflector regions
        boundary = self._notlast_x_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dx = self._hxyz[boundary + (np.newaxis, 0)]
        neig_D = neig_dc[boundary_grps]
        neig_dx = neig_hxyz[boundary + (np.newaxis, 0)]
        alb = ref_albedo[boundary_grps]
        is_adj_ref = self._is_adj_ref_right
        dtilde = np.where(is_adj_ref, (2.0 * D * (1.0 - alb)) /
                          (4.0 * D * (1.0 + alb) + (1.0 - alb) * dx),
                          (2.0 * D * neig_D) / (neig_dx * D + dx * neig_D))
        self._dtilde[boundary_grps + (1,)] = dtilde

        # Define reflector albedo for all cells on the back surface, in case
        # a cell borders a reflector region on the back
        current_in_back = self._current[:,:,:,_CURRENTS['in_back'],:]
        current_out_back = self._current[:,:,:,_CURRENTS['out_back'],:]
        ref_albedo = np.divide(current_in_back, current_out_back,
                               where=current_out_back > 1.0e-10,
                               out=np.ones_like(current_out_back))

        # Diffusion coefficient of neighbor to back
        neig_dc = np.roll(self._diffcof, 1, axis=1)
        # Cell dimensions of neighbor to back
        neig_hxyz = np.roll(self._hxyz, 1, axis=1)

        # Define dtilde at back surface for all mesh cells not on back boundary
        # Dtilde is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notfirst_y_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dy = self._hxyz[boundary + (np.newaxis, 1)]
        neig_D = neig_dc[boundary_grps]
        neig_dy = neig_hxyz[boundary + (np.newaxis, 1)]
        alb = ref_albedo[boundary_grps]
        is_adj_ref = self._is_adj_ref_back
        dtilde = np.where(is_adj_ref, (2.0 * D * (1.0 - alb)) /
                          (4.0 * D * (1.0 + alb) + (1.0 - alb) * dy),
                          (2.0 * D * neig_D) / (neig_dy * D + dy * neig_D))
        self._dtilde[boundary_grps + (2,)] = dtilde

        # Define reflector albedo for all cells on the front surface, in case
        # a cell borders a reflector region in the front
        current_in_front = self._current[:,:,:,_CURRENTS['in_front'],:]
        current_out_front = self._current[:,:,:,_CURRENTS['out_front'],:]
        ref_albedo = np.divide(current_in_front, current_out_front,
                               where=current_out_front > 1.0e-10,
                               out=np.ones_like(current_out_front))

        # Diffusion coefficient of neighbor to front
        neig_dc = np.roll(self._diffcof, -1, axis=1)
        # Cell dimensions of neighbor to front
        neig_hxyz = np.roll(self._hxyz, -1, axis=1)

        # Define dtilde at front surface for all mesh cells not on front
        # boundary. Dtilde is defined differently for regions that do and don't
        # neighbor reflector regions
        boundary = self._notlast_y_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dy = self._hxyz[boundary + (np.newaxis, 1)]
        neig_D = neig_dc[boundary_grps]
        neig_dy = neig_hxyz[boundary + (np.newaxis, 1)]
        alb = ref_albedo[boundary_grps]
        is_adj_ref = self._is_adj_ref_front
        dtilde = np.where(is_adj_ref, (2.0 * D * (1.0 - alb)) /
                          (4.0 * D * (1.0 + alb) + (1.0 - alb) * dy),
                          (2.0 * D * neig_D) / (neig_dy * D + dy * neig_D))
        self._dtilde[boundary_grps + (3,)] = dtilde

        # Define reflector albedo for all cells on the bottom surface, in case
        # a cell borders a reflector region on the bottom
        current_in_bottom = self._current[:,:,:,_CURRENTS['in_bottom'],:]
        current_out_bottom = self._current[:,:,:,_CURRENTS['out_bottom'],:]
        ref_albedo = np.divide(current_in_bottom, current_out_bottom,
                               where=current_out_bottom > 1.0e-10,
                               out=np.ones_like(current_out_bottom))

        # Diffusion coefficient of neighbor to bottom
        neig_dc = np.roll(self._diffcof, 1, axis=2)
        # Cell dimensions of neighbor to bottom
        neig_hxyz = np.roll(self._hxyz, 1, axis=2)

        # Define dtilde at bottom surface for all mesh cells not on bottom
        # boundary. Dtilde is defined differently for regions that do and don't
        # neighbor reflector regions
        boundary = self._notfirst_z_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dz = self._hxyz[boundary + (np.newaxis, 2)]
        neig_D = neig_dc[boundary_grps]
        neig_dz = neig_hxyz[boundary + (np.newaxis, 2)]
        alb = ref_albedo[boundary_grps]
        is_adj_ref = self._is_adj_ref_bottom
        dtilde = np.where(is_adj_ref, (2.0 * D * (1.0 - alb)) /
                          (4.0 * D * (1.0 + alb) + (1.0 - alb) * dz),
                          (2.0 * D * neig_D) / (neig_dz * D + dz * neig_D))
        self._dtilde[boundary_grps + (4,)] = dtilde

        # Define reflector albedo for all cells on the top surface, in case
        # a cell borders a reflector region on the top
        current_in_top = self._current[:,:,:,_CURRENTS['in_top'],:]
        current_out_top = self._current[:,:,:,_CURRENTS['out_top'],:]
        ref_albedo = np.divide(current_in_top, current_out_top,
                               where=current_out_top > 1.0e-10,
                               out=np.ones_like(current_out_top))

        # Diffusion coefficient of neighbor to top
        neig_dc = np.roll(self._diffcof, -1, axis=2)
        # Cell dimensions of neighbor to top
        neig_hxyz = np.roll(self._hxyz, -1, axis=2)

        # Define dtilde at top surface for all mesh cells not on top boundary
        # Dtilde is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notlast_z_accel
        boundary_grps = boundary + (slice(None),)
        D = self._diffcof[boundary_grps]
        dz = self._hxyz[boundary + (np.newaxis, 2)]
        neig_D = neig_dc[boundary_grps]
        neig_dz = neig_hxyz[boundary + (np.newaxis, 2)]
        alb = ref_albedo[boundary_grps]
        is_adj_ref = self._is_adj_ref_top
        dtilde = np.where(is_adj_ref, (2.0 * D * (1.0 - alb)) /
                          (4.0 * D * (1.0 + alb) + (1.0 - alb) * dz),
                          (2.0 * D * neig_D) / (neig_dz * D + dz * neig_D))
        self._dtilde[boundary_grps + (5,)] = dtilde

    def _compute_dhat(self):
        """ Computes the nonlinear coupling coefficient using a vectorized
        numpy approach. Aggregate values for the dhat multidimensional array
        are populated by first defining values on the problem boundary, and then
        for all other regions. For indices not lying by a boundary, dhat values
        are distinguished between regions that neighbor a reflector region and
        regions that don't neighbor a reflector

        """
        # Define current in each direction
        current_in_left = self._current[:,:,:,_CURRENTS['in_left'],:]
        current_out_left = self._current[:,:,:,_CURRENTS['out_left'],:]
        current_in_right = self._current[:,:,:,_CURRENTS['in_right'],:]
        current_out_right = self._current[:,:,:,_CURRENTS['out_right'],:]
        current_in_back = self._current[:,:,:,_CURRENTS['in_back'],:]
        current_out_back = self._current[:,:,:,_CURRENTS['out_back'],:]
        current_in_front = self._current[:,:,:,_CURRENTS['in_front'],:]
        current_out_front = self._current[:,:,:,_CURRENTS['out_front'],:]
        current_in_bottom = self._current[:,:,:,_CURRENTS['in_bottom'],:]
        current_out_bottom = self._current[:,:,:,_CURRENTS['out_bottom'],:]
        current_in_top = self._current[:,:,:,_CURRENTS['in_top'],:]
        current_out_top = self._current[:,:,:,_CURRENTS['out_top'],:]

        dx = self._hxyz[:,:,:,np.newaxis,0]
        dy = self._hxyz[:,:,:,np.newaxis,1]
        dz = self._hxyz[:,:,:,np.newaxis,2]
        dxdydz = np.prod(self._hxyz, axis=3)[:,:,:,np.newaxis]

        # Define net current on each face
        net_current_left = (current_in_left - current_out_left) / dxdydz * dx
        net_current_right = (current_out_right - current_in_right) / dxdydz * \
            dx
        net_current_back = (current_in_back - current_out_back) / dxdydz * dy
        net_current_front = (current_out_front - current_in_front) / dxdydz * \
            dy
        net_current_bottom = (current_in_bottom - current_out_bottom) / \
            dxdydz * dz
        net_current_top = (current_out_top - current_in_top) / dxdydz * dz

        # Define flux in each cell
        cell_flux = self._flux / dxdydz
        # Extract indices of coremap that are accelerated
        is_accel = self._coremap != _CMFD_NOACCEL

        # Define dhat at left surface for all mesh cells on left boundary
        boundary = self._first_x_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_left[boundary_grps]
        dtilde = self._dtilde[boundary + (slice(None), 0)]
        flux = cell_flux[boundary_grps]
        self._dhat[boundary_grps + (0,)] = (net_current + dtilde * flux) / flux

        # Define dhat at right surface for all mesh cells on right boundary
        boundary = self._last_x_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_right[boundary_grps]
        dtilde = self._dtilde[boundary + (slice(None), 1)]
        flux = cell_flux[boundary_grps]
        self._dhat[boundary_grps + (1,)] = (net_current - dtilde * flux) / flux

        # Define dhat at back surface for all mesh cells on back boundary
        boundary = self._first_y_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_back[boundary_grps]
        dtilde = self._dtilde[boundary + (slice(None), 2)]
        flux = cell_flux[boundary_grps]
        self._dhat[boundary_grps + (2,)] = (net_current + dtilde * flux) / flux

        # Define dhat at front surface for all mesh cells on front boundary
        boundary = self._last_y_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_front[boundary_grps]
        dtilde = self._dtilde[boundary + (slice(None), 3)]
        flux = cell_flux[boundary_grps]
        self._dhat[boundary_grps + (3,)] = (net_current - dtilde * flux) / flux

        # Define dhat at bottom surface for all mesh cells on bottom boundary
        boundary = self._first_z_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_bottom[boundary_grps]
        dtilde = self._dtilde[boundary + (slice(None), 4)]
        flux = cell_flux[boundary_grps]
        self._dhat[boundary_grps + (4,)] = (net_current + dtilde * flux) / flux

        # Define dhat at top surface for all mesh cells on top boundary
        boundary = self._last_z_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_top[boundary_grps]
        dtilde = self._dtilde[boundary + (slice(None), 5)]
        flux = cell_flux[boundary_grps]
        self._dhat[boundary_grps + (5,)] = (net_current - dtilde * flux) / flux

        # Cell flux of neighbor to left
        neig_flux = np.roll(self._flux, 1, axis=0) / dxdydz

        # Define dhat at left surface for all mesh cells not on left boundary
        # Dhat is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notfirst_x_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_left[boundary_grps]
        dtilde = self._dtilde[boundary_grps + (0,)]
        flux = cell_flux[boundary_grps]
        flux_left = neig_flux[boundary_grps]
        is_adj_ref = self._is_adj_ref_left
        dhat = np.where(is_adj_ref, (net_current + dtilde * flux) / flux,
                        (net_current - dtilde * (flux_left - flux)) /
                        (flux_left + flux))
        self._dhat[boundary_grps + (0,)] = dhat

        # Cell flux of neighbor to right
        neig_flux = np.roll(self._flux, -1, axis=0) / dxdydz

        # Define dhat at right surface for all mesh cells not on right boundary
        # Dhat is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notlast_x_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_right[boundary_grps]
        dtilde = self._dtilde[boundary_grps + (1,)]
        flux = cell_flux[boundary_grps]
        flux_right = neig_flux[boundary_grps]
        is_adj_ref = self._is_adj_ref_right
        dhat = np.where(is_adj_ref, (net_current - dtilde * flux) / flux,
                        (net_current + dtilde * (flux_right - flux)) /
                        (flux_right + flux))
        self._dhat[boundary_grps + (1,)] = dhat

        # Cell flux of neighbor to back
        neig_flux = np.roll(self._flux, 1, axis=1) / dxdydz

        # Define dhat at back surface for all mesh cells not on back boundary
        # Dhat is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notfirst_y_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_back[boundary_grps]
        dtilde = self._dtilde[boundary_grps + (2,)]
        flux = cell_flux[boundary_grps]
        flux_back = neig_flux[boundary_grps]
        is_adj_ref = self._is_adj_ref_back
        dhat = np.where(is_adj_ref, (net_current + dtilde * flux) / flux,
                        (net_current - dtilde * (flux_back - flux)) /
                        (flux_back + flux))
        self._dhat[boundary_grps + (2,)] = dhat

        # Cell flux of neighbor to front
        neig_flux = np.roll(self._flux, -1, axis=1) / dxdydz

        # Define dhat at front surface for all mesh cells not on front boundary
        # Dhat is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notlast_y_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_front[boundary_grps]
        dtilde = self._dtilde[boundary_grps + (3,)]
        flux = cell_flux[boundary_grps]
        flux_front = neig_flux[boundary_grps]
        is_adj_ref = self._is_adj_ref_front
        dhat = np.where(is_adj_ref, (net_current - dtilde * flux) / flux,
                        (net_current + dtilde * (flux_front - flux)) /
                        (flux_front + flux))
        self._dhat[boundary_grps + (3,)] = dhat

        # Cell flux of neighbor to bottom
        neig_flux = np.roll(self._flux, 1, axis=2) / dxdydz

        # Define dhat at bottom surface for all mesh cells not on bottom
        # boundary. Dhat is defined differently for regions that do and don't
        # neighbor reflector regions
        boundary = self._notfirst_z_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_bottom[boundary_grps]
        dtilde = self._dtilde[boundary_grps + (4,)]
        flux = cell_flux[boundary_grps]
        flux_bottom = neig_flux[boundary_grps]
        is_adj_ref = self._is_adj_ref_bottom
        dhat = np.where(is_adj_ref, (net_current + dtilde * flux) / flux,
                        (net_current - dtilde * (flux_bottom - flux)) /
                        (flux_bottom + flux))
        self._dhat[boundary_grps + (4,)] = dhat

        # Cell flux of neighbor to top
        neig_flux = np.roll(self._flux, -1, axis=2) / dxdydz

        # Define dhat at top surface for all mesh cells not on top boundary
        # Dhat is defined differently for regions that do and don't neighbor
        # reflector regions
        boundary = self._notlast_z_accel
        boundary_grps = boundary + (slice(None),)
        net_current = net_current_top[boundary_grps]
        dtilde = self._dtilde[boundary_grps + (5,)]
        flux = cell_flux[boundary_grps]
        flux_top = neig_flux[boundary_grps]
        is_adj_ref = self._is_adj_ref_top
        dhat = np.where(is_adj_ref, (net_current - dtilde * flux) / flux,
                        (net_current + dtilde * (flux_top - flux)) /
                        (flux_top + flux))
        self._dhat[boundary_grps + (5,)] = dhat
