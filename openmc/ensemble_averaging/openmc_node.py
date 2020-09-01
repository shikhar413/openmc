"""
This module runs all OpenMC-related functions when running ensemble averaging.
Specifically, it creates and sends all CMFD tallies to the CMFD node
and receives updated source weights to update the particle weights in
the OpenMC source bank.

References
----------

.. [Smith] K. Smith, "Nodal method storage reduction by non-linear
   iteration", *Trans. Am. Nucl. Soc.*, **44**, 265 (1983).

"""

from contextlib import contextmanager
from numbers import Integral, Real
import sys
import time
import logging

import numpy as np
import h5py
from mpi4py import MPI

import openmc.lib
from openmc.checkvalue import check_type, check_greater_than
from openmc.exceptions import OpenMCError

# Maximum/minimum neutron energies
_ENERGY_MAX_NEUTRON = np.inf
_ENERGY_MIN_NEUTRON = 0.


class OpenMCNode(object):
    r"""Class for running OpenMC node when running CMFD with ensemble averaging.

    Attributes
    ----------
    tally_begin : int
        Batch number at which CMFD tallies should begin accummulating
    solver_begin: int
        Batch number at which CMFD solver should start executing
    mesh : openmc.cmfd.CMFDMesh
        Structured mesh to be used for acceleration
    ref_d : list of floats
        List of reference diffusion coefficients to fix CMFD parameters to
    ea_run_strategy : {'bulk-synch', 'eager-asynch', 'redez-asynch'}
        Specifies type of ensemble averaging run strategy to employ
    use_logger : bool
        Whether or not to log events to log file
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

    weight_clipping : float
        Weight clipping to control the maximum allowed change in weight in
        the presence of CMFD feedback. During prolongation, the weight factor
        in any CMFD mesh cell is clipped to
        [1/(1+weight_clipping), 1+weight_clipping]
    n_threads : int
        Number of threads per process allocated to run OpenMC
    indices : numpy.ndarray
        Stores spatial and group dimensions as [nx, ny, nz, ng]
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
    seed_begin : int
        Seed number at which to start independent OpenMC runs
    n_inactive : int
        Number of inactive batches to run for each independent seed
    n_particles : int
        Number of particles to run for each independent seed
    global_comm : mpi4py.MPI.Intracomm
        MPI intercommunicator to comunicate between CMFD and OpenMC nodes
    local_comm : mpi4py.MPI.Intracomm
        MPI intercommunicator to communicate locally between CMFD nodes
    time_openmcnode : float
        Time in OpenMC node, in seconds
    time_sendtallies : float
        Time taken to send tallies to CMFD node, in seconds
    time_waitcmfdsrc : float
        Time waiting for weightfactors from CMFD node
    """

    def __init__(self):
        """ Constructor for OpenMCNode class. Default values for instance variables
        set in this method.

        """
        # Variables that users can modify
        self._n_threads = 1
        self._n_particles = 1000
        self._n_inactive = 10
        self._weight_clipping = 0.2

        # Variables defined by EnsAvgCMFDRun class
        self._tally_begin = None
        self._solver_begin = None
        self._ref_d = None
        self._use_logger = None
        self._ea_run_strategy = None
        self._window_type = None
        self._mesh = None
        self._n_procs_per_seed = None
        self._n_seeds = None
        self._seed_begin = None
        self._openmc_verbosity = None
        self._verbosity = None
        self._n_batches = None
        self._global_comm = None
        self._local_comm = None

        # External variables used during runtime but users cannot control
        self._set_reference_params = False
        self._indices = np.zeros(4, dtype=np.int32)
        self._egrid = None
        self._mesh_id = None
        self._tally_ids = None
        self._energy_filters = None
        self._sourcecounts = None
        self._weightfactors = None
        self._reset_every = None
        self._current_batch = 0
        self._cmfd_src = None
        self._time_openmcnode = None
        self._time_sendtallies = None
        self._time_waitcmfdsrc = None

    @property
    def tally_begin(self):
        return self._tally_begin

    @property
    def solver_begin(self):
        return self._solver_begin

    @property
    def ref_d(self):
        return self._ref_d

    @property
    def ea_run_strategy(self):
        return self._ea_run_strategy

    @property
    def use_logger(self):
        return self._use_logger

    @property
    def window_type(self):
        return self._window_type

    @property
    def mesh(self):
        return self._mesh

    @property
    def weight_clipping(self):
        return self._weight_clipping

    @property
    def n_threads(self):
        return self._n_threads

    @property
    def indices(self):
        return self._indices

    @property
    def global_comm(self):
        return self._global_comm

    @property
    def local_comm(self):
        return self._local_comm

    @property
    def n_seeds(self):
        return self._n_seeds

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
    def n_procs_per_seed(self):
        return self._n_procs_per_seed

    @property
    def n_batches(self):
        return self._n_batches

    @property
    def n_inactive(self):
        return self._n_inactive

    @property
    def n_particles(self):
        return self._n_particles

    @weight_clipping.setter
    def weight_clipping(self, weight_clipping):
        check_type('CMFD weight clipping', weight_clipping, Real)
        check_greater_than('CMFD weight clipping', weight_clipping, 0., True)
        self._weight_clipping = weight_clipping

    @n_threads.setter
    def n_threads(self, threads):
        check_type('OpenMC threads', threads, Integral)
        check_greater_than('OpenMC threads', threads, 0)
        self._n_threads = threads

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

    # All error checking for following methods done in EnsAvgCMFDRun class
    @tally_begin.setter
    def tally_begin(self, begin):
        self._tally_begin = begin

    @solver_begin.setter
    def solver_begin(self, begin):
        self._solver_begin = begin

    @ref_d.setter
    def ref_d(self, ref_d):
        self._ref_d = ref_d

    @ea_run_strategy.setter
    def ea_run_strategy(self, ea_run_strategy):
        self._ea_run_strategy = ea_run_strategy

    @use_logger.setter
    def use_logger(self, use_logger):
        self._use_logger = use_logger

    @window_type.setter
    def window_type(self, window_type):
        self._window_type = window_type

    @mesh.setter
    def mesh(self, cmfd_mesh):
        self._mesh = cmfd_mesh

    @global_comm.setter
    def global_comm(self, comm):
        self._global_comm = comm

    @local_comm.setter
    def local_comm(self, comm):
        self._local_comm = comm

    @n_seeds.setter
    def n_seeds(self, n_seeds):
        self._n_seeds = n_seeds

    @seed_begin.setter
    def seed_begin(self, begin):
        self._seed_begin = begin

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

    @openmc_verbosity.setter
    def openmc_verbosity(self, verbosity):
        self._openmc_verbosity = verbosity

    @n_procs_per_seed.setter
    def n_procs_per_seed(self, procs):
        self._n_procs_per_seed = procs

    @n_batches.setter
    def n_batches(self, batches):
        self._n_batches = batches

    @contextmanager
    def run_in_memory(self, **kwargs):
        """ Context manager for running OpenMCNode functions.

        This function can be used with a 'with' statement to ensure the
        OpenMCNode class is properly initialized/finalized. For example::

            from openmc.ensemble_averaging.openmc_node import OpenMCNode
            node = OpenMCNode()
            with node.run_in_memory():
                status = 0
                do_stuff_before_simulation_start()
                while status == 0:
                    status = node.next_batch()
                    do_stuff_between_batches()

        Parameters
        ----------
        **kwargs
            All keyword arguments used to initialize OpenMCNode class.

        """
        # Extract arguments passed from EnsAvgCMFDRun class
        global_args = kwargs['global_args']
        openmc_args = kwargs['openmc_args']

        self._initialize_ea_params(global_args, openmc_args)
        if self._verbosity >= 1:
            self._write_summary()

        # Run and pass arguments to C API run_in_memory function
        args = ['-s', str(self._n_threads)]
        with openmc.lib.run_in_memory(args=args, intracomm=self._local_comm):
            self.init()
            yield
            self.finalize()

    def init(self):
        """ Initialize OpenMCNode instance in memory and set up
        necessary CMFD parameters.

        """
        if self._use_logger:
            openmc.lib.settings.use_logger = True
        self._log_event('Started OpenMC node init')

        # Configure OpenMC parameters
        self._configure_openmc()

        # Configure CMFD parameters
        self._configure_cmfd()

        # Create tally objects
        self._create_cmfd_tally()

        # Initialize simulation
        openmc.lib.simulation_init()

        # Set cmfd_run variable to True through C API
        openmc.lib.settings.cmfd_run = True

        self._log_event('Finished OpenMC node init')

    def next_batch(self):
        """ Run next batch for OpenMCNode.

        Returns
        -------
        int
            Status after running a batch (0=normal, 1=reached maximum number of
            batches, 2=tally triggers reached)

        """
        self._log_event('Started next batch, current batch {}'.format(self._current_batch))

        # Start timer for OpenMC node
        if openmc.lib.master():
            time_start_openmc = time.time()

        # Increment current batch
        self._current_batch += 1

        if self._current_batch > self._n_batches:
            return 1

        # Check to set CMFD tallies as active
        if self._tally_begin == self._current_batch:
            tallies = openmc.lib.tallies
            for tally_id in self._tally_ids:
                tallies[tally_id].active = True

        # Check to reset tallies
        if self._reset_every:
            self._cmfd_tally_reset()

        # Aggregate CMFD tallies by running next batch in OpenMC
        status = openmc.lib.next_batch()

        self._log_event('Finished running OpenMC'.format(self._current_batch))

        # Broadcast CMFD tallies to CMFD node
        if openmc.lib.master():
            if self._current_batch >= self._tally_begin:
                time_start_sendtallies = time.time()
                self._send_tallies_to_cmfd_node()
                log_str = 'Sent tallies to CMFD'
                if self._ea_run_strategy == 'rendez-asynch' and self._current_batch >= self._solver_begin:
                    log_str += ' and received CMFD source'
                self._log_event(log_str)
                time_stop_sendtallies = time.time()
                self._time_sendtallies += (time_stop_sendtallies -
                                           time_start_sendtallies)

        if self._current_batch >= self._solver_begin:
            # Count bank sites in CMFD mesh
            outside = self._count_bank_sites()

            # Check and raise error if source sites exist outside of CMFD mesh
            if openmc.lib.master() and outside:
                raise OpenMCError('Source sites outside of the CMFD mesh')

            self._log_event('Finished counting bank sites')

            # Receive updated weight factors from CMFD node and update source
            if self._ea_run_strategy != 'rendez-asynch' and openmc.lib.master():
                time_start_recvcmfdsrc = time.time()
                self._recv_cmfdsrc_from_cmfd_node()
                self._log_event('Finished recieving CMFD source')
                time_stop_recvcmfdsrc = time.time()
                self._time_waitcmfdsrc += (time_stop_recvcmfdsrc -
                                           time_start_recvcmfdsrc)

            # Reweight source based on updated weight factors
            self._cmfd_reweight()
            self._log_event('Finished CMFD reweight')

        # Stop timer for OpenMC node
        if openmc.lib.master():
            time_stop_openmc = time.time()
            self._time_openmcnode += time_stop_openmc - time_start_openmc

        self._log_event('Finished next batch, current batch {}'.format(self._current_batch))
        return status

    def finalize(self):
        """ Finalize simulation by calling
        :func:`openmc.lib.simulation_finalize`.

        """
        self._log_event('Started OpenMC node finalize')

        # Finalize simuation
        openmc.lib.simulation_finalize()
        if openmc.lib.master():
            self._send_timing_stats_to_cmfd_node()

        self._log_event('Finished OpenMC node finalize')

    def _initialize_ea_params(self, global_args, openmc_args):
        """ Initialize global parameters inherited from EnsAvgCMFDRun class """
        # Initialize global parameters inherited from EnsAvgCMFDRun class
        for param in global_args:
            setattr(self, param, global_args[param])

        # Initialize OpenMC parameters inherited from EnsAvgCMFDRun class
        for param in openmc_args:
            setattr(self, param, openmc_args[param])

        rank = self._global_comm.Get_rank()
        self._seed_num = int((rank-self._n_procs_per_seed) /
                             self._n_procs_per_seed) + self._seed_begin

    def _write_summary(self):
        """ Write summary of OpenMC node parameters """
        openmc_params = ['n_threads', 'n_particles', 'n_inactive',
                         'weight_clipping']
        rank = self._global_comm.Get_rank()
        outstr = "********* PROCESS {}: OPENMC NODE, SEED {} *********\n".format(rank,
                                                                       self._seed_num)
        for param in openmc_params:
            param_repr = str(getattr(self, param))
            outstr += "     {}: {}\n".format(param, param_repr)
        outstr += "**************************************************\n"
        print(outstr)
        sys.stdout.flush()

    def _configure_openmc(self):
        """ Configure OpenMC parameters through OpenMC lib """
        openmc.lib.settings.seed = self._seed_num
        openmc.lib.settings.verbosity = self._openmc_verbosity
        openmc.lib.settings.inactive = self._n_inactive
        openmc.lib.settings.set_batches(self._n_batches)
        openmc.lib.settings.particles = self._n_particles

        # Initialize timers
        self._time_openmcnode = 0.0
        self._time_sendtallies = 0.0
        self._time_waitcmfdsrc = 0.0

    def _configure_cmfd(self):
        """ Configure CMFD parameters and set CMFD input variables """
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
            self._energy_filters = True
        else:
            self._egrid = np.array([_ENERGY_MIN_NEUTRON, _ENERGY_MAX_NEUTRON])
            self._indices[3] = 1
            self._energy_filters = False

        # Check CMFD tallies accummulated before feedback turned on
        if self._solver_begin < self._tally_begin:
            raise ValueError('Tally begin must be less than or equal to '
                             'solver begin')

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

    def _set_tally_window(self):
        """ Set parameters to handle different tally window options """
        # Set tallies to reset every batch if window_type is not none
        if self._window_type != 'none':
            self._reset_every = True

    def _cmfd_tally_reset(self):
        """ Reset all CMFD tallies in memory """
        # Print message
        if (self.verbosity >= 2 and openmc.lib.master() and
            not self._reset_every):
            print(' CMFD tallies reset')
            sys.stdout.flush()

        # Reset CMFD tallies
        tallies = openmc.lib.tallies
        for tally_id in self._tally_ids:
            tallies[tally_id].reset()

    def _count_bank_sites(self):
        """ Determine the number of fission bank sites in each cell of a given
        mesh and energy group structure.

        Returns
        -------
        bool
            Whether any source sites outside of CMFD mesh were found

        """
        # Initialize variables
        m = openmc.lib.meshes[self._mesh_id]
        source_bank = openmc.lib.source_bank()
        energy = self._egrid
        sites_outside = np.zeros(1, dtype=bool)
        nxnynz = np.prod(self._indices[0:3])
        ng = self._indices[3]

        outside = np.zeros(1, dtype=bool)
        self._sourcecounts = np.zeros((nxnynz, ng))
        count = np.zeros(self._sourcecounts.shape)

        # Get location and energy of each particle in source bank
        source_xyz = source_bank['r']
        source_energies = source_bank['E']
        source_weights = source_bank['wgt']

        # Convert xyz location to mesh index and ravel index to scalar
        mesh_locations = np.floor((source_xyz - m.lower_left) / m.width)
        mesh_bins = mesh_locations[:,2] * m.dimension[1] * m.dimension[0] + \
            mesh_locations[:,1] * m.dimension[0] + mesh_locations[:,0]

        # Check if any source locations lie outside of defined CMFD mesh
        if np.any(mesh_bins < 0) or np.any(mesh_bins >= np.prod(m.dimension)):
            idx = np.where((mesh_locations >= m.dimension) | (mesh_locations < 0))
            unique_locs = []
            new_locs = []
            for i in idx[0]:
                source_loc = openmc.lib.source_bank()['r'][i]
                if list(source_loc) in unique_locs:
                    openmc.lib.source_bank()['r'][i] = new_locs[unique_locs.index(list(source_loc))]
                else:
                    new_loc = np.zeros((3,))
                    for j in range(3):
                        loc = source_loc[j]
                        if loc < m.lower_left[j]:
                            new_loc[j] = np.random.uniform(m.lower_left[j], m.lower_left[j] + m.width[j])
                        elif loc > m.upper_right[j]:
                            new_loc[j] = np.random.uniform(m.upper_right[j] - m.width[j], m.upper_right[j])
                        else:
                            new_loc[j] = loc
                    unique_locs.append(list(source_loc))
                    new_locs.append(new_loc)
                    prev_loc = source_loc
                    print("Source location changed from", source_loc, "to", new_loc)
                    openmc.lib.source_bank()['r'][i] = new_loc
                    sys.stdout.flush()
            outside[0] = True

            # Get location and energy of each particle in source bank
            source_xyz = source_bank['r']
            source_energies = source_bank['E']
            source_weights = source_bank['wgt']

            # Convert xyz location to mesh index and ravel index to scalar
            mesh_locations = np.floor((source_xyz - m.lower_left) / m.width)
            mesh_bins = mesh_locations[:,2] * m.dimension[1] * m.dimension[0] + \
            mesh_locations[:,1] * m.dimension[0] + mesh_locations[:,0]

        # Determine which energy bin each particle's energy belongs to
        # Separate into cases bases on where source energies lies on egrid
        energy_bins = np.zeros(len(source_energies), dtype=int)
        idx = np.where(source_energies < energy[0])
        energy_bins[idx] = 0
        idx = np.where(source_energies > energy[-1])
        energy_bins[idx] = ng - 1
        idx = np.where((source_energies >= energy[0]) &
                       (source_energies <= energy[-1]))
        energy_bins[idx] = np.digitize(source_energies[idx], energy) - 1

        # Determine all unique combinations of mesh bin and energy bin, and
        # count number of particles that belong to these combinations
        idx, inverse = np.unique(np.array([mesh_bins, energy_bins]), axis=1,
                                 return_inverse=True)
        counts = np.bincount(inverse, weights=source_weights)

        # Store counts to appropriate mesh-energy combination
        count[idx[0].astype(int), idx[1].astype(int)] = counts

        # Collect values of count from all processors
        self._local_comm.Reduce(count, self._sourcecounts, MPI.SUM)
        # Check if there were sites outside the mesh for any processor
        self._local_comm.Reduce(outside, sites_outside, MPI.LOR)

        return sites_outside[0]

    def _log_event(self, log_msg):
        if self._use_logger and openmc.lib.master():
            current_time = time.time()
            logging.info(log_msg + ", time={}".format(current_time))

    def _send_tallies_to_cmfd_node(self):
        """ Aggregate CMFD tallies and transfer to CMFD node """
        keff = openmc.lib.keff()[0]

        # Get tallies in-memory
        tallies = openmc.lib.tallies

        # Get flux from CMFD tally 0
        tally_id = self._tally_ids[0]
        flux = tallies[tally_id].results[:,0,1]

        # Get total rr from CMFD tally 0
        totalrr = tallies[tally_id].results[:,1,1]

        # Get scattering rr from CMFD tally 1
        # flux is repeated to account for extra dimensionality of scattering xs
        tally_id = self._tally_ids[1]
        scattrr = tallies[tally_id].results[:,0,1]

        # Get nu-fission rr from CMFD tally 1
        nfissrr = tallies[tally_id].results[:,1,1]
        num_realizations = tallies[tally_id].num_realizations

        # Get surface currents from CMFD tally 2
        tally_id = self._tally_ids[2]
        current = tallies[tally_id].results[:,0,1]

        current_batch = self._current_batch

        if self._set_reference_params:
            tally_data = np.concatenate((flux, totalrr, scattrr, nfissrr,
                                         current, [num_realizations, keff,
                                                   current_batch]))
        else:
            # Get p1 scatter rr from CMFD tally 3
            tally_id = self._tally_ids[3]
            p1scattrr = tallies[tally_id].results[:,0,1]

            tally_data = np.concatenate((flux, totalrr, scattrr, nfissrr,
                                         current, p1scattrr,
                                         [num_realizations, keff,
                                          current_batch]))

        if self._ea_run_strategy == 'eager-asynch':
            self._global_comm.Isend(tally_data, dest=0, tag=0)
        else:
            self._global_comm.Send(tally_data, dest=0, tag=0)

        if self._verbosity >= 2:
            source = self._global_comm.Get_rank()
            outstr = "{:>11s}Sending tally data for batch {} from process {} to {}"
            print(outstr.format('', self._current_batch, source, 0))
            sys.stdout.flush()

        if self._ea_run_strategy == 'rendez-asynch' and self._current_batch >= self._solver_begin:
            self._recv_cmfdsrc_from_cmfd_node()

    def _send_timing_stats_to_cmfd_node(self):
        """ Transfer timing stats to CMFD node for printing """
        timing_data = np.array([self._time_openmcnode, self._time_sendtallies,
                                self._time_waitcmfdsrc])
        if self._ea_run_strategy == 'eager-asynch':
            self._global_comm.Isend(timing_data, dest=0, tag=1)
        else:
            self._global_comm.Send(timing_data, dest=0, tag=1)
        if self._verbosity >= 2:
            source = self._global_comm.Get_rank()
            outstr = "{:>11s}Sending timing data from process {} to {}"
            print(outstr.format('', source, 0))
            sys.stdout.flush()

    def _recv_cmfdsrc_from_cmfd_node(self):
        """ Receive weightfactors from CMFD node and update source """
        self._cmfd_src = np.empty(self._indices, dtype=np.float64)
        if self._ea_run_strategy == 'eager-asynch':
            # Wait for CMFD source from CMFD node on first data transfer
            if self._current_batch == self._solver_begin:
                req = self._global_comm.Irecv(self._cmfd_src, source=0)
                req.Wait()
            # Otherwise keep requesting for newest available CMFD source
            else:
                new_req_avail = True
                while (new_req_avail):
                    req = self._global_comm.Irecv(self._cmfd_src, source=0)
                    new_req_avail = req.Test()
                    if new_req_avail:
                        req.Wait()
                    else:
                        # One too many requests submitted by this point
                        # The most recent request must be cancelled
                        req.Cancel()
        else:
            self._global_comm.Recv(self._cmfd_src, source=0)

        if self._verbosity >= 2:
            dest = self._global_comm.Get_rank()
            outstr = "{:>11s}Process {} received CMFD source from process {}"
            print(outstr.format('', dest, 0))
            sys.stdout.flush()

    def _cmfd_reweight(self):
        """ Perform weighting of particles in source bank """
        nx, ny, nz, ng = self._indices

        if openmc.lib.master():
            with np.errstate(divide='ignore', invalid='ignore'):
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
            with np.errstate(divide='ignore', invalid='ignore'):
                self._weightfactors = (np.divide(self._cmfd_src * norm,
                                       sourcecounts, where=div_condition,
                                       out=np.ones_like(self._cmfd_src),
                                       dtype=np.float32))
            ub = 1. + self._weight_clipping
            self._weightfactors[self._weightfactors > ub] = ub
            lb = 1./(1. + self._weight_clipping)
            self._weightfactors[self._weightfactors < lb] = lb

        self._weightfactors = self.local_comm.bcast(
                              self._weightfactors)

        m = openmc.lib.meshes[self._mesh_id]
        energy = self._egrid

        # Get locations and energies of all particles in source bank
        source_xyz = openmc.lib.source_bank()['r']
        source_energies = openmc.lib.source_bank()['E']

        # Convert xyz location to the CMFD mesh index
        mesh_ijk = np.floor((source_xyz - m.lower_left)/m.width).astype(int)

        # Determine which energy bin each particle's energy belongs to
        # Separate into cases bases on where source energies lies on egrid
        energy_bins = np.zeros(len(source_energies), dtype=int)
        idx = np.where(source_energies < energy[0])
        energy_bins[idx] = ng - 1
        idx = np.where(source_energies > energy[-1])
        energy_bins[idx] = 0
        idx = np.where((source_energies >= energy[0]) &
                       (source_energies <= energy[-1]))
        energy_bins[idx] = ng - np.digitize(source_energies[idx], energy)

        # Determine weight factor of each particle based on its mesh index
        # and energy bin and updates its weight
        openmc.lib.source_bank()['wgt'] *= self._weightfactors[
                mesh_ijk[:,0], mesh_ijk[:,1], mesh_ijk[:,2], energy_bins]

        if np.any(source_energies < energy[0]):
            print(' WARNING: Source point below energy grid')
            sys.stdout.flush()
        if np.any(source_energies > energy[-1]):
            print(' WARNING: Source point above energy grid')
            sys.stdout.flush()

    def _create_cmfd_tally(self):
        """ Create all tallies in-memory used to solve CMFD problem """
        # Create Mesh object based on CMFDMesh, stored internally
        cmfd_mesh = openmc.lib.RegularMesh()
        # Store id of mesh object
        self._mesh_id = cmfd_mesh.id
        # Set dimension and parameters of mesh object
        cmfd_mesh.dimension = self._mesh.dimension
        cmfd_mesh.set_parameters(lower_left=self._mesh.lower_left,
                                 upper_right=self._mesh.upper_right,
                                 width=self._mesh.width)

        # Create mesh Filter object, stored internally
        mesh_filter = openmc.lib.MeshFilter()
        # Set mesh for Mesh Filter
        mesh_filter.mesh = cmfd_mesh

        # Set up energy filters, if applicable
        if self._energy_filters:
            # Create Energy Filter object, stored internally
            energy_filter = openmc.lib.EnergyFilter()
            # Set bins for Energy Filter
            energy_filter.bins = self._egrid

            # Create Energy Out Filter object, stored internally
            energyout_filter = openmc.lib.EnergyoutFilter()
            # Set bins for Energy Filter
            energyout_filter.bins = self._egrid

        # Create Mesh Surface Filter object, stored internally
        meshsurface_filter = openmc.lib.MeshSurfaceFilter()
        # Set mesh for Mesh Surface Filter
        meshsurface_filter.mesh = cmfd_mesh

        # Create Legendre Filter object, stored internally
        legendre_filter = openmc.lib.LegendreFilter()
        # Set order for Legendre Filter
        legendre_filter.order = 1

        # Create CMFD tallies, stored internally
        n_tallies = 4
        self._tally_ids = []
        for i in range(n_tallies):
            # Skip computation of p1 scattering tally if reference diffusion
            # parameters given
            if self._set_reference_params and i == 3:
                continue

            cmfd_tally = openmc.lib.Tally()
            # Set nuclide bins
            cmfd_tally.nuclides = ['total']
            self._tally_ids.append(cmfd_tally.id)

            # Set attributes of CMFD flux, total tally
            if i == 0:
                # Set filters for tally
                if self._energy_filters:
                    cmfd_tally.filters = [mesh_filter, energy_filter]
                else:
                    cmfd_tally.filters = [mesh_filter]
                # Set scores, type, and estimator for tally
                cmfd_tally.scores = ['flux', 'total']
                cmfd_tally.type = 'volume'
                cmfd_tally.estimator = 'analog'

            # Set attributes of CMFD neutron production tally
            elif i == 1:
                # Set filters for tally
                if self._energy_filters:
                    cmfd_tally.filters = [mesh_filter, energy_filter,
                                          energyout_filter]
                else:
                    cmfd_tally.filters = [mesh_filter]
                # Set scores, type, and estimator for tally
                cmfd_tally.scores = ['nu-scatter', 'nu-fission']
                cmfd_tally.type = 'volume'
                cmfd_tally.estimator = 'analog'

            # Set attributes of CMFD surface current tally
            elif i == 2:
                # Set filters for tally
                if self._energy_filters:
                    cmfd_tally.filters = [meshsurface_filter, energy_filter]
                else:
                    cmfd_tally.filters = [meshsurface_filter]
                # Set scores, type, and estimator for tally
                cmfd_tally.scores = ['current']
                cmfd_tally.type = 'mesh-surface'
                cmfd_tally.estimator = 'analog'

            # Set attributes of CMFD P1 scatter tally
            elif i == 3:
                # Set filters for tally
                if self._energy_filters:
                    cmfd_tally.filters = [mesh_filter, legendre_filter,
                                          energy_filter]
                else:
                    cmfd_tally.filters = [mesh_filter, legendre_filter]
                # Set scores for tally
                cmfd_tally.scores = ['scatter']
                cmfd_tally.type = 'volume'
                cmfd_tally.estimator = 'analog'
