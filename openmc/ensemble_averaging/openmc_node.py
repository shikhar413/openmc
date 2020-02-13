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
from numbers import Integral
import sys
import time
import warnings

import numpy as np
import h5py

import openmc.lib
from openmc.checkvalue import (check_type, check_length, check_value,
                               check_greater_than)
from openmc.exceptions import OpenMCError

# Maximum/minimum neutron energies
_ENERGY_MAX_NEUTRON = np.inf
_ENERGY_MIN_NEUTRON = 0.


class OpenMCNode(object):
    # TODO broadcast sourcecounts to CMFDNode
    # TODO how to deal with freeing CMFD memory?
    # TODO deal with cmfd_reweight
    # TODO Add parameter for OpenMC verbosity
    r"""Class for running OpenMC node when running CMFD with ensemble averaging.

    Attributes
    ----------
    tally_begin : int
        Batch number at which CMFD tallies should begin accummulating
    solver_begin: int
        Batch number at which CMFD solver should start executing
    mesh : openmc.cmfd.CMFDMesh
        Structured mesh to be used for acceleration
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
    indices : numpy.ndarray
        Stores spatial and group dimensions as [nx, ny, nz, ng]
    intracomm : mpi4py.MPI.Intracomm or None
        MPI intercommunicator for running MPI commands

    """

    def __init__(self):
        """Constructor for CMFDRun class. Default values for instance variables
        set in this method.

        """
        # Variables that users can modify
        self._tally_begin = 1
        self._solver_begin = 1
        self._mesh = None
        self._window_type = 'none'
        self._window_size = 10
        self._intracomm = None

        # External variables used during runtime but users cannot control
        self._indices = np.zeros(4, dtype=np.int32)
        self._egrid = None
        self._mesh_id = None
        self._tally_ids = None
        self._energy_filters = None
        self._openmc_src_rate = None
        self._flux_rate = None
        self._total_rate = None
        self._p1scatt_rate = None
        self._scatt_rate = None
        self._nfiss_rate = None
        self._current_rate = None
        self._sourcecounts = None
        self._weightfactors = None
        self._reset_every = None

    @property
    def tally_begin(self):
        return self._tally_begin

    @property
    def solver_begin(self):
        return self._solver_begin

    @property
    def mesh(self):
        return self._mesh

    @property
    def window_type(self):
        return self._window_type

    @property
    def window_size(self):
        return self._window_size

    @property
    def indices(self):
        return self._indices

    @tally_begin.setter
    def tally_begin(self, begin):
        check_type('CMFD tally begin batch', begin, Integral)
        check_greater_than('CMFD tally begin batch', begin, 0)
        self._tally_begin = begin

    @solver_begin.setter
    def solver_begin(self, begin):
        check_type('CMFD feedback begin batch', begin, Integral)
        check_greater_than('CMFD feedback begin batch', begin, 0)
        self._solver_begin = begin

    @mesh.setter
    def mesh(self, cmfd_mesh):
        check_type('CMFD mesh', cmfd_mesh, CMFDMesh)

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

    @window_type.setter
    def window_type(self, window_type):
        check_type('CMFD window type', window_type, str)
        check_value('CMFD window type', window_type,
                    ['none', 'rolling', 'expanding'])
        self._window_type = window_type

    @window_size.setter
    def window_size(self, window_size):
        check_type('CMFD window size', window_size, Integral)
        check_greater_than('CMFD window size', window_size, 0)
        if self._window_type != 'rolling':
            warn_msg = 'Window size will have no effect on CMFD simulation ' \
                       'unless window type is set to "rolling".'
            warnings.warn(warn_msg, RuntimeWarning)
        self._window_size = window_size

    @contextmanager
    def run_in_memory(self, **kwargs):
        """ Context manager for running CMFD functions with OpenMC shared
        library functions.

        This function can be used with a 'with' statement to ensure the
        CMFDRun class is properly initialized/finalized. For example::

            from openmc import cmfd
            cmfd_run = cmfd.CMFDRun()
            with cmfd_run.run_in_memory():
                do_stuff_before_simulation_start()
                for _ in cmfd_run.iter_batches():
                    do_stuff_between_batches()

        Parameters
        ----------
        **kwargs
            All keyword arguments passed to :func:`openmc.lib.run_in_memory`.

        """
        print(kwargs)
        yield
        '''
        # Run and pass arguments to C API run_in_memory function
        with openmc.lib.run_in_memory(**kwargs):
            self.init()
            yield
            self.finalize()
        '''

    def init(self):
        """ Initialize OpenMC instance in memory and set up 
        necessary CMFD parameters.

        """
        # Configure CMFD parameters
        self._configure_cmfd()

        # Create tally objects
        self._create_cmfd_tally()

        # Initialize simulation
        openmc.lib.simulation_init()

        # Set cmfd_run variable to True through C API
        openmc.lib.settings.cmfd_run = True

    def next_batch(self):
        """ Run next batch for CMFDRun.

        Returns
        -------
        int
            Status after running a batch (0=normal, 1=reached maximum number of
            batches, 2=tally triggers reached)

        """
        status = 1
        '''
        # Check to reset tallies
        if self._reset_every:
            self._cmfd_tally_reset()

        # Run next batch
        status = openmc.lib.next_batch()

        # Aggregate CMFD tallies and broadcast to CMFDNode
        if openmc.lib.master():
            if openmc.lib.current_batch() >= self._tally_begin:
                self._compute_xs()
        # TODO if current_batch >= self._solver_begin:
        #    compute sourcecounts
        #    if master:
        #        broadcast tallies AND sourcecounts to CMFD node
        '''
        return status

    def finalize(self):
        """ Finalize simulation by calling
        :func:`openmc.lib.simulation_finalize` and print out CMFD timing
        information.

        """
        # Finalize simuation
        openmc.lib.simulation_finalize()

    def _configure_cmfd(self):
        """Initialize CMFD parameters and set CMFD input variables"""
        # Define all variables necessary for running CMFD
        self._initialize_cmfd()

    def _initialize_cmfd(self):
        """Sets values of CMFD instance variables based on user input,
           separating between variables that only exist on all processes
           and those that only exist on the master process

        """
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

        # Define all variables that will exist only on master process
        if openmc.lib.master():
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

    def _set_tally_window(self):
        """Sets parameters to handle different tally window options"""
        # Set parameters for expanding window
        if self._window_type == 'expanding':
            self._reset_every = True
            self._window_size = 1
        # Set parameters for rolling window
        elif self.window_type == 'rolling':
            self._reset_every = True
        # Set parameters for default case, with no window
        else:
            self._window_size = 1
            self._reset_every = False

    def _cmfd_tally_reset(self):
        """Resets all CMFD tallies in memory"""
        # Print message
        if (openmc.lib.settings.verbosity >= 6 and openmc.lib.master() and
                not self._reset_every):
            print(' CMFD tallies reset')
            sys.stdout.flush()

        # Reset CMFD tallies
        tallies = openmc.lib.tallies
        for tally_id in self._tally_ids:
            tallies[tally_id].reset()

    def _cmfd_reweight(self):
        """Performs weighting of particles in source bank"""
        # Get spatial dimensions and energy groups
        nx, ny, nz, ng = self._indices

        # Count bank site in mesh and reverse due to egrid structured
        outside = self._count_bank_sites()

        # Check and raise error if source sites exist outside of CMFD mesh
        if openmc.lib.master() and outside:
            raise OpenMCError('Source sites outside of the CMFD mesh')

        # TODO broadcast sourcecounts to CMFDNode
        # TODO get weightfactors from CMFDNode

        m = openmc.lib.meshes[self._mesh_id]
        energy = self._egrid
        ng = self._indices[3]

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

        if openmc.lib.master() and np.any(source_energies < energy[0]):
            print(' WARNING: Source point below energy grid')
            sys.stdout.flush()
        if openmc.lib.master() and np.any(source_energies > energy[-1]):
            print(' WARNING: Source point above energy grid')
            sys.stdout.flush()

    def _count_bank_sites(self):
        """Determines the number of fission bank sites in each cell of a given
        mesh and energy group structure.
        Returns
        -------
        bool
            Wheter any source sites outside of CMFD mesh were found

        """
        # Initialize variables
        m = openmc.lib.meshes[self._mesh_id]
        bank = openmc.lib.source_bank()
        energy = self._egrid
        sites_outside = np.zeros(1, dtype=bool)
        nxnynz = np.prod(self._indices[0:3])
        ng = self._indices[3]

        outside = np.zeros(1, dtype=bool)
        self._sourcecounts = np.zeros((nxnynz, ng))
        count = np.zeros(self._sourcecounts.shape)

        # Get location and energy of each particle in source bank
        source_xyz = openmc.lib.source_bank()['r']
        source_energies = openmc.lib.source_bank()['E']

        # Convert xyz location to mesh index and ravel index to scalar
        mesh_locations = np.floor((source_xyz - m.lower_left) / m.width)
        mesh_bins = mesh_locations[:,2] * m.dimension[1] * m.dimension[0] + \
            mesh_locations[:,1] * m.dimension[0] + mesh_locations[:,0]

        # Check if any source locations lie outside of defined CMFD mesh
        if np.any(mesh_bins < 0) or np.any(mesh_bins >= np.prod(m.dimension)):
            outside[0] = True

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
        idx, counts = np.unique(np.array([mesh_bins, energy_bins]), axis=1,
                                return_counts=True)

        # Store counts to appropriate mesh-energy combination
        count[idx[0].astype(int), idx[1].astype(int)] = counts

        if have_mpi:
            # Collect values of count from all processors
            self._intracomm.Reduce(count, self._sourcecounts, MPI.SUM)
            # Check if there were sites outside the mesh for any processor
            self._intracomm.Reduce(outside, sites_outside, MPI.LOR)
        # Deal with case if MPI not defined (only one proc)
        else:
            sites_outside = outside
            self._sourcecounts = count

        return sites_outside[0]

    def _compute_xs(self):
        """Takes CMFD tallies from OpenMC and computes macroscopic cross
        sections, flux, and diffusion coefficients for each mesh cell using
        a tally window scheme

        """
        # Update window size for expanding window if necessary
        num_cmfd_batches = openmc.lib.current_batch() - self._tally_begin + 1
        if (self._window_type == 'expanding' and
                num_cmfd_batches == self._window_size * 2):
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

        # Get tallies in-memory
        tallies = openmc.lib.tallies

        # Get flux from CMFD tally 0
        tally_id = self._tally_ids[0]
        flux = tallies[tally_id].results[:,0,1]

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

        # Get total rr from CMFD tally 0
        totalrr = tallies[tally_id].results[:,1,1]

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

        # Get scattering rr from CMFD tally 1
        # flux is repeated to account for extra dimensionality of scattering xs
        tally_id = self._tally_ids[1]
        scattrr = tallies[tally_id].results[:,0,1]

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

        # Get nu-fission rr from CMFD tally 1
        nfissrr = tallies[tally_id].results[:,1,1]
        num_realizations = tallies[tally_id].num_realizations

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

        # Openmc source distribution is sum of nu-fission rr in incoming
        # energies
        openmc_src = np.sum(reshape_nfissrr, axis=3)

        # Bank OpenMC source distribution from current batch to
        # openmc_src_rate
        self._openmc_src_rate = np.append(self._openmc_src_rate, openmc_src,
                                          axis=4)

        # Get surface currents from CMFD tally 2
        tally_id = self._tally_ids[2]
        current = tallies[tally_id].results[:,0,1]

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

        # Get p1 scatter rr from CMFD tally 3
        tally_id = self._tally_ids[3]
        p1scattrr = tallies[tally_id].results[:,0,1]

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

    def _create_cmfd_tally(self):
        """Creates all tallies in-memory that are used to solve CMFD problem"""
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

            # Set all tallies to be active from beginning
            cmfd_tally.active = True


