//! \file eigenvalue.h
//! \brief Data/functions related to k-eigenvalue calculations

#ifndef OPENMC_EIGENVALUE_H
#define OPENMC_EIGENVALUE_H

#include <array>
#include <cstdint> // for int64_t
#include <vector>

#include "xtensor/xtensor.hpp"
#include <hdf5.h>

#include "openmc/particle.h"
#include "pugixml.hpp"
#include "openmc/xml_interface.h"

namespace openmc {

class ConvergenceTally
{
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  //----------------------------------------------------------------------------
  // Methods

  void from_xml(pugi::xml_node node);

  //----------------------------------------------------------------------------
  // Accessors

  int dimension() const { return dimension_; }
  void set_dimension(int dimension_);

  int axial_order() const { return axial_order_; }
  void set_axial_order(int axial_order);

  int radial_order() const { return radial_order_; }
  void set_radial_order(int radial_order);

  double x() const { return x_; }
  void set_x(double x) { x_ = x; }

  double y() const { return y_; }
  void set_y(double y) { y_ = y; }

  double r() const { return r_; }
  void set_r(double r) { r_ = r; }

  double min() const { return min_; }
  double max() const { return max_; }
  void set_minmax(double min, double max);

  //----------------------------------------------------------------------------
  // Data members

protected:
  //! Dimension of FET expansion: 1=axial, 2=radial, 3=axial-radial
  int dimension_;

  //! Cartesian x coordinate for the origin of radial expansion.
  double x_;

  //! Cartesian y coordinate for the origin of radial expansion.
  double y_;

  //! Maximum radius from the origin covered by radial expansion.
  double r_;

  //! The minimum coordinate along the reference axis that the axial expansion covers.
  //! Assumes z-axis as reference axis
  double min_;

  //! The maximum coordinate along the reference axis that the expansion covers.
  //! Assumes z-axis as reference axis
  double max_;

  //! Order of polynomial for radial expansion
  int radial_order_;

  //! Order of polynomial for axial expansion
  int axial_order_;

};

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

extern double keff_generation; //!<  Single-generation k on each processor
extern std::array<double, 2> k_sum; //!< Used to reduce sum and sum_sq
extern std::vector<double> entropy; //!< Shannon entropy at each generation
extern std::vector<double> conv_tally;  //! Convergence tally data
extern xt::xtensor<double, 1> source_frac; //!< Source fraction for UFS

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

//! Collect/normalize the tracklength keff from each process
void calculate_generation_keff();

//! Calculate mean/standard deviation of keff during active generations
//!
//! This function sets the global variables keff and keff_std which represent
//! the mean and standard deviation of the mean of k-effective over active
//! generations. It also broadcasts the value from the master process.
void calculate_average_keff();

#ifdef _OPENMP
//! Join threadprivate fission banks into a single fission bank
//!
//! Note that this operation is necessarily sequential to preserve the order of
//! the bank when using varying numbers of threads.
void join_bank_from_threads();
#endif

//! Calculates a minimum variance estimate of k-effective
//!
//! The minimum variance estimate is based on a linear combination of the
//! collision, absorption, and tracklength estimates. The theory behind this can
//! be found in M. Halperin, "Almost linearly-optimum combination of unbiased
//! estimates," J. Am. Stat. Assoc., 56, 36-43 (1961),
//! doi:10.1080/01621459.1961.10482088. The implementation here follows that
//! described in T. Urbatsch et al., "Estimation and interpretation of keff
//! confidence intervals in MCNP," Nucl. Technol., 111, 169-182 (1995).
//!
//! \param[out] k_combined Estimate of k-effective and its standard deviation
//! \return Error status
extern "C" int openmc_get_keff(double* k_combined);
extern "C" int openmc_get_keff_gen(double* keff);

//! Sample/redistribute source sites from accumulated fission sites
void synchronize_bank();

//! Calculates the Shannon entropy of the fission source distribution to assess
//! source convergence
void shannon_entropy();

//! Calculates the convergence tally from fission bank for 1d case
void convergence_tally_1d();
//! Calculates the convergence tally from fission bank for 2d case
void convergence_tally_2d();

//! Determines the source fraction in each UFS mesh cell and reweights the
//! source bank so that the sum of the weights is equal to n_particles. The
//! 'source_frac' variable is used later to bias the production of fission sites
void ufs_count_sites();

//! Get UFS weight corresponding to particle's location
extern "C" double ufs_get_weight(const Particle* p);

//! Write data related to k-eigenvalue to statepoint
//! \param[in] group HDF5 group
void write_eigenvalue_hdf5(hid_t group);

//! Read data related to k-eigenvalue from statepoint
//! \param[in] group HDF5 group
void read_eigenvalue_hdf5(hid_t group);

} // namespace openmc

#endif // OPENMC_EIGENVALUE_H
