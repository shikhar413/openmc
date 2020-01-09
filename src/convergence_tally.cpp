#include "openmc/convergence_tally.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "openmc/bank.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/simulation.h"


#include "openmc/eigenvalue.h"

namespace openmc {

//==============================================================================
// ConvergenceTally implementation
//==============================================================================

ConvergenceTally::ConvergenceTally(pugi::xml_node node)
{
  auto dim = std::stoi(get_node_value(node, "dimension"));
  set_dimension(dim);

  // Set axial properties
  if (dim == 1 || dim == 3) {
    set_axial_order(std::stoi(get_node_value(node, "axial_order")));
    double min = std::stod(get_node_value(node, "min"));
    double max = std::stod(get_node_value(node, "max"));
    set_minmax(min, max);
    n_bins_ = axial_order_ + 1;
  }

  // Set radial properties
  if (dim == 2 || dim == 3) {
    set_radial_order(std::stoi(get_node_value(node, "radial_order")));
    x_ = std::stod(get_node_value(node, "x"));
    y_ = std::stod(get_node_value(node, "y"));
    r_ = std::stod(get_node_value(node, "r"));
    n_bins_ = ((radial_order_+1) * (radial_order_+2)) / 2;
  }
  // TODO: set n_bins_ properly for dim = 3
}

void
ConvergenceTally::set_radial_order(int order)
{
  if (order < 0) {
    throw std::invalid_argument{"Convergence tally radial order must be non-negative."};
  }
  radial_order_ = order;
}

void
ConvergenceTally::set_axial_order(int order)
{
  if (order < 0) {
    throw std::invalid_argument{"Convergence tally axial order must be non-negative."};
  }
  axial_order_ = order;
}

void
ConvergenceTally::set_dimension(int dimension)
{
  if (dimension > 3 || dimension < 1) {
    throw std::runtime_error{"Dimension for ConvergenceTally must be '1', '2', or '3'"};
  }
  dimension_ = dimension;
}

void
ConvergenceTally::set_minmax(double min, double max)
{
  if (max < min) {
    throw std::invalid_argument{"Maximum value must be greater than minimum value"};
  }
  min_ = min;
  max_ = max;
}

void
ConvergenceTally::compute()
{
  if (dimension_ == 1) compute_1d();
  else if (dimension_ == 2) compute_2d();
}

void
ConvergenceTally::compute_1d()
{
  // Assumes z axis define Legendre axis
  // Array storing convergence tally results
  double res[n_bins_] = {0};
  const int nthreads = omp_get_max_threads();

  // Store bank_size, fission_bank on stack since std::vector not threadsafe
  int bank_size = simulation::fission_bank.size();
  const auto& fission_bank = simulation::fission_bank;

  // Allocate and initialize res_private to 0.0
  double *res_private = new double[n_bins_*nthreads];
  for(int i=0; i < n_bins_*nthreads; i++) res_private[i] = 0.;

  #pragma omp parallel
  {
    const int ithread = omp_get_thread_num();

    // Compute P_n for each particle in fission bank on each thread
    // Store to res_private
    #pragma omp for
    for (auto i = 0; i < bank_size; i++) {
      const auto& bank = fission_bank[i];
      double x_norm = 2.0 * (bank.r.z - min_)/(max_ - min_) - 1.0;
      double res_tmp[n_bins_] = {0};
      calc_pn_c(axial_order_, x_norm, res_tmp);
      for (auto j = 0; j < n_bins_; j++) {
        res_private[ithread*n_bins_+j] += res_tmp[j];
      }
    }

    // Collapse res_private to res
    #pragma omp for
    for(int i=0; i < n_bins_; i++) {
      for(int t=0; t < nthreads; t++) {
        res[i] += res_private[n_bins_*t + i];
      }
    }
  }
  delete[] res_private;

  std::vector<double> results_local;
  results_.clear();
  simulation::conv_results.clear();
  for(int i = 0; i < n_bins_; i++) {
    results_local.push_back(res[i]);
    results_.push_back(0.0);
    simulation::conv_results.push_back(0.0);
  }

#ifdef OPENMC_MPI
  //MPI_Reduce(results_local.data(), results_.data(), results_local.size(),
  MPI_Reduce(results_local.data(), simulation::conv_results.data(), results_local.size(),
            MPI_DOUBLE, MPI_SUM, 0, mpi::intracomm);
#endif
}

void
ConvergenceTally::compute_2d()
{
  // Assumes x,y axes define plane for unit disk
  // Array storing convergence tally results
  double res[n_bins_] = {0};
  const int nthreads = omp_get_max_threads();

  // Store bank_size, fission_bank on stack since std::vector not threadsafe
  int bank_size = simulation::fission_bank.size();
  const auto& fission_bank = simulation::fission_bank;

  // Allocate and initialize res_private to 0.0
  double *res_private = new double[n_bins_*nthreads];
  for(int i=0; i < n_bins_*nthreads; i++) res_private[i] = 0.;

  #pragma omp parallel
  {
    const int ithread = omp_get_thread_num();

    // Compute Z_n for each particle in fission bank on each thread
    // Store to res_private
    #pragma omp for
    for (auto i = 0; i < bank_size; i++) {
      const auto& bank = fission_bank[i];
      double b_x = bank.r.x - x_;
      double b_y = bank.r.y - y_;
      double b_r = std::sqrt(b_x*b_x + b_y*b_y) / r_;
      double theta = std::atan2(b_y, b_x);

      if (b_r <= 1.0) {
        // Compute and return the Zernike weights.
        double res_tmp[n_bins_] = {0};
        calc_zn(radial_order_, b_r, theta, res_tmp);
        for (auto j = 0; j < n_bins_; j++)
          res_private[ithread*n_bins_+j] += res_tmp[j];
      }
    }

    // Collapse res_private to res
    #pragma omp for
    for(int i=0; i < n_bins_; i++) {
      for(int t=0; t < nthreads; t++) {
        res[i] += res_private[n_bins_*t + i];
      }
    }
  }
  delete[] res_private;

  std::vector<double> results_local;
  results_.clear();
  simulation::conv_results.clear();
  for(int i = 0; i < n_bins_; i++) {
    results_local.push_back(res[i]);
    results_.push_back(0.0);
    simulation::conv_results.push_back(0.0);
  }

#ifdef OPENMC_MPI
  //MPI_Reduce(results_local.data(), results_.data(), results_local.size(),
  MPI_Reduce(results_local.data(), simulation::conv_results.data(), results_local.size(),
            MPI_DOUBLE, MPI_SUM, 0, mpi::intracomm);
#endif
}

//==============================================================================
// C-API functions
//==============================================================================

extern "C" int openmc_get_convergence_tally(double** tally_data, int32_t* n)
{
  //if (simulation::conv_tally->results().size() == 0) {
  if (simulation::conv_results.size() == 0) {
    set_errmsg("Convergence tally has not been allocated");
    return OPENMC_E_ALLOCATE;
  }
  else {
    //*tally_data = simulation::conv_tally->results().data();
    //*n = simulation::conv_tally->results().size();
    *tally_data = simulation::conv_results.data();
    *n = simulation::conv_results.size();
    return 0;
  }
}

} // namespace openmc
