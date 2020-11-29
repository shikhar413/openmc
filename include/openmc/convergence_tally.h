//! \file convergence_tally.h
//! \brief Data/functions related to FET convergence tally implementation

#ifndef OPENMC_CONVERGENCE_TALLY_H
#define OPENMC_CONVERGENCE_TALLY_H

#include <stdexcept> // for invalid_argument

#include "pugixml.hpp"

#include "openmc/xml_interface.h"

namespace openmc
{

class ConvergenceTally
{
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors
  ConvergenceTally(pugi::xml_node node);

  //----------------------------------------------------------------------------
  // Methods

  //! Calculate FET convergence results
  void compute();

  //! Scale FET convergence results by total source weight
  void scale();

  //----------------------------------------------------------------------------
  // Accessors

  int dimension() const { return dimension_; }
  void set_dimension(int dimension_);

  int axial_order() const { return axial_order_; }
  void set_axial_order(int order);

  int radial_order() const { return radial_order_; }
  void set_radial_order(int order);

  double x() const { return x_; }
  void set_x(double x) { x_ = x; }

  double y() const { return y_; }
  void set_y(double y) { y_ = y; }

  double r() const { return r_; }
  void set_r(double r) { r_ = r; }

  double min() const { return min_; }
  double max() const { return max_; }
  void set_minmax(double min, double max);

  int n_bins() const { return n_bins_; }

  // FET convergence tally results for each bin
  std::vector<double> results;

private:
  //----------------------------------------------------------------------------
  // Private methods

  //! Calculate FET convergence results for 1d Legendre expansion
  void compute_1d();

  //! Calculate FET convergence results for 2d Zernike expansion
  void compute_2d();

  //! Calculate FET convergence results for 3d Zernike-Legendre expansion
  void compute_3d();

  //----------------------------------------------------------------------------
  // Data members

protected:
  //! Dimension of FET expansion: 1=axial, 2=radial, 3=radial-axial
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

  //! Number of bins for expansion
  int n_bins_;
};

}  // namespace openmc
#endif // OPENMC_CONVERGENCE_TALLY_H
