// Boost.Multiprecision support for the DirectLUSolvers solvers.
//
// The solvers work with a cpp_bin_float scalar without this header -- every
// tolerance derives from the scalar's own epsilon. What it adds is speed in the
// extended-precision residuals of iterative refinement (LeftRightLU,
// MultifrontalQR, and RobustLU through them). Those emulate twice the working
// precision with error-free transformations by default, which for a software
// type costs ~25 library calls per multiply-add; here they accumulate in a
// cpp_bin_float with twice the mantissa instead, where the product is exact in
// one multiply. See ExtendedPrecision in LeftRightLUExtendedResidual.h.
//
// Include it IN PLACE OF the Boost headers, in every translation unit that uses
// a solver with a cpp_bin_float scalar: the specialization below has to be the
// one every instantiation sees.
//
//   #include <BoostMultiprecisionSupport>
//   #include <LeftRightLU>
//   using quad = boost::multiprecision::cpp_bin_float_quad;
//   Eigen::LeftRightLU<Eigen::SparseMatrix<quad>> lu(A);
//
// Needs Boost.Multiprecision (header-only).
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef DIRECT_LU_BOOST_MULTIPRECISION_SUPPORT_H
#define DIRECT_LU_BOOST_MULTIPRECISION_SUPPORT_H

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/eigen.hpp>

#include "LeftRightLUExtendedResidual.h"

namespace Eigen {
namespace left_right_lu {

/** A cpp_bin_float of D digits accumulates in one of 2D digits, in the same
  * base, with the default (far wider) exponent range: the product of two
  * working values is exact, and neither it nor the running sum can overflow
  * where the working type would not. Expression templates are off in the
  * accumulator -- it only ever sees one operation per statement. */
template <unsigned Digits, boost::multiprecision::backends::digit_base_type Base, class Allocator,
          class Exponent, Exponent MinExponent, Exponent MaxExponent,
          boost::multiprecision::expression_template_option ExpressionTemplates>
struct ExtendedPrecision<boost::multiprecision::number<
    boost::multiprecision::backends::cpp_bin_float<Digits, Base, Allocator, Exponent, MinExponent, MaxExponent>,
    ExpressionTemplates>> {
  typedef boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float<2 * Digits, Base>,
                                        boost::multiprecision::et_off>
      type;
};

}  // namespace left_right_lu
}  // namespace Eigen

#endif  // DIRECT_LU_BOOST_MULTIPRECISION_SUPPORT_H
