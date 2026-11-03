// mpfix support for the DirectLUSolvers solvers (https://github.com/rhiestan/mpfix).
//
// The solvers work with an mpfix::Float scalar through mpfix/eigen.h alone --
// every tolerance derives from the scalar's own epsilon. What this header adds
// is speed in the extended-precision residuals of iterative refinement
// (LeftRightLU, MultifrontalQR, and RobustLU through them). Those emulate twice
// the working precision with error-free transformations by default, which for
// a software type costs ~17 operations per multiply-add; here they accumulate
// in an mpfix::Float with twice the precision instead, where the product is
// exact in one multiply. See ExtendedPrecision in LeftRightLUExtendedResidual.h.
//
// Include it IN PLACE OF mpfix/eigen.h, in every translation unit that uses a
// solver with an mpfix scalar: the specialization below has to be the one
// every instantiation sees.
//
//   #include <MpfixSupport>
//   #include <LeftRightLU>
//   Eigen::LeftRightLU<Eigen::SparseMatrix<mpfix::binary128>> lu(A);
//
// Needs mpfix (header-only) and C++20.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef DIRECT_LU_MPFIX_SUPPORT_H
#define DIRECT_LU_MPFIX_SUPPORT_H

#include "mpfix/eigen.h"

#include "LeftRightLUExtendedResidual.h"

namespace Eigen {
namespace left_right_lu {

/** A Float of P bits accumulates in one of 2P bits, with one more exponent bit
  * (as far as mpfix allows): the product of two working values -- including
  * two subnormals, or two near the overflow threshold -- is exact, and the
  * running sum cannot overflow where the working type would not. */
template <int P, int E>
struct ExtendedPrecision<mpfix::Float<P, E>> {
  typedef mpfix::Float<2 * P, (E < 29 ? E + 1 : E)> type;
};

}  // namespace left_right_lu
}  // namespace Eigen

#endif  // DIRECT_LU_MPFIX_SUPPORT_H
