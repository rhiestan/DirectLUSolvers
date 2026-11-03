// The solvers with a software floating-point scalar (Boost.Multiprecision's
// cpp_bin_float_quad: 113-bit mantissa, eps ~1.9e-34), run side by side with
// double through the same templated checks. Every tolerance is a multiple of
// the scalar's own epsilon, so a check that passes for double and fails for
// quad points at something in the solver tied to double -- a hard-wired
// tolerance, a std:: call that bypasses the type's own math, an fma the type
// does not round once. References are dense Eigen decompositions in the same
// scalar.
//
// Needs Boost.Multiprecision (header-only); built only with
// -DDLU_WITH_BOOST_MULTIPRECISION=ON.

// In place of the Boost headers: also routes the refinement residuals through a
// cpp_bin_float of twice the mantissa (ExtendedPrecision).
#include <BoostMultiprecisionSupport>

#include <LeftRightLU>
#include <MultifrontalQR>
#include <PointBlockLU>
#include <RobustLU>
#include <SupernodalLDLT>
#include <SupernodalLU>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "testing/Check.h"
#include "testing/MultiprecisionChecks.h"

using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::note;
using Eigen::Index;
using quad = boost::multiprecision::cpp_bin_float_quad;
using Ref600 = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<600>>;

using namespace mp_checks;

template <>
struct mp_checks::ScalarName<quad> {
  static constexpr const char* value = "quad";
};

// ---------------------------------------------------------------------------
// Extended-precision residual
// ---------------------------------------------------------------------------

static_assert(std::is_void<Eigen::left_right_lu::ExtendedPrecision<double>::type>::value,
              "double keeps the error-free-transformation emulation");
static_assert(std::numeric_limits<Eigen::left_right_lu::ExtendedPrecision<quad>::type>::digits >=
                  2 * std::numeric_limits<quad>::digits,
              "quad accumulates in a native type with at least twice its mantissa");

int main() {
  std::printf("--- extended residual ---\n");
  extendedResidualAccuracy<double, Ref600>();
  extendedResidualAccuracy<quad, Ref600>();
  runQR<double>();
  runQR<quad>();
  runLU<double>();
  runLU<quad>();
  runOthers<double>();
  runOthers<quad>();
  return lu_testing::summarize("multiprecision");
}
