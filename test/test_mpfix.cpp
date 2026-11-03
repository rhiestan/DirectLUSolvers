// The solvers with mpfix scalars (binary128, binary256, binary512: fixed-size
// IEEE 754 types, https://github.com/rhiestan/mpfix), side by side with double
// through the same templated checks as test_multiprecision.
//
// Needs mpfix (header-only, C++20); built only with -DDLU_WITH_MPFIX=ON.

// In place of mpfix/eigen.h: also routes the refinement residuals through an
// mpfix::Float of twice the precision (ExtendedPrecision).
#include <MpfixSupport>

#include <limits>
#include <type_traits>

#include "testing/Check.h"
#include "testing/MultiprecisionChecks.h"

using namespace mp_checks;
using mpfix::binary128;
using mpfix::binary256;
using mpfix::binary512;
// Reference for the residual sums: well beyond binary512's 978-bit accumulator.
using Ref = mpfix::Float<2048, 29>;

template <>
struct mp_checks::ScalarName<binary128> {
  static constexpr const char* value = "binary128";
};
template <>
struct mp_checks::ScalarName<binary256> {
  static constexpr const char* value = "binary256";
};
template <>
struct mp_checks::ScalarName<binary512> {
  static constexpr const char* value = "binary512";
};

template <typename T>
constexpr bool accumulatesWide() {
  using Wide = typename Eigen::left_right_lu::ExtendedPrecision<T>::type;
  return std::numeric_limits<Wide>::digits >= 2 * std::numeric_limits<T>::digits &&
         std::numeric_limits<Wide>::max_exponent >= std::numeric_limits<T>::max_exponent &&
         std::numeric_limits<Wide>::min_exponent <= std::numeric_limits<T>::min_exponent;
}
static_assert(accumulatesWide<binary128>() && accumulatesWide<binary256>() && accumulatesWide<binary512>(),
              "every mpfix width accumulates in a Float with twice its precision and a wider exponent range");

template <typename T>
void runAll() {
  extendedResidualAccuracy<T, Ref>();
  runQR<T>();
  runLU<T>();
  runOthers<T>();
}

int main() {
  runAll<double>();
  runAll<binary128>();
  runAll<binary256>();
  runAll<binary512>();
  return lu_testing::summarize("mpfix");
}
