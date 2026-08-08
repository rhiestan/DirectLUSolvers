// Scalar-type coverage for both solvers.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   ctest --test-dir build -R test_scalar_types --output-on-failure
//
// WHY THIS EXISTS
//
// The README states the solvers accept "any scalar Eigen supports (double,
// float, std::complex<double>, ...)", but every other suite uses double. Two
// things were therefore claimed and never checked:
//
//   * float, where the auto static-pivot threshold sqrt(eps)*max|A| is ~1e-4
//     rather than ~1e-8, so the pivot-replacement logic operates in a very
//     different regime;
//   * complex, which is the only case where adjoint() and transpose() DIFFER.
//     test_supernodal_lu.cpp even says so explicitly ("for real scalars equals
//     transpose()"), so the conjugating path was entirely untested. Anything
//     that should conjugate but merely transposes -- the adjoint solve, the
//     determinant, equilibration, matching -- would pass every existing test.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <complex>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"

using lu_testing::check;
using lu_testing::checkTrue;

namespace {

template <typename Scalar>
using SpMat = Eigen::SparseMatrix<Scalar>;
template <typename Scalar>
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template <typename Scalar>
using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// A diagonally dominant matrix with a symmetric PATTERN and general values. For
// a complex Scalar the values are deliberately NOT hermitian, so A^H != A^T and
// the adjoint path is genuinely exercised.
template <typename Scalar>
SpMat<Scalar> makeMatrix(int n, double offDiagProb, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::vector<Eigen::Triplet<Scalar>> t;
  auto value = [&] {
    // For real Scalar the imaginary argument is discarded by the cast.
    return Scalar(typename Eigen::NumTraits<Scalar>::Real(uni(rng)));
  };
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (prob(rng) < offDiagProb) {
        t.emplace_back(i, j, value());
        t.emplace_back(j, i, value());
      }
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, Scalar(typename Eigen::NumTraits<Scalar>::Real(n)));
  SpMat<Scalar> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// Complex specialisation: give every entry a non-trivial imaginary part so that
// A^H differs from A^T everywhere it matters.
SpMat<std::complex<double>> makeComplexMatrix(int n, double offDiagProb, unsigned seed) {
  typedef std::complex<double> C;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::vector<Eigen::Triplet<C>> t;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (prob(rng) < offDiagProb) {
        t.emplace_back(i, j, C(uni(rng), uni(rng)));
        t.emplace_back(j, i, C(uni(rng), uni(rng)));  // not hermitian, not symmetric
      }
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, C(n + uni(rng), 0.3 * uni(rng)));
  SpMat<C> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

template <typename Scalar>
Vec<Scalar> onesLike(Eigen::Index n) {
  Vec<Scalar> v(n);
  v.setOnes();
  return v;
}

// --------------------------------------------------------------------------
//  float
// --------------------------------------------------------------------------

template <template <typename, typename, typename> class SolverTpl, typename Ordering,
          typename Executor>
void testFloat(const char* who) {
  typedef SolverTpl<SpMat<float>, Ordering, Executor> Solver;
  const int n = 200;
  SpMat<float> A = makeMatrix<float>(n, 0.05, 11);
  Vec<float> xTrue = onesLike<float>(n);
  for (int i = 0; i < n; ++i) xTrue(i) = 1.0f + 0.01f * static_cast<float>(i % 7);
  Vec<float> b = A * xTrue;

  Solver s;
  s.compute(A);
  checkTrue(s.info() == Eigen::Success, std::string(who) + " [float]: factorizes");
  const Vec<float> x = s.solve(b);
  const double resid = static_cast<double>((A * x - b).norm() / b.norm());
  // float carries ~7 decimal digits; 1e-5 is a real accuracy demand here, not a
  // rubber stamp.
  check(resid < 1e-5, std::string(who) + " [float]: relative residual", resid);
}

// --------------------------------------------------------------------------
//  complex
// --------------------------------------------------------------------------

template <template <typename, typename, typename> class SolverTpl, typename Ordering,
          typename Executor>
void testComplex(const char* who) {
  typedef std::complex<double> C;
  typedef SolverTpl<SpMat<C>, Ordering, Executor> Solver;
  const int n = 150;
  SpMat<C> A = makeComplexMatrix(n, 0.05, 23);
  const std::string tag = std::string(who) + " [complex]";

  Vec<C> xTrue(n);
  for (int i = 0; i < n; ++i) xTrue(i) = C(1.0 + 0.01 * i, 0.5 - 0.003 * i);
  const Vec<C> b = A * xTrue;

  Solver s;
  s.compute(A);
  checkTrue(s.info() == Eigen::Success, tag + ": factorizes");

  const Vec<C> x = s.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(resid < 1e-10, tag + ": A x = b residual", resid);
  const double err = (x - xTrue).norm() / xTrue.norm();
  check(err < 1e-10, tag + ": solution error", err);

  // --- the transpose/adjoint distinction, invisible for real scalars --------
  const Vec<C> bT = A.transpose() * xTrue;
  const Vec<C> xT = s.transpose().solve(bT);
  const double residT = (A.transpose() * xT - bT).norm() / bT.norm();
  check(residT < 1e-10, tag + ": transpose() solves A^T x = b", residT);

  const Vec<C> bH = A.adjoint() * xTrue;
  const Vec<C> xH = s.adjoint().solve(bH);
  const double residH = (A.adjoint() * xH - bH).norm() / bH.norm();
  check(residH < 1e-10, tag + ": adjoint() solves A^H x = b", residH);

  // If adjoint() merely transposed, this test would still pass above only by
  // accident; assert the two really are different operators on this matrix, so a
  // regression that collapses one into the other cannot hide.
  const double opDiff = (Mat<C>(A.transpose()) - Mat<C>(A.adjoint())).norm() /
                        Mat<C>(A.adjoint()).norm();
  checkTrue(opDiff > 1e-3, tag + ": A^T and A^H are genuinely different here");
  const Vec<C> xTofH = s.transpose().solve(bH);  // wrong operator on purpose
  const double wrong = (A.adjoint() * xTofH - bH).norm() / bH.norm();
  checkTrue(wrong > 1e-6, tag + ": transpose() does NOT satisfy the adjoint system");

  // --- multiple right-hand sides ------------------------------------------
  Mat<C> X(n, 3);
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < n; ++i) X(i, j) = C(0.5 + 0.01 * i, -0.25 + 0.02 * j);
  const Mat<C> B = A * X;
  const Mat<C> Xs = s.solve(B);
  const double residM = (A * Xs - B).norm() / B.norm();
  check(residM < 1e-10, tag + ": multi-RHS residual", residM);
}

// determinant() gets its own small matrix on purpose. det scales like the
// product of the pivots, so a diagonally dominant n=150 system has
// |det| ~ 150^150 ~ 1e326 -- past the top of double, where every solver
// (including Eigen::SparseLU) returns inf and any comparison is vacuous. n=14
// keeps it comfortably representable so the check tests the SIGN AND
// CONJUGATION logic, which is what can actually be wrong for complex scalars.
void testComplexDeterminant() {
  typedef std::complex<double> C;
  const int n = 14;
  SpMat<C> A = makeComplexMatrix(n, 0.30, 5);
  // Damp the diagonal so the product stays far from overflow.
  for (int j = 0; j < A.outerSize(); ++j)
    for (SpMat<C>::InnerIterator it(A, j); it; ++it)
      if (it.row() == it.col()) it.valueRef() = C(1.4, 0.2);

  Eigen::SparseLU<SpMat<C>> ref;
  ref.compute(A);
  if (ref.info() != Eigen::Success) {
    lu_testing::note("complex determinant: reference SparseLU failed; skipped");
    return;
  }
  const C expected = ref.determinant();
  checkTrue(std::isfinite(std::abs(expected)) && std::abs(expected) > 1e-12,
            "complex determinant: reference value is usable");

  Eigen::SupernodalLU<SpMat<C>> s;
  s.compute(A);
  const double dSnlu = std::abs(s.determinant() - expected) / std::abs(expected);
  check(dSnlu < 1e-8, "SupernodalLU [complex]: determinant matches SparseLU", dSnlu);

  Eigen::LeftRightLU<SpMat<C>> t;
  t.compute(A);
  const double dLrlu = std::abs(t.determinant() - expected) / std::abs(expected);
  check(dLrlu < 1e-8, "LeftRightLU [complex]: determinant matches SparseLU", dLrlu);

  // logAbsDeterminant() exists precisely because determinant() overflows; check
  // it agrees here, where both are representable. Both solvers now provide it.
  const double refLog = std::log(std::abs(expected));
  const double scale = std::max(1.0, std::abs(refLog));
  check(std::abs(t.logAbsDeterminant() - refLog) / scale < 1e-8,
        "LeftRightLU [complex]: logAbsDeterminant agrees",
        std::abs(t.logAbsDeterminant() - refLog) / scale);
  check(std::abs(s.logAbsDeterminant() - refLog) / scale < 1e-8,
        "SupernodalLU [complex]: logAbsDeterminant agrees",
        std::abs(s.logAbsDeterminant() - refLog) / scale);

  // determinantSign() must carry the complex PHASE, not just +/-1: sign * exp(log|det|)
  // has to reconstruct the determinant.
  for (int which = 0; which < 2; ++which) {
    const C sign = which == 0 ? s.determinantSign() : t.determinantSign();
    const double logAbs = which == 0 ? s.logAbsDeterminant() : t.logAbsDeterminant();
    const C rebuilt = sign * std::exp(logAbs);
    const double rel = std::abs(rebuilt - expected) / std::abs(expected);
    check(rel < 1e-8,
          std::string(which == 0 ? "SupernodalLU" : "LeftRightLU") +
              " [complex]: sign * exp(logAbs) reconstructs det",
          rel);
  }
}

// determinant() overflowing on a moderately sized system is not hypothetical --
// it is what broke the first version of the test above. Pin the documented
// remedy: logAbsDeterminant() must stay finite exactly where determinant() does
// not.
void testDeterminantOverflow() {
  const int n = 150;
  Eigen::SparseMatrix<double> A(n, n);
  for (int i = 0; i < n; ++i) A.insert(i, i) = 150.0;
  A.makeCompressed();
  const double expectedLog = n * std::log(150.0);

  // Both solvers must behave the same way here. SupernodalLU gained
  // logAbsDeterminant() only after this test showed LeftRightLU had the only
  // escape hatch from the overflow.
  {
    Eigen::SupernodalLU<Eigen::SparseMatrix<double>> s;
    s.compute(A);
    checkTrue(!std::isfinite(s.determinant()),
              "SupernodalLU: determinant() overflows on 150^150, as expected");
    const double logDet = s.logAbsDeterminant();
    checkTrue(std::isfinite(logDet),
              "SupernodalLU: logAbsDeterminant() stays finite where determinant() overflows");
    check(std::abs(logDet - expectedLog) / expectedLog < 1e-10,
          "SupernodalLU: logAbsDeterminant() value is correct",
          std::abs(logDet - expectedLog) / expectedLog);
    checkTrue(s.determinantSign() == 1.0, "SupernodalLU: determinantSign() is +1 here");
  }
  {
    Eigen::LeftRightLU<Eigen::SparseMatrix<double>> t;
    t.compute(A);
    checkTrue(!std::isfinite(t.determinant()),
              "LeftRightLU: determinant() overflows on 150^150, as expected");
    const double logDet = t.logAbsDeterminant();
    checkTrue(std::isfinite(logDet),
              "LeftRightLU: logAbsDeterminant() stays finite where determinant() overflows");
    check(std::abs(logDet - expectedLog) / expectedLog < 1e-10,
          "LeftRightLU: logAbsDeterminant() value is correct",
          std::abs(logDet - expectedLog) / expectedLog);
  }

  // A negative-determinant case, so the sign is not trivially +1: an odd number
  // of negative pivots must give -1 while log|det| stays unchanged.
  {
    Eigen::SparseMatrix<double> B(n, n);
    for (int i = 0; i < n; ++i) B.insert(i, i) = (i == 0) ? -150.0 : 150.0;
    B.makeCompressed();
    Eigen::SupernodalLU<Eigen::SparseMatrix<double>> s;
    s.compute(B);
    checkTrue(s.determinantSign() == -1.0, "SupernodalLU: determinantSign() detects a sign flip");
    check(std::abs(s.logAbsDeterminant() - expectedLog) / expectedLog < 1e-10,
          "SupernodalLU: log|det| unchanged by the sign flip",
          std::abs(s.logAbsDeterminant() - expectedLog) / expectedLog);
  }
}

// Equilibration and matching both reason about magnitudes; for complex scalars
// that means std::abs of a complex number, not a sign. Exercise them on a badly
// scaled complex system where they must actually engage.
void testComplexScalingAndMatching() {
  typedef std::complex<double> C;
  const int n = 120;
  SpMat<C> A = makeComplexMatrix(n, 0.06, 77);
  for (int j = 0; j < A.outerSize(); ++j)
    for (SpMat<C>::InnerIterator it(A, j); it; ++it)
      it.valueRef() *= std::pow(10.0, (it.row() % 9) - 4);  // rows span ~1e-4..1e4

  Vec<C> xTrue(n);
  for (int i = 0; i < n; ++i) xTrue(i) = C(1.0, 0.25);
  const Vec<C> b = A * xTrue;

  Eigen::SupernodalLU<SpMat<C>> s;
  s.setEquilibration(true);
  s.setMatching(true);
  s.compute(A);
  const Vec<C> x = s.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(resid < 1e-8, "complex: equilibration+matching on a badly scaled system", resid);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("Scalar-type coverage (float, complex<double>)\n");

  std::printf("float:\n");
  testFloat<Eigen::SupernodalLU, Eigen::AMDOrdering<int>, Eigen::supernodal_lu::SerialExecutor>(
      "SupernodalLU");
  testFloat<Eigen::LeftRightLU, Eigen::AMDOrdering<int>, Eigen::supernodal_lu::SerialExecutor>(
      "LeftRightLU");

  std::printf("complex<double>:\n");
  testComplex<Eigen::SupernodalLU, Eigen::AMDOrdering<int>, Eigen::supernodal_lu::SerialExecutor>(
      "SupernodalLU");
  testComplex<Eigen::LeftRightLU, Eigen::AMDOrdering<int>, Eigen::supernodal_lu::SerialExecutor>(
      "LeftRightLU");

  std::printf("complex robustness:\n");
  testComplexDeterminant();
  testDeterminantOverflow();
  testComplexScalingAndMatching();

  return lu_testing::summarize("Scalar types");
}
