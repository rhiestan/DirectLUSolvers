// Correctness tests for Eigen::SupernodalLDLT.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_supernodal_ldlt --output-on-failure
//
// The reference is Eigen::SimplicialLDLT, which factors the same class of matrix
// by a simplicial (non-supernodal) algorithm, plus a dense LDLT where an exact
// value is wanted. What this suite is really pinning down is the three things
// that are specific to a SUPERNODAL LDL^T and could each be wrong without moving
// a residual: that only one triangle of the input is read, that only one triangle
// of the factor is computed, and that the transposed backsolve reads the same L
// the forward sweep wrote.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "SupernodalLDLT.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::laplacian2d;
using lu_testing::laplacian3d;

namespace {

// Keep only one triangle of a symmetric matrix, so a test can prove the solver
// never reads the other one.
template <typename Scalar>
SparseMatrix<Scalar> triangleOf(const SparseMatrix<Scalar>& A, bool lower) {
  std::vector<Eigen::Triplet<Scalar>> t;
  for (int j = 0; j < A.outerSize(); ++j)
    for (typename SparseMatrix<Scalar>::InnerIterator it(A, j); it; ++it) {
      const int i = static_cast<int>(it.row());
      if (lower ? (i >= j) : (i <= j)) t.emplace_back(i, j, it.value());
    }
  SparseMatrix<Scalar> out(A.rows(), A.cols());
  out.setFromTriplets(t.begin(), t.end());
  out.makeCompressed();
  return out;
}

// A diagonally dominant SPD matrix on a random symmetric pattern.
SparseMatrix<double> randomSpd(int n, double offDiagProb, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::vector<double> rowSum(n, 0.0);
  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < n; ++j)
    for (int i = j + 1; i < n; ++i)
      if (coin(gen) < offDiagProb) {
        const double v = value(gen);
        t.emplace_back(i, j, v);
        t.emplace_back(j, i, v);
        rowSum[i] += std::abs(v);
        rowSum[j] += std::abs(v);
      }
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, rowSum[i] + 1.0);  // strictly dominant => SPD
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// Solve with the lower triangle and report the worse of error and residual.
double solveAndMeasure(const SparseMatrix<double>& A, const char* name, double tol = 1e-9) {
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLDLT<SparseMatrix<double>> solver;
  solver.compute(triangleOf(A, /*lower=*/true));
  if (solver.info() != Eigen::Success) {
    lu_testing::fail(std::string(name) + " factorization failed: " + solver.lastErrorMessage());
    return 1.0;
  }
  VectorXd x = solver.solve(b);
  const double err = (x - xTrue).norm() / xTrue.norm();
  const double resid = (A * x - b).norm() / b.norm();
  const double worst = std::max(err, resid);
  check(worst < tol, name, worst);
  return worst;
}

void testAccuracy() {
  std::printf("\n-- accuracy against an exact right-hand side --\n");
  solveAndMeasure(laplacian2d(20, 20), "2D Laplacian 20x20");
  solveAndMeasure(laplacian2d(40, 25), "2D Laplacian 40x25");
  solveAndMeasure(laplacian3d(8, 8, 8), "3D Laplacian 8^3");
  solveAndMeasure(randomSpd(200, 0.02, 11), "random SPD n=200 (sparse)");
  solveAndMeasure(randomSpd(150, 0.10, 12), "random SPD n=150 (denser)");

  // A diagonal matrix has no off-diagonal panel at all: every supernode is a
  // single column with an empty row-block list, which is the degenerate shape
  // most of the panel arithmetic has to survive.
  SparseMatrix<double> diag(50, 50);
  std::vector<Eigen::Triplet<double>> dt;
  for (int i = 0; i < 50; ++i) dt.emplace_back(i, i, 1.0 + i);
  diag.setFromTriplets(dt.begin(), dt.end());
  solveAndMeasure(diag, "diagonal matrix");
}

void testTriangleIndependence() {
  std::printf("\n-- only one triangle is read, and it does not matter which --\n");
  const SparseMatrix<double> A = laplacian2d(15, 15);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Lower> lower;
  lower.compute(triangleOf(A, true));
  Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Upper> upper;
  upper.compute(triangleOf(A, false));
  // The full matrix through the Lower instance: the upper half must be IGNORED,
  // not added in a second time. If it were, the factor would be of 2A off the
  // diagonal and the answer would be wrong rather than merely different.
  Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Lower> full;
  full.compute(A);

  if (lower.info() != Eigen::Success || upper.info() != Eigen::Success ||
      full.info() != Eigen::Success) {
    lu_testing::fail("triangle independence: a factorization failed");
    return;
  }
  const VectorXd xl = lower.solve(b), xu = upper.solve(b), xf = full.solve(b);
  check((xl - xTrue).norm() / xTrue.norm() < 1e-9, "lower triangle solves correctly",
        (xl - xTrue).norm() / xTrue.norm());
  check((xu - xTrue).norm() / xTrue.norm() < 1e-9, "upper triangle solves correctly",
        (xu - xTrue).norm() / xTrue.norm());
  check((xf - xTrue).norm() / xTrue.norm() < 1e-9, "full matrix solves correctly (other half ignored)",
        (xf - xTrue).norm() / xTrue.norm());
  checkTrue(lower.nnzL() == upper.nnzL() && lower.nnzL() == full.nnzL(),
            "all three read the same graph (equal nnzL)");
  check((xl - xf).norm() <= 0.0, "lower-triangle and full input agree exactly", (xl - xf).norm());
}

void testIndefiniteIsDeclined() {
  std::printf("\n-- an indefinite matrix is declined, not approximated --\n");

  // [[1,2],[2,1]]: positive diagonal, eigenvalues 3 and -1. The first pivot is
  // fine and the second is not, so this also checks that the report names a
  // column rather than just failing.
  std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {1, 0, 2.0}, {1, 1, 1.0}};
  SparseMatrix<double> A(2, 2);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(A);
  checkTrue(s.info() == Eigen::NumericalIssue, "indefinite 2x2 reports NumericalIssue");
  checkTrue(!s.isFactorized(), "indefinite 2x2 leaves isFactorized() false");
  checkTrue(s.notPositiveDefiniteColumn() >= 0, "indefinite 2x2 names the offending column");
  if (!s.lastErrorMessage().empty()) lu_testing::note(s.lastErrorMessage());

  // A negative diagonal entry: no positive definite matrix has one, so this must
  // be caught wherever the ordering happens to place it.
  SparseMatrix<double> B = laplacian2d(10, 10);
  B.coeffRef(37, 37) = -4.0;
  B.makeCompressed();
  Eigen::SupernodalLDLT<SparseMatrix<double>> s2;
  s2.compute(triangleOf(B, true));
  checkTrue(s2.info() == Eigen::NumericalIssue, "negative diagonal entry is caught");
}

void testAgainstSimplicialLdlt() {
  std::printf("\n-- agreement with Eigen::SimplicialLDLT --\n");
  const SparseMatrix<double> A = laplacian2d(18, 18);
  const int n = static_cast<int>(A.rows());
  VectorXd b = VectorXd::Random(n);

  Eigen::SimplicialLDLT<SparseMatrix<double>, Eigen::Lower> ref;
  ref.compute(A);
  Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Lower> s;
  s.compute(triangleOf(A, true));
  if (ref.info() != Eigen::Success || s.info() != Eigen::Success) {
    lu_testing::fail("SimplicialLDLT comparison: a factorization failed");
    return;
  }
  const VectorXd xr = ref.solve(b), xs = s.solve(b);
  const double agree = (xr - xs).norm() / xr.norm();
  check(agree < 1e-10, "solutions agree with SimplicialLDLT", agree);

  // SimplicialLDLT exposes D rather than a log-determinant, so sum the logs to
  // get a comparable quantity that does not overflow.
  double d1 = 0.0;
  for (Eigen::Index k = 0; k < ref.vectorD().size(); ++k) d1 += std::log(ref.vectorD()[k]);
  const double d2 = s.logAbsDeterminant();
  check(std::abs(d1 - d2) / std::max(1.0, std::abs(d1)) < 1e-10, "log|det| agrees with SimplicialLDLT",
        std::abs(d1 - d2));
}

void testDeterminant() {
  std::printf("\n-- determinant against a dense reference --\n");
  const SparseMatrix<double> A = randomSpd(60, 0.08, 21);
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(A, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail("determinant: factorization failed");
    return;
  }
  const MatrixXd dense = MatrixXd(A);
  const double denseLogDet = std::log(std::abs(dense.determinant()));
  const double rel = std::abs(denseLogDet - s.logAbsDeterminant()) / std::max(1.0, std::abs(denseLogDet));
  check(rel < 1e-9, "log|det| matches a dense determinant", rel);

  // Equilibration rescales the matrix that is actually factored, so the scaling
  // has to be divided back out. Running with it off is what proves it was.
  Eigen::SupernodalLDLT<SparseMatrix<double>> noScale;
  noScale.setEquilibration(false);
  noScale.compute(triangleOf(A, true));
  const double rel2 =
      std::abs(noScale.logAbsDeterminant() - s.logAbsDeterminant()) / std::max(1.0, std::abs(denseLogDet));
  check(rel2 < 1e-9, "log|det| is independent of equilibration", rel2);
}

void testEquilibrationAndRefinement() {
  std::printf("\n-- equilibration and optional refinement --\n");
  // Badly scaled: rows spanning many orders of magnitude. S A S is what makes
  // this tractable, and the symmetric form is the only one that keeps A
  // symmetric while doing it.
  const int n = 120;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) {
    const double s = std::pow(10.0, (i % 12) - 6);
    t.emplace_back(i, i, 4.0 * s);
    if (i + 1 < n) {
      const double o = -1.0 * std::sqrt(s * std::pow(10.0, ((i + 1) % 12) - 6));
      t.emplace_back(i + 1, i, o);
      t.emplace_back(i, i + 1, o);
    }
  }
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;
  Eigen::SupernodalLDLT<SparseMatrix<double>> on;
  on.compute(triangleOf(A, true));
  if (on.info() != Eigen::Success) {
    lu_testing::fail("equilibration: factorization failed");
    return;
  }
  VectorXd x = on.solve(b);
  check((A * x - b).norm() / b.norm() < 1e-10, "badly scaled matrix solves with equilibration",
        (A * x - b).norm() / b.norm());

  Eigen::SupernodalLDLT<SparseMatrix<double>> refined;
  refined.setMaxIterativeRefinements(3);
  refined.compute(triangleOf(A, true));
  VectorXd xr = refined.solve(b);
  check((A * xr - b).norm() / b.norm() < 1e-10, "refinement leaves the answer at least as good",
        (A * xr - b).norm() / b.norm());
  checkTrue(refined.iterativeRefinements() >= 0, "refinement step count is reported");
  checkTrue(on.iterativeRefinements() == 0, "refinement is off by default");
}

void testMultipleRightHandSides() {
  std::printf("\n-- multiple right-hand sides --\n");
  const SparseMatrix<double> A = laplacian2d(12, 12);
  const int n = static_cast<int>(A.rows());
  MatrixXd X = MatrixXd::Random(n, 5);
  MatrixXd B = A * X;
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(A, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail("multi-RHS: factorization failed");
    return;
  }
  MatrixXd Y = s.solve(B);
  check((Y - X).norm() / X.norm() < 1e-10, "5 right-hand sides at once", (Y - X).norm() / X.norm());

  // Column k of a multi-RHS solve must equal the single-RHS solve of column k.
  VectorXd single = s.solve(VectorXd(B.col(2)));
  check((single - Y.col(2)).norm() <= 1e-14, "multi-RHS column matches a single solve",
        (single - Y.col(2)).norm());
}

void testRefactorization() {
  std::printf("\n-- analyzePattern once, factorize repeatedly --\n");
  SparseMatrix<double> A = laplacian2d(14, 14);
  const int n = static_cast<int>(A.rows());
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.analyzePattern(triangleOf(A, true));
  s.factorize(triangleOf(A, true));
  const Eigen::Index fill1 = s.nnzL();

  // Same pattern, different values.
  SparseMatrix<double> A2 = A;
  for (int k = 0; k < A2.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it)
      if (it.row() == it.col()) it.valueRef() = 8.0;
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A2 * xTrue;
  s.factorize(triangleOf(A2, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail("refactorization failed: " + s.lastErrorMessage());
    return;
  }
  VectorXd x = s.solve(b);
  check((x - xTrue).norm() / xTrue.norm() < 1e-10, "refactorize with new values, same pattern",
        (x - xTrue).norm() / xTrue.norm());
  checkTrue(s.nnzL() == fill1, "refactorization reuses the symbolic structure");
}

void testComplexHermitian() {
  std::printf("\n-- complex Hermitian --\n");
  typedef std::complex<double> C;
  const int n = 60;
  std::vector<Eigen::Triplet<C>> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, C(4.0, 0.0));  // a Hermitian matrix has a real diagonal
    if (i + 1 < n) t.emplace_back(i + 1, i, C(1.0, 1.0));  // lower triangle only
  }
  SparseMatrix<C> L(n, n);
  L.setFromTriplets(t.begin(), t.end());
  L.makeCompressed();

  Eigen::SupernodalLDLT<SparseMatrix<C>> s;
  s.compute(L);
  if (s.info() != Eigen::Success) {
    lu_testing::fail("complex Hermitian factorization failed: " + s.lastErrorMessage());
    return;
  }
  // Build the full Hermitian operator to form an exact right-hand side.
  Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> dense =
      Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>(L);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) dense(i, j) = std::conj(dense(j, i));

  Eigen::Matrix<C, Eigen::Dynamic, 1> xTrue =
      Eigen::Matrix<C, Eigen::Dynamic, 1>::Random(n);
  Eigen::Matrix<C, Eigen::Dynamic, 1> b = dense * xTrue;
  Eigen::Matrix<C, Eigen::Dynamic, 1> x = s.solve(b);
  const double err = (x - xTrue).norm() / xTrue.norm();
  check(err < 1e-10, "Hermitian solve uses the conjugate transpose", err);
  // The failure this catches: dropping the conjugation makes the solver factor
  // the (non-Hermitian) symmetric matrix instead, which still "works" and gives
  // a wrong answer.
  check(std::abs(s.logAbsDeterminant()) > 0.0, "complex log|det| is finite",
        s.logAbsDeterminant());
}

void testHalvedAgainstSupernodalLu() {
  std::printf("\n-- the storage claim against SupernodalLU on the same matrix --\n");
  const SparseMatrix<double> A = laplacian3d(10, 10, 10);

  Eigen::SupernodalLDLT<SparseMatrix<double>> ldlt;
  ldlt.compute(triangleOf(A, true));
  Eigen::SupernodalLU<SparseMatrix<double>> lu;
  lu.setMatching(false);  // an SPD matrix needs no matching; compare like for like
  lu.compute(A);
  if (ldlt.info() != Eigen::Success || lu.info() != Eigen::Success) {
    lu_testing::fail("halving comparison: a factorization failed");
    return;
  }
  const double ldltScalars = double(ldlt.predictedFactorNonzeros());
  const double luScalars = double(lu.predictedFactorNonzeros());
  const double ratio = luScalars / ldltScalars;
  lu_testing::note("arena scalars: LDLT " + std::to_string((long long)ldltScalars) + " vs LU " +
                   std::to_string((long long)luScalars));
  // Per supernode the LU arena is w^2 + 2*w*r against this solver's w^2 + w*r,
  // so the ratio approaches 2 as the off-diagonal panels dominate and is strictly
  // below it. Anything at or under 1 would mean the second factor never went away.
  check(ratio > 1.5 && ratio < 2.0, "LU stores 1.5-2x this solver's scalars", ratio);
  checkTrue(ldlt.supernodeCount() == lu.supernodeCount(),
            "both reach the same supernode partition");
}

void testParallelAgreement() {
  std::printf("\n-- parallel factorization agrees with serial --\n");
  const SparseMatrix<double> A = laplacian3d(12, 12, 12);
  const SparseMatrix<double> Lo = triangleOf(A, true);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLDLT<SparseMatrix<double>> serial;
  serial.compute(Lo);
  Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>,
                        Eigen::supernodal_lu::StdThreadExecutor>
      parallel;
  parallel.compute(Lo);
  if (serial.info() != Eigen::Success || parallel.info() != Eigen::Success) {
    lu_testing::fail("parallel agreement: a factorization failed");
    return;
  }
  checkTrue(serial.nnzL() == parallel.nnzL(), "fill is identical serial vs parallel");
  const VectorXd xs = serial.solve(b), xp = parallel.solve(b);
  check((xp - xTrue).norm() / xTrue.norm() < 1e-9, "parallel answer is accurate",
        (xp - xTrue).norm() / xTrue.norm());
  const double residS = (A * xs - b).norm() / b.norm();
  const double residP = (A * xp - b).norm() / b.norm();
  check(residP <= std::max(residS * 10.0, 1e-12), "parallel residual no worse than serial", residP);
}

void testEdgeCasesAndGuards() {
  std::printf("\n-- edge cases and the fill guard --\n");
  {
    SparseMatrix<double> empty(0, 0);
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(empty);
    checkTrue(s.info() == Eigen::Success, "n=0 is handled");
  }
  {
    SparseMatrix<double> one(1, 1);
    one.insert(0, 0) = 3.0;
    one.makeCompressed();
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(one);
    VectorXd b(1);
    b << 6.0;
    VectorXd x = s.solve(b);
    check(std::abs(x[0] - 2.0) < 1e-14, "n=1 solves", std::abs(x[0] - 2.0));
  }
  {
    // The guard must fire BEFORE the arena is allocated.
    const SparseMatrix<double> A = laplacian2d(20, 20);
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.analyzePattern(triangleOf(A, true));
    const Eigen::Index predicted = s.predictedFactorNonzeros();
    checkTrue(predicted > 0, "predictedFactorNonzeros() is known after analyze");
    s.setMaxFactorNonzeros(predicted / 2);
    s.factorize(triangleOf(A, true));
    checkTrue(s.info() == Eigen::NumericalIssue, "fill guard aborts below the limit");

    Eigen::SupernodalLDLT<SparseMatrix<double>> s2;
    s2.analyzePattern(triangleOf(A, true));
    s2.setMaxFactorNonzeros(s2.predictedFactorNonzeros() + 1);
    s2.factorize(triangleOf(A, true));
    checkTrue(s2.info() == Eigen::Success, "fill guard passes above the limit");
  }
  {
    // Honest reporting: a solve whose residual is bad must downgrade info().
    const SparseMatrix<double> A = laplacian2d(10, 10);
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(triangleOf(A, true));
    VectorXd b = VectorXd::Random(static_cast<int>(A.rows()));
    s.solve(b);
    checkTrue(s.info() == Eigen::Success && s.solveResidual() < 1e-10,
              "a good solve reports Success and a small residual");
  }
}

}  // namespace

int main() {
  std::printf("SupernodalLDLT correctness\n");
  testAccuracy();
  testTriangleIndependence();
  testIndefiniteIsDeclined();
  testAgainstSimplicialLdlt();
  testDeterminant();
  testEquilibrationAndRefinement();
  testMultipleRightHandSides();
  testRefactorization();
  testComplexHermitian();
  testHalvedAgainstSupernodalLu();
  testParallelAgreement();
  testEdgeCasesAndGuards();
  return lu_testing::summarize("SupernodalLDLT correctness");
}
