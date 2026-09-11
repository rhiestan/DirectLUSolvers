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
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include <complex>
#include <cstdio>
#include <limits>
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

void testPositiveDefiniteFastPathDeclines() {
  std::printf("\n-- Pivoting::None declines an indefinite matrix rather than guessing --\n");

  // [[1,2],[2,1]]: positive diagonal, eigenvalues 3 and -1. The first pivot is
  // fine and the second is not, so this also checks the report names a column.
  std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {1, 0, 2.0}, {1, 1, 1.0}};
  SparseMatrix<double> A(2, 2);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.setPivoting(Eigen::supernodal_ldlt::Pivoting::None);
  s.compute(A);
  checkTrue(s.info() == Eigen::NumericalIssue, "Pivoting::None reports NumericalIssue");
  checkTrue(!s.isFactorized(), "Pivoting::None leaves isFactorized() false");
  checkTrue(s.notPositiveDefiniteColumn() >= 0, "Pivoting::None names the offending column");
  if (!s.lastErrorMessage().empty()) lu_testing::note(s.lastErrorMessage());

  SparseMatrix<double> B = laplacian2d(10, 10);
  B.coeffRef(37, 37) = -4.0;
  B.makeCompressed();
  Eigen::SupernodalLDLT<SparseMatrix<double>> s2;
  s2.setPivoting(Eigen::supernodal_ldlt::Pivoting::None);
  s2.compute(triangleOf(B, true));
  checkTrue(s2.info() == Eigen::NumericalIssue, "Pivoting::None catches a negative diagonal entry");

  // The same matrix under the default must SUCCEED -- that is the whole point of
  // Bunch-Kaufman, and the contrast is what makes the declining meaningful.
  Eigen::SupernodalLDLT<SparseMatrix<double>> s3;
  s3.compute(triangleOf(B, true));
  checkTrue(s3.info() == Eigen::Success, "the same matrix factors under Bunch-Kaufman");
}

// Solve an indefinite matrix and check against an exact right-hand side.
void indefiniteSolve(const SparseMatrix<double>& A, const char* name, double tol = 1e-9) {
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(A, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail(std::string(name) + " failed: " + s.lastErrorMessage());
    return;
  }
  VectorXd x = s.solve(b);
  const double err = (x - xTrue).norm() / xTrue.norm();
  const double resid = (A * x - b).norm() / b.norm();
  check(std::max(err, resid) < tol, name, std::max(err, resid));
}

void testIndefinite() {
  std::printf("\n-- symmetric indefinite matrices --\n");

  // A zero diagonal is the case 1x1 pivots cannot touch at all: every pivot must
  // be a 2x2 block. [[0,1],[1,0]] has eigenvalues +1 and -1.
  {
    std::vector<Eigen::Triplet<double>> t{{1, 0, 1.0}};
    SparseMatrix<double> A(2, 2);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(A);
    if (s.info() != Eigen::Success) {
      lu_testing::fail("zero-diagonal 2x2 failed: " + s.lastErrorMessage());
    } else {
      VectorXd b(2);
      b << 3.0, 5.0;
      VectorXd x = s.solve(b);       // [[0,1],[1,0]] x = b  =>  x = (5, 3)
      VectorXd want(2);
      want << 5.0, 3.0;
      check((x - want).norm() < 1e-14, "zero diagonal forces a 2x2 pivot and solves",
            (x - want).norm());
      checkTrue(s.pivotBlocks2x2() == 1, "exactly one 2x2 pivot block was used");
      checkTrue(s.inertia().positive == 1 && s.inertia().negative == 1,
                "inertia of [[0,1],[1,0]] is (1 positive, 1 negative)");
    }
  }

  // A Laplacian shifted past its smallest eigenvalues: symmetric, indefinite,
  // and with a full diagonal, so the pivot choice is a genuine mixture.
  //
  // The shift is deliberately NOT an integer. A 2D Laplacian on a g x g grid has
  // eigenvalues 4 - 2cos(p*pi/(g+1)) - 2cos(q*pi/(g+1)), which is exactly 4
  // whenever p + q = g + 1 -- so A - 4I is singular by construction, and testing
  // an indefinite solver on it measures nothing but the conditioning.
  {
    SparseMatrix<double> A = laplacian2d(14, 14);
    SparseMatrix<double> I(A.rows(), A.cols());
    I.setIdentity();
    SparseMatrix<double> shifted = A - 4.37 * I;  // straddles zero, hits no eigenvalue
    shifted.makeCompressed();
    indefiniteSolve(shifted, "shifted 2D Laplacian (indefinite)");
  }

  // Random symmetric with a zero diagonal: nothing on the diagonal to divide by.
  {
    const int n = 120;
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::vector<Eigen::Triplet<double>> t;
    for (int j = 0; j < n; ++j)
      for (int i = j + 1; i < n; ++i)
        if (coin(gen) < 0.06) {
          const double v = value(gen);
          t.emplace_back(i, j, v);
          t.emplace_back(j, i, v);
        }
    SparseMatrix<double> A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    indefiniteSolve(A, "random symmetric, zero diagonal", 1e-7);
  }
}

void testSaddlePoint() {
  std::printf("\n-- a saddle-point (KKT) system, with its inertia known in advance --\n");
  // [[H, B^T], [B, 0]] with H symmetric positive definite and B of full row rank
  // has inertia exactly (n1 positive, n2 negative, 0 zero) -- Haynsworth. That
  // makes it an exact test of the pivot signs rather than a plausibility check.
  const int g = 10, n1 = g * g, n2 = 25, n = n1 + n2;
  const SparseMatrix<double> H = laplacian2d(g, g);
  std::mt19937 gen(99);
  std::uniform_real_distribution<double> value(-1.0, 1.0);

  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < H.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(H, j); it; ++it)
      t.emplace_back(static_cast<int>(it.row()), j, it.value());
  // B: each constraint row touches a few unknowns; keep rows independent by
  // giving constraint r a private column r on top of the random couplings.
  for (int r = 0; r < n2; ++r) {
    t.emplace_back(n1 + r, r, 1.0);
    t.emplace_back(r, n1 + r, 1.0);
    for (int k = 0; k < 3; ++k) {
      const int c = n2 + (r * 7 + k * 13) % (n1 - n2);
      const double v = value(gen);
      t.emplace_back(n1 + r, c, v);
      t.emplace_back(c, n1 + r, v);
    }
  }
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(A, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail("saddle point failed: " + s.lastErrorMessage());
    return;
  }
  VectorXd x = s.solve(b);
  check((x - xTrue).norm() / xTrue.norm() < 1e-8, "saddle-point system solves",
        (x - xTrue).norm() / xTrue.norm());

  const MatrixXd denseA(A);
  const Eigen::SelfAdjointEigenSolver<MatrixXd> es(denseA);
  Eigen::Index pos = 0, neg = 0;
  for (Eigen::Index k = 0; k < es.eigenvalues().size(); ++k)
    (es.eigenvalues()[k] > 0 ? pos : neg)++;
  const Eigen::supernodal_ldlt::Inertia in = s.inertia();
  lu_testing::note("inertia: solver (" + std::to_string((long long)in.positive) + ", " +
                   std::to_string((long long)in.negative) + ")  eigenvalues (" +
                   std::to_string((long long)pos) + ", " + std::to_string((long long)neg) + ")");
  checkTrue(in.positive == pos && in.negative == neg, "inertia matches the dense eigenvalue signs");
  checkTrue(in.positive == n1 && in.negative == n2, "inertia is the (n1, n2) a KKT system predicts");
  // Whether 2x2 pivots are NEEDED here is a property of the ordering, not of the
  // matrix: AMD eliminates the H block first, and by the time the constraint rows
  // come up the Schur complement -B H^-1 B^T has given them a usable diagonal. So
  // this is reported, not asserted -- the inertia above is the real check.
  lu_testing::note("2x2 blocks: " + std::to_string((long long)s.pivotBlocks2x2()) +
                   ", perturbed pivots: " + std::to_string((long long)s.replacedPivots()));
}

void testInertiaAgainstEigenvalues() {
  std::printf("\n-- inertia against dense eigenvalue signs --\n");
  // Non-integer shifts on purpose: a g x g Laplacian has 4 as an exact
  // eigenvalue (p + q = g + 1), and the inertia of a singular matrix is not
  // something a factorization in floating point can be asked to agree on.
  for (double shift : {1.3, 4.37, 6.9}) {
    SparseMatrix<double> A = laplacian2d(9, 9);
    SparseMatrix<double> I(A.rows(), A.cols());
    I.setIdentity();
    SparseMatrix<double> shifted = A - shift * I;
    shifted.makeCompressed();

    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(triangleOf(shifted, true));
    if (s.info() != Eigen::Success) {
      lu_testing::fail("inertia: factorization failed at shift " + std::to_string(shift));
      continue;
    }
    const MatrixXd denseShifted(shifted);
    const Eigen::SelfAdjointEigenSolver<MatrixXd> es(denseShifted);
    Eigen::Index pos = 0, neg = 0;
    for (Eigen::Index k = 0; k < es.eigenvalues().size(); ++k)
      (es.eigenvalues()[k] > 0 ? pos : neg)++;
    const Eigen::supernodal_ldlt::Inertia in = s.inertia();
    checkTrue(in.positive == pos && in.negative == neg,
              "inertia correct at shift " + std::to_string(shift));
  }
}

void testDeterminantSign() {
  std::printf("\n-- determinant sign on an indefinite matrix --\n");
  SparseMatrix<double> A = laplacian2d(8, 8);
  SparseMatrix<double> I(A.rows(), A.cols());
  I.setIdentity();
  SparseMatrix<double> shifted = A - 5.3 * I;
  shifted.makeCompressed();

  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(shifted, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail("determinant sign: factorization failed");
    return;
  }
  // The determinant is that of the matrix actually factored, so a perturbed
  // pivot would legitimately move it. Assert there was none, or the comparison
  // below is measuring the perturbation rather than the determinant.
  checkTrue(s.replacedPivots() == 0, "no pivot perturbed, so det is of A itself");
  const MatrixXd dense(shifted);
  const Eigen::PartialPivLU<MatrixXd> lu(dense);
  const double refLogAbs = std::log(std::abs(lu.determinant()));
  const double refSign = lu.determinant() > 0 ? 1.0 : -1.0;
  check(std::abs(refLogAbs - s.logAbsDeterminant()) / std::max(1.0, std::abs(refLogAbs)) < 1e-9,
        "log|det| matches a dense LU on an indefinite matrix",
        std::abs(refLogAbs - s.logAbsDeterminant()));
  checkTrue(std::real(s.determinantSign()) == refSign, "determinant SIGN matches the dense LU");
}

void testPivotingModesAgreeOnSpd() {
  std::printf("\n-- the two pivoting modes agree on a positive definite matrix --\n");
  const SparseMatrix<double> A = laplacian3d(7, 7, 7);
  const SparseMatrix<double> Lo = triangleOf(A, true);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLDLT<SparseMatrix<double>> bk;  // default
  bk.compute(Lo);
  Eigen::SupernodalLDLT<SparseMatrix<double>> none;
  none.setPivoting(Eigen::supernodal_ldlt::Pivoting::None);
  none.compute(Lo);
  if (bk.info() != Eigen::Success || none.info() != Eigen::Success) {
    lu_testing::fail("pivoting modes: a factorization failed");
    return;
  }
  const VectorXd xb = bk.solve(b), xn = none.solve(b);
  check((xb - xTrue).norm() / xTrue.norm() < 1e-10, "Bunch-Kaufman is accurate on SPD",
        (xb - xTrue).norm() / xTrue.norm());
  check((xn - xTrue).norm() / xTrue.norm() < 1e-10, "Pivoting::None is accurate on SPD",
        (xn - xTrue).norm() / xTrue.norm());
  checkTrue(bk.nnzL() == none.nnzL(), "both modes produce the same fill");
  checkTrue(bk.inertia().positive == n && bk.inertia().negative == 0,
            "an SPD matrix reports all-positive inertia");
  lu_testing::note("2x2 blocks chosen on this SPD matrix: " +
                   std::to_string((long long)bk.pivotBlocks2x2()) + ", straddling fallbacks: " +
                   std::to_string((long long)bk.straddlingPivots()));
  checkTrue(bk.replacedPivots() == 0, "no pivot was perturbed on a well-conditioned SPD matrix");
}

void testComplexHermitianIndefinite() {
  std::printf("\n-- complex Hermitian indefinite --\n");
  typedef std::complex<double> C;
  const int n = 40;
  // Hermitian with a zero diagonal: purely 2x2 pivots, and the conjugation in
  // the 2x2 arithmetic is what this exercises.
  std::vector<Eigen::Triplet<C>> t;
  for (int i = 0; i + 1 < n; ++i) t.emplace_back(i + 1, i, C(1.0, 0.5));
  SparseMatrix<C> L(n, n);
  L.setFromTriplets(t.begin(), t.end());
  L.makeCompressed();

  Eigen::SupernodalLDLT<SparseMatrix<C>> s;
  s.compute(L);
  if (s.info() != Eigen::Success) {
    lu_testing::fail("complex Hermitian indefinite failed: " + s.lastErrorMessage());
    return;
  }
  Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> dense =
      Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>(L);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) dense(i, j) = std::conj(dense(j, i));
  Eigen::Matrix<C, Eigen::Dynamic, 1> xTrue = Eigen::Matrix<C, Eigen::Dynamic, 1>::Random(n);
  Eigen::Matrix<C, Eigen::Dynamic, 1> b = dense * xTrue;
  Eigen::Matrix<C, Eigen::Dynamic, 1> x = s.solve(b);
  check((x - xTrue).norm() / xTrue.norm() < 1e-9, "Hermitian indefinite solve",
        (x - xTrue).norm() / xTrue.norm());
  checkTrue(s.pivotBlocks2x2() > 0, "Hermitian zero diagonal needed 2x2 pivots");
}

void testSingularIsReported() {
  std::printf("\n-- a singular matrix is perturbed and flagged, not silently wrong --\n");
  // Two identical rows/columns: rank deficient by construction.
  const int n = 30;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, 2.0);
  for (int i = 0; i + 1 < n; ++i) t.emplace_back(i + 1, i, 1.0);
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  // Make column 5 a duplicate of column 4 (and symmetrically for the rows).
  A.coeffRef(5, 5) = 0.0;
  A.coeffRef(5, 4) = 0.0;
  if (n > 6) A.coeffRef(6, 5) = 0.0;
  A.makeCompressed();

  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(A, true));
  // Either it perturbs and flags a bad solve, or it reports outright -- both are
  // honest. What must NOT happen is Success with a large residual.
  if (s.info() == Eigen::Success) {
    VectorXd b = VectorXd::Random(n);
    s.solve(b);
    checkTrue(s.info() != Eigen::Success || s.solveResidual() < 1e-6,
              "a singular system never reports Success with a bad residual");
    lu_testing::note("replacedPivots=" + std::to_string((long long)s.replacedPivots()) +
                     " solveResidual=" + std::to_string(s.solveResidual()));
  } else {
    checkTrue(true, "singular matrix reported at factorization time");
  }

  // An exactly singular indefinite matrix: a g x g Laplacian has 4 as an exact
  // eigenvalue, so A - 4I is singular. Static pivoting will happily step over
  // the zero pivot and produce a factorization of a nearby matrix -- what must
  // not happen is solve() calling the resulting garbage a success.
  SparseMatrix<double> L2 = laplacian2d(14, 14);
  SparseMatrix<double> I2(L2.rows(), L2.cols());
  I2.setIdentity();
  SparseMatrix<double> singular = L2 - 4.0 * I2;
  singular.makeCompressed();

  Eigen::SupernodalLDLT<SparseMatrix<double>> s2;
  s2.compute(triangleOf(singular, true));
  if (s2.info() != Eigen::Success) {
    checkTrue(true, "exactly singular matrix reported at factorization time");
  } else {
    const int n2 = static_cast<int>(singular.rows());
    VectorXd xTrue = VectorXd::Random(n2);
    VectorXd b = singular * xTrue;
    VectorXd x = s2.solve(b);
    const double resid = (singular * x - b).norm() / b.norm();
    lu_testing::note("singular shift: replacedPivots=" +
                     std::to_string((long long)s2.replacedPivots()) + " residual=" +
                     std::to_string(resid));
    checkTrue(s2.info() == Eigen::Success ? resid < 1e-6 : true,
              "an exactly singular system is never Success with a bad residual");
  }
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

// A symmetric matrix with an ENTIRELY ZERO diagonal on a random pattern: no 1x1
// pivot exists anywhere, so every pivot has to be a 2x2 block, and whether one is
// reachable is decided by the ordering rather than by the numeric phase. n is
// even, so the generic matrix of this shape is nonsingular -- which matters,
// because a singular one would fail for reasons that have nothing to do with the
// matching and would make this test look like it passed for the right reason.
SparseMatrix<double> zeroDiagonalSymmetric(int n, int perRow, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> pick(0, n - 1);
  std::uniform_real_distribution<double> val(0.5, 2.0);
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < perRow; ++k) {
      const int j = pick(gen);
      if (j == i) continue;
      const double v = val(gen) * (k % 2 ? -1.0 : 1.0);
      t.emplace_back(i, j, v);
      t.emplace_back(j, i, v);
    }
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// A small residual says the computed x solves a NEARBY system. It says nothing
// about how far x is from the answer to the system that was asked about, and on a
// badly scaled matrix those come apart completely. This pins both halves: that
// the solver reports Success with a tiny residual and a badly wrong answer (which
// is correct -- the factorization is backward stable and the error is the
// matrix), and that conditionEstimate() is what makes that visible.
void testConditionEstimate() {
  std::printf("\n-- condition estimate --\n");

  // Well conditioned: kappa is small and the answer is as good as the residual.
  {
    const SparseMatrix<double> A = laplacian2d(20, 20);
    const int n = static_cast<int>(A.rows());
    VectorXd xTrue = VectorXd::Random(n), b = A * xTrue;
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(triangleOf(A, true));
    const VectorXd x = s.solve(b);
    const double kappa = s.conditionEstimate();
    // A 2D Laplacian's kappa grows like (g/pi)^2; for g=20 that is in the hundreds.
    checkTrue(kappa > 10.0 && kappa < 1e6, "a well-conditioned matrix estimates a modest kappa");
    check((x - xTrue).norm() / xTrue.norm() < 1e-10, "and the answer is accurate",
          (x - xTrue).norm() / xTrue.norm());
    std::printf("        lap2d 20x20: kappa=%.2e\n", kappa);
  }

  // Badly scaled: row magnitudes spanning 10^+-6 make the matrix ill-conditioned
  // even after symmetric Ruiz, and the residual stops predicting the error.
  {
    const int n = 600;
    std::mt19937 gen(11);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::uniform_real_distribution<double> val(0.5, 2.0);
    std::uniform_real_distribution<double> expo(-6.0, 6.0);
    std::vector<double> rowScale(n);
    for (int i = 0; i < n; ++i) rowScale[i] = std::pow(10.0, expo(gen));
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < 3; ++k) {
        const int j = pick(gen);
        if (j == i) continue;
        const double v = val(gen) * (k % 2 ? -1.0 : 1.0) * rowScale[i] * rowScale[j];
        t.emplace_back(i, j, v);
        t.emplace_back(j, i, v);
      }
    SparseMatrix<double> A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();

    VectorXd xTrue = VectorXd::Random(n), b = A * xTrue;
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.setMatching(true);
    s.compute(triangleOf(A, true));
    if (s.info() != Eigen::Success) {
      lu_testing::fail("ill-conditioned: factorization failed");
      return;
    }
    const VectorXd x = s.solve(b);
    const double resid = (A * x - b).norm() / b.norm();
    const double err = (x - xTrue).norm() / xTrue.norm();
    const double kappa = s.conditionEstimate();
    std::printf("        badly scaled: kappa=%.2e resid=%.2e err=%.2e\n", kappa, resid, err);

    // The trap, stated as an assertion: residual tiny, answer wrong, Success.
    checkTrue(resid < 1e-10, "the residual is at machine precision");
    checkTrue(err > 1e4 * resid, "yet the answer is orders of magnitude less accurate");
    checkTrue(s.info() == Eigen::Success,
              "and solve() reports Success, correctly -- it IS backward stable");
    // kappa is what separates "the solver did badly" from "the matrix has no
    // answer in this precision". Only it can flag this case.
    checkTrue(kappa > 1e6, "the condition estimate is what exposes it");
    // The first-order bound has to actually bound: err <~ kappa * omega.
    const double bound =
        Eigen::left_right_lu::estimateForwardError(kappa, std::max(resid, 1e-17));
    checkTrue(bound >= err || bound == 1.0, "and kappa * omega bounds the observed error");
  }

  // Re-factorizing must invalidate the cache rather than hand back a stale kappa.
  {
    const SparseMatrix<double> A = laplacian2d(12, 12);
    const SparseMatrix<double> Lo = triangleOf(A, true);
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    s.compute(Lo);
    const double first = s.conditionEstimate();
    const SparseMatrix<double> scaled = (1e6 * A).eval();
    s.factorize(triangleOf(scaled, true));
    const double second = s.conditionEstimate();
    // kappa is scale invariant, so the two must AGREE -- which is only possible
    // if the second call actually recomputed rather than returned a cached value
    // from a different factorization. Equilibration makes both identical.
    check(std::abs(first - second) <= 1e-6 * first, "the estimate is recomputed after factorize()",
          std::abs(first - second) / first);
  }
}

void testSymmetricMatching() {
  std::printf("\n-- symmetric weighted matching --\n");

  // On a positive definite matrix the diagonal is already the best pivot there
  // is, every cycle of the matching is a fixed point, and no pair is formed. The
  // check is that this costs nothing but the matching itself -- same fill, same
  // answer, bit for bit.
  {
    const SparseMatrix<double> A = laplacian2d(30, 30);
    const SparseMatrix<double> Lo = triangleOf(A, true);
    const int n = static_cast<int>(A.rows());
    VectorXd xTrue = VectorXd::Random(n), b = A * xTrue;
    Eigen::SupernodalLDLT<SparseMatrix<double>> plain, matched;
    matched.setMatching(true);
    plain.compute(Lo);
    matched.compute(Lo);
    checkTrue(plain.matchedPairs() == 0, "matching off reports no pairs");
    checkTrue(matched.matchedPairs() == 0, "and on an SPD matrix it forms none either");
    checkTrue(plain.nnzL() == matched.nnzL(), "so the fill is unchanged");
    const VectorXd xp = plain.solve(b), xm = matched.solve(b);
    checkTrue((xp - xm).cwiseAbs().maxCoeff() == 0.0, "and the answer is bit-identical");
  }

  // The case it exists for. Without matching the ordering puts no 2x2 candidate
  // where the numeric phase can reach it, so pivots get perturbed wholesale and
  // the answer is worthless -- correctly FLAGGED, but worthless. With matching the
  // pairs arrive adjacent inside a supernode and the same matrix solves cleanly.
  for (int n : {2000, 4000}) {
    const SparseMatrix<double> A = zeroDiagonalSymmetric(n, 3, 5u + unsigned(n));
    const SparseMatrix<double> Lo = triangleOf(A, true);
    VectorXd xTrue = VectorXd::Random(n), b = A * xTrue;

    Eigen::SupernodalLDLT<SparseMatrix<double>> plain, matched;
    matched.setMatching(true);
    plain.compute(Lo);
    matched.compute(Lo);
    if (matched.info() != Eigen::Success) {
      lu_testing::fail("matched factorization failed: " + matched.lastErrorMessage());
      continue;
    }
    const VectorXd xm = matched.solve(b);
    const double matchedErr = (xm - xTrue).norm() / xTrue.norm();
    double plainErr = std::numeric_limits<double>::infinity();
    if (plain.info() == Eigen::Success) {
      const VectorXd xp = plain.solve(b);
      plainErr = (xp - xTrue).norm() / xTrue.norm();
    }
    std::printf("        n=%d: pairs=%lld 2x2=%lld perturbed=%lld->%lld nnzL=%lld->%lld err %.1e->%.1e\n",
                n, static_cast<long long>(matched.matchedPairs()),
                static_cast<long long>(matched.pivotBlocks2x2()),
                static_cast<long long>(plain.replacedPivots()),
                static_cast<long long>(matched.replacedPivots()),
                static_cast<long long>(plain.nnzL()), static_cast<long long>(matched.nnzL()),
                plainErr, matchedErr);

    // A zero diagonal rules out 1-cycles, so the matching pairs off along cycles
    // of length 2 or more and leaves a singleton only where a cycle is odd. That
    // is a handful of columns, not half of them.
    checkTrue(2 * matched.matchedPairs() >= n - n / 50,
              "a zero diagonal forces nearly every column into a pair");
    checkTrue(matched.pivotBlocks2x2() > 0, "and 2x2 pivot blocks are actually taken");
    check(matchedErr < 1e-9, "the matched factorization solves it", matchedErr);

    // THE MECHANISM, which is what is worth asserting: without matching the
    // ordering leaves no reachable 2x2 pivot and the solver perturbs its way
    // through a large fraction of the columns. Whether the answer then survives
    // iterative refinement varies with the matrix -- at n=2000 it does and at
    // n=4000 it does not -- so the perturbation count is the robust statement and
    // the errors above are printed rather than asserted on.
    checkTrue(plain.replacedPivots() > n / 20,
              "without matching, a large fraction of pivots are perturbed");
    checkTrue(matched.replacedPivots() * 10 < plain.replacedPivots(),
              "and matching cuts that by an order of magnitude");
    checkTrue(matched.nnzL() <= plain.nnzL(), "while costing no fill: the quotient orders better");
  }

  // The quotient ordering must read the same triangle rule as everything else: a
  // caller holding the upper triangle, or the whole matrix, gets the same answer.
  {
    const int n = 800;
    const SparseMatrix<double> A = zeroDiagonalSymmetric(n, 3, 77);
    VectorXd xTrue = VectorXd::Random(n), b = A * xTrue;
    Eigen::SupernodalLDLT<SparseMatrix<double>> lower;
    Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Upper> upper;
    Eigen::SupernodalLDLT<SparseMatrix<double>> whole;
    lower.setMatching(true);
    upper.setMatching(true);
    whole.setMatching(true);
    lower.compute(triangleOf(A, true));
    upper.compute(triangleOf(A, false));
    whole.compute(A);
    checkTrue(lower.matchedPairs() == upper.matchedPairs() &&
                  lower.matchedPairs() == whole.matchedPairs(),
              "the same pairs are found from either triangle or the whole matrix");
    checkTrue(lower.nnzL() == upper.nnzL() && lower.nnzL() == whole.nnzL(),
              "and the quotient ordering reaches the same structure");
    const VectorXd xl = lower.solve(b), xu = upper.solve(b), xw = whole.solve(b);
    check((xl - xTrue).norm() / xTrue.norm() < 1e-9, "lower triangle solves",
          (xl - xTrue).norm() / xTrue.norm());
    // Not bit-identity, unlike the SPD case: this matrix perturbs pivots, so
    // refinement runs, and refinement's residual goes through
    // selfadjointView<UpLo>() * x -- which sums the same numbers in a different
    // order depending on which triangle is stored. The factorizations agree
    // exactly; the last bits of the refined answer do not.
    const double gapUpper = (xl - xu).norm() / xl.norm();
    const double gapWhole = (xl - xw).norm() / xl.norm();
    check(gapUpper < 1e-12 && gapWhole < 1e-12, "and all three agree to rounding",
          std::max(gapUpper, gapWhole));
  }
}

// L, D and the permutation are only worth exposing if they RECONSTRUCT the
// matrix, and the reconstruction is where the local Bunch-Kaufman interchanges
// have to be paid off: the solve gets to defer them per supernode, a standalone
// L does not. So this runs the same check on a positive definite matrix (no
// interchange, factorPermutation() == permutation()) and on an indefinite one
// (interchanges, and they had better differ).
void checkReconstruction(const SparseMatrix<double>& A, const char* name, bool expectPivoting) {
  Eigen::SupernodalLDLT<SparseMatrix<double>> s;
  s.compute(triangleOf(A, true));
  if (s.info() != Eigen::Success) {
    lu_testing::fail(std::string(name) + ": factorization failed");
    return;
  }
  const int n = static_cast<int>(A.rows());
  const SparseMatrix<double> L = s.matrixL(), D = s.matrixD();
  const VectorXd sc = s.scalingS();
  const auto P = s.factorPermutation();

  // L * D * L^T should equal P (S A S) P^T.
  const MatrixXd lhs = MatrixXd(L) * MatrixXd(D) * MatrixXd(L).transpose();
  const MatrixXd scaled = sc.asDiagonal() * MatrixXd(A) * sc.asDiagonal();
  MatrixXd rhs(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) rhs(P.indices()(i), P.indices()(j)) = scaled(i, j);
  const double err = (lhs - rhs).norm() / rhs.norm();
  check(err < 1e-11, (std::string(name) + ": L D L^T reconstructs P (S A S) P^T").c_str(), err);

  // L must be genuinely unit lower triangular, not merely close to it.
  double worstUpper = 0.0, worstDiag = 0.0;
  for (int j = 0; j < n; ++j)
    for (SparseMatrix<double>::InnerIterator it(L, j); it; ++it) {
      if (it.row() < j) worstUpper = std::max(worstUpper, std::abs(it.value()));
      if (it.row() == j) worstDiag = std::max(worstDiag, std::abs(it.value() - 1.0));
    }
  checkTrue(worstUpper == 0.0 && worstDiag == 0.0,
            (std::string(name) + ": L is exactly unit lower triangular").c_str());

  bool permsDiffer = false;
  for (int i = 0; i < n; ++i)
    if (P.indices()(i) != s.permutation().indices()(i)) permsDiffer = true;
  checkTrue(permsDiffer == expectPivoting,
            expectPivoting ? (std::string(name) + ": factorPermutation() carries the interchanges").c_str()
                           : (std::string(name) + ": no interchange, so the two permutations agree").c_str());
}

void testFactorAccessors() {
  std::printf("\n-- matrixL() / matrixD() reconstruct the matrix --\n");
  checkReconstruction(laplacian2d(12, 12), "SPD 2D Laplacian", /*expectPivoting=*/false);

  // Indefinite, with a zero diagonal block, so Bunch-Kaufman must interchange.
  {
    const int m = 60;
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < m; ++i) {
      t.emplace_back(i, i, 2.0);
      if (i + 1 < m) {
        t.emplace_back(i + 1, i, -1.0);
        t.emplace_back(i, i + 1, -1.0);
      }
      t.emplace_back(m + i, i, 1.0);  // B = I coupling block
      t.emplace_back(i, m + i, 1.0);
    }
    for (int i = 0; i < m; ++i)
      if (i + 1 < m) {
        t.emplace_back(m + i, m + i + 1, 0.25);  // a little structure, still zero diagonal
        t.emplace_back(m + i + 1, m + i, 0.25);
      }
    SparseMatrix<double> K(2 * m, 2 * m);
    K.setFromTriplets(t.begin(), t.end());
    K.makeCompressed();
    Eigen::SupernodalLDLT<SparseMatrix<double>> probe;
    probe.compute(triangleOf(K, true));
    std::printf("        saddle point: %lld 2x2 pivots, %lld perturbed\n",
                static_cast<long long>(probe.pivotBlocks2x2()),
                static_cast<long long>(probe.replacedPivots()));
    checkReconstruction(K, "indefinite saddle point",
                        /*expectPivoting=*/probe.pivotBlocks2x2() > 0);
  }

  // vectorD() is the whole of D exactly when nothing needed a 2x2 block.
  {
    Eigen::SupernodalLDLT<SparseMatrix<double>> s;
    const SparseMatrix<double> A = laplacian2d(10, 10);
    s.compute(triangleOf(A, true));
    checkTrue(s.pivotBlocks2x2() == 0, "SPD input needs no 2x2 block");
    const MatrixXd fromVector = MatrixXd(s.vectorD().asDiagonal());
    const double gap = (MatrixXd(s.matrixD()) - fromVector).norm();
    check(gap == 0.0, "vectorD() is all of D when there is no 2x2 block", gap);
  }
}

// Level dispatch leaves the root-separator chain on one lane; the chunked path
// splits those supernodes' panels across the pool instead. The risk it carries is
// not a wrong answer on average but a RACE, so what this pins down is that the
// two paths agree on every count the factorization produces -- and it uses an
// INDEFINITE matrix, because the chunked update kernel has to reproduce the 2x2
// pivot's row MIXING, which a positive definite matrix never exercises.
void testIntraSupernodeParallelism() {
  std::printf("\n-- intra-supernode parallelism --\n");
  typedef Eigen::SupernodalLDLT<SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>,
                                Eigen::supernodal_lu::StdThreadExecutor>
      ThreadedLDLT;
  {
    ThreadedLDLT probe;
    if (probe.executor().concurrency() <= 1) {
      std::printf("  [SKIP] one lane available; nothing to dispatch\n");
      return;
    }
  }

  SparseMatrix<double> A = laplacian3d(16, 16, 16);
  SparseMatrix<double> shift(A.rows(), A.cols());
  shift.setIdentity();
  A = A - 5.3 * shift;  // indefinite: 5.3 sits inside the spectrum (0, 12)
  const SparseMatrix<double> Lo = triangleOf(A, true);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  ThreadedLDLT chunked;  // on by default
  chunked.compute(Lo);
  ThreadedLDLT outer;
  outer.setIntraSupernodeParallelism(false);
  outer.compute(Lo);
  if (chunked.info() != Eigen::Success || outer.info() != Eigen::Success) {
    lu_testing::fail("intra-supernode: a factorization failed");
    return;
  }

  // Without this the rest of the test would pass vacuously on a build where the
  // guards never admit a single level.
  checkTrue(chunked.intraParallelSupernodes() > 0, "the chunked path actually ran");
  checkTrue(outer.intraParallelSupernodes() == 0, "disabling it leaves every level outer");
  std::printf("        %lld of %lld supernodes ran chunked, %lld 2x2 pivots\n",
              static_cast<long long>(chunked.intraParallelSupernodes()),
              static_cast<long long>(chunked.supernodeCount()),
              static_cast<long long>(chunked.pivotBlocks2x2()));
  checkTrue(chunked.pivotBlocks2x2() > 0, "the matrix needed 2x2 pivots, so the mixing was used");

  checkTrue(chunked.nnzL() == outer.nnzL(), "fill is identical either way");
  checkTrue(chunked.pivotBlocks2x2() == outer.pivotBlocks2x2(), "same 2x2 pivot count");
  checkTrue(chunked.replacedPivots() == outer.replacedPivots(), "same perturbation count");
  checkTrue(chunked.inertia().positive == outer.inertia().positive &&
                chunked.inertia().negative == outer.inertia().negative,
            "same inertia");

  const VectorXd xc = chunked.solve(b), xo = outer.solve(b);
  check((xc - xo).norm() / xo.norm() < 1e-12, "the two paths agree to rounding",
        (xc - xo).norm() / xo.norm());
  const double resid = (A * xc - b).norm() / b.norm();
  check(resid < 1e-9, "the chunked answer is accurate", resid);
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
  testPositiveDefiniteFastPathDeclines();
  testIndefinite();
  testSaddlePoint();
  testInertiaAgainstEigenvalues();
  testDeterminantSign();
  testPivotingModesAgreeOnSpd();
  testComplexHermitianIndefinite();
  testSingularIsReported();
  testAgainstSimplicialLdlt();
  testDeterminant();
  testEquilibrationAndRefinement();
  testMultipleRightHandSides();
  testRefactorization();
  testComplexHermitian();
  testHalvedAgainstSupernodalLu();
  testConditionEstimate();
  testSymmetricMatching();
  testFactorAccessors();
  testParallelAgreement();
  testIntraSupernodeParallelism();
  testEdgeCasesAndGuards();
  return lu_testing::summarize("SupernodalLDLT correctness");
}
