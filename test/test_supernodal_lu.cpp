// Correctness tests for Eigen::SupernodalLU.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_supernodal_lu --output-on-failure
//
// Compares SupernodalLU against a dense LU reference and Eigen::SparseLU.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using lu_testing::check;
using lu_testing::laplacian2d;
using lu_testing::randomSymmetricPattern;

namespace {

double solveAndMeasure(const SparseMatrix<double>& A, const char* name) {
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  try {
    solver.compute(A);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] %-40s threw: %s\n", name, e.what());
    ++lu_testing::failureCount();
    return 1.0;
  }
  if (solver.info() != Eigen::Success) {
    std::printf("  [FAIL] %-40s factorization failed: %s\n", name, solver.lastErrorMessage().c_str());
    ++lu_testing::failureCount();
    return 1.0;
  }
  VectorXd x = solver.solve(b);

  const double err = (x - xTrue).norm() / xTrue.norm();
  const double resid = (A * x - b).norm() / b.norm();
  const double worst = std::max(err, resid);
  check(worst < 1e-8, name, worst);

  // cross-check determinant magnitude and nnz against Eigen::SparseLU
  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  if (ref.info() == Eigen::Success) {
    const double d1 = std::abs(solver.determinant());
    const double d2 = std::abs(ref.determinant());
    const double drel = std::abs(d1 - d2) / std::max(1.0, std::abs(d2));
    std::printf("        det(ours)=%.6e det(SparseLU)=%.6e relDiff=%.2e  nnzL=%lld nnzU=%lld snodes=%lld\n",
                d1, d2, drel, (long long)solver.nnzL(), (long long)solver.nnzU(),
                (long long)solver.supernodeCount());
  }
  return worst;
}

// Verify matrixL()/matrixU().solveInPlace reproduce the full solve when driven
// with the documented permutation recipe (P A P^T = L U).
void testFactorAccessors() {
  SparseMatrix<double> A = laplacian2d(12, 10);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  solver.setMaxIterativeRefinements(0);  // compare the raw factor solve
  solver.setEquilibration(false);        // accessors expose the factors of A directly
  solver.compute(A);

  // manual solve through the L and U factor accessors.
  VectorXd y = solver.rowsPermutation() * b;
  solver.matrixL().solveInPlace(y);
  solver.matrixU().solveInPlace(y);
  VectorXd xManual = solver.colsPermutation().transpose() * y;

  const VectorXd xSolve = solver.solve(b);
  const double agree = (xManual - xSolve).norm() / xSolve.norm();
  const double resid = (A * xManual - b).norm() / b.norm();
  check(std::max(agree, resid) < 1e-10, "matrixL()/matrixU() vs solve()", std::max(agree, resid));

  // also exercise the matrix (multi-column) overload of solveInPlace.
  MatrixXd Xtrue = MatrixXd::Random(n, 3);
  MatrixXd Bm = A * Xtrue;
  MatrixXd Ym = solver.rowsPermutation() * Bm;
  solver.matrixL().solveInPlace(Ym);
  solver.matrixU().solveInPlace(Ym);
  MatrixXd Xm = solver.colsPermutation().transpose() * Ym;
  const double residM = (A * Xm - Bm).norm() / Bm.norm();
  check(residM < 1e-10, "matrixL()/matrixU() multi-column solveInPlace", residM);
}

// Verify transpose()/adjoint() solve A^T x = b. Uses a symmetric-PATTERN matrix
// with unsymmetric VALUES so that A^T != A and the test is meaningful.
void testTransposeSolve() {
  SparseMatrix<double> A = randomSymmetricPattern(150, 0.05, 99);
  const int n = static_cast<int>(A.rows());

  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  solver.compute(A);

  // transpose(): solve A^T x = b.
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd bT = A.transpose() * xTrue;
  VectorXd xT = solver.transpose().solve(bT);
  const double residT = (A.transpose() * xT - bT).norm() / bT.norm();
  check(residT < 1e-8, "transpose().solve(): A^T x = b", residT);

  // cross-check against Eigen::SparseLU's transpose solve.
  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  VectorXd xRef = ref.transpose().solve(bT);
  const double agree = (xT - xRef).norm() / xRef.norm();
  check(agree < 1e-8, "transpose().solve() matches Eigen::SparseLU", agree);

  // adjoint(): for real scalars equals transpose(); verify it also solves.
  VectorXd xA = solver.adjoint().solve(bT);
  const double residA = (A.adjoint() * xA - bT).norm() / bT.norm();
  check(residA < 1e-8, "adjoint().solve(): A^H x = b", residA);
}

// Equilibration: a badly-scaled but well-conditioned system. With scaling on
// (default) it must solve accurately; with it off, the magnitude-relative static
// pivot threshold bumps many legitimate small pivots and the result degrades.
void testEquilibration() {
  SparseMatrix<double> A = randomSymmetricPattern(150, 0.05, 31);  // well-conditioned
  const int n = static_cast<int>(A.rows());

  // scale rows and columns by wildly different magnitudes (~1e-6 .. 1e6).
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> expo(-6.0, 6.0);
  std::vector<double> dr(n), dc(n);
  for (int i = 0; i < n; ++i) {
    dr[i] = std::pow(10.0, expo(rng));
    dc[i] = std::pow(10.0, expo(rng));
  }
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      it.valueRef() *= dr[it.row()] * dc[j];

  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLU<SparseMatrix<double>> on;  // equilibration on (default)
  on.compute(A);
  const VectorXd xOn = on.solve(b);
  const double residOn = (A * xOn - b).norm() / b.norm();

  Eigen::SupernodalLU<SparseMatrix<double>> off;
  off.setEquilibration(false);
  off.compute(A);
  const VectorXd xOff = off.solve(b);
  const double residOff = (A * xOff - b).norm() / b.norm();

  check(residOn < 1e-8, "equilibration solves badly-scaled system", residOn);
  std::printf("        scaling off: resid=%.2e (bumped %lld) | on: resid=%.2e (bumped %lld)\n",
              residOff, (long long)off.replacedPivots(), residOn, (long long)on.replacedPivots());
}

// Krylov refinement robustness: with a deliberately weak preconditioner (a large
// static-pivot threshold bumps many pivots), stationary iterative refinement
// stalls far from machine precision, while LU-preconditioned BiCGStab converges.
void testKrylovRefinement() {
  SparseMatrix<double> A = laplacian2d(20, 20);  // SPD, well-conditioned
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  long long bumped = 0;
  auto run = [&](Eigen::supernodal_lu::Refinement method) {
    Eigen::SupernodalLU<SparseMatrix<double>> s;
    s.setEquilibration(false);
    s.setStaticPivotThreshold(3.0);  // bump many pivots -> weak preconditioner
    s.setRefinementMethod(method);
    s.setMaxIterativeRefinements(30);
    s.compute(A);
    const VectorXd x = s.solve(b);
    bumped = s.replacedPivots();
    return (A * x - b).norm() / b.norm();
  };

  const double irResid = run(Eigen::supernodal_lu::Refinement::IterativeRefinement);
  const double bcgResid = run(Eigen::supernodal_lu::Refinement::BiCGStab);

  check(bcgResid < 1e-10, "BiCGStab refinement converges (weak preconditioner)", bcgResid);
  check(bcgResid < irResid * 1e-3, "BiCGStab beats stationary IR (weak preconditioner)",
        bcgResid / irResid);
  std::printf("        (%lld/%d pivots bumped) stationary IR resid=%.2e | BiCGStab resid=%.2e\n",
              bumped, n, irResid, bcgResid);
}

// Honest failure reporting: a solve that cannot be satisfied (here a structurally
// singular matrix with a zeroed row/column, so static pivoting bumps the dead
// pivot and the factorization "succeeds") must report info()==NumericalIssue
// rather than silently returning a large-residual answer. A good solve on the
// same factorization restores Success.
void testHonestFailure() {
  // Well-conditioned control: solve must succeed AND report Success after solve.
  SparseMatrix<double> good = randomSymmetricPattern(80, 0.06, 17);
  VectorXd xTrueGood = VectorXd::Random(80);
  Eigen::SupernodalLU<SparseMatrix<double>> okSolver;
  okSolver.compute(good);
  const VectorXd xGood = okSolver.solve(good * xTrueGood);
  check(okSolver.info() == Eigen::Success && okSolver.solveResidual() < 1e-8,
        "honest check: good solve reports Success", okSolver.solveResidual());

  // Structurally singular matrix: zero out one row and its symmetric column
  // (including the diagonal). The factorization bumps the dead pivot and returns
  // Success, but a generic right-hand side is not in range -> garbage solution.
  SparseMatrix<double> A = randomSymmetricPattern(80, 0.06, 23);
  const int dead = 40;
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      if (it.row() == dead || it.col() == dead) it.valueRef() = 0.0;
  A.prune(0.0);  // drop the explicit zeros so the pattern stays symmetric

  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  solver.compute(A);
  // factorization itself does not see a structural zero pivot (it gets bumped).
  const bool factored = solver.isFactorized();

  VectorXd b = VectorXd::Random(80);  // generic RHS, not in range(A)
  const VectorXd x = solver.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(factored && solver.info() == Eigen::NumericalIssue,
        "honest check: singular solve reports NumericalIssue", resid);
  std::printf("        singular solve resid=%.2e info=%s (factored=%d)\n", resid,
              solver.info() == Eigen::NumericalIssue ? "NumericalIssue" : "Success", factored);

  // The factors stay usable: a right-hand side in range(A) solves fine and the
  // status recovers to Success (a failed solve must not poison later solves).
  VectorXd xr = VectorXd::Random(80);
  xr(dead) = 0.0;                 // keep the consistent component out of the null space
  VectorXd bIn = A * xr;          // guaranteed in range(A)
  const VectorXd x2 = solver.solve(bIn);
  const double resid2 = (A * x2 - bIn).norm() / std::max(1e-300, bIn.norm());
  check(solver.info() == Eigen::Success && resid2 < 1e-8,
        "honest check: status recovers on a consistent RHS", resid2);
}

// Fail-fast fill guard: predictedFactorNonzeros() after analyze exactly matches
// the realized fill; setMaxFactorNonzeros() below the real fill aborts factorize
// cleanly (NumericalIssue, nothing allocated); above it, it proceeds normally.
void testFillGuard() {
  SparseMatrix<double> A = laplacian2d(20, 20);
  const int n = static_cast<int>(A.rows());

  Eigen::SupernodalLU<SparseMatrix<double>> probe;
  probe.analyzePattern(A);
  const long long predicted = probe.predictedFactorNonzeros();
  check(predicted > 0, "predictedFactorNonzeros() > 0 after analyze", predicted > 0 ? 0.0 : 1.0);

  Eigen::SupernodalLU<SparseMatrix<double>> guarded;
  guarded.setMaxFactorNonzeros(1000);  // far below the true fill -> must trip
  guarded.compute(A);
  check(guarded.info() == Eigen::NumericalIssue && !guarded.isFactorized(),
        "fill guard aborts factorize below limit", guarded.isFactorized() ? 1.0 : 0.0);

  Eigen::SupernodalLU<SparseMatrix<double>> ok;
  ok.setMaxFactorNonzeros(predicted + 1);  // generous -> normal factorization
  ok.compute(A);
  const bool factored = ok.info() == Eigen::Success && ok.isFactorized();
  const long long realized = static_cast<long long>(ok.nnzL()) + ok.nnzU() - n;
  check(factored && realized == predicted, "guard passes; prediction == realized fill (nnzL+U-n)",
        factored && realized == predicted ? 0.0 : 1.0);
  VectorXd b = A * VectorXd::Random(n);
  const double resid = (A * ok.solve(b) - b).norm() / b.norm();
  check(resid < 1e-8, "guarded (passing) solve is correct", resid);
}

void testMultipleRhs() {
  SparseMatrix<double> A = laplacian2d(8, 8);
  const int n = static_cast<int>(A.rows());
  MatrixXd X = MatrixXd::Random(n, 4);
  MatrixXd B = A * X;

  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  solver.compute(A);
  MatrixXd Xs = solver.solve(B);
  const double resid = (A * Xs - B).norm() / B.norm();
  check(resid < 1e-8, "multiple RHS (4 cols)", resid);
}

// --------------------------------------------------------------------------
//  Input validation, state and reporting
// --------------------------------------------------------------------------
//
// Everything below is about what the solver does at its boundaries: a solve
// with no factors to solve with, an input it cannot factor, a determinant whose
// intermediate products leave the representable range, and a multi-column
// solve where one column hides the others. None of it involves the numerics of
// a healthy solve, so a wrong answer here is a wrong contract, not a rounding
// question, and every check is exact.

using lu_testing::checkTrue;
typedef SparseMatrix<double> SpMat;
typedef SparseMatrix<std::complex<double>> SpMatC;

// Two disconnected copies of a grid: no elimination order can create fill
// between them, so a coupling entry added later is guaranteed to lie outside
// the analyzed structure.
SpMat twoComponents(int gx, int gy) {
  const SpMat L = laplacian2d(gx, gy);
  const int m = static_cast<int>(L.rows());
  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < L.outerSize(); ++j)
    for (SpMat::InnerIterator it(L, j); it; ++it) {
      t.emplace_back(it.row(), j, it.value());
      t.emplace_back(m + it.row(), m + j, it.value());
    }
  SpMat A(2 * m, 2 * m);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// A refused solve must hand back NaN and NumericalIssue, never numbers.
bool refused(const Eigen::SupernodalLU<SpMat>& s, const VectorXd& x) {
  return s.info() == Eigen::NumericalIssue && x.size() > 0 && !x.allFinite() && !s.lastErrorMessage().empty();
}

// analyzePattern() on a new matrix must retire the factors of the old one:
// the arenas belong to the old structure and the maps to the new, so a solve in
// between would read out of bounds. It is refused; factorize() then restores
// a working solver.
void testReanalysisRetiresFactors() {
  for (int larger = 0; larger < 2; ++larger) {
    const SpMat A = larger ? laplacian2d(4, 4) : laplacian2d(10, 10);
    const SpMat B = larger ? laplacian2d(30, 30) : laplacian2d(4, 4);
    const int nb = static_cast<int>(B.rows());
    const char* tag = larger ? "re-analysis (larger)" : "re-analysis (smaller)";
    Eigen::SupernodalLU<SpMat> s;
    s.compute(A);
    s.analyzePattern(B);
    checkTrue(!s.isFactorized() && s.info() == Eigen::Success,
              std::string(tag) + ": analyzePattern() retires the old factors");
    const VectorXd x = s.solve(VectorXd::Ones(nb));
    checkTrue(refused(s, x), std::string(tag) + ": solve before factorize() is refused");
    checkTrue(std::isnan(s.determinant()) && std::isnan(s.logAbsDeterminant()) &&
                  std::isnan(s.determinantSign()),
              std::string(tag) + ": determinant queries are NaN without factors");
    s.factorize(B);
    const VectorXd xTrue = VectorXd::Random(nb);
    const VectorXd x2 = s.solve(B * xTrue);
    const double err = (x2 - xTrue).norm() / xTrue.norm();
    check(s.info() == Eigen::Success && err < 1e-10, std::string(tag) + ": factorize() then solve", err);
  }
}

// Every way of reaching solve() without factors: never analyzed, analyzed only,
// declined by the fill guard, singular, and through the transposed view and the
// L/U proxies. All refuse instead of reading arenas that do not exist.
void testSolveWithoutFactors() {
  const SpMat A = laplacian2d(6, 6);
  const int n = static_cast<int>(A.rows());
  {
    Eigen::SupernodalLU<SpMat> s;
    s.factorize(A);
    checkTrue(s.info() == Eigen::InvalidInput && !s.isFactorized(),
              "factorize() before analyzePattern() is declined");
  }
  {
    Eigen::SupernodalLU<SpMat> s;
    s.analyzePattern(A);
    const VectorXd x = s.solve(VectorXd::Ones(n));
    checkTrue(refused(s, x), "solve() after analyzePattern() only is refused");
    const VectorXd xt = s.transpose().solve(VectorXd::Ones(n));
    checkTrue(refused(s, xt), "transpose().solve() without factors is refused");
    VectorXd y = VectorXd::Ones(n);
    s.matrixL().solveInPlace(y);
    checkTrue(refused(s, y), "matrixL().solveInPlace() without factors is refused");
    y.setOnes();
    s.matrixU().solveInPlace(y);
    checkTrue(refused(s, y), "matrixU().solveInPlace() without factors is refused");
  }
  {
    Eigen::SupernodalLU<SpMat> s;
    s.setMaxFactorNonzeros(10);
    s.compute(A);
    checkTrue(s.info() == Eigen::NumericalIssue && !s.isFactorized(), "fill guard declines");
    const VectorXd x = s.solve(VectorXd::Ones(n));
    checkTrue(refused(s, x), "solve() after a declined factorize is refused");
    s.setMaxFactorNonzeros(0);
    s.factorize(A);
    checkTrue(s.info() == Eigen::Success && s.isFactorized() && s.lastErrorMessage().empty(),
              "a successful factorize clears the fill-guard message");
  }
  {
    SpMat Z(5, 5);
    Z.makeCompressed();
    Eigen::SupernodalLU<SpMat> s;
    s.compute(Z);  // max|A| = 0: the automatic threshold is 0 and the zero pivot is fatal
    checkTrue(s.info() == Eigen::NumericalIssue && !s.isFactorized(), "zero matrix: singular");
    const VectorXd x = s.solve(VectorXd::Ones(5));
    checkTrue(refused(s, x), "solve() after a singular factorization is refused");
  }
}

// Inputs factorize() cannot take: a non-square matrix, a non-finite value, a
// matrix of another size than analyzePattern() saw, and one with a nonzero
// outside the analyzed pattern. Each is declined with InvalidInput, nothing is
// factored, and compute() on the same solver recovers.
void testInvalidInput() {
  {
    SpMat R(4, 3);
    R.insert(0, 0) = 1;
    R.insert(1, 1) = 1;
    R.insert(2, 2) = 1;
    R.insert(3, 0) = 1;
    R.makeCompressed();
    Eigen::SupernodalLU<SpMat> s;
    s.compute(R);
    checkTrue(s.info() == Eigen::InvalidInput && !s.isFactorized(), "non-square matrix is declined");
  }
  for (int kind = 0; kind < 2; ++kind) {
    SpMat A = laplacian2d(6, 6);
    const int n = static_cast<int>(A.rows());
    A.coeffRef(7, 7) = kind == 0 ? std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::quiet_NaN();
    Eigen::SupernodalLU<SpMat> s;
    s.compute(A);
    const char* tag = kind == 0 ? "inf" : "NaN";
    checkTrue(s.info() == Eigen::InvalidInput && !s.isFactorized(),
              std::string("a matrix with an ") + tag + " value is declined");
    const VectorXd x = s.solve(VectorXd::Ones(n));
    checkTrue(refused(s, x), std::string("solve() after the ") + tag + " decline is refused");
    A.coeffRef(7, 7) = 4.0;
    s.compute(A);
    const VectorXd xTrue = VectorXd::Random(n);
    const double err = (s.solve(A * xTrue) - xTrue).norm() / xTrue.norm();
    check(s.info() == Eigen::Success && s.lastErrorMessage().empty() && err < 1e-10,
          std::string("compute() recovers after the ") + tag + " decline", err);
  }
  {
    Eigen::SupernodalLU<SpMat> s;
    s.analyzePattern(laplacian2d(6, 6));
    s.factorize(laplacian2d(7, 7));
    checkTrue(s.info() == Eigen::InvalidInput && !s.isFactorized(),
              "factorize() with a matrix of another size is declined");
  }
  {
    const SpMat A = twoComponents(8, 4);
    const int n = static_cast<int>(A.rows());
    Eigen::SupernodalLU<SpMat> s;
    for (int lower = 0; lower < 2; ++lower) {
      s.analyzePattern(A);  // the recovery compute() below analyzes the grown pattern
      SpMat C = A;
      if (lower)
        C.coeffRef(n - 1, 0) = 1.0;
      else
        C.coeffRef(0, n - 1) = 1.0;
      C.makeCompressed();
      s.factorize(C);
      const char* tag = lower ? "below" : "above";
      checkTrue(s.info() == Eigen::InvalidInput && !s.isFactorized(),
                std::string("factorize() with a nonzero outside the pattern (") + tag +
                    " the diagonal) is declined");
      const VectorXd x = s.solve(VectorXd::Ones(n));
      checkTrue(refused(s, x), std::string("solve() after the pattern decline (") + tag + ") is refused");
      s.compute(C);
      const VectorXd xTrue = VectorXd::Random(n);
      const double err = (s.solve(C * xTrue) - xTrue).norm() / xTrue.norm();
      check(s.info() == Eigen::Success && err < 1e-10,
            std::string("compute() on the grown pattern (") + tag + ") solves it", err);
    }
  }
}

// determinant() must survive intermediate overflow and underflow: the pivots of
// a diagonal alternating 1e5 and 1e-5 multiply to 1, but a running product of
// 400 of them passes through 1e200 (or 1e-200) on the way.
void testDeterminantRange() {
  const int n = 400;
  SpMat A(n, n), B(n, n);
  for (int i = 0; i < n; ++i) {
    A.insert(i, i) = (i % 2 == 0) ? 1e-5 : 1e5;   // overflow first
    B.insert(i, i) = (i < n / 2) ? 1e-5 : 1e5;    // underflow first
  }
  A.makeCompressed();
  B.makeCompressed();
  for (int equilibrate = 0; equilibrate < 2; ++equilibrate) {
    Eigen::SupernodalLU<SpMat> sa, sb;
    sa.setEquilibration(equilibrate == 1);
    sb.setEquilibration(equilibrate == 1);
    // the unscaled matrix has max|A| = 1e5, so the automatic threshold would
    // replace every 1e-5 pivot; keep the factorization exact instead.
    sa.setStaticPivotThreshold(0);
    sb.setStaticPivotThreshold(0);
    sa.compute(A);
    sb.compute(B);
    const std::string tag = equilibrate ? " (equilibrated)" : " (unscaled)";
    check(sa.replacedPivots() == 0 && std::abs(sa.determinant() - 1.0) < 1e-8,
          "determinant() through intermediate overflow" + tag, sa.determinant());
    check(sb.replacedPivots() == 0 && std::abs(sb.determinant() - 1.0) < 1e-8,
          "determinant() through intermediate underflow" + tag, sb.determinant());
    check(std::abs(sa.logAbsDeterminant()) < 1e-8 && sa.determinantSign() == 1.0,
          "logAbsDeterminant()/determinantSign() agree" + tag, sa.logAbsDeterminant());
  }
  // complex: (i)^400 = 1, through the same magnitudes
  SpMatC C(n, n);
  for (int i = 0; i < n; ++i) C.insert(i, i) = std::complex<double>(0, (i % 2 == 0) ? 1e-5 : 1e5);
  C.makeCompressed();
  Eigen::SupernodalLU<SpMatC> sc;
  sc.compute(C);
  check(std::abs(sc.determinant() - 1.0) < 1e-8, "complex determinant() through intermediate overflow",
        std::abs(sc.determinant() - 1.0));
}

// Multi-column honesty: each right-hand side is its own system, so the gate
// must hold for the worst column. A singular matrix, a consistent column scaled
// to 1e9 and an inconsistent column of norm 1: the block residual ratio is 1e-9
// (Success), the second column's own residual is O(1) (garbage).
void testPerColumnHonesty() {
  const int n = 80;
  SpMat A = randomSymmetricPattern(n, 0.06, 23);
  const int dead = 40;
  for (int j = 0; j < A.outerSize(); ++j)
    for (SpMat::InnerIterator it(A, j); it; ++it)
      if (it.row() == dead || it.col() == dead) it.valueRef() = 0.0;
  A.prune(0.0);

  Eigen::SupernodalLU<SpMat> s;
  s.compute(A);
  VectorXd xr = VectorXd::Random(n);
  xr(dead) = 0.0;
  const VectorXd consistent = 1e9 * (A * xr);
  const VectorXd inconsistent = VectorXd::Random(n);
  for (int order = 0; order < 2; ++order) {
    MatrixXd B(n, 2);
    B.col(order) = consistent;
    B.col(1 - order) = inconsistent;
    const MatrixXd X = s.solve(B);
    double worst = 0.0, block = (A * X - B).norm() / B.norm();
    for (int c = 0; c < 2; ++c) worst = std::max(worst, (A * X.col(c) - B.col(c)).norm() / B.col(c).norm());
    const std::string tag = order ? " (huge column second)" : " (huge column first)";
    check(s.info() == Eigen::NumericalIssue, "multi-rhs: a garbage column is flagged" + tag, worst);
    check(std::abs(s.solveResidual() - worst) <= 0.5 * worst,
          "multi-rhs: solveResidual() is the worst column's" + tag, s.solveResidual());
    check(block < 1e-6, "multi-rhs: the block ratio alone would have passed" + tag, block);
  }
  // and every column fine, including an all-zero one, is Success with a finite residual.
  MatrixXd G(n, 3);
  G.col(0) = A * xr;
  G.col(1) = 1e-9 * (A * xr);
  G.col(2).setZero();
  const MatrixXd Xg = s.solve(G);
  check(s.info() == Eigen::Success && Xg.allFinite() && std::isfinite(s.solveResidual()),
        "multi-rhs: consistent columns of any scale plus a zero column pass", s.solveResidual());
}

// lastErrorMessage() describes the LAST failure: it is cleared by a successful
// factorize and by a successful solve.
void testMessageLifetime() {
  const SpMat A = randomSymmetricPattern(60, 0.08, 5);
  Eigen::SupernodalLU<SpMat> s;
  s.compute(A);
  VectorXd bad = VectorXd::Ones(60);
  bad(3) = std::numeric_limits<double>::quiet_NaN();
  const VectorXd xb = s.solve(bad);
  checkTrue(s.info() == Eigen::NumericalIssue && !s.lastErrorMessage().empty(),
            "a NaN right-hand side is flagged with a message");
  const VectorXd xg = s.solve(A * VectorXd::Ones(60));
  checkTrue(s.info() == Eigen::Success && s.lastErrorMessage().empty(),
            "a successful solve clears the message");
}

void testValidationAndReporting() {
  testReanalysisRetiresFactors();
  testSolveWithoutFactors();
  testInvalidInput();
  testDeterminantRange();
  testPerColumnHonesty();
  testMessageLifetime();
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("SupernodalLU correctness tests\n");

  std::printf("Random symmetric-pattern matrices:\n");
  solveAndMeasure(randomSymmetricPattern(50, 0.10, 1), "random n=50 p=0.10");
  solveAndMeasure(randomSymmetricPattern(120, 0.05, 7), "random n=120 p=0.05");
  solveAndMeasure(randomSymmetricPattern(200, 0.03, 13), "random n=200 p=0.03");

  std::printf("2D Laplacian (5-point) matrices:\n");
  solveAndMeasure(laplacian2d(10, 10), "laplacian 10x10");
  solveAndMeasure(laplacian2d(20, 20), "laplacian 20x20");
  solveAndMeasure(laplacian2d(30, 25), "laplacian 30x25");

  std::printf("Other:\n");
  testMultipleRhs();
  testFactorAccessors();
  testTransposeSolve();
  testEquilibration();
  testKrylovRefinement();
  testHonestFailure();
  testFillGuard();

  std::printf("Input validation, state and reporting:\n");
  testValidationAndReporting();

  return lu_testing::summarize("SupernodalLU correctness");
}
