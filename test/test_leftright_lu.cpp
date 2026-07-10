// Correctness tests for Eigen::LeftRightLU (PARDISO-style left-right-looking LU
// with a barrier-free dynamic scheduler and in-block complete pivoting).
// Build (from repo root):
//   clang++ -std=c++17 -O2 -pthread -I eigen -I DirectLUSolvers/src \
//       DirectLUSolvers/test/test_leftright_lu.cpp -o build/test_leftright_lu
//
// Covers: direct solves, multiple RHS, factor accessors, transpose/adjoint,
// equilibration, complete vs partial vs no in-block pivoting, log-determinant,
// honest failure reporting, and parallel(dynamic-scheduler)-vs-serial agreement.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "LeftRightLU.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
namespace lr = Eigen::left_right_lu;

namespace {

int g_failures = 0;

void check(bool ok, const char* name, double residual) {
  std::printf("  [%s] %-46s residual = %.3e\n", ok ? "PASS" : "FAIL", name, residual);
  if (!ok) ++g_failures;
}

SparseMatrix<double> randomSymmetricPattern(int n, double offDiagProb, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (prob(rng) < offDiagProb) {
        triplets.emplace_back(i, j, uni(rng));
        triplets.emplace_back(j, i, uni(rng));
      }
  for (int i = 0; i < n; ++i) triplets.emplace_back(i, i, n + uni(rng));
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

SparseMatrix<double> laplacian2d(int gx, int gy) {
  const int n = gx * gy;
  std::vector<Eigen::Triplet<double>> triplets;
  auto id = [gx](int x, int y) { return y * gx + x; };
  for (int y = 0; y < gy; ++y)
    for (int x = 0; x < gx; ++x) {
      const int i = id(x, y);
      triplets.emplace_back(i, i, 4.0);
      if (x > 0) triplets.emplace_back(i, id(x - 1, y), -1.0);
      if (x + 1 < gx) triplets.emplace_back(i, id(x + 1, y), -1.0);
      if (y > 0) triplets.emplace_back(i, id(x, y - 1), -1.0);
      if (y + 1 < gy) triplets.emplace_back(i, id(x, y + 1), -1.0);
    }
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

// A symmetric-pattern matrix with a numerically WEAK diagonal but strong
// off-diagonal entries: exactly the case where in-block pivoting matters. The
// diagonal is small; large entries sit off-diagonal (still pattern-symmetric).
SparseMatrix<double> weakDiagonal(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i + 1 < n; ++i) {
    // strong off-diagonal couplings
    triplets.emplace_back(i, i + 1, 2.0 + uni(rng));
    triplets.emplace_back(i + 1, i, 2.0 + uni(rng));
  }
  for (int i = 0; i < n; i += 3) {  // a few longer-range couplings for real fill
    const int j = (i + 5) % n;
    triplets.emplace_back(std::min(i, j), std::max(i, j), 1.5 + uni(rng));
    triplets.emplace_back(std::max(i, j), std::min(i, j), 1.5 + uni(rng));
  }
  for (int i = 0; i < n; ++i) triplets.emplace_back(i, i, 1e-3 * uni(rng));  // weak diagonal
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

template <typename Solver>
double solveResidual(Solver& solver, const SparseMatrix<double>& A, const VectorXd& b) {
  const VectorXd x = solver.solve(b);
  return (A * x - b).norm() / b.norm();
}

double solveAndMeasure(const SparseMatrix<double>& A, const char* name) {
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    std::printf("  [FAIL] %-46s factorization failed: %s\n", name, solver.lastErrorMessage().c_str());
    ++g_failures;
    return 1.0;
  }
  VectorXd x = solver.solve(b);
  const double err = (x - xTrue).norm() / xTrue.norm();
  const double resid = (A * x - b).norm() / b.norm();
  const double worst = std::max(err, resid);
  check(worst < 1e-8, name, worst);

  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  if (ref.info() == Eigen::Success) {
    const double d1 = std::abs(solver.determinant());
    const double d2 = std::abs(ref.determinant());
    const double drel = std::abs(d1 - d2) / std::max(1.0, std::abs(d2));
    std::printf("        det(ours)=%.6e det(SparseLU)=%.6e relDiff=%.2e  nnzL=%lld snodes=%lld\n",
                d1, d2, drel, (long long)solver.nnzL(), (long long)solver.supernodeCount());
  }
  return worst;
}

void testMultipleRhs() {
  SparseMatrix<double> A = laplacian2d(8, 8);
  const int n = static_cast<int>(A.rows());
  MatrixXd X = MatrixXd::Random(n, 4);
  MatrixXd B = A * X;
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  MatrixXd Xs = solver.solve(B);
  check((A * Xs - B).norm() / B.norm() < 1e-8, "multiple RHS (4 cols)", (A * Xs - B).norm() / B.norm());
}

void testFactorAccessors() {
  SparseMatrix<double> A = laplacian2d(12, 10);
  const int n = static_cast<int>(A.rows());
  VectorXd b = A * VectorXd::Random(n);
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.setMaxIterativeRefinements(0);
  solver.setEquilibration(false);
  solver.compute(A);

  VectorXd y = solver.rowsPermutation() * b;
  solver.matrixL().solveInPlace(y);
  solver.matrixU().solveInPlace(y);
  VectorXd xManual = solver.colsPermutation().transpose() * y;
  const double resid = (A * xManual - b).norm() / b.norm();
  check(resid < 1e-10, "matrixL()/matrixU() reproduce the solve", resid);
}

void testTransposeSolve() {
  SparseMatrix<double> A = randomSymmetricPattern(150, 0.05, 99);
  const int n = static_cast<int>(A.rows());
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);

  VectorXd bT = A.transpose() * VectorXd::Random(n);
  VectorXd xT = solver.transpose().solve(bT);
  check((A.transpose() * xT - bT).norm() / bT.norm() < 1e-8, "transpose().solve(): A^T x = b",
        (A.transpose() * xT - bT).norm() / bT.norm());

  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  VectorXd xRef = ref.transpose().solve(bT);
  check((xT - xRef).norm() / xRef.norm() < 1e-8, "transpose().solve() matches Eigen::SparseLU",
        (xT - xRef).norm() / xRef.norm());

  VectorXd bA = A.adjoint() * VectorXd::Random(n);
  VectorXd xA = solver.adjoint().solve(bA);
  check((A.adjoint() * xA - bA).norm() / bA.norm() < 1e-8, "adjoint().solve(): A^H x = b",
        (A.adjoint() * xA - bA).norm() / bA.norm());
}

// In-block pivoting: a weak-diagonal matrix. Complete pivoting (default) should
// solve accurately; compare the three modes. All go through matching + static
// pivoting + refinement, so all should be usable, but this exercises the row+col
// interchange path and its solve folding directly.
void testCompletePivoting() {
  SparseMatrix<double> A = weakDiagonal(300, 7);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  auto run = [&](lr::Pivoting mode, const char* name) {
    Eigen::LeftRightLU<SparseMatrix<double>> s;
    s.setPivoting(mode);
    s.compute(A);
    const double resid = solveResidual(s, A, b);
    std::printf("        %-10s resid=%.2e bumped=%lld\n", name, resid, (long long)s.replacedPivots());
    return resid;
  };
  const double complete = run(lr::Pivoting::Complete, "complete");
  const double partial = run(lr::Pivoting::Partial, "partial");
  run(lr::Pivoting::None, "none");

  check(complete < 1e-8, "complete pivoting solves weak-diagonal system", complete);
  check(partial < 1e-8, "partial pivoting solves weak-diagonal system", partial);

  // determinant sign/magnitude with column swaps must still match Eigen::SparseLU.
  Eigen::LeftRightLU<SparseMatrix<double>> s;
  s.compute(A);
  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  const double drel = std::abs(std::abs(s.determinant()) - std::abs(ref.determinant())) /
                      std::max(1.0, std::abs(ref.determinant()));
  check(drel < 1e-6, "determinant matches SparseLU under complete pivoting", drel);
}

// Force the column-swap path: with matching OFF, the raw weak diagonal makes
// in-block complete pivoting actually interchange rows AND columns. This is the
// hardest code to get right (the per-supernode column permutation Q_s folded
// through the forward/backward and transpose solves and the determinant sign).
void testCompletePivotingColumnSwaps() {
  SparseMatrix<double> A = weakDiagonal(200, 3);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.setMatching(false);                    // no diagonal help -> pivoting must work
  solver.setPivoting(lr::Pivoting::Complete);
  solver.compute(A);
  check(solver.info() == Eigen::Success, "complete pivoting factorizes (matching off)",
        solver.info() == Eigen::Success ? 0.0 : 1.0);

  const VectorXd x = solver.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(resid < 1e-8, "column-swap path: forward solve correct", resid);

  VectorXd bT = A.transpose() * xTrue;
  const VectorXd xT = solver.transpose().solve(bT);
  const double residT = (A.transpose() * xT - bT).norm() / bT.norm();
  check(residT < 1e-8, "column-swap path: transpose solve correct", residT);

  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  const double drel = std::abs(std::abs(solver.determinant()) - std::abs(ref.determinant())) /
                      std::max(1.0, std::abs(ref.determinant()));
  const bool signOk = (solver.determinant() > 0) == (ref.determinant() > 0);
  check(drel < 1e-6 && signOk, "column-swap path: determinant magnitude & sign", drel);
  std::printf("        det(ours)=%+.6e det(SparseLU)=%+.6e\n", solver.determinant(), ref.determinant());
}

void testLogDeterminant() {
  SparseMatrix<double> A = laplacian2d(15, 15);  // SPD -> positive determinant
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  const double logAbs = solver.logAbsDeterminant();
  const double fromDet = std::log(std::abs(solver.determinant()));
  const double rel = std::abs(logAbs - fromDet) / std::max(1.0, std::abs(fromDet));
  check(rel < 1e-10, "logAbsDeterminant() consistent with log|determinant()|", rel);
  check(solver.determinantSign() > 0.0, "determinantSign() positive for SPD matrix",
        solver.determinantSign() > 0 ? 0.0 : 1.0);
  std::printf("        log|det| = %.6e  sign = %+.0f\n", logAbs, solver.determinantSign());
}

void testEquilibration() {
  SparseMatrix<double> A = randomSymmetricPattern(150, 0.05, 31);
  const int n = static_cast<int>(A.rows());
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> expo(-6.0, 6.0);
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      it.valueRef() *= std::pow(10.0, expo(rng));  // wreck the scaling
  VectorXd b = A * VectorXd::Random(n);
  Eigen::LeftRightLU<SparseMatrix<double>> on;
  on.compute(A);
  const double residOn = solveResidual(on, A, b);
  check(residOn < 1e-8, "equilibration solves badly-scaled system", residOn);
}

void testHonestFailure() {
  SparseMatrix<double> A = randomSymmetricPattern(80, 0.06, 23);
  const int dead = 40;
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      if (it.row() == dead || it.col() == dead) it.valueRef() = 0.0;
  A.prune(0.0);

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  const bool factored = solver.isFactorized();
  VectorXd b = VectorXd::Random(80);
  const VectorXd x = solver.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(factored && solver.info() == Eigen::NumericalIssue,
        "honest check: singular solve reports NumericalIssue", resid);

  VectorXd xr = VectorXd::Random(80);
  xr(dead) = 0.0;
  VectorXd bIn = A * xr;
  const VectorXd x2 = solver.solve(bIn);
  const double resid2 = (A * x2 - bIn).norm() / std::max(1e-300, bIn.norm());
  check(solver.info() == Eigen::Success && resid2 < 1e-8,
        "honest check: status recovers on a consistent RHS", resid2);
}

// Fail-fast fill guard: predictedFactorNonzeros() is available after analyze and
// exactly matches the realized fill; setMaxFactorNonzeros() below the real fill
// makes factorize() abort cleanly (NumericalIssue, nothing allocated) instead of
// attempting the allocation; above it, factorization proceeds normally.
void testFillGuard() {
  SparseMatrix<double> A = laplacian2d(20, 20);
  const int n = static_cast<int>(A.rows());

  Eigen::LeftRightLU<SparseMatrix<double>> probe;
  probe.analyzePattern(A);
  const long long predicted = probe.predictedFactorNonzeros();
  check(predicted > 0, "predictedFactorNonzeros() > 0 after analyze", predicted > 0 ? 0.0 : 1.0);

  Eigen::LeftRightLU<SparseMatrix<double>> guarded;
  guarded.setMaxFactorNonzeros(1000);  // far below the true fill -> must trip
  guarded.compute(A);
  check(guarded.info() == Eigen::NumericalIssue && !guarded.isFactorized(),
        "fill guard aborts factorize below limit", guarded.isFactorized() ? 1.0 : 0.0);

  Eigen::LeftRightLU<SparseMatrix<double>> ok;
  ok.setMaxFactorNonzeros(predicted + 1);  // generous -> normal factorization
  ok.compute(A);
  const bool factored = ok.info() == Eigen::Success && ok.isFactorized();
  const long long realized = static_cast<long long>(ok.nnzL()) + ok.nnzU() - n;
  check(factored && realized == predicted, "guard passes; prediction == realized fill (nnzL+U-n)",
        factored && realized == predicted ? 0.0 : 1.0);
  VectorXd b = A * VectorXd::Random(n);
  const double resid = (A * ok.solve(b) - b).norm() / b.norm();
  check(resid < 1e-8, "guarded (passing) solve is correct", resid);
  std::printf("        predicted=%lld  realized(nnzL+U-n)=%lld\n", predicted, realized);
}

// Parallel dynamic scheduler vs serial: same matrix, StdThreadExecutor vs the
// serial default must agree to solver accuracy (bit-identity is NOT a goal). Also
// checks the scheduler completes without deadlock and reports no error.
void testParallelVsSerial() {
  SparseMatrix<double> A = laplacian2d(60, 60);  // 3600 unknowns, wide tree
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> serial;
  serial.compute(A);
  const VectorXd xs = serial.solve(b);
  const double residS = (A * xs - b).norm() / b.norm();

  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                     Eigen::supernodal_lu::StdThreadExecutor>
      parallel;
  parallel.compute(A);
  if (parallel.info() != Eigen::Success) {
    std::printf("  [FAIL] %-46s parallel factorization failed: %s\n", "parallel scheduler",
                parallel.lastErrorMessage().c_str());
    ++g_failures;
    return;
  }
  const VectorXd xp = parallel.solve(b);
  const double residP = (A * xp - b).norm() / b.norm();
  const double agree = (xp - xs).norm() / xs.norm();

  check(residP < 1e-8, "parallel dynamic scheduler solves accurately", residP);
  check(agree < 1e-8, "parallel solution agrees with serial", agree);
  std::printf("        threads=%d serial resid=%.2e parallel resid=%.2e agree=%.2e\n",
              parallel.executor().concurrency(), residS, residP, agree);

  // stress: run the parallel factorization several times to shake out any
  // scheduler race/deadlock (different matrices, repeated compute()).
  for (int rep = 0; rep < 5; ++rep) {
    SparseMatrix<double> M = randomSymmetricPattern(400, 0.02, 100 + rep);
    Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                       Eigen::supernodal_lu::StdThreadExecutor>
        p;
    p.compute(M);
    VectorXd bb = M * VectorXd::Random(400);
    const double r = solveResidual(p, M, bb);
    if (!(p.info() == Eigen::Success && r < 1e-8)) {
      check(false, "parallel scheduler stress repetition", r);
      return;
    }
  }
  check(true, "parallel scheduler stress (5 repetitions, no deadlock)", 0.0);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("LeftRightLU correctness tests\n");

  std::printf("Random symmetric-pattern matrices:\n");
  solveAndMeasure(randomSymmetricPattern(50, 0.10, 1), "random n=50 p=0.10");
  solveAndMeasure(randomSymmetricPattern(120, 0.05, 7), "random n=120 p=0.05");
  solveAndMeasure(randomSymmetricPattern(200, 0.03, 13), "random n=200 p=0.03");

  std::printf("2D Laplacian (5-point) matrices:\n");
  solveAndMeasure(laplacian2d(10, 10), "laplacian 10x10");
  solveAndMeasure(laplacian2d(20, 20), "laplacian 20x20");
  solveAndMeasure(laplacian2d(30, 25), "laplacian 30x25");

  std::printf("Features:\n");
  testMultipleRhs();
  testFactorAccessors();
  testTransposeSolve();
  testCompletePivoting();
  testCompletePivotingColumnSwaps();
  testLogDeterminant();
  testEquilibration();
  testHonestFailure();
  testFillGuard();
  testParallelVsSerial();

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
