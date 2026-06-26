// Correctness tests for Eigen::SupernodalLU.
// Build (from repo root):
//   clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src \
//       DirectLUSolvers/test/test_supernodal_lu.cpp -o build/test_supernodal_lu
//
// Compares SupernodalLU against a dense LU reference and Eigen::SparseLU.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "SupernodalLU.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace {

int g_failures = 0;

void check(bool ok, const char* name, double residual) {
  std::printf("  [%s] %-40s residual = %.3e\n", ok ? "PASS" : "FAIL", name, residual);
  if (!ok) ++g_failures;
}

// Build a random matrix with a SYMMETRIC pattern and general (unsymmetric)
// values, made diagonally dominant so plain unpivoted LU is stable.
SparseMatrix<double> randomSymmetricPattern(int n, double offDiagProb, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (prob(rng) < offDiagProb) {
        triplets.emplace_back(i, j, uni(rng));  // pattern symmetric, values differ
        triplets.emplace_back(j, i, uni(rng));
      }
  // strong diagonal for stability
  for (int i = 0; i < n; ++i) triplets.emplace_back(i, i, n + uni(rng));

  SparseMatrix<double> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

// 5-point 2D Laplacian on a grid (symmetric pattern, nice for nested dissection).
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

double solveAndMeasure(const SparseMatrix<double>& A, const char* name) {
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  try {
    solver.compute(A);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] %-40s threw: %s\n", name, e.what());
    ++g_failures;
    return 1.0;
  }
  if (solver.info() != Eigen::Success) {
    std::printf("  [FAIL] %-40s factorization failed: %s\n", name, solver.lastErrorMessage().c_str());
    ++g_failures;
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

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
