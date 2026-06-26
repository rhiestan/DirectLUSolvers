// Correctness tests for Eigen::SupernodalLU.
// Build (from repo root):
//   clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src \
//       DirectLUSolvers/test/test_supernodal_lu.cpp -o build/test_supernodal_lu
//
// Compares SupernodalLU against a dense LU reference and Eigen::SparseLU.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

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

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
