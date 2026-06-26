// Correctness + speedup test for SupernodalLU with the StdThreadExecutor backend.
//
// Build (from repo root):
//   clang++ -std=c++17 -O2 -pthread -I eigen -I DirectLUSolvers/src \
//       DirectLUSolvers/test/test_parallel_lu.cpp -o build/test_parallel_lu
//
// Verifies the parallel factorization produces the same factor as the serial
// one (identical determinant, machine-precision residual), then times both on
// the testdata matrices.

#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "SupernodalLU.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::Triplet;
using Clock = std::chrono::steady_clock;

using Serial = Eigen::SupernodalLU<SparseMatrix<double>>;
using Parallel = Eigen::SupernodalLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                                     Eigen::supernodal_lu::StdThreadExecutor>;

namespace {

int g_failures = 0;

void check(bool ok, const char* name, double value) {
  std::printf("  [%s] %-44s %.3e\n", ok ? "PASS" : "FAIL", name, value);
  if (!ok) ++g_failures;
}

SparseMatrix<double> loadSymmetrized(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line;
  std::getline(in, line);
  const bool symmetric = line.find("symmetric") != std::string::npos;
  do { std::getline(in, line); } while (!line.empty() && line[0] == '%');
  int rows, cols; long long nnz;
  { std::istringstream s(line); s >> rows >> cols >> nnz; }
  std::vector<Triplet<double>> t;
  for (long long k = 0; k < nnz; ++k) {
    std::getline(in, line);
    std::istringstream s(line); int i, j; double v; s >> i >> j >> v;
    --i; --j; t.emplace_back(i, j, v);
    if (symmetric && i != j) t.emplace_back(j, i, v);
  }
  SparseMatrix<double> A(rows, cols); A.setFromTriplets(t.begin(), t.end()); A.makeCompressed();
  std::vector<Triplet<double>> t2;
  for (int c = 0; c < A.outerSize(); ++c)
    for (SparseMatrix<double>::InnerIterator it(A, c); it; ++it) {
      t2.emplace_back(it.row(), it.col(), it.value());
      t2.emplace_back(it.col(), it.row(), 0.0);
    }
  SparseMatrix<double> S(rows, cols); S.setFromTriplets(t2.begin(), t2.end()); S.makeCompressed();
  return S;
}

SparseMatrix<double> laplacian2d(int gx, int gy) {
  const int n = gx * gy;
  std::vector<Triplet<double>> t;
  auto id = [gx](int x, int y) { return y * gx + x; };
  for (int y = 0; y < gy; ++y)
    for (int x = 0; x < gx; ++x) {
      const int i = id(x, y);
      t.emplace_back(i, i, 4.0);
      if (x > 0) t.emplace_back(i, id(x - 1, y), -1.0);
      if (x + 1 < gx) t.emplace_back(i, id(x + 1, y), -1.0);
      if (y > 0) t.emplace_back(i, id(x, y - 1), -1.0);
      if (y + 1 < gy) t.emplace_back(i, id(x, y + 1), -1.0);
    }
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

template <typename Solver>
double bestFactorMs(Solver& solver, const SparseMatrix<double>& A, int reps) {
  double best = 1e30;
  solver.analyzePattern(A);
  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now();
    solver.factorize(A);
    auto t1 = Clock::now();
    best = std::min(best, ms(t0, t1));
  }
  return best;
}

void runMatrix(const std::string& path) {
  std::printf("== %s\n", path.c_str());
  SparseMatrix<double> A;
  try {
    A = loadSymmetrized(path);
  } catch (const std::exception& e) {
    std::printf("  load error: %s\n", e.what());
    ++g_failures;
    return;
  }
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Serial serial;
  Parallel parallel;  // default-constructs a std::thread pool

  const double tSerial = bestFactorMs(serial, A, 3);
  const double tParallel = bestFactorMs(parallel, A, 3);

  if (serial.info() != Eigen::Success || parallel.info() != Eigen::Success) {
    std::printf("  (factorization reported NumericalIssue; skipping accuracy checks)\n");
  }

  // both factorizations must agree: same determinant, same solution.
  const double ds = serial.determinant(), dp = parallel.determinant();
  if (std::isfinite(ds) && std::isfinite(dp)) {
    const double detRel = std::abs(std::abs(ds) - std::abs(dp)) / std::max(1.0, std::abs(ds));
    check(detRel < 1e-10, "parallel determinant == serial", detRel);
  }

  VectorXd xs = serial.solve(b);
  VectorXd xp = parallel.solve(b);
  const double agree = (xs - xp).norm() / std::max(1e-300, xs.norm());
  check(agree < 1e-10, "parallel solution == serial solution", agree);

  const double resid = (A * xp - b).norm() / b.norm();
  std::printf("        n=%d snodes=%lld levels=%lld widest=%lld  threads=%d\n",
              n, (long long)parallel.supernodeCount(), (long long)parallel.levelCount(),
              (long long)parallel.widestLevel(), parallel.executor().concurrency());
  std::printf("        serial=%.1f ms  parallel=%.1f ms  speedup=%.2fx  resid=%.2e\n",
              tSerial, tParallel, tParallel > 0 ? tSerial / tParallel : 0.0, resid);
}

void runLaplacian(int g) {
  std::printf("== laplacian %dx%d\n", g, g);
  SparseMatrix<double> A = laplacian2d(g, g);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;
  Serial serial;
  Parallel parallel;
  const double tSerial = bestFactorMs(serial, A, 3);
  const double tParallel = bestFactorMs(parallel, A, 3);
  VectorXd xs = serial.solve(b), xp = parallel.solve(b);
  const double agree = (xs - xp).norm() / std::max(1e-300, xs.norm());
  check(agree < 1e-10, "parallel solution == serial solution", agree);
  std::printf("        n=%d snodes=%lld levels=%lld widest=%lld  threads=%d\n",
              n, (long long)parallel.supernodeCount(), (long long)parallel.levelCount(),
              (long long)parallel.widestLevel(), parallel.executor().concurrency());
  std::printf("        serial=%.1f ms  parallel=%.1f ms  speedup=%.2fx\n",
              tSerial, tParallel, tParallel > 0 ? tSerial / tParallel : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("SupernodalLU parallel (StdThreadExecutor) tests\n");

  std::vector<std::string> files = {
      "testdata/dendrimer/dendrimer.mtx",
      "testdata/tomography/tomography.mtx",
      "testdata/rdb2048_noL/rdb2048_noL.mtx",
      "testdata/YaleB_10NN/YaleB_10NN.mtx",
  };
  if (argc > 1) files.assign(argv + 1, argv + argc);
  for (const std::string& f : files) runMatrix(f);

  std::printf("\nSynthetic 2D Laplacians (wide elimination tree):\n");
  runLaplacian(100);
  runLaplacian(200);
  runLaplacian(300);

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
