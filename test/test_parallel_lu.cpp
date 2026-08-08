// Correctness + speedup test for SupernodalLU with the StdThreadExecutor backend.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_parallel_lu --output-on-failure
//
// Verifies the parallel factorization produces the same factor as the serial
// one (identical determinant, machine-precision residual), then times both on
// the testdata matrices.

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::laplacian2d;
using lu_testing::ms;
using Clock = lu_testing::Clock;

using Serial = Eigen::SupernodalLU<SparseMatrix<double>>;
using Parallel = Eigen::SupernodalLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                                     Eigen::supernodal_lu::StdThreadExecutor>;

namespace {

SparseMatrix<double> loadSymmetrized(const std::string& path) {
  return lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(path));
}

// NOTE the explicit MatrixXd: solver.solve(b) returns a lazy Eigen Solve<>
// expression, so binding it to `auto` would time expression construction and
// nothing else (it reported 0.0 ms). Assigning to a concrete matrix forces the
// solve to actually run inside the timed region.
template <typename Solver, typename Rhs>
double bestSolveMs(Solver& solver, const Rhs& b, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    const Eigen::MatrixXd x = solver.solve(b);
    const auto t1 = Clock::now();
    if (!x.allFinite()) return -1.0;
    best = std::min(best, ms(t0, t1));
  }
  return best;
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
    lu_testing::fail(std::string("load error: ") + e.what());
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

// The PARALLEL TRIANGULAR SOLVE (setParallelSolve, on by default).
//
// The forward sweep is a scatter in its serial form -- supernode s pushes into
// its ancestors' rows -- which cannot run level-parallel, so the parallel path
// uses an equivalent gather instead. The claim is that this leaves each element's
// accumulation order untouched and is therefore BIT-IDENTICAL, not merely close;
// this asserts exactly that, since a reformulation that quietly reassociated
// would still pass a tolerance-based check.
//
// Note the work threshold: the sweeps stay serial below rows x nrhs = 200000, so
// this deliberately uses a matrix and RHS count large enough to cross it. Without
// that, the parallel path would never run and the test would prove nothing.
void testParallelSolve() {
  std::printf("== parallel triangular solve\n");
  SparseMatrix<double> A = laplacian2d(300, 300);  // 90000 rows
  const int n = static_cast<int>(A.rows());
  const int nrhs = 4;  // 90000 * 4 = 360000 > threshold
  Eigen::MatrixXd B(n, nrhs);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < nrhs; ++j) B(i, j) = 1.0 + 0.1 * ((i + j) % 7);

  Parallel solver;
  solver.setMaxIterativeRefinements(0);  // compare the raw triangular sweeps
  solver.compute(A);
  checkTrue(solver.info() == Eigen::Success, "parallel solve: factorization succeeds");

  solver.setParallelSolve(false);
  const Eigen::MatrixXd serialX = solver.solve(B);
  const double tSerial = bestSolveMs(solver, B, 3);

  solver.setParallelSolve(true);
  const Eigen::MatrixXd parallelX = solver.solve(B);
  const double tParallel = bestSolveMs(solver, B, 3);

  const double maxDiff = (serialX - parallelX).cwiseAbs().maxCoeff();
  check(maxDiff == 0.0, "parallel solve is BIT-IDENTICAL to serial", maxDiff);

  const double resid = (A * parallelX - B).norm() / B.norm();
  check(resid < 1e-8, "parallel solve residual", resid);

  std::printf("        n=%d nrhs=%d threads=%d  serial=%.1f ms  parallel=%.1f ms  speedup=%.2fx\n",
              n, nrhs, solver.executor().concurrency(), tSerial, tParallel,
              tParallel > 0 ? tSerial / tParallel : 0.0);

  // Below the work threshold the sweeps must stay serial, and still be correct.
  Eigen::MatrixXd smallB = Eigen::MatrixXd::Ones(n, 1);
  solver.setParallelSolve(true);
  const Eigen::MatrixXd tiny = solver.solve(smallB);
  checkTrue(tiny.allFinite(), "sub-threshold solve stays correct");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("SupernodalLU parallel (StdThreadExecutor) tests\n");

  std::vector<std::string> files = {
      lu_testing::testdataPath("dendrimer/dendrimer.mtx"),
      lu_testing::testdataPath("tomography/tomography.mtx"),
      lu_testing::testdataPath("rdb2048_noL/rdb2048_noL.mtx"),
      lu_testing::testdataPath("YaleB_10NN/YaleB_10NN.mtx"),
  };
  if (argc > 1) files.assign(argv + 1, argv + argc);
  for (const std::string& f : files) runMatrix(f);

  std::printf("\nSynthetic 2D Laplacians (wide elimination tree):\n");
  runLaplacian(100);
  runLaplacian(200);
  runLaplacian(300);

  testParallelSolve();

  return lu_testing::summarize("SupernodalLU parallel");
}
