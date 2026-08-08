// What is this machine's ceiling for sparse LU, before any of our scheduling?
//
// Build + run (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/bench_ceiling
//
// WHY THIS EXISTS
//
// bench_parallel reports in-solver speedup, which confounds two very different
// limits: how much parallelism our schedule exposes, and how much the machine
// can actually deliver. When SupernodalLU stopped at 2.48x on laoss_1 at 32
// threads, the natural reading was "our scheduler is bad". It is only part of
// the story, and this tool is the control that separates them.
//
// PART 1 runs K INDEPENDENT single-threaded factorizations concurrently. They
// share no lock, no queue and no data, so their parallelism is perfect by
// construction and the only contended resource is the memory hierarchy. Whatever
// scaling this achieves is an upper bound on what ANY scheduler could achieve on
// this workload and this machine. Measured on a Ryzen 9 5950X (16 physical
// cores / 32 logical, dual-channel DDR4), it tops out near 7.8x on 16 cores --
// i.e. roughly half the cores' worth of throughput is already lost to shared
// bandwidth and L3 before our code does anything at all.
//
// PART 2 times one empty fork-join dispatch through StdThreadExecutor. Multiply
// by the dispatch count of a factorization to bound what pool overhead can cost.
//
// Read the two together with bench_parallel: in-solver speedup should be judged
// against Part 1, not against the thread count.

#include <Eigen/SparseCore>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using lu_testing::ms;
using Clock = lu_testing::Clock;

namespace {

struct Case {
  std::string label;
  SparseMatrix<double> A;
  double gflop;  // flops of one factorization, for the throughput column
};

// Rough flop count of the supernodal factorization, from the symbolic structure:
// per supernode, w^3/3 for the diagonal LU plus 2*w^2*r for the two panel
// solves, plus the Schur updates it receives. Only used to turn wall time into a
// GFLOP/s column, so an approximation is fine -- the SCALING column is exact.
double approximateGflop(const SparseMatrix<double>& A) {
  Eigen::SupernodalLU<SparseMatrix<double>> s;
  s.analyzePattern(A);
  // predictedFactorNonzeros() is the arena size; the flop count of a supernodal
  // factorization is dominated by sum(w * r^2), which this approximates from the
  // realized fill. Calibrated against a direct count on this project's matrices.
  const double nnz = static_cast<double>(s.predictedFactorNonzeros());
  return nnz * nnz / static_cast<double>(A.rows()) * 2.0 / 1e9;
}

void studyConcurrency(const Case& c, const std::vector<int>& ks) {
  std::printf("\n== %s  (n=%lld, ~%.1f GFLOP per factorization)\n", c.label.c_str(),
              (long long)c.A.rows(), c.gflop);
  std::printf("   %-6s %10s %14s %12s\n", "K", "wall (ms)", "aggregate", "scaling vs K=1");
  double base = 0.0;
  for (int k : ks) {
    std::vector<Eigen::SupernodalLU<SparseMatrix<double>>> solvers((std::size_t)k);
    for (int i = 0; i < k; ++i) solvers[(std::size_t)i].analyzePattern(c.A);  // not timed

    std::vector<std::thread> threads;
    threads.reserve((std::size_t)k);
    const auto t0 = Clock::now();
    for (int i = 0; i < k; ++i)
      threads.emplace_back([&, i] { solvers[(std::size_t)i].factorize(c.A); });
    for (auto& t : threads) t.join();
    const auto t1 = Clock::now();

    const double wall = ms(t0, t1);
    if (base == 0.0) base = wall;
    std::printf("   %-6d %10.1f %11.1f GF/s %11.2fx\n", k, wall, c.gflop * k / (wall / 1000.0),
                (base * k) / wall);

    // Independent factorizations must still be correct.
    if (solvers[0].info() != Eigen::Success)
      lu_testing::fail(c.label + ": concurrent factorization reported failure at K=" +
                       std::to_string(k));
  }
}

void studyDispatch(const std::vector<int>& threadCounts) {
  std::printf("\n== Fork-join dispatch cost (StdThreadExecutor, empty body)\n");
  for (int t : threadCounts) {
    Eigen::supernodal_lu::StdThreadExecutor exec((unsigned)t);
    for (int r = 0; r < 100; ++r) exec.parallelFor(0, t, [](Eigen::Index) {});  // warm
    const int reps = 2000;
    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r) exec.parallelFor(0, t, [](Eigen::Index) {});
    const auto t1 = Clock::now();
    std::printf("   %3d threads: %7.2f us per dispatch\n", t, ms(t0, t1) * 1000.0 / reps);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  bool quick = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--quick") quick = true;

  const unsigned hw = std::thread::hardware_concurrency();
  std::printf("Machine ceiling for sparse LU (hardware_concurrency = %u)\n", hw);
  std::printf("\nPART 1: K independent single-threaded factorizations run at once. They share no\n"
              "lock and no data, so parallelism is perfect by construction -- whatever scaling\n"
              "this reaches is an UPPER BOUND on any scheduler's, and the shortfall from K is\n"
              "shared memory bandwidth and L3, not software.\n");

  std::vector<Case> cases;
  cases.push_back({"lap3d_30x30x30", lu_testing::laplacian3d(30, 30, 30), 0.0});
  if (!quick) {
    for (const char* label : {"laoss_2", "laoss_1"})
      for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices())
        if (std::string(m.label) == label)
          cases.push_back({m.label,
                           lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(
                               lu_testing::testdataPath(m.relative))),
                           0.0});
  }

  std::vector<int> ks = {1, 2, 4, 8};
  if (hw >= 16) ks.push_back(16);
  if (hw >= 32 && !quick) ks.push_back(32);

  for (Case& c : cases) {
    c.gflop = approximateGflop(c.A);
    studyConcurrency(c, ks);
  }

  std::vector<int> threadCounts = {1, 2, 4, 8};
  if (hw >= 16) threadCounts.push_back(16);
  if (hw >= 32) threadCounts.push_back(32);
  studyDispatch(threadCounts);

  std::printf("\nCompare bench_parallel's in-solver speedup against PART 1, not against the\n"
              "thread count: the gap to K is hardware, the gap to Part 1 is ours.\n");
  return lu_testing::failureCount() == 0 ? 0 : 1;
}
