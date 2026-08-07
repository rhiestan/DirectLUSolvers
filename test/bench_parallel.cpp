// Thread-count scaling benchmark with per-phase timing.
//
// Build + run (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/bench_parallel
//
// WHY THIS EXISTS
//
// test_parallel_lu times the parallel factorization at ONE thread count and
// reports a single speedup number. That is enough to prove the parallel path
// agrees with the serial one, but not enough to decide where parallel work is
// worth doing. Two things are invisible in it:
//
//   1. Per-phase cost. compute() is analyzePattern() + factorize(), and then
//      solve() runs separately. Only factorize() is parallelized at all. If
//      analyze and solve together are a large share of wall-clock, Amdahl caps
//      the achievable speedup no matter how good the factorization scheduler
//      gets -- and the "many right-hand sides against one factorization" use
//      case the README advertises makes solve() the dominant term.
//   2. Which MECHANISM pays. SupernodalLU has two: elimination-tree level
//      parallelism, and intra-supernode chunking (setIntraSupernodeParallelism).
//      LeftRightLU has a third, its barrier-free DAG scheduler. Measuring them
//      separately says which one to invest in.
//
// So this sweeps thread counts and reports every phase separately, per solver,
// with the intra-supernode mechanism toggled on and off.
//
// USAGE
//
//   bench_parallel                       # the built-in scaling set
//   bench_parallel --threads 1,4,16      # explicit thread counts
//   bench_parallel --reps 5              # take the best of N timed repetitions
//   bench_parallel --rhs 16              # columns for the multi-RHS solve row
//   bench_parallel --quick               # synthetic matrices only, fast
//   bench_parallel path/to/A.mtx ...     # explicit matrices
//
// Timings are best-of-reps (not mean): the minimum is the least noisy estimator
// of achievable time on a loaded desktop.

#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/PooledExecutor.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::ms;
using lu_testing::PooledExecutor;
using Clock = lu_testing::Clock;

// Both solvers instantiated on the reconfigurable pooled executor, so one
// solver type covers the whole sweep (see testing/PooledExecutor.h for why
// StdThreadExecutor cannot be reconfigured in place).
using Snlu = Eigen::SupernodalLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, PooledExecutor>;
using Lrlu = Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, PooledExecutor>;

namespace {

struct Options {
  std::vector<int> threads;
  int reps = 3;
  int rhs = 8;
  bool quick = false;
  std::vector<std::string> matrices;
};

// One measured phase: its name, and its best time at each thread count.
struct PhaseRow {
  std::string name;
  std::vector<double> timeMs;  // parallel to Options::threads
  bool parallelizable = true;  // false => a flat row is expected, not a defect
};

double bestOf(int reps, const std::function<void()>& body) {
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    body();
    const auto t1 = Clock::now();
    best = std::min(best, ms(t0, t1));
  }
  return best;
}

// ---------------------------------------------------------------------------
//  Per-matrix sweep
// ---------------------------------------------------------------------------

struct MatrixResult {
  std::string label;
  Eigen::Index n = 0, nnz = 0;
  Eigen::Index snodes = 0, levels = 0, widest = 0;
  std::vector<PhaseRow> rows;
  // Worst relative residual seen across the sweep. A timing benchmark that does
  // not check its answers can happily report a speedup for a broken
  // factorization, so every thread count is verified as it is measured.
  double worstResid = 0.0;
};

MatrixResult sweep(const std::string& label, const SparseMatrix<double>& A, const Options& opt) {
  MatrixResult result;
  result.label = label;
  result.n = A.rows();
  result.nnz = A.nonZeros();

  const Eigen::Index n = A.rows();
  const VectorXd b = A * VectorXd::Ones(n);
  const MatrixXd B = A * MatrixXd::Ones(n, opt.rhs);

  const std::size_t T = opt.threads.size();
  PhaseRow analyze{"analyze (symbolic)", std::vector<double>(T, 0.0), /*parallelizable=*/false};
  PhaseRow factorLevels{"SNLU factor, levels only", std::vector<double>(T, 0.0), true};
  PhaseRow factorIntra{"SNLU factor, levels+intra", std::vector<double>(T, 0.0), true};
  PhaseRow factorDag{"LRLU factor, DAG scheduler", std::vector<double>(T, 0.0), true};
  PhaseRow solve1{"SNLU solve, 1 rhs", std::vector<double>(T, 0.0), false};
  PhaseRow solveK{"SNLU solve, " + std::to_string(opt.rhs) + " rhs", std::vector<double>(T, 0.0), false};

  for (std::size_t ti = 0; ti < T; ++ti) {
    const int threads = opt.threads[ti];

    // --- SupernodalLU, intra-supernode parallelism OFF (level parallelism only)
    {
      Snlu s;
      s.executor() = PooledExecutor(threads);
      s.setIntraSupernodeParallelism(false);
      analyze.timeMs[ti] = bestOf(opt.reps, [&] { s.analyzePattern(A); });
      factorLevels.timeMs[ti] = bestOf(opt.reps, [&] { s.factorize(A); });
      if (ti == 0) {
        result.snodes = s.supernodeCount();
        result.levels = s.levelCount();
        result.widest = s.widestLevel();
      }
    }

    // --- SupernodalLU, both mechanisms (the shipping default)
    {
      Snlu s;
      s.executor() = PooledExecutor(threads);
      s.setIntraSupernodeParallelism(true);
      s.analyzePattern(A);
      factorIntra.timeMs[ti] = bestOf(opt.reps, [&] { s.factorize(A); });

      // Solve, measured on the SAME factorization with refinement disabled, so
      // this is the triangular solve alone and not the BiCGStab matvecs on top.
      s.setMaxIterativeRefinements(0);
      VectorXd x;
      MatrixXd X;
      solve1.timeMs[ti] = bestOf(opt.reps, [&] { x = s.solve(b); });
      solveK.timeMs[ti] = bestOf(opt.reps, [&] { X = s.solve(B); });

      const double resid = (A * x - b).norm() / b.norm();
      result.worstResid = std::max(result.worstResid, resid);
    }

    // --- LeftRightLU, barrier-free dynamic scheduler
    {
      Lrlu s;
      s.executor() = PooledExecutor(threads);
      s.analyzePattern(A);
      factorDag.timeMs[ti] = bestOf(opt.reps, [&] { s.factorize(A); });
      s.setMaxIterativeRefinements(0);
      const VectorXd x = s.solve(b);
      result.worstResid = std::max(result.worstResid, (A * x - b).norm() / b.norm());
    }
  }

  result.rows = {analyze, factorLevels, factorIntra, factorDag, solve1, solveK};
  return result;
}

// ---------------------------------------------------------------------------
//  Reporting
// ---------------------------------------------------------------------------

void printMatrixResult(const MatrixResult& r, const Options& opt) {
  std::printf("\n== %s  (n=%lld nnz=%lld, %lld supernodes, %lld levels, widest %lld)\n",
              r.label.c_str(), (long long)r.n, (long long)r.nnz, (long long)r.snodes,
              (long long)r.levels, (long long)r.widest);

  std::printf("  %-28s", "phase \\ threads");
  for (int t : opt.threads) std::printf(" %9d", t);
  std::printf(" %9s\n", "speedup");

  for (const PhaseRow& row : r.rows) {
    std::printf("  %-28s", row.name.c_str());
    for (double t : row.timeMs) std::printf(" %9.1f", t);
    const double first = row.timeMs.front();
    const double last = row.timeMs.back();
    const double speedup = last > 0.0 ? first / last : 0.0;
    std::printf(" %8.2fx%s\n", speedup, row.parallelizable ? "" : "  (serial phase)");
  }

  // Amdahl view at the highest thread count: where does the time actually go?
  const std::size_t last = opt.threads.size() - 1;
  const double analyzeMs = r.rows[0].timeMs[last];
  const double factorMs = r.rows[2].timeMs[last];   // levels+intra, the default
  const double solveMs = r.rows[4].timeMs[last];    // 1 rhs
  const double solveKMs = r.rows[5].timeMs[last];
  const double totalOne = analyzeMs + factorMs + solveMs;
  if (totalOne > 0.0) {
    std::printf("  at %d threads, one solve:   analyze %.0f%%  factor %.0f%%  solve %.0f%%\n",
                opt.threads[last], 100.0 * analyzeMs / totalOne, 100.0 * factorMs / totalOne,
                100.0 * solveMs / totalOne);
    // The advertised sweet spot: many right-hand sides against one factorization.
    const double totalK = analyzeMs + factorMs + solveKMs;
    std::printf("  at %d threads, %d rhs:      analyze %.0f%%  factor %.0f%%  solve %.0f%%\n",
                opt.threads[last], opt.rhs, 100.0 * analyzeMs / totalK,
                100.0 * factorMs / totalK, 100.0 * solveKMs / totalK);
  }

  // A speedup measured on a wrong answer is not a speedup.
  lu_testing::check(r.worstResid < 1e-8, r.label + ": worst residual over the sweep", r.worstResid);
}

std::vector<int> parseThreadList(const std::string& spec) {
  std::vector<int> out;
  std::size_t pos = 0;
  while (pos <= spec.size()) {
    const std::size_t comma = spec.find(',', pos);
    const std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!tok.empty()) {
      const int v = std::atoi(tok.c_str());
      if (v > 0) out.push_back(v);
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

// Powers of two from 1 up to the machine's concurrency, with the exact
// concurrency appended when it is not itself a power of two.
std::vector<int> defaultThreadList() {
  const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  std::vector<int> out;
  for (int t = 1; t <= hw; t *= 2) out.push_back(t);
  if (out.empty() || out.back() != hw) out.push_back(hw);
  return out;
}

struct Case {
  std::string label;
  std::function<SparseMatrix<double>()> build;
};

// Matrices big enough for parallelism to be measurable. Small testdata matrices
// factor in single-digit milliseconds, where dispatch overhead dominates and
// scaling numbers say nothing useful -- they are deliberately not here.
std::vector<Case> scalingSet(const Options& opt) {
  std::vector<Case> cases;
  cases.push_back({"lap2d_300x300", [] { return lu_testing::laplacian2d(300, 300); }});
  cases.push_back({"lap3d_30x30x30", [] { return lu_testing::laplacian3d(30, 30, 30); }});
  if (opt.quick) return cases;

  for (const char* label : {"laoss_2", "laoss_1"}) {
    for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices()) {
      if (std::string(m.label) != label) continue;
      const std::string path = lu_testing::testdataPath(m.relative);
      cases.push_back({m.label, [path] {
                         return lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(path));
                       }});
    }
  }
  return cases;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--threads" && i + 1 < argc) opt.threads = parseThreadList(argv[++i]);
    else if (arg == "--reps" && i + 1 < argc) opt.reps = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--rhs" && i + 1 < argc) opt.rhs = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--quick") opt.quick = true;
    else if (arg.rfind("--", 0) == 0) {
      std::printf("usage: %s [--threads 1,2,4,...] [--reps N] [--rhs K] [--quick] [matrix.mtx ...]\n",
                  argv[0]);
      return 2;
    } else {
      opt.matrices.push_back(arg);
    }
  }
  if (opt.threads.empty()) opt.threads = defaultThreadList();

  std::printf("DirectLUSolvers parallel scaling benchmark\n");
  std::printf("  hardware concurrency: %u\n", std::thread::hardware_concurrency());
  std::printf("  thread counts:        ");
  for (int t : opt.threads) std::printf("%d ", t);
  std::printf("\n  reps (best-of):       %d\n", opt.reps);
  std::printf("  multi-rhs columns:    %d\n", opt.rhs);
  std::printf("\nTimings in ms, best of %d. 'speedup' compares the last thread count to the first.\n",
              opt.reps);
  std::printf("Solve rows have refinement disabled, so they measure the triangular solve alone.\n");

  std::vector<Case> cases;
  if (!opt.matrices.empty()) {
    for (const std::string& path : opt.matrices)
      cases.push_back({lu_testing::matrixLabel(path), [path] {
                         return lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(path));
                       }});
  } else {
    cases = scalingSet(opt);
  }

  for (const Case& c : cases) {
    SparseMatrix<double> A;
    try {
      A = c.build();
    } catch (const std::exception& e) {
      lu_testing::fail(c.label + ": " + e.what());
      continue;
    }
    printMatrixResult(sweep(c.label, A, opt), opt);
  }

  std::printf("\n");
  return lu_testing::failureCount() == 0 ? 0 : 1;
}
