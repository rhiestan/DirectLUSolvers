// Per-matrix solver/ordering shootout with per-phase timing.
//
// Build + run (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/bench_solvers
//
// WHY THIS EXISTS
//
// compare_testdata answers "does every solver get the right answer on the whole
// corpus, and roughly how fast" -- one cold run per solver, one factor+solve
// number per cell, one ordering each. That is the right shape for a sweep of 14
// matrices, and the wrong shape for the question that follows it: given THIS
// matrix, which configuration should I actually use? Three things it cannot
// show:
//
//   1. Cold-start cost lands in the measurement. MKL's first pardiso() call
//      spins up its thread pool; on a 1015-row matrix that is ~500 ms of the
//      ~1 ms of real work, so PARDISO looks 500x slower than it is. Every solver
//      here is warmed up before the timed repetitions.
//   2. Which PHASE costs. analyzePattern is a large share of wall clock on a
//      dense-ish matrix (a third of it for METIS on setfos_2), and it is the
//      phase you skip entirely when refactorizing the same pattern. A single
//      factor+solve total hides that.
//   3. The ORDERING, which on an unsymmetric pattern moves the answer further
//      than the choice of solver does -- and not always in the direction fill
//      predicts. On setfos_2, COLAMD carries 47% more fill than METIS and still
//      factors faster, because it leaves 39 wide supernodes against METIS's 321
//      narrow ones and wins the difference back in BLAS-3 efficiency.
//
// Thread counts are stated per row rather than assumed: PARDISO is multithreaded
// by default while Eigen::SparseLU never is, so an unqualified comparison of the
// two is meaningless.
//
// USAGE
//
//   bench_solvers                        # the Tier::Small testdata corpus
//   bench_solvers --quick                # synthetic matrices only, no testdata
//   bench_solvers --threads 1,4,16       # explicit thread counts
//   bench_solvers --reps 7               # best of N timed repetitions
//   bench_solvers --no-matching          # our solvers with setMatching(false)
//   bench_solvers path/to/A.mtx ...      # explicit matrices
//
// METIS rows appear when built with -DDLU_WITH_METIS=ON, PARDISO rows with
// -DDLU_WITH_PARDISO=ON.

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#ifdef HAVE_METIS
#include <iostream>  // Eigen/MetisSupport uses std::cerr without including it
#include <Eigen/MetisSupport>
#endif

#ifdef HAVE_PARDISO
#include <Eigen/PardisoSupport>
#include <mkl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "LeftRightLU.h"
#include "PointBlockLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::supernodal_lu::PooledExecutor;
using lu_testing::ms;
using Clock = lu_testing::Clock;

typedef SparseMatrix<double> SpMat;

// Both solvers on the reconfigurable pooled executor, so one type covers every
// thread count (StdThreadExecutor cannot be reassigned into a live solver).
typedef Eigen::LeftRightLU<SpMat, Eigen::AMDOrdering<int>, PooledExecutor> LrluAmd;
typedef Eigen::LeftRightLU<SpMat, Eigen::COLAMDOrdering<int>, PooledExecutor> LrluColamd;
typedef Eigen::SupernodalLU<SpMat, Eigen::AMDOrdering<int>, PooledExecutor> SnluAmd;
#ifdef HAVE_METIS
typedef Eigen::LeftRightLU<SpMat, Eigen::MetisOrdering<int>, PooledExecutor> LrluMetis;
#endif

// PointBlockLU is single-threaded by design and takes no executor, so it needs
// its own runner below rather than runOurs().
typedef Eigen::PointBlockLU<SpMat, Eigen::COLAMDOrdering<int>> PblkColamd;

namespace {

// Residual above which a row is called a failure, matching compare_testdata.
const double kResidTolerance = 1e-6;

struct Options {
  std::vector<int> threads;
  int reps = 5;
  bool quick = false;
  bool matching = true;
  std::vector<std::string> matrices;
};

struct Row {
  std::string solver;
  int threads = 1;
  double analyzeMs = 0, factorMs = 0, solveMs = 0;
  double err = 0, resid = 0;
  long long fill = -1;
  bool ok = false;
  std::string note;

  double total() const { return analyzeMs + factorMs + solveMs; }
};

// Run `body` reps+1 times -- the first is a discarded warm-up -- and keep the
// MINIMUM of each phase. The minimum, not the mean: a slow repetition on a
// loaded desktop only ever means the machine did something else too, so the
// minimum is the least noisy estimator of achievable time.
template <typename Body>
Row measure(const std::string& solver, int threads, int reps, Body body) {
  Row r;
  r.solver = solver;
  r.threads = threads;
  r.analyzeMs = r.factorMs = r.solveMs = 1e300;
  for (int rep = 0; rep <= reps; ++rep) {
    Row one;
    if (!body(one)) {
      r.note = one.note;
      return r;
    }
    if (rep == 0) continue;  // warm-up
    r.analyzeMs = std::min(r.analyzeMs, one.analyzeMs);
    r.factorMs = std::min(r.factorMs, one.factorMs);
    r.solveMs = std::min(r.solveMs, one.solveMs);
    r.err = one.err;
    r.resid = one.resid;
    r.fill = one.fill;
  }
  r.ok = true;
  return r;
}

// SupernodalLU needs a pattern-symmetrized copy, so `input` may differ from the
// `A` the answer is scored against -- the two are the same linear system.
template <typename Solver>
Row runOurs(const std::string& name, int threads, const Options& opt, const SpMat& input,
            const SpMat& A, const VectorXd& b, const VectorXd& xTrue) {
  return measure(name, threads, opt.reps, [&](Row& one) {
    Solver s;
    s.executor() = PooledExecutor(threads);
    if (!opt.matching) s.setMatching(false);
    const auto t0 = Clock::now();
    s.analyzePattern(input);
    const auto t1 = Clock::now();
    s.factorize(input);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed: " + s.lastErrorMessage();
      return false;
    }
    const VectorXd x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    return true;
  });
}

// PointBlockLU's headline number is the REPLAY, not the first factorization:
// analyzePattern() plus one full factorize() happen once per pattern, and every
// Newton step after that pays only the replay. Timing the first factorization
// instead would measure a cost the target workload amortizes to nothing, so the
// factor column here is the replay and the row says so.
template <typename Solver>
Row runPointBlock(const std::string& name, const Options& opt, const SpMat& A, const VectorXd& b,
                  const VectorXd& xTrue) {
  Row r;
  r.solver = name;
  r.threads = 1;
  Solver s;
  const auto t0 = Clock::now();
  s.analyzePattern(A);
  const auto t1 = Clock::now();
  s.factorize(A);  // full factorization: records the pattern and the pivots
  if (s.info() != Eigen::Success) {
    r.note = "factorize failed: " + s.lastErrorMessage();
    return r;
  }
  r.analyzeMs = ms(t0, t1);
  r.factorMs = 1e300;
  r.solveMs = 1e300;
  VectorXd x;
  for (int rep = 0; rep <= opt.reps; ++rep) {
    const auto a0 = Clock::now();
    s.factorize(A);
    const auto a1 = Clock::now();
    x = s.solve(b);
    const auto a2 = Clock::now();
    if (rep == 0) continue;  // warm-up
    r.factorMs = std::min(r.factorMs, ms(a0, a1));
    r.solveMs = std::min(r.solveMs, ms(a1, a2));
  }
  r.err = (x - xTrue).norm() / xTrue.norm();
  r.resid = (A * x - b).norm() / b.norm();
  r.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
  r.ok = true;
  return r;
}

Row runSparseLU(const Options& opt, const SpMat& A, const VectorXd& b, const VectorXd& xTrue) {
  return measure("Eigen::SparseLU", 1, opt.reps, [&](Row& one) {
    Eigen::SparseLU<SpMat> s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed: " + s.lastErrorMessage();
      return false;
    }
    const VectorXd x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    return true;
  });
}

#ifdef HAVE_PARDISO
Row runPardiso(int threads, const Options& opt, const SpMat& A, const VectorXd& b,
               const VectorXd& xTrue) {
  return measure("MKL PARDISO", threads, opt.reps, [&](Row& one) {
    mkl_set_num_threads(threads);
    Eigen::PardisoLU<SpMat> s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed";
      return false;
    }
    const VectorXd x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = s.pardisoParameterArray()[17];  // IPARM(18): nonzeros in the LU factors
    return true;
  });
}
#endif

void printRow(const Row& r) {
  if (!r.ok) {
    std::printf("  %-28s %3d  %s\n", r.solver.c_str(), r.threads, r.note.c_str());
    return;
  }
  std::printf("  %-28s %3d %9.2f %9.2f %8.2f %9.2f  %9.2e %9.2e", r.solver.c_str(), r.threads,
              r.analyzeMs, r.factorMs, r.solveMs, r.total(), r.err, r.resid);
  if (r.fill >= 0)
    std::printf(" %12lld\n", r.fill);
  else
    std::printf(" %12s\n", "-");
}

std::vector<int> parseThreadList(const char* text) {
  std::vector<int> out;
  const std::string s(text);
  std::size_t begin = 0;
  while (begin <= s.size()) {
    const std::size_t comma = s.find(',', begin);
    const std::string piece = s.substr(begin, comma - begin);
    if (!piece.empty()) {
      const int t = std::atoi(piece.c_str());
      if (t > 0) out.push_back(t);
    }
    if (comma == std::string::npos) break;
    begin = comma + 1;
  }
  return out;
}

std::vector<int> defaultThreadList() {
  const unsigned hw = std::thread::hardware_concurrency();
  std::vector<int> out{1};
  if (hw > 1) out.push_back(static_cast<int>(hw));
  return out;
}

struct Case {
  std::string label;
  SpMat A;
};

// --quick uses synthetic matrices so the suite runs on a checkout without
// testdata/. upwind2d is here rather than another Laplacian because the
// ordering axis this benchmark exists to measure only opens up on an
// unsymmetric pattern.
std::vector<Case> buildCases(const Options& opt, int& loadFailures) {
  std::vector<Case> cases;
  if (opt.quick) {
    cases.push_back({"lap2d_60x60", lu_testing::laplacian2d(60, 60)});
    cases.push_back({"upwind_40x40", lu_testing::upwind2d(40, 40)});
    return cases;
  }

  std::vector<std::string> paths = opt.matrices;
  if (paths.empty()) paths = lu_testing::benchmarkPaths(lu_testing::Tier::Small);
  for (const std::string& path : paths) {
    try {
      cases.push_back({lu_testing::matrixLabel(path), lu_testing::loadMatrixMarket(path)});
    } catch (const std::exception& e) {
      std::printf("%s: load failed: %s\n", path.c_str(), e.what());
      ++loadFailures;
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
    else if (arg == "--quick") opt.quick = true;
    else if (arg == "--no-matching") opt.matching = false;
    else if (arg.rfind("--", 0) == 0) {
      std::printf("usage: %s [--quick] [--threads 1,2,4,...] [--reps N] [--no-matching] "
                  "[matrix.mtx ...]\n", argv[0]);
      return 2;
    } else {
      opt.matrices.push_back(arg);
    }
  }
  if (opt.threads.empty()) opt.threads = defaultThreadList();

  std::printf("DirectLUSolvers per-matrix solver/ordering benchmark\n");
  std::printf("  reps (best-of, after a warm-up): %d\n", opt.reps);
  std::printf("  thread counts:                   ");
  for (int t : opt.threads) std::printf("%d ", t);
  std::printf("\n  our solvers' matching:           %s\n", opt.matching ? "on" : "off");
  std::printf("  METIS rows:                      %s\n",
#ifdef HAVE_METIS
              "yes");
#else
              "no (build with -DDLU_WITH_METIS=ON)");
#endif
  std::printf("  PARDISO rows:                    %s\n",
#ifdef HAVE_PARDISO
              "yes");
#else
              "no (build with -DDLU_WITH_PARDISO=ON)");
#endif
  std::printf("\nTimes in ms. fill = nnzL+nnzU as each solver reports it; ours and SparseLU\n"
              "count the diagonal in both factors, PARDISO's IPARM(18) counts it once, so\n"
              "compare fill only up to an offset of n.\n");

  int failures = 0;
  std::vector<Case> cases = buildCases(opt, failures);

  for (const Case& c : cases) {
    const SpMat& A = c.A;
    const SpMat Asym = lu_testing::ensureSymmetricPattern(A);
    std::srand(12345);  // same xTrue for every solver and every matrix
    const VectorXd xTrue = VectorXd::Random(A.rows());
    const VectorXd b = A * xTrue;

    std::printf("\n=== %s\n", c.label.c_str());
    std::printf("    n=%lld  nnz=%lld  nnz/row=%.1f  patternSymmetry=%.3f  symmetrized nnz=%lld\n",
                (long long)A.rows(), (long long)A.nonZeros(),
                double(A.nonZeros()) / double(A.rows()), lu_testing::patternSymmetry(A),
                (long long)Asym.nonZeros());
    std::printf("  %-28s %3s %9s %9s %8s %9s  %9s %9s %12s\n", "solver", "thr", "analyze",
                "factor", "solve", "total", "err", "resid", "fill");

    std::vector<Row> rows;
    for (int t : opt.threads) {
      rows.push_back(runOurs<LrluAmd>("LeftRightLU AMD", t, opt, A, A, b, xTrue));
      rows.push_back(runOurs<LrluColamd>("LeftRightLU COLAMD", t, opt, A, A, b, xTrue));
#ifdef HAVE_METIS
      rows.push_back(runOurs<LrluMetis>("LeftRightLU METIS", t, opt, A, A, b, xTrue));
#endif
      rows.push_back(runOurs<SnluAmd>("SupernodalLU AMD (on Asym)", t, opt, Asym, A, b, xTrue));
    }
    rows.push_back(runPointBlock<PblkColamd>("PointBlockLU COLAMD (replay)", opt, A, b, xTrue));
    rows.push_back(runSparseLU(opt, A, b, xTrue));
#ifdef HAVE_PARDISO
    for (int t : opt.threads) rows.push_back(runPardiso(t, opt, A, b, xTrue));
#endif

    const Row* best = nullptr;
    for (const Row& r : rows) {
      printRow(r);
      // Only OUR solvers gate the exit code -- this benchmark is not a bug
      // report against Eigen or MKL.
      const bool ours = r.solver.rfind("LeftRightLU", 0) == 0 ||
                        r.solver.rfind("SupernodalLU", 0) == 0;
      if (ours && !(r.ok && std::isfinite(r.resid) && r.resid < kResidTolerance)) ++failures;
      if (r.ok && (!best || r.total() < best->total())) best = &r;
    }
    if (best)
      std::printf("  fastest: %s (%d thread%s) at %.2f ms\n", best->solver.c_str(), best->threads,
                  best->threads == 1 ? "" : "s", best->total());
  }

  std::printf("\n%d configuration(s) of our solvers failed to reach resid < %.0e.\n", failures,
              kResidTolerance);
  return failures == 0 ? 0 : 1;
}
