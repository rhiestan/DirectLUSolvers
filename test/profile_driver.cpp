// VTune profiling driver: per-phase timing over the SuiteSparse corpus.
//
// Not a test. It exists so a profiler sees analyzePattern / factorize / solve
// as three separately-attributable phases instead of one compute() blob, and so
// a run can be restricted to one solver and one phase.
//
//   profile_driver [--solver snlu|lrlu] [--phase all|analyze|factorize|solve]
//                  [--reps N] [--matrix <group/name> ...]
//
// Phases are wrapped in ITT task markers when built with DLU_WITH_ITT, so
// VTune's "Task Type" grouping splits the timeline without a second run.

#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

#ifdef DLU_WITH_ITT
#include <ittnotify.h>
namespace {
__itt_domain* ittDomain() {
  static __itt_domain* d = __itt_domain_create("DirectLUSolvers");
  return d;
}
struct IttTask {
  explicit IttTask(const char* name) {
    __itt_string_handle* h = __itt_string_handle_create(name);
    __itt_task_begin(ittDomain(), __itt_null, __itt_null, h);
  }
  ~IttTask() { __itt_task_end(ittDomain()); }
};
}  // namespace
#define DLU_ITT_TASK(name) IttTask dlu_itt_scoped_task(name)
#else
#define DLU_ITT_TASK(name) ((void)0)
#endif

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::supernodal_lu::PooledExecutor;
using Clock = std::chrono::steady_clock;

// Threaded instantiations. PooledExecutor (not StdThreadExecutor) so the thread
// count can be set after construction.
using SnluMT = Eigen::SupernodalLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, PooledExecutor>;
using LrluMT = Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, PooledExecutor>;

namespace {

constexpr long long kFillLimit = 120LL * 1000 * 1000;

double secondsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

VectorXd deterministicRhs(const SparseMatrix<double>& A) {
  VectorXd x(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i)
    x(i) = 1.0 + 0.5 * std::sin(static_cast<double>(i));
  return A * x;
}

struct Timing {
  double analyze = 0.0, factorize = 0.0, solve = 0.0;
  long long fill = 0;
  bool ok = false;
};

enum class Phase { All, Analyze, Factorize, Solve };

template <typename Solver>
Timing profileOne(const SparseMatrix<double>& A, const VectorXd& b, Phase phase, int reps,
                  int threads) {
  Timing t;
  Solver s;
  s.setMaxFactorNonzeros(kFillLimit);
  if (threads > 1) s.executor() = PooledExecutor(threads);

  // analyzePattern is idempotent, so repeating it is legitimate; factorize and
  // solve are repeated against the one analysis, which is how a caller with
  // many right-hand sides actually uses these solvers.
  const bool wantAnalyze = (phase == Phase::All || phase == Phase::Analyze);
  const bool wantFact = (phase == Phase::All || phase == Phase::Factorize);
  const bool wantSolve = (phase == Phase::All || phase == Phase::Solve);

  try {
    const int analyzeReps = wantAnalyze ? reps : 1;
    for (int r = 0; r < analyzeReps; ++r) {
      DLU_ITT_TASK("analyzePattern");
      const auto t0 = Clock::now();
      s.analyzePattern(A);
      t.analyze += secondsSince(t0);
    }
    t.analyze /= analyzeReps;
    if (s.info() != Eigen::Success) return t;

    const int factReps = wantFact ? reps : 1;
    for (int r = 0; r < factReps; ++r) {
      DLU_ITT_TASK("factorize");
      const auto t0 = Clock::now();
      s.factorize(A);
      t.factorize += secondsSince(t0);
    }
    t.factorize /= factReps;
    if (s.info() != Eigen::Success) return t;
    t.fill = static_cast<long long>(s.nnzL()) + s.nnzU();

    if (wantSolve) {
      VectorXd x;
      for (int r = 0; r < reps; ++r) {
        DLU_ITT_TASK("solve");
        const auto t0 = Clock::now();
        x = s.solve(b);
        t.solve += secondsSince(t0);
      }
      t.solve /= reps;
      t.ok = (s.info() == Eigen::Success);
    } else {
      t.ok = true;
    }
  } catch (const std::exception& e) {
    std::printf("    threw: %s\n", e.what());
  }
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  std::string solver = "both";
  Phase phase = Phase::All;
  int reps = 1;
  int threads = 1;
  std::string synthetic;
  std::vector<std::string> want;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--solver" && i + 1 < argc) solver = argv[++i];
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (a == "--matrix" && i + 1 < argc) want.push_back(argv[++i]);
    else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (a == "--synthetic" && i + 1 < argc) synthetic = argv[++i];
    else if (a == "--phase" && i + 1 < argc) {
      const std::string p = argv[++i];
      if (p == "all") phase = Phase::All;
      else if (p == "analyze") phase = Phase::Analyze;
      else if (p == "factorize") phase = Phase::Factorize;
      else if (p == "solve") phase = Phase::Solve;
      else { std::printf("unknown phase '%s'\n", p.c_str()); return 2; }
    } else {
      std::printf("usage: %s [--solver snlu|lrlu|both] [--phase all|analyze|factorize|solve]\n"
                  "          [--reps N] [--matrix <group/name>]...\n", argv[0]);
      return 2;
    }
  }
  if (reps < 1) reps = 1;

  // A synthetic run replaces the corpus entirely: these are the regular-grid
  // Laplacians whose elimination trees have a big root separator, which is the
  // shape that exposes scheduler tail behaviour.
  if (!synthetic.empty()) {
    SparseMatrix<double> A;
    if (synthetic == "lap2d") A = lu_testing::laplacian2d(300, 300);
    else if (synthetic == "lap3d") A = lu_testing::laplacian3d(30, 30, 30);
    else { std::printf("unknown --synthetic '%s'\n", synthetic.c_str()); return 2; }
    const VectorXd b = deterministicRhs(A);
    std::printf("%-30s %9s %10s %10s %10s %10s %12s\n", "matrix", "n", "solver",
                "analyze/s", "factor/s", "solve/s", "fill");
    auto show = [&](const char* who, const Timing& t) {
      std::printf("%-30s %9lld %10s %10.4f %10.4f %10.4f %12lld\n", synthetic.c_str(),
                  (long long)A.rows(), who, t.analyze, t.factorize, t.solve, t.fill);
    };
    if (solver == "snlu" || solver == "both")
      show("SNLU", profileOne<SnluMT>(A, b, phase, reps, threads));
    if (solver == "lrlu" || solver == "both")
      show("LRLU", profileOne<LrluMT>(A, b, phase, reps, threads));
    return 0;
  }

  std::vector<lu_testing::SuiteSparseMatrix> selected;
  for (const auto& m : lu_testing::suitesparseMatrices()) {
    if (!m.available) continue;
    if (!want.empty() &&
        std::find(want.begin(), want.end(), m.label()) == want.end())
      continue;
    selected.push_back(m);
  }
  if (selected.empty()) {
    std::printf("No matrices selected (downloaded corpus empty?).\n");
    return 1;
  }

  std::printf("%-30s %9s %10s %10s %10s %10s %12s\n", "matrix", "n", "solver",
              "analyze/s", "factor/s", "solve/s", "fill");

  double totAnalyze = 0, totFact = 0, totSolve = 0, totLoad = 0;
  for (const auto& m : selected) {
    SparseMatrix<double> A;
    double loadSeconds = 0.0;
    try {
      const auto tLoad = Clock::now();
      A = lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(m.path));
      loadSeconds = secondsSince(tLoad);
    } catch (const std::exception& e) {
      std::printf("%-30s load failed: %s\n", m.label().c_str(), e.what());
      continue;
    }
    totLoad += loadSeconds;
    const VectorXd b = deterministicRhs(A);

    auto report = [&](const char* who, const Timing& t) {
      std::printf("%-30s %9lld %10s %10.4f %10.4f %10.4f %12lld%s\n", m.label().c_str(),
                  (long long)A.rows(), who, t.analyze, t.factorize, t.solve, t.fill,
                  t.ok ? "" : "  (declined/bad)");
      totAnalyze += t.analyze; totFact += t.factorize; totSolve += t.solve;
    };

    if (solver == "snlu" || solver == "both")
      report("SNLU", profileOne<SnluMT>(A, b, phase, reps, threads));
    if (solver == "lrlu" || solver == "both")
      report("LRLU", profileOne<LrluMT>(A, b, phase, reps, threads));
  }

  std::printf("\ntotal: load %.3fs  analyze %.3fs  factorize %.3fs  solve %.3fs\n",
              totLoad, totAnalyze, totFact, totSolve);
  return 0;
}
