// Fill and accuracy regression tests for SupernodalLU and LeftRightLU.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_regression --output-on-failure
//
// WHY THIS EXISTS
//
// The other suites gate on the residual, and compare_testdata gates on
// resid < 1e-6. That catches a solver returning a wrong answer -- but it does
// NOT catch the failure mode that matters most here: a change that leaves every
// residual at machine precision while inflating FILL by orders of magnitude.
// Applying the fill-reducing permutation in the wrong direction, for instance,
// keeps residuals perfect and shows up only as a 250-350x larger factor on 3D
// matrices; no residual-based check can see it.
//
// So this suite pins nnzL + nnzU per (matrix, solver) against a checked-in
// baseline. Fill is a deterministic function of the pattern and the ordering, so
// a mismatch beyond the tolerance is a real structural change -- either a
// regression, or an improvement that should be reviewed and re-baselined
// deliberately rather than silently absorbed.
//
// USAGE
//
//   test_regression                    # check against test/baselines/testdata.baseline
//   test_regression --synthetic-only   # only the generated matrices; needs no testdata/
//   test_regression --tier small       # skip the large 3D FEM systems
//   test_regression --tolerance 0.10   # allow 10% fill drift (default 5%)
//   test_regression --check-time       # also gate factor+solve time (wide band, machine-dependent)
//   test_regression --update           # REGENERATE the baseline file, then exit
//
// Re-baseline only when you have understood why the fill moved. `--update`
// rewrites every entry, so read the diff before committing it.

#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

#ifdef HAVE_METIS
#include "SupernodalLUMetis.h"
#endif

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::ms;
using lu_testing::Tier;
using Clock = lu_testing::Clock;

namespace {

// Default fill tolerance. Fill is deterministic for a fixed pattern + ordering,
// so this is slack for benign changes (an amalgamation tweak moving a handful of
// structural zeros), not for genuine structural drift.
constexpr double kDefaultFillTolerance = 0.05;

// A solve is required to reach this, unless the baseline records that this
// matrix never did (some testdata matrices are singular or near-singular by
// construction) -- in which case a 10x degradation from the recorded value fails.
constexpr double kResidTolerance = 1e-6;
constexpr double kResidSlack = 10.0;

// Only gated with --check-time. Wall-clock is machine- and load-dependent, so
// the band is deliberately wide: this catches an algorithmic blow-up, not noise.
constexpr double kTimeSlack = 3.0;

// ---------------------------------------------------------------------------
//  Measurement
// ---------------------------------------------------------------------------

struct Record {
  std::string matrix;
  std::string solver;
  long long n = 0, nnz = 0;
  long long nnzL = -1, nnzU = -1;  // -1 == the solver declined / failed
  long long snodes = 0;
  double resid = 0.0;
  double timeMs = 0.0;

  long long fill() const { return nnzL + nnzU; }
  bool factored() const { return nnzL >= 0; }
};

// A deterministic right-hand side. Eigen's VectorXd::Random draws from the
// global C rand() state, which makes a recorded residual irreproducible; this
// does not.
VectorXd deterministicRhs(const SparseMatrix<double>& A, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  VectorXd xTrue(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) xTrue(i) = uni(rng);
  return A * xTrue;
}

// `configure` runs before compute(), for solver options that are set at run time
// rather than chosen by type (MC64 matching, for instance).
template <typename Solver, typename Configure>
Record measure(const std::string& matrix, const std::string& solverName,
               const SparseMatrix<double>& A, Configure configure) {
  Record r;
  r.matrix = matrix;
  r.solver = solverName;
  r.n = A.rows();
  r.nnz = A.nonZeros();

  const VectorXd b = deterministicRhs(A, 20260807u);
  try {
    Solver solver;
    configure(solver);
    const auto t0 = Clock::now();
    solver.compute(A);
    if (solver.info() != Eigen::Success) return r;  // declined: nnzL stays -1
    const VectorXd x = solver.solve(b);
    const auto t1 = Clock::now();
    r.timeMs = ms(t0, t1);
    r.nnzL = solver.nnzL();
    r.nnzU = solver.nnzU();
    r.snodes = static_cast<long long>(solver.supernodeCount());
    r.resid = (A * x - b).norm() / b.norm();
  } catch (const std::exception&) {
    r.nnzL = -1;
  }
  return r;
}

template <typename Solver>
Record measure(const std::string& matrix, const std::string& solverName,
               const SparseMatrix<double>& A) {
  return measure<Solver>(matrix, solverName, A, [](Solver&) {});
}

// ---------------------------------------------------------------------------
//  Cases
// ---------------------------------------------------------------------------

struct Case {
  std::string label;
  bool synthetic;
  Tier tier;
  std::function<SparseMatrix<double>()> build;
};

std::vector<Case> buildCases() {
  std::vector<Case> cases;

  // Synthetic: fully determined by this source file, so these run on a checkout
  // without testdata/ and pin the ordering/amalgamation behaviour on structures
  // whose fill is well understood (a 3D Laplacian is the shape that exposed the
  // ordering-direction bug).
  cases.push_back({"lap2d_60x60", true, Tier::Small, [] { return lu_testing::laplacian2d(60, 60); }});
  cases.push_back({"lap2d_120x120", true, Tier::Small, [] { return lu_testing::laplacian2d(120, 120); }});
  cases.push_back({"lap3d_16x16x16", true, Tier::Small, [] { return lu_testing::laplacian3d(16, 16, 16); }});
  cases.push_back({"lap3d_20x20x20", true, Tier::Small, [] { return lu_testing::laplacian3d(20, 20, 20); }});
  cases.push_back({"random_400", true, Tier::Small,
                   [] { return lu_testing::randomSymmetricPattern(400, 0.02, 4242); }});
  cases.push_back({"weakdiag_500", true, Tier::Small, [] { return lu_testing::weakDiagonal(500, 7); }});

  // Real matrices from the shared registry. pre2 (Tier::Huge) is excluded: the
  // solvers decline it by design, so there is no fill to pin.
  for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices()) {
    if (m.tier == Tier::Huge) continue;
    const std::string path = lu_testing::testdataPath(m.relative);
    // Loaded RAW -- the symmetric-pattern copy SupernodalLU needs is made per
    // case below, so LeftRightLU can be pinned on the unsymmetric original it
    // actually supports.
    cases.push_back({m.label, false, m.tier,
                     [path] { return lu_testing::loadMatrixMarket(path); }});
  }
  return cases;
}

// ---------------------------------------------------------------------------
//  Baseline file I/O
// ---------------------------------------------------------------------------

std::string baselinePath() {
  if (const char* env = std::getenv("DLU_BASELINE_FILE")) {
    if (env[0] != '\0') return std::string(env);
  }
  return std::string(DLU_BASELINE_FILE);
}

typedef std::map<std::pair<std::string, std::string>, Record> BaselineMap;

BaselineMap readBaseline(const std::string& path, bool* found) {
  BaselineMap out;
  std::ifstream in(path);
  *found = static_cast<bool>(in);
  if (!in) return out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    Record r;
    if (!(iss >> r.matrix >> r.solver >> r.n >> r.nnz >> r.nnzL >> r.nnzU >> r.snodes >> r.resid >>
          r.timeMs))
      continue;
    out[{r.matrix, r.solver}] = r;
  }
  return out;
}

void writeBaseline(const std::string& path, const std::vector<Record>& records) {
  std::ofstream out(path);
  if (!out) {
    lu_testing::fail("cannot write baseline file " + path);
    return;
  }
  out << "# DirectLUSolvers fill/accuracy regression baselines.\n"
      << "#\n"
      << "# Regenerate with:  test_regression --update\n"
      << "# Checked by:       test_regression   (see test/test_regression.cpp)\n"
      << "#\n"
      << "# Fill (nnzL+nnzU) is the load-bearing column: it is a deterministic function of\n"
      << "# the sparsity pattern and the fill-reducing ordering, so a change here is a real\n"
      << "# structural change in the solver, not numerical noise. A large jump is exactly\n"
      << "# the signature of an ordering-direction mistake, which leaves every residual at\n"
      << "# machine precision while inflating 3D factors 250-350x.\n"
      << "#\n"
      << "# nnzL = nnzU = -1 means the solver declined the matrix (info() != Success);\n"
      << "# that is a pinned behaviour too -- a solver that suddenly accepts it has changed.\n"
      << "#\n"
      << "# resid is the achieved relative residual, recorded for reference. The gate is\n"
      << "# resid <= max(1e-6, 10 * recorded), so a matrix that legitimately cannot reach\n"
      << "# 1e-6 (singular by construction) is still held to its own accuracy.\n"
      << "#\n"
      << "# time_ms is informational; it is only gated under --check-time.\n"
      << "#\n"
      << "# matrix solver n nnz nnzL nnzU supernodes resid time_ms\n";
  for (const Record& r : records) {
    out << r.matrix << ' ' << r.solver << ' ' << r.n << ' ' << r.nnz << ' ' << r.nnzL << ' '
        << r.nnzU << ' ' << r.snodes << ' ';
    // enough digits to be meaningful, few enough to keep the diff readable
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3e %.1f", r.resid, r.timeMs);
    out << buf << '\n';
  }
}

// ---------------------------------------------------------------------------
//  Comparison
// ---------------------------------------------------------------------------

double relativeDelta(long long observed, long long baseline) {
  if (baseline == 0) return observed == 0 ? 0.0 : 1.0;
  return static_cast<double>(observed - baseline) / static_cast<double>(baseline);
}

void compareAgainstBaseline(const Record& r, const BaselineMap& baseline, double fillTolerance,
                            bool checkTime) {
  const std::string key = r.matrix + "/" + r.solver;
  const auto it = baseline.find({r.matrix, r.solver});
  if (it == baseline.end()) {
    lu_testing::fail("no baseline entry for " + key + " (run --update to add it)");
    return;
  }
  const Record& b = it->second;

  // A changed matrix invalidates every other comparison for this row, so this
  // is reported on its own rather than as a confusing fill mismatch.
  if (r.n != b.n || r.nnz != b.nnz) {
    lu_testing::fail(key + ": matrix changed (n " + std::to_string(b.n) + "->" +
                     std::to_string(r.n) + ", nnz " + std::to_string(b.nnz) + "->" +
                     std::to_string(r.nnz) + "); baseline is stale");
    return;
  }

  if (r.factored() != b.factored()) {
    lu_testing::fail(key + (r.factored() ? ": now factors, baseline recorded a decline"
                                         : ": now declines, baseline recorded a factorization"));
    return;
  }
  if (!r.factored()) {
    checkTrue(true, key + ": still declined (as baselined)");
    return;
  }

  const double fillDelta = relativeDelta(r.fill(), b.fill());
  check(std::abs(fillDelta) <= fillTolerance, key + ": fill nnzL+nnzU", fillDelta);
  if (std::abs(fillDelta) > fillTolerance)
    lu_testing::note("fill " + std::to_string(b.fill()) + " -> " + std::to_string(r.fill()) +
                     " (" + std::to_string(fillDelta * 100.0) + "%)");

  const double residLimit = std::max(kResidTolerance, b.resid * kResidSlack);
  check(std::isfinite(r.resid) && r.resid <= residLimit, key + ": residual", r.resid);

  // Supernode count is informational: a different partition with the same fill
  // is a legitimate amalgamation change, not a regression.
  if (r.snodes != b.snodes)
    lu_testing::note(key + ": supernodes " + std::to_string(b.snodes) + " -> " +
                     std::to_string(r.snodes) + " (informational)");

  if (checkTime && b.timeMs > 0.0) {
    const double ratio = r.timeMs / b.timeMs;
    check(ratio <= kTimeSlack, key + ": factor+solve time ratio", ratio);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  bool update = false, syntheticOnly = false, checkTime = false;
  double fillTolerance = kDefaultFillTolerance;
  Tier maxTier = Tier::Large;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--update") update = true;
    else if (arg == "--synthetic-only") syntheticOnly = true;
    else if (arg == "--check-time") checkTime = true;
    else if (arg == "--tolerance" && i + 1 < argc) fillTolerance = std::atof(argv[++i]);
    else if (arg == "--tier" && i + 1 < argc) {
      const std::string t = argv[++i];
      if (t == "small") maxTier = Tier::Small;
      else if (t == "large") maxTier = Tier::Large;
      else if (t == "huge") maxTier = Tier::Huge;
      else { std::printf("unknown tier '%s'\n", t.c_str()); return 2; }
    } else {
      std::printf("usage: %s [--update] [--synthetic-only] [--tier small|large|huge]\n"
                  "          [--tolerance FRAC] [--check-time]\n", argv[0]);
      return 2;
    }
  }

  std::printf("DirectLUSolvers fill/accuracy regression\n");
  std::printf("  baseline:  %s\n", baselinePath().c_str());
  std::printf("  mode:      %s\n", update ? "UPDATE (rewriting baselines)" : "check");
  if (!update) std::printf("  fill tol:  %.1f%%\n", fillTolerance * 100.0);
  std::printf("  matrices:  %s\n", syntheticOnly ? "synthetic only" : "synthetic + testdata");

  bool baselineFound = false;
  const BaselineMap baseline = readBaseline(baselinePath(), &baselineFound);
  if (!update && !baselineFound) {
    std::printf("\nERROR: baseline file not found. Generate it once with:\n"
                "  %s --update\n", argv[0]);
    return 1;
  }

  std::vector<Record> records;
  for (const Case& c : buildCases()) {
    if (syntheticOnly && !c.synthetic) continue;
    if (static_cast<int>(c.tier) > static_cast<int>(maxTier)) continue;

    SparseMatrix<double> A;
    try {
      A = c.build();
    } catch (const std::exception& e) {
      lu_testing::fail(c.label + ": load/build failed: " + e.what());
      continue;
    }

    // Each solver is pinned on the input it is designed for. SupernodalLU
    // requires a symmetric pattern, so it gets the padded copy; LeftRightLU
    // symmetrizes internally, AFTER matching, and padding its input first is the
    // documented mistake (102x fill on gemat11) -- so pinning it on the padded
    // copy would guard a path no caller should take.
    const SparseMatrix<double> Asym = lu_testing::ensureSymmetricPattern(A);
    std::printf("== %s (n=%lld nnz=%lld, symmetrized nnz=%lld)\n", c.label.c_str(),
                (long long)A.rows(), (long long)A.nonZeros(), (long long)Asym.nonZeros());

    std::vector<Record> row;
    row.push_back(measure<Eigen::SupernodalLU<SparseMatrix<double>>>(c.label, "SupernodalLU", Asym));
    row.push_back(measure<Eigen::LeftRightLU<SparseMatrix<double>>>(c.label, "LeftRightLU", A));
    // MC64 is a separate symbolic path -- a different permutation, so different
    // fill -- and it is deterministic, so it is pinnable like the rest. Without
    // this it would be the one code path a fill regression could slip through.
    row.push_back(measure<Eigen::SupernodalLU<SparseMatrix<double>>>(
        c.label, "SupernodalLU+MC64", Asym, [](Eigen::SupernodalLU<SparseMatrix<double>>& s) {
          s.setMatchingMethod(Eigen::supernodal_lu::MatchingMethod::MC64);
        }));
#ifdef HAVE_METIS
    row.push_back(measure<Eigen::SupernodalLUMetis<SparseMatrix<double>>>(
        c.label, "SupernodalLU+METIS", Asym));
#endif

    for (const Record& r : row) {
      if (r.factored())
        std::printf("   %-20s fill=%-12lld snodes=%-7lld resid=%.2e  %.1f ms\n", r.solver.c_str(),
                    r.fill(), r.snodes, r.resid, r.timeMs);
      else
        std::printf("   %-20s declined (info() != Success)\n", r.solver.c_str());
      if (!update) compareAgainstBaseline(r, baseline, fillTolerance, checkTime);
      records.push_back(r);
    }
  }

  if (update) {
    // Preserve entries this run did not cover (e.g. the testdata rows when
    // updating with --synthetic-only) so a partial update is not a silent
    // deletion of the rest of the baseline.
    std::map<std::pair<std::string, std::string>, Record> merged = baseline;
    for (const Record& r : records) merged[{r.matrix, r.solver}] = r;
    std::vector<Record> all;
    all.reserve(merged.size());
    for (const auto& kv : merged) all.push_back(kv.second);
    writeBaseline(baselinePath(), all);
    std::printf("\nWrote %zu baseline entries (%zu measured this run) to\n  %s\n", all.size(),
                records.size(), baselinePath().c_str());
    std::printf("Review the diff before committing -- a fill change is a real change.\n");
    return lu_testing::failureCount() == 0 ? 0 : 1;
  }

  return lu_testing::summarize("Regression");
}
