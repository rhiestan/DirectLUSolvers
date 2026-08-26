// Does the block triangular form pay? BTF on against BTF off, same solver, same
// matrix, same ordering, so every difference in a row is BTF's doing.
//
// Build + run (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/bench_btf                    # testdata + SuiteSparse corpora
//   ./build/bench_btf --quick            # synthetic only, no downloads needed
//   ./build/bench_btf --reps 15 ted_B    # one matrix, tighter estimate
//
// WHY THIS EXISTS
//
// LeftRightLU permutes to block triangular form by default and factors only the
// diagonal blocks. The fill effect of that is pinned per matrix by
// test_regression, and its correctness by test_btf -- but neither answers what a
// caller actually wants to know, which is whether turning it on made their
// factorization faster, and what it costs when it cannot help. No other
// benchmark here can answer that: they all run the shipping configuration, so
// BTF is either on in every row or absent from the comparison entirely.
//
// WHAT THE ROWS MEAN
//
// Matrices fall into three groups, and the summary separates them because their
// answers are completely different:
//
//   1. IRREDUCIBLE (one block). BTF found nothing and cost one O(n + nnz) sweep.
//      Every symmetric-pattern matrix is here -- its strongly connected
//      components are just its connected components -- so this is where every
//      PDE/FEM system lands. The measured cost is inside run-to-run noise.
//   2. REDUCIBLE, FILL UNCHANGED. The matrix split, but the split bought no
//      fill. This group PAYS: each block is ordered separately, and many small
//      AMD calls cost more than one large one. The cost is confined to
//      analyzePattern -- factorize and solve are untouched -- which is exactly
//      the phase a refactorization workflow skips, so it amortizes to nothing in
//      a Newton loop.
//   3. REDUCIBLE, FILL REDUCED. The case BTF exists for. Fill is superlinear in
//      block size, so the gain is not proportional to how much the matrix split:
//      a matrix that reduces to blocks of size 1 is fully triangular after
//      matching and has nothing left to factor at all.
//
// The residual columns matter as much as the timing. BTF changes which
// operations happen, so a speedup that arrived with a worse answer is not a win
// -- and on a structurally singular matrix the residual gets LOUDER, because BTF
// stops spurious fill from masking the singularity. Both are flagged by the
// solver either way; this benchmark only reports, it does not judge.

#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "LeftRightLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::ms;
using Clock = lu_testing::Clock;

typedef SparseMatrix<double> SpMat;

namespace {

struct Options {
  int reps = 5;
  bool quick = false;
  bool testdata = true;
  bool suitesparse = true;
  std::vector<std::string> filters;
};

struct Run {
  bool ok = false;
  double analyzeMs = 0, factorMs = 0, solveMs = 0;
  double resid = 0;
  long long fill = 0;
  long long blocks = 0, largest = 0, offDiag = 0;
  std::string note;

  double total() const { return analyzeMs + factorMs + solveMs; }
};

// Warm-up then minimum-of-N per phase, the same estimator bench_solvers uses:
// a slow repetition on a loaded desktop only ever means the machine did
// something else too, so the minimum is the least noisy estimate of achievable
// time.
template <typename Solver>
Run measure(const SpMat& A, const VectorXd& b, bool btf, int reps) {
  Run r;
  r.analyzeMs = r.factorMs = r.solveMs = 1e300;
  for (int rep = 0; rep <= reps; ++rep) {
    Solver s;
    s.setBlockTriangularForm(btf);
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      r.note = "declined: " + s.lastErrorMessage();
      return r;
    }
    const VectorXd x = s.solve(b);
    const auto t3 = Clock::now();
    if (rep == 0) continue;  // warm-up
    r.analyzeMs = std::min(r.analyzeMs, ms(t0, t1));
    r.factorMs = std::min(r.factorMs, ms(t1, t2));
    r.solveMs = std::min(r.solveMs, ms(t2, t3));
    // Not min-reduced: these are properties of the factorization, identical in
    // every repetition. Taking the last is taking the only one.
    r.resid = (A * x - b).norm() / b.norm();
    r.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    r.blocks = static_cast<long long>(s.btfBlockCount());
    r.largest = static_cast<long long>(s.largestBtfBlock());
    r.offDiag = static_cast<long long>(s.btfOffDiagonalNonzeros());
  }
  r.ok = true;
  return r;
}

// Deterministic, so a reported residual is reproducible; scaled per row so the
// right-hand side is not accidentally orthogonal to anything interesting.
VectorXd deterministicRhs(const SpMat& A) {
  VectorXd x(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) x(i) = 1.0 + 0.5 * std::sin(static_cast<double>(i));
  return A * x;
}

enum Group { kIrreducible = 0, kNoFillChange = 1, kFillReduced = 2, kGroupCount = 3 };

const char* groupName(int g) {
  switch (g) {
    case kIrreducible: return "irreducible (1 block)";
    case kNoFillChange: return "reducible, fill unchanged";
    default: return "reducible, fill reduced";
  }
}

struct Summary {
  std::vector<double> speedup[kGroupCount];
  double fillOn[kGroupCount] = {0, 0, 0};
  double fillOff[kGroupCount] = {0, 0, 0};
  double timeOn = 0, timeOff = 0;
};

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t mid = v.size() / 2;
  return v.size() % 2 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

template <typename Solver>
void sweep(const char* ordering, const std::vector<std::pair<std::string, SpMat>>& corpus,
           const Options& opt, Summary& sum) {
  std::printf("\n=== LeftRightLU, %s ordering\n", ordering);
  std::printf("%-26s %7s %8s %5s | %11s %11s %6s | %15s %17s %13s %7s | %9s %9s\n", "matrix",
              "blocks", "largest", "%of n", "fill off", "fill on", "ratio", "analyze off>on",
              "factor off>on", "solve off>on", "speedup", "resid off", "resid on");

  for (const auto& entry : corpus) {
    const SpMat& A = entry.second;
    const VectorXd b = deterministicRhs(A);
    const Run off = measure<Solver>(A, b, false, opt.reps);
    const Run on = measure<Solver>(A, b, true, opt.reps);
    std::printf("%-26s ", entry.first.c_str());
    if (!off.ok || !on.ok) {
      std::printf("%s\n", (off.ok ? on.note : off.note).c_str());
      continue;
    }

    const double pct = 100.0 * static_cast<double>(on.largest) / static_cast<double>(A.rows());
    const double fillRatio = static_cast<double>(on.fill) / static_cast<double>(off.fill);
    const double speedup = off.total() / on.total();
    std::printf("%7lld %8lld %4.0f%% | %11lld %11lld %6.3f | %6.2f>%-7.2f %7.2f>%-8.2f "
                "%5.2f>%-6.2f %6.2fx | %9.1e %9.1e\n",
                on.blocks, on.largest, pct, off.fill, on.fill, fillRatio, off.analyzeMs,
                on.analyzeMs, off.factorMs, on.factorMs, off.solveMs, on.solveMs, speedup,
                off.resid, on.resid);

    const int g = on.blocks <= 1 ? kIrreducible : (fillRatio > 0.99 ? kNoFillChange : kFillReduced);
    sum.speedup[g].push_back(speedup);
    sum.fillOn[g] += static_cast<double>(on.fill);
    sum.fillOff[g] += static_cast<double>(off.fill);
    sum.timeOn += on.total();
    sum.timeOff += off.total();
  }
}

void report(const char* ordering, const Summary& s) {
  std::printf("%s\n", ordering);
  std::size_t rows = 0;
  for (int g = 0; g < kGroupCount; ++g) {
    const std::vector<double>& sp = s.speedup[g];
    rows += sp.size();
    if (sp.empty()) continue;
    std::printf("  %-28s n=%2zu  speedup median %.2fx  min %.2fx  max %.2fx  fill %.3fx\n",
                groupName(g), sp.size(), median(sp), *std::min_element(sp.begin(), sp.end()),
                *std::max_element(sp.begin(), sp.end()),
                s.fillOff[g] > 0 ? s.fillOn[g] / s.fillOff[g] : 1.0);
  }
  std::vector<double> all;
  for (int g = 0; g < kGroupCount; ++g)
    all.insert(all.end(), s.speedup[g].begin(), s.speedup[g].end());
  // Corpus total AND median, deliberately: the total is dominated by whichever
  // matrix is slowest, the median says what a typical matrix sees, and on this
  // corpus the two differ by a lot. Quoting either alone misleads.
  std::printf("  %zu matrices: corpus total %.1f ms off -> %.1f ms on (%.2fx), median row %.2fx\n",
              rows, s.timeOff, s.timeOn, s.timeOn > 0 ? s.timeOff / s.timeOn : 1.0, median(all));
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) opt.reps = std::atoi(argv[++i]);
    else if (a == "--quick") opt.quick = true;
    else if (a == "--testdata-only") opt.suitesparse = false;
    else if (a == "--suitesparse-only") opt.testdata = false;
    else if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
      std::printf("usage: %s [--reps N] [--quick] [--testdata-only|--suitesparse-only] "
                  "[name-substring ...]\n", argv[0]);
      return 2;
    } else {
      opt.filters.push_back(a);  // substring filter on the matrix label
    }
  }
  auto keep = [&opt](const std::string& label) {
    if (opt.filters.empty()) return true;
    for (const std::string& f : opt.filters)
      if (label.find(f) != std::string::npos) return true;
    return false;
  };

  std::vector<std::pair<std::string, SpMat>> corpus;
  if (opt.quick) {
    // Enough to exercise the code paths without needing anything downloaded: an
    // upwind grid is fully triangular after matching (n singleton blocks), a
    // Laplacian is irreducible, and a random unsymmetric pattern splits
    // partially. Group 2 -- reducible but no fill change, the one that costs --
    // has no synthetic stand-in here; it needs the real corpus.
    if (keep("upwind2d_60x60")) corpus.emplace_back("upwind2d_60x60", lu_testing::upwind2d(60, 60));
    if (keep("lap2d_60x60")) corpus.emplace_back("lap2d_60x60", lu_testing::laplacian2d(60, 60));
    if (keep("unsym_2000"))
      corpus.emplace_back("unsym_2000", lu_testing::randomUnsymmetricPattern(2000, 0.002, 7u));
  } else {
    if (opt.testdata) {
      for (const auto& m : lu_testing::benchmarkMatrices()) {
        if (m.tier != lu_testing::Tier::Small || !keep(m.label)) continue;
        try {
          corpus.emplace_back(m.label,
                              lu_testing::loadMatrixMarket(lu_testing::testdataPath(m.relative)));
        } catch (const std::exception&) {
          std::printf("  (skipped %s: not present)\n", m.label);
        }
      }
    }
    if (opt.suitesparse) {
      for (const auto& m : lu_testing::suitesparseMatrices()) {
        if (!m.available || m.tier != lu_testing::Tier::Small || !keep(m.label())) continue;
        try {
          corpus.emplace_back(m.label(), lu_testing::loadMatrixMarket(m.path));
        } catch (const std::exception&) {
        }
      }
    }
  }

  std::printf("Block triangular form: on vs off, %zu matrices, best of %d after a warm-up\n",
              corpus.size(), opt.reps);
  if (corpus.empty()) {
    std::printf("Nothing to measure. Run with --quick, or fetch the corpora:\n"
                "  python test/matrices/fetch_suitesparse.py\n");
    return 0;
  }

  Summary amd, colamd;
  sweep<Eigen::LeftRightLU<SpMat, Eigen::AMDOrdering<int>>>("AMD", corpus, opt, amd);
  sweep<Eigen::LeftRightLU<SpMat, Eigen::COLAMDOrdering<int>>>("COLAMD", corpus, opt, colamd);

  std::printf("\n=== summary\n");
  report("AMD", amd);
  report("COLAMD", colamd);
  // Always 0: every outcome here is information, including a residual that got
  // worse. Whether a solve is acceptable is test_suitesparse's judgement to make,
  // and the fill is test_regression's -- this is a benchmark, not a gate.
  return 0;
}
