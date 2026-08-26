// Correctness sweep over the curated SuiteSparse corpus.
//
// Setup (once):
//   python test/matrices/fetch_suitesparse.py
// Then:
//   ctest --test-dir build -R test_suitesparse --output-on-failure
//
// WHY THIS EXISTS
//
// The hand-picked matrices in testdata/ are few and were chosen by what this
// project happened to encounter. They are all things these solvers handle. A
// corpus drawn from the SuiteSparse collection and stratified on PATTERN
// SYMMETRY deliberately includes matrices they should NOT handle well, because
// the interesting question is not "does it solve" but "does it behave correctly
// when it cannot solve".
//
// WHAT COUNTS AS PASSING
//
// Both solvers factor a symmetric pattern. Given a strongly unsymmetric pattern
// they must add structural zeros, fill can explode, and the numerics can go
// badly wrong. That is allowed. What is NOT allowed is doing so quietly. So for
// each matrix exactly one of these must hold:
//
//   * it factors and solve() reports Success -- then the residual must be small;
//   * it factors and solve() reports NumericalIssue -- an honest self-report,
//     which this suite confirms really was a bad solve (flagging a GOOD solve
//     as a failure would be its own defect);
//   * it is DECLINED at factorization -- info() == NumericalIssue plus a
//     diagnostic, via the setMaxFactorNonzeros guard, nothing huge allocated.
//
// The last two are PASSES. What fails is an UNFLAGGED wrong answer, a crash, or
// an attempt to allocate the machine into swap. That is exactly the honesty
// contract the README claims, and before this suite existed nothing tested it
// on matrices that actually break the solvers.
//
// The corpus is OPTIONAL: with nothing downloaded the suite reports how to get
// it and passes, so a fresh checkout is not broken by a missing download.

#include <Eigen/SparseCore>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::Tier;

namespace {

// Cap the factor at ~120M scalars (~1 GB of doubles) per solver. Chosen so the
// suite cannot exhaust a developer machine on the deliberately-pathological
// members of the corpus, while being far above what any well-conditioned
// member needs.
constexpr long long kFillLimit = 120LL * 1000 * 1000;

// A solve is judged against the same threshold the solver itself uses.
constexpr double kResidTolerance = 1e-6;

struct Outcome {
  bool factored = false;       // compute() succeeded
  bool declined = false;       // factorization refused (fill guard, breakdown)
  bool solveClaimedOk = false; // info() == Success AFTER solve()
  double resid = 0.0;          // residual WE measure, independently
  long long fill = 0;
  double fillRatio = 0.0;
  std::string note;
};

template <typename Solver>
Outcome run(const SparseMatrix<double>& A, const VectorXd& b) {
  Outcome o;
  Solver s;
  s.setMaxFactorNonzeros(kFillLimit);
  try {
    s.compute(A);
    if (s.info() != Eigen::Success) {
      o.declined = true;
      o.note = s.lastErrorMessage();
      return o;
    }
    const VectorXd x = s.solve(b);
    o.factored = true;
    // Crucially, read info() AFTER solve(): the documented contract is that
    // solve() measures the true residual itself and downgrades info() to
    // NumericalIssue rather than returning a bad answer quietly. Judging the
    // residual without reading this would punish the solver for being honest.
    o.solveClaimedOk = (s.info() == Eigen::Success);
    if (!o.solveClaimedOk) o.note = s.lastErrorMessage();
    o.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    o.fillRatio = static_cast<double>(o.fill) / static_cast<double>(A.nonZeros());
    o.resid = (A * x - b).norm() / b.norm();
  } catch (const std::exception& e) {
    o.note = std::string("threw: ") + e.what();
  }
  return o;
}

VectorXd deterministicRhs(const SparseMatrix<double>& A) {
  // Ones, scaled per row so the RHS is not accidentally orthogonal to anything
  // interesting; deterministic so a reported residual is reproducible.
  VectorXd x(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i)
    x(i) = 1.0 + 0.5 * std::sin(static_cast<double>(i));
  return A * x;
}

const char* bandOf(double psym) {
  if (psym >= 0.999) return "sym";
  if (psym >= 0.5) return "partial";
  return "unsym";
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  Tier maxTier = Tier::Small;  // "quick" tier by default
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tier" && i + 1 < argc) {
      const std::string t = argv[++i];
      if (t == "quick") maxTier = Tier::Small;
      else if (t == "standard") maxTier = Tier::Large;
      else if (t == "large") maxTier = Tier::Huge;
      else { std::printf("unknown tier '%s'\n", t.c_str()); return 2; }
    } else {
      std::printf("usage: %s [--tier quick|standard|large]\n", argv[0]);
      return 2;
    }
  }

  std::printf("SuiteSparse corpus sweep\n");
  std::printf("  manifest: %s\n", lu_testing::suitesparseManifestPath().c_str());

  const std::vector<lu_testing::SuiteSparseMatrix> all = lu_testing::suitesparseMatrices();
  if (all.empty()) {
    std::printf("\nNo manifest found -- corpus not configured. Nothing to test.\n");
    return lu_testing::summarize("SuiteSparse");
  }

  std::vector<lu_testing::SuiteSparseMatrix> selected;
  for (const auto& m : all)
    if (static_cast<int>(m.tier) <= static_cast<int>(maxTier)) selected.push_back(m);

  long long present = 0;
  for (const auto& m : selected) present += m.available ? 1 : 0;
  std::printf("  %lld of %zu matrices in this tier are downloaded", present, selected.size());
  if (present < static_cast<long long>(selected.size()))
    std::printf(" -- run: python test/matrices/fetch_suitesparse.py");
  std::printf("\n  fill guard: %lld scalars per solver\n\n", kFillLimit);

  if (present == 0) {
    std::printf("Corpus not downloaded; skipping (this is not a failure).\n");
    return lu_testing::summarize("SuiteSparse");
  }

  std::printf("%-30s %-8s %6s %9s %11s   %s\n", "matrix", "band", "psym", "n", "fill/nnz",
              "outcome");

  long long factored = 0, flagged = 0, declined = 0;
  for (const auto& m : selected) {
    if (!m.available) continue;

    SparseMatrix<double> A;
    try {
      A = lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(m.path));
    } catch (const std::exception& e) {
      lu_testing::fail(m.label() + ": load failed: " + e.what());
      continue;
    }
    const VectorXd b = deterministicRhs(A);

    const Outcome snlu = run<Eigen::SupernodalLU<SparseMatrix<double>>>(A, b);
    const Outcome lrlu = run<Eigen::LeftRightLU<SparseMatrix<double>>>(A, b);

    std::printf("%-30s %-8s %6.2f %9lld ", m.label().c_str(), bandOf(m.patternSymmetry),
                m.patternSymmetry, (long long)A.rows());
    if (snlu.factored) std::printf("%10.1fx   ", snlu.fillRatio);
    else std::printf("%11s   ", "-");

    // THE CONTRACT: a solver must never quietly return a wrong answer. It may
    // solve accurately, refuse to factor, or return a bad answer it has itself
    // flagged -- but not the fourth thing.
    auto judge = [&](const Outcome& o, const char* who) {
      const std::string key = m.label() + " [" + who + "]";
      if (o.factored && o.solveClaimedOk) {
        check(std::isfinite(o.resid) && o.resid < kResidTolerance,
              key + ": claimed Success, residual", o.resid);
        return;
      }
      if (o.factored && !o.solveClaimedOk) {
        // Honest self-report. Confirm it really is bad -- flagging a GOOD solve
        // as a failure would be its own defect.
        lu_testing::checkTrue(!std::isfinite(o.resid) || o.resid >= kResidTolerance,
                              key + ": flagged a solve that was genuinely bad");
        return;
      }
      if (o.declined) {
        lu_testing::checkTrue(!o.note.empty(), key + ": decline carries a diagnostic");
        return;
      }
      lu_testing::fail(key + ": neither solved nor declined cleanly (" + o.note + ")");
    };

    if (snlu.factored && snlu.solveClaimedOk) {
      std::printf("solved  resid %.1e", snlu.resid);
      ++factored;
    } else if (snlu.factored) {
      std::printf("flagged bad solve (resid %.1e)", snlu.resid);
      ++flagged;
    } else if (snlu.declined) {
      std::printf("declined (fill guard)");
      ++declined;
    } else {
      std::printf("ERROR");
    }
    std::printf("\n");

    judge(snlu, "SupernodalLU");
    judge(lrlu, "LeftRightLU");

    // The two solvers share the symbolic pipeline, so they must agree on
    // whether a matrix is feasible at all. A divergence means one of the numeric
    // cores has drifted from the shared analysis.
    lu_testing::checkTrue(snlu.factored == lrlu.factored,
                          m.label() + ": both solvers agree on feasibility");
    // Fill, though, is no longer shared: LeftRightLU permutes to block
    // triangular form and factors the diagonal blocks only, which SupernodalLU
    // does not do. So the contract is one-sided -- LeftRightLU may do far
    // better, and must not do meaningfully worse. (It bites even on the
    // pre-symmetrized matrices fed here, because a symmetric pattern still
    // splits into its connected components, and ordering those separately beats
    // ordering them together.)
    //
    // The tolerance is not slack for its own sake: BTF orders each block
    // independently, and a fill-reducing ordering is a heuristic, so a matrix
    // that barely splits can come out a per cent or two worse than one global
    // ordering. Anything beyond that is a real regression.
    if (snlu.factored && lrlu.factored) {
      const bool ok = lrlu.fill <= snlu.fill + snlu.fill / 20;
      lu_testing::checkTrue(ok, m.label() + ": LeftRightLU fill <= SupernodalLU fill");
      if (lrlu.fill != snlu.fill)
        std::printf("        BTF: LeftRightLU fill %lld vs SupernodalLU %lld (%.2fx)\n", lrlu.fill,
                    snlu.fill, static_cast<double>(lrlu.fill) / static_cast<double>(snlu.fill));
    }
  }

  std::printf("\n%lld solved, %lld returned a bad answer the solver itself flagged, %lld declined\n"
              "at factorization. The latter two are expected on the strongly unsymmetric band and\n"
              "count as correct behaviour -- what would fail here is an UNflagged bad answer.\n",
              factored, flagged, declined);
  return lu_testing::summarize("SuiteSparse");
}
