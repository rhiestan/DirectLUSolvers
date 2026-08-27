// RobustLU: the fallback ladder.
//
//   ctest --test-dir build -R test_robust_lu --output-on-failure
//
// WHY THIS EXISTS
//
// The ladder's value is not "it solves more matrices" -- it is that it makes the
// right decision for the right reason, and stops when no decision helps. Those
// are separable properties and this suite tests them separately:
//
//   1. COST. On a matrix the first rung handles, the ladder must do exactly one
//      factorization. A ladder that quietly doubles everyone's work to rescue a
//      minority is a bad trade, and this is the check that would catch it.
//   2. ESCALATION. On a matrix the first rung fails, it must reach a strategy
//      that works -- and record why it moved.
//   3. STOPPING. On a structurally singular or hopelessly ill-conditioned
//      matrix it must stop immediately rather than burn full factorizations
//      rediscovering a property of the matrix. This is the property that cost
//      the most design effort, so it gets the most direct test: count the
//      attempts.
//   4. HONESTY. It must never report Success with an untrustworthy answer, and
//      its report() must match what actually happened.
//
// The SuiteSparse corpus is the proving ground when present, because the
// expected outcome per matrix was measured before the ladder was written. The
// suite still passes without it, on constructed matrices.

#include <Eigen/SparseCore>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "RobustLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::note;
namespace rlu = Eigen::robust_lu;

typedef SparseMatrix<double> SpMat;
typedef Eigen::RobustLU<SpMat> Robust;

namespace {

VectorXd deterministicRhs(const SpMat& A) {
  VectorXd x(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) x[i] = 1.0 + 0.5 * std::sin(double(i));
  return A * x;
}

// A matrix with a structurally empty column: no zero-free diagonal exists, so
// no LU does either, whatever the pivoting.
SpMat structurallySingular(int n) {
  std::vector<Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    if (j == n / 2) continue;  // this column stays empty
    t.emplace_back(j, j, 3.0);
    if (j > 0) t.emplace_back(j - 1, j, 1.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// Backward-stable and yet completely wrong: kappa ~ 1e19, so no rung can help.
//
// The superdiagonal is -1.7 rather than -2 on purpose. With -2 the whole
// factorization is exact in binary floating point, so the answer comes out
// perfect however enormous kappa is -- and flagging THAT matrix as
// ill-conditioned would be a false alarm rather than a test. At -1.7 everything
// rounds, and the computed answer really is wrong by orders of magnitude while
// its residual stays at 1e-14. That gap is the thing being tested.
SpMat hopelesslyIllConditioned(int n) {
  std::vector<Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    t.emplace_back(j, j, 1.0);
    if (j > 0) t.emplace_back(j - 1, j, -1.7);
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// Rank n-1 by construction, and analytically so: row k and column k are both
// empty. The range of A therefore excludes e_k, which makes the inconsistency of
// any right-hand side exactly its k-th component -- so the least-squares answer,
// its residual and its optimality are all known in closed form rather than
// measured against a tolerance.
SpMat rankDeficientDiagonal(int n, int k) {
  std::vector<Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    if (j == k) continue;
    t.emplace_back(j, j, double(2 + (j % 5)));
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

void testCostsNothingWhenTheFirstRungWorks() {
  std::printf("  a well-behaved matrix costs exactly one factorization\n");
  const SpMat A = lu_testing::laplacian2d(25, 25);
  const VectorXd b = deterministicRhs(A);

  Robust s;
  s.compute(A);
  const VectorXd x = s.solve(b);

  checkTrue(s.info() == Eigen::Success, "solves");
  checkTrue(s.attempts().size() == 1, "exactly one attempt was made");
  checkTrue(s.strategy() == rlu::Strategy::Default, "and it was the default strategy");
  checkTrue(s.outcome() == rlu::Outcome::Solved, "outcome is Solved");
  check((A * x - b).norm() / b.norm() < 1e-12, "residual is small", (A * x - b).norm() / b.norm());
  check(s.backwardError() < 1e-14, "backward error is at machine precision", s.backwardError());

  // The same answer LeftRightLU alone would give: the ladder must not perturb a
  // case it does not need to touch.
  Eigen::LeftRightLU<SpMat> plain;
  plain.compute(A);
  const VectorXd xp = plain.solve(b);
  checkTrue((x - xp).cwiseAbs().maxCoeff() == 0.0,
            "and it is bit-identical to LeftRightLU on its own");
}

void testStructuralSingularityEscalatesToRankRevealing() {
  std::printf("  structural singularity escalates to the rank-revealing rung\n");
  // Before step 5 this was a dead end: no zero-free diagonal means no LU, and
  // the ladder could only report that. Now it is the one diagnosis that reaches
  // the QR rung WITHOUT an LU failure first, because there is no point trying
  // MC64 or partial pivoting on a matrix that has no LU at all.
  const SpMat A = structurallySingular(200);
  const VectorXd b = VectorXd::Ones(200);

  Robust s;
  s.compute(A);
  const VectorXd x = s.solve(b);

  checkTrue(s.attempts().size() == 2, "one LU rung, then straight to rank-revealing");
  checkTrue(s.attempts()[1].strategy == rlu::Strategy::RankRevealing,
            "no MC64 or partial-pivoting rung was wasted on it");
  checkTrue(s.outcome() == rlu::Outcome::RankDeficient, "and it is rescued, not merely diagnosed");
  checkTrue(s.rank() >= 0 && s.rank() < 200, "with a rank below n, as constructed");
  checkTrue(x.allFinite(), "the answer is finite");
  // A^T r == 0 is what makes it a least-squares answer rather than just some vector.
  check((A.transpose() * (A * x - b)).norm() < 1e-10, "and it is the least-squares minimiser",
        (A.transpose() * (A * x - b)).norm());
  std::printf("        rank=%lld of 200, %s\n", (long long)s.rank(),
              rlu::outcomeName(s.outcome()));
}

void testStopsWhenConditioningIsHopeless() {
  std::printf("  a hopeless condition number stops the ladder immediately\n");
  // kappa ~ 3e21 -- backward stable, and no digit of the answer survives.
  const SpMat A = hopelesslyIllConditioned(80);
  VectorXd xTrue(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) xTrue[i] = 1.0 + 0.1 * std::sin(3.0 * double(i));
  const VectorXd b = A * xTrue;

  Robust s;
  s.compute(A);
  const VectorXd x = s.solve(b);

  // The flag has to be earned: this answer really is wrong, and its residual
  // really does look fine. Without both halves the test would pass on a matrix
  // that was merely large.
  const double trueError = (x - xTrue).norm() / xTrue.norm();
  checkTrue(trueError > 1.0, "the computed answer is in fact completely wrong");
  checkTrue((A * x - b).norm() / b.norm() < 1e-6, "while its residual looks perfectly fine");

  checkTrue(s.attempts().size() == 1, "no escalation was attempted");
  checkTrue(s.outcome() == rlu::Outcome::IllConditioned, "diagnosed as ill-conditioned");
  checkTrue(s.info() != Eigen::Success, "and not reported as a success");
  checkTrue(s.backwardError() < 1e-13,
            "even though the factorization itself was backward stable");
  checkTrue(s.report().find("the matrix, not the solver") != std::string::npos,
            "the report says whose fault it is");
  checkTrue(s.report().find("worst-case bound") != std::string::npos,
            "and does not overclaim: a condition number bounds, it does not measure");
  checkTrue(x.allFinite(), "an answer is still returned, flagged");
  std::printf("        kappa=%.2e omega=%.1e -- %s\n", s.conditionEstimate(), s.backwardError(),
              rlu::outcomeName(s.outcome()));
}

void testEscalationCanBeCapped() {
  std::printf("  setMaxStrategy caps the climb\n");
  const SpMat A = lu_testing::laplacian2d(20, 20);
  Robust s;
  s.setMaxStrategy(rlu::Strategy::Default);
  s.compute(A);
  checkTrue(s.attempts().size() == 1, "capped at the first rung");
  checkTrue(s.info() == Eigen::Success, "and still solves an easy matrix");
}

void testReportIsConsistent() {
  std::printf("  report() matches the attempt log\n");
  const SpMat A = lu_testing::laplacian2d(15, 15);
  Robust s;
  s.compute(A);
  const std::string r = s.report();
  checkTrue(!r.empty(), "report is non-empty");
  checkTrue(r.find("solved") != std::string::npos, "it states the outcome");
  // One numbered line per attempt, no more and no fewer: a report that drifts
  // from the log is worse than no report.
  std::size_t lines = 0;
  for (std::size_t i = 0; i + 2 < r.size(); ++i)
    if (r[i] == ' ' && r[i + 1] == ' ' && r[i + 2] >= '1' && r[i + 2] <= '9') ++lines;
  checkTrue(lines == s.attempts().size(), "one line per attempt");
}

void testRankRevealingOnAConsistentSingularSystem() {
  std::printf("  rank-deficient but consistent: solved, with the rank reported\n");
  const int n = 200, k = 77;
  const SpMat A = rankDeficientDiagonal(n, k);
  VectorXd xTrue = VectorXd::Zero(n);
  for (int i = 0; i < n; ++i)
    if (i != k) xTrue[i] = 1.0 + 0.25 * double(i % 7);
  const VectorXd b = A * xTrue;  // b_k == 0, so the system IS consistent

  Robust s;
  s.compute(A);
  const VectorXd x = s.solve(b);

  checkTrue(s.info() == Eigen::Success, "a verified answer is produced");
  checkTrue(s.outcome() == rlu::Outcome::RankDeficient, "reported as rank deficient");
  checkTrue(s.isLeastSquares(), "and flagged as a least-squares answer");
  checkTrue(s.rank() == n - 1, "the rank is exactly n-1, as constructed");
  check((A * x - b).norm() / b.norm() < 1e-12, "and it solves the system",
        (A * x - b).norm() / b.norm());
  // The basic solution puts a zero in the free variable. Stating it is the point
  // -- a caller who needs the minimum-norm solution needs to know this is not it.
  checkTrue(x[k] == 0.0, "the free variable is zero: this is the BASIC solution");
  checkTrue(s.report().find("BASIC least-squares") != std::string::npos,
            "and the report says so rather than implying A x = b was solved");
}

void testRankRevealingOnAnInconsistentSystem() {
  std::printf("  rank-deficient and INCONSISTENT: least squares, verified\n");
  // The branch the SuiteSparse corpus cannot reach, because every corpus
  // right-hand side is built as b = A*x and is therefore consistent. Here b has
  // a nonzero k-th component that no x can reproduce, so the minimum achievable
  // residual is exactly |b_k| -- known, not estimated.
  const int n = 150, k = 40;
  const SpMat A = rankDeficientDiagonal(n, k);
  VectorXd b = VectorXd::Ones(n);

  Robust s;
  s.compute(A);
  const VectorXd x = s.solve(b);
  const VectorXd r = A * x - b;

  checkTrue(s.info() == Eigen::Success, "a verified least-squares answer is produced");
  checkTrue(s.outcome() == rlu::Outcome::RankDeficient, "reported as rank deficient");
  checkTrue(s.rank() == n - 1, "the rank is exactly n-1");

  // The residual must be the unavoidable one and nothing more: r == -b_k e_k.
  check(std::abs(r.norm() - 1.0) < 1e-12, "the residual is exactly the unavoidable component",
        r.norm());
  checkTrue(std::abs(r[k] + b[k]) < 1e-12, "and it lies entirely in the unreachable direction");
  // Least-squares optimality: A^T r == 0, which is what makes it the minimiser
  // rather than merely a vector with a smallish residual.
  check((A.transpose() * r).norm() < 1e-12, "A^T r == 0, so it really is the minimiser",
        (A.transpose() * r).norm());
}

void testRankRevealingFillGuard() {
  std::printf("  the fill guard declines rather than running for minutes\n");
  const SpMat A = rankDeficientDiagonal(120, 33);
  const VectorXd b = VectorXd::Ones(120);

  Robust s;
  s.setMaxRankRevealingFill(1);  // nothing can fit under this
  s.compute(A);

  checkTrue(s.outcome() == rlu::Outcome::StructurallySingular,
            "with the rung guarded off, the honest answer is structurally singular");
  checkTrue(s.info() != Eigen::Success, "and it is not claimed as a success");
  bool declined = false;
  for (const rlu::Attempt& a : s.attempts())
    if (a.strategy == rlu::Strategy::RankRevealing && !a.factored &&
        a.note.find("fill guard") != std::string::npos)
      declined = true;
  checkTrue(declined, "the declined rung is logged with its reason, not silently skipped");

  // Lift the guard and the same matrix is rescued: the guard is a budget, not a
  // capability limit.
  Robust s2;
  s2.compute(A);
  const VectorXd x = s2.solve(b);
  checkTrue(s2.outcome() == rlu::Outcome::RankDeficient, "unguarded, the same matrix is rescued");
  checkTrue(x.allFinite(), "with a finite answer");
}

void testHealthyMatrixNeverReachesTheRung() {
  std::printf("  a healthy matrix never reaches the rank-revealing rung\n");
  const SpMat A = lu_testing::laplacian2d(20, 20);
  Robust s;
  s.compute(A);
  for (const rlu::Attempt& a : s.attempts())
    checkTrue(a.strategy != rlu::Strategy::RankRevealing,
              "QR was never attempted on a matrix the first rung handles");
  checkTrue(s.rank() == -1, "and no rank is reported, since none was computed");
  checkTrue(!s.isLeastSquares(), "nor is the answer mislabelled as least squares");
}

// ---------------------------------------------------------------------------
//  The corpus: outcomes measured BEFORE the ladder was written
// ---------------------------------------------------------------------------

void testCorpus() {
  std::printf("  the SuiteSparse corpus\n");
  const std::vector<lu_testing::SuiteSparseMatrix> all = lu_testing::suitesparseMatrices();
  long long present = 0;
  for (const auto& m : all)
    if (m.available && m.tier == lu_testing::Tier::Small) ++present;
  if (present == 0) {
    note("corpus not downloaded -- skipping (run python test/matrices/fetch_suitesparse.py)");
    return;
  }

  // These matrices need MORE than the first rung. Which rung rescues them is
  // deliberately NOT pinned, and that is a finding rather than a hedge: their
  // rung-0 elimination has a growth factor of 1e+15 (fd12) to 1e+24 (lhr10c), so
  // the resulting backward error is essentially noise, and the noise moves with
  // the compiler flags. Built with -mfma this suite sees fd12 rescued by MC64;
  // built without it, fd12 squeaks past the first rung outright. Both are
  // correct behaviour -- pinning either one would pin the rounding, not the
  // ladder. What IS invariant is that the answer is trustworthy at the end.
  const char* hard[] = {"Muite/Chebyshev3", "JGD_CAG/CAG_mat1916", "Hohn/fd12",
                        "Mallya/lhr10c", "Shyy/shyy41",
                        // Rescued only by the rank-revealing rung: no LU
                        // configuration reaches a usable answer on it at all.
                        "Bai/rw5151", "Pajek/SmaGri"};

  long long firstRung = 0, escalated = 0, stopped = 0;
  for (const auto& m : all) {
    if (!m.available || m.tier != lu_testing::Tier::Small) continue;
    SpMat A;
    try {
      A = lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(m.path));
    } catch (const std::exception&) {
      continue;
    }
    const VectorXd b = deterministicRhs(A);

    Robust s;
    s.setMaxFactorNonzeros(120LL * 1000 * 1000);
    s.compute(A);
    const VectorXd x = s.solve(b);
    const double resid = (A * x - b).norm() / b.norm();

    // THE CONTRACT, and it is the same one test_suitesparse enforces: a claimed
    // success must be a real one. Everything else the ladder does is an
    // optimisation on top of this.
    if (s.info() == Eigen::Success)
      check(std::isfinite(resid) && resid < 1e-6, m.label() + ": claimed success is real", resid);
    else
      checkTrue(!s.lastErrorMessage().empty(), m.label() + ": a failure carries a diagnostic");

    bool isHard = false;
    for (const char* h : hard)
      if (m.label() == h) isHard = true;

    if (isHard) {
      const bool ok = s.outcome() == rlu::Outcome::Solved ||
                      s.outcome() == rlu::Outcome::RankDeficient;
      checkTrue(ok, m.label() + ": the ladder reaches a trustworthy answer");
      checkTrue(std::isfinite(resid) && resid < 1e-6, m.label() + ": and it really is one");
      if (s.attempts().size() > 1) ++escalated;
      std::printf("        %-24s %s (%zu attempts)\n", m.label().c_str(),
                  rlu::strategyName(s.strategy()), s.attempts().size());
    } else if (s.outcome() == rlu::Outcome::Solved &&
               s.strategy() == rlu::Strategy::Default) {
      // The cost property, per matrix: an untouched matrix pays for one
      // factorization and nothing more.
      checkTrue(s.attempts().size() == 1, m.label() + ": solved on the first rung, one attempt");
      ++firstRung;
    } else {
      ++stopped;
      std::printf("        %-24s %-22s via %s (%zu attempts)\n", m.label().c_str(),
                  rlu::outcomeName(s.outcome()), rlu::strategyName(s.strategy()),
                  s.attempts().size());
      // Even when it gives up, it must not have thrashed.
      checkTrue(s.attempts().size() <= 4, m.label() + ": gave up without thrashing");
    }
  }
  std::printf("        %lld solved on the first rung, %lld rescued by escalation, %lld stopped\n",
              firstRung, escalated, stopped);
  checkTrue(firstRung > 0, "the corpus exercised the no-escalation path");
  // The ladder has to actually ladder: if nothing ever escalates, every rung
  // below the first is dead code and this suite is not testing it.
  checkTrue(escalated > 0, "and the escalation path");
  checkTrue(stopped > 0, "and the stopping path");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("RobustLU fallback ladder\n");

  testCostsNothingWhenTheFirstRungWorks();
  testStructuralSingularityEscalatesToRankRevealing();
  testStopsWhenConditioningIsHopeless();
  testEscalationCanBeCapped();
  testReportIsConsistent();
  testRankRevealingOnAConsistentSingularSystem();
  testRankRevealingOnAnInconsistentSystem();
  testRankRevealingFillGuard();
  testHealthyMatrixNeverReachesTheRung();
  testCorpus();

  return lu_testing::summarize("RobustLU");
}
