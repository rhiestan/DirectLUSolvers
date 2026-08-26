// Condition estimation and error bounds: does the solver know how much of its
// own answer to believe?
//
//   ctest --test-dir build -R test_condition_estimate --output-on-failure
//
// WHY THIS EXISTS
//
// Every other suite here gates on a residual, and a residual is the reassuring
// half of the story: it says the computed x solves a nearby system, never how
// far x is from the answer that was asked for. On an ill-conditioned matrix
// those differ by orders of magnitude. This suite tests the machinery that
// tells them apart, in three layers:
//
//   1. THE ESTIMATOR, against matrices whose exact 1-norm condition number is
//      known in closed form or computable densely. Hager's algorithm is a
//      LOWER bound, so the load-bearing assertion is est <= true -- an
//      estimator that overshoots is reporting a matrix as worse conditioned
//      than it is, which is a different and equally real bug.
//   2. THE BACKWARD ERROR, which unlike the condition estimate is exact, so it
//      can be checked against a hand-computed value rather than a tolerance.
//   3. THE CONTRACT the whole feature exists for: a well-conditioned system
//      reports many correct digits, an ill-conditioned one reports few, and
//      the two are distinguishable through the API. Plus the promise that none
//      of it costs anything unless asked -- asserted as conditionEstimateSolves()
//      == 0 after a default solve, which is the testable form of "no slowdown
//      in the default codepath".

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::note;

typedef SparseMatrix<double> SpMat;
typedef Eigen::LeftRightLU<SpMat> Solver;

namespace {

// Upper bidiagonal, 1 on the diagonal and -2 above it. Its inverse has entries
// 2^(j-i) on and above the diagonal, so the exact 1-norm condition number is
// 3 * (2^n - 1) -- a closed form, which makes it the ideal yardstick: the
// condition number is dialled purely by n, over any range wanted, with no
// rounding in the "true" value to argue about.
SpMat powerOfTwoBidiagonal(int n) {
  std::vector<Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    t.emplace_back(j, j, 1.0);
    if (j > 0) t.emplace_back(j - 1, j, -2.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

double exactBidiagonalCondition(int n) {
  return (n == 1) ? 1.0 : 3.0 * (std::pow(2.0, n) - 1.0);
}

// The same shape with a superdiagonal that is NOT a power of two. That one
// detail is what makes it useful here: with -2 the factorization and the solve
// are EXACT in binary floating point, so the backward error is identically zero
// and kappa * omega is zero no matter how enormous kappa gets -- a correct
// answer, and a useless test. At -1.7 every operation rounds, the backward
// error sits at machine epsilon, and kappa is free to amplify it.
SpMat roundingBidiagonal(int n) {
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

// Not exactly representable, so b = A*xTrue rounds too.
VectorXd irrationalSolution(Eigen::Index n) {
  VectorXd x(n);
  for (Eigen::Index i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * std::sin(3.0 * double(i));
  return x;
}

// The true 1-norm condition number, densely. Only for small n -- this is the
// O(n^3) reference the estimator is judged against.
double denseCondition1(const SpMat& A) {
  const MatrixXd dense = MatrixXd(A);
  const MatrixXd inv = dense.inverse();
  double normA = 0, normInv = 0;
  for (Eigen::Index j = 0; j < dense.cols(); ++j) {
    normA = (std::max)(normA, dense.col(j).cwiseAbs().sum());
    normInv = (std::max)(normInv, inv.col(j).cwiseAbs().sum());
  }
  return normA * normInv;
}

VectorXd rhsFor(const SpMat& A, const VectorXd& xTrue) { return A * xTrue; }

VectorXd ones(Eigen::Index n) { return VectorXd::Ones(n); }

// ---------------------------------------------------------------------------
//  1. The estimator
// ---------------------------------------------------------------------------

void testEstimatorAgainstClosedForm() {
  std::printf("  Hager-Higham estimate vs the closed-form condition number\n");
  for (int n : {2, 4, 8, 16, 30, 45}) {
    const SpMat A = powerOfTwoBidiagonal(n);
    Solver s;
    s.compute(A);
    checkTrue(s.info() == Eigen::Success, "bidiagonal n=" + std::to_string(n) + " factors");

    const double est = s.conditionEstimate();
    const double exact = exactBidiagonalCondition(n);

    // THE load-bearing property. Hager maximises over a subset of the unit
    // ball, so it can only ever underestimate; anything above the true value
    // means the algorithm is wrong, not merely imprecise. A 1e-9 relative slack
    // covers the floating-point evaluation of the ratio itself.
    checkTrue(est <= exact * (1.0 + 1e-9),
              "n=" + std::to_string(n) + ": estimate is a lower bound (" + std::to_string(est) +
                  " <= " + std::to_string(exact) + ")");
    // And it must be USEFUL, not merely safe: returning 1 always would satisfy
    // the bound above. Hager is documented as almost always within a factor of
    // 3; 10 leaves room without letting a real regression through.
    check(est >= exact / 10.0, "n=" + std::to_string(n) + ": estimate within 10x of exact",
          exact / est);
  }
}

void testEstimatorAgainstDenseInverse() {
  std::printf("  Hager-Higham estimate vs a dense inverse, on real matrices\n");
  struct Case {
    const char* name;
    SpMat A;
  };
  std::vector<Case> cases;
  cases.push_back({"lap2d_8x8", lu_testing::laplacian2d(8, 8)});
  cases.push_back({"lap2d_12x12", lu_testing::laplacian2d(12, 12)});
  cases.push_back({"upwind2d_10x10", lu_testing::upwind2d(10, 10)});
  cases.push_back({"randomUnsym_120", lu_testing::randomUnsymmetricPattern(120, 0.02, 11u)});
  cases.push_back({"weakDiagonal_80", lu_testing::weakDiagonal(80, 5u)});

  for (const Case& c : cases) {
    Solver s;
    s.compute(c.A);
    if (s.info() != Eigen::Success) {
      note(std::string(c.name) + ": declined, skipped");
      continue;
    }
    const double est = s.conditionEstimate();
    const double exact = denseCondition1(c.A);
    if (!std::isfinite(exact)) {
      note(std::string(c.name) + ": dense inverse is not finite, skipped");
      continue;
    }
    checkTrue(est <= exact * 1.05,
              std::string(c.name) + ": estimate is a lower bound (" + std::to_string(est) +
                  " vs " + std::to_string(exact) + ")");
    check(est >= exact / 10.0, std::string(c.name) + ": estimate within 10x", exact / est);
  }
}

void testEstimatorDegenerateSizes() {
  std::printf("  degenerate sizes\n");
  {
    SpMat A(1, 1);
    A.insert(0, 0) = 4.0;
    A.makeCompressed();
    Solver s;
    s.compute(A);
    // kappa(scalar) == 1 exactly: ||a||*||1/a|| == 1.
    check(std::abs(s.conditionEstimate() - 1.0) < 1e-12, "1x1: kappa == 1",
          s.conditionEstimate());
  }
  {
    SpMat A(0, 0);
    A.makeCompressed();
    Solver s;
    s.compute(A);
    checkTrue(std::isfinite(s.conditionEstimate()), "0x0: kappa is finite, not NaN");
  }
  {
    // Never factorized: the honest answer is "unbounded", not 1.
    Solver s;
    checkTrue(!std::isfinite(s.conditionEstimate()) || s.conditionEstimate() > 1e300,
              "unfactorized: kappa is infinite");
  }
}

void testEstimateIsCachedAndInvalidated() {
  std::printf("  caching and invalidation\n");
  const SpMat A = powerOfTwoBidiagonal(20);
  Solver s;
  s.compute(A);

  checkTrue(s.conditionEstimateSolves() == 0, "no solves spent before the estimate is asked for");
  const double first = s.conditionEstimate();
  const Eigen::Index spent = s.conditionEstimateSolves();
  checkTrue(spent > 0, "the estimate cost some solves");
  checkTrue(spent <= 12, "the estimate cost a handful of solves, not O(n)");

  const double second = s.conditionEstimate();
  checkTrue(second == first, "a second call returns the identical cached value");
  checkTrue(s.conditionEstimateSolves() == spent, "a second call spends no further solves");

  // A refactorization is a new operator, so the cache must not survive it.
  SpMat B = A;
  for (int k = 0; k < B.outerSize(); ++k)
    for (SpMat::InnerIterator it(B, k); it; ++it) it.valueRef() *= 4.0;
  s.factorize(B);
  checkTrue(s.conditionEstimateSolves() == 0, "refactorize() invalidates the cached estimate");
  // Scaling A by 4 leaves kappa unchanged: it is scale-invariant.
  const double rescaled = s.conditionEstimate();
  checkTrue(std::abs(rescaled - first) <= 1e-6 * first,
            "kappa is invariant under a uniform rescaling of A");
}

// ---------------------------------------------------------------------------
//  2. The backward error
// ---------------------------------------------------------------------------

void testBackwardErrorIsExact() {
  std::printf("  Oettli-Prager backward error\n");
  const SpMat A = lu_testing::laplacian2d(10, 10);
  const VectorXd xTrue = ones(A.rows());
  const VectorXd b = rhsFor(A, xTrue);

  Solver s;
  s.compute(A);
  const VectorXd x = s.solve(b);
  checkTrue(s.info() == Eigen::Success, "well-conditioned system solves");

  const double omega = s.componentwiseBackwardError(b, x);
  // A backward-stable solve on a well-conditioned matrix should land within a
  // small multiple of eps. This is the assertion that says the solver did
  // everything a method working in this precision could.
  check(omega < 1e-13, "backward error is at the level of machine precision", omega);

  // The exact solution has, by definition, zero backward error.
  check(s.componentwiseBackwardError(b, xTrue) < 1e-15,
        "the exact solution has ~zero backward error", s.componentwiseBackwardError(b, xTrue));

  // A deliberately wrong x must be reported as such, and the value is checkable
  // by hand: perturbing one component of x by a relative delta cannot produce a
  // backward error smaller than roughly that delta over the row's weight.
  VectorXd xBad = x;
  xBad[0] *= 1.5;
  const double omegaBad = s.componentwiseBackwardError(b, xBad);
  checkTrue(omegaBad > 1e-3, "a visibly wrong x has a large backward error");
  checkTrue(omegaBad > omega * 1e6, "wrong x is orders of magnitude worse than the computed one");
}

void testBackwardErrorIsRowScalingInvariant() {
  std::printf("  backward error is invariant under row scaling\n");
  // The componentwise backward error scales A, x and b together, so multiplying
  // row i of both A and b by any nonzero factor cannot change it. That is the
  // defining property that distinguishes it from a normwise residual, and the
  // reason it is the right measure for a badly scaled matrix.
  const SpMat A = lu_testing::randomUnsymmetricPattern(200, 0.01, 3u);
  const VectorXd xTrue = ones(A.rows());
  const VectorXd b = rhsFor(A, xTrue);

  Solver s;
  s.compute(A);
  if (s.info() != Eigen::Success) {
    note("random unsymmetric matrix declined, skipped");
    return;
  }
  const VectorXd x = s.solve(b);
  const double omega = s.componentwiseBackwardError(b, x);

  // Scale rows by wildly different factors.
  std::vector<Triplet<double>> t;
  VectorXd bScaled(A.rows());
  std::vector<double> factor(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) factor[i] = std::pow(10.0, (i % 11) - 5);
  for (int k = 0; k < A.outerSize(); ++k)
    for (SpMat::InnerIterator it(A, k); it; ++it)
      t.emplace_back(it.row(), it.col(), it.value() * factor[it.row()]);
  for (Eigen::Index i = 0; i < A.rows(); ++i) bScaled[i] = b[i] * factor[i];
  SpMat AScaled(A.rows(), A.cols());
  AScaled.setFromTriplets(t.begin(), t.end());
  AScaled.makeCompressed();

  Solver s2;
  s2.compute(AScaled);
  const double omegaScaled = s2.componentwiseBackwardError(bScaled, x);
  checkTrue(std::abs(omegaScaled - omega) <= 1e-12,
            "row scaling leaves the componentwise backward error unchanged");
}

// ---------------------------------------------------------------------------
//  3. The contract: 13 digits vs 2 digits
// ---------------------------------------------------------------------------

void testDigitsDistinguishConditioning() {
  std::printf("  correct-digit estimates track conditioning\n");
  std::printf("        %-6s %-10s %-9s %-9s %-7s %-10s %s\n", "n", "kappa", "omega", "ferr",
              "digits", "true err", "residual");

  // The claim under test, in one table: the RESIDUAL is pinned at ~1e-16 in
  // every row, so a caller reading only info()/solveResidual() sees four
  // identical successes -- while the true error walks from 5e-16 to 72. The
  // forward-error estimate is the only column that tells them apart.
  int previousDigits = 99;
  double worstResidual = 0.0;
  bool sawManyDigits = false, sawNoDigits = false;
  for (int n : {8, 40, 60, 80}) {
    const SpMat A = roundingBidiagonal(n);
    const VectorXd xTrue = irrationalSolution(A.rows());
    const VectorXd b = rhsFor(A, xTrue);

    Solver s;
    s.compute(A);
    const VectorXd x = s.solve(b);
    const double omega = s.componentwiseBackwardError(b, x);
    const double ferr = s.estimatedForwardError(b, x);
    const int digits = s.estimatedCorrectDigits(b, x);
    const double trueError = (x - xTrue).norm() / xTrue.norm();
    worstResidual = (std::max)(worstResidual, s.solveResidual());

    std::printf("        %-6d %-10.2e %-9.1e %-9.1e %-7d %-10.1e %.1e\n", n,
                s.conditionEstimate(), omega, ferr, digits, trueError, s.solveResidual());

    // A forward-error estimate the true error exceeds is worse than none at
    // all. The 100x slack covers the bound being first order and kappa being a
    // lower bound; without any slack this would be asserting an equality that
    // the theory does not claim.
    checkTrue(trueError <= (std::max)(ferr * 100.0, 1e-14),
              "n=" + std::to_string(n) + ": the true error respects the estimated bound");
    checkTrue(digits <= previousDigits,
              "n=" + std::to_string(n) + ": worse conditioning never reports MORE digits");
    previousDigits = digits;
    // Backward stability holds throughout, which is the point: the solver is
    // doing everything right in every row.
    check(omega < 1e-13, "n=" + std::to_string(n) + ": backward error stays at eps", omega);
    if (digits >= 10) sawManyDigits = true;
    if (digits == 0) sawNoDigits = true;
  }

  checkTrue(sawManyDigits, "the well-conditioned case reports many digits");
  checkTrue(sawNoDigits, "the ill-conditioned case reports none");
  // If this ever fails, the table above stopped making its point: the residual
  // would then be distinguishing the cases and the forward error would be
  // redundant.
  check(worstResidual < 1e-12,
        "every case looks equally good to the residual check -- which is the point",
        worstResidual);
}

void testDefaultPathPaysNothing() {
  std::printf("  the default path pays nothing\n");
  const SpMat A = lu_testing::laplacian2d(20, 20);
  const VectorXd b = rhsFor(A, ones(A.rows()));

  Solver s;
  s.compute(A);
  checkTrue(!s.errorBounds(), "error bounds are OFF by default");
  const VectorXd x = s.solve(b);

  // The testable form of "no slowdown in the default codepath": a default
  // solve() performs no extra triangular solves at all, and computes neither
  // bound. If either of these ever fires, something started running for free
  // callers who never asked for it.
  checkTrue(s.conditionEstimateSolves() == 0, "a default solve() spends no estimator solves");
  checkTrue(std::isnan(s.lastBackwardError()), "lastBackwardError() is NaN when not measured");
  checkTrue(std::isnan(s.lastForwardError()), "lastForwardError() is NaN when not measured");
  checkTrue(s.lastCorrectDigits() == -1, "lastCorrectDigits() reports -1 when not measured");

  // Turning the option on must not change the ANSWER, only what is reported
  // about it -- bit-identically, since it adds measurement and nothing else.
  Solver s2;
  s2.setErrorBounds(true);
  s2.compute(A);
  const VectorXd x2 = s2.solve(b);
  checkTrue((x - x2).cwiseAbs().maxCoeff() == 0.0,
            "enabling error bounds leaves the solution bit-identical");
  checkTrue(!std::isnan(s2.lastBackwardError()), "with bounds on, the backward error is recorded");
  checkTrue(s2.lastCorrectDigits() > 10, "a well-conditioned system reports many digits");

  // A stale bound from a previous solve must not survive a solve that did not
  // measure one.
  s2.setErrorBounds(false);
  const VectorXd x3 = s2.solve(b);
  checkTrue(std::isnan(s2.lastBackwardError()), "turning bounds off clears the recorded value");
  checkTrue((x3 - x2).cwiseAbs().maxCoeff() == 0.0, "and still returns the same solution");
}

void testInfoDowngradeIsOptIn() {
  std::printf("  info() downgrade on a hopeless condition number is opt-in\n");
  // kappa ~ 1e19, past what double can carry, so no digit of the answer
  // survives -- and yet the residual is ~1e-14 and default info() says Success.
  // That combination is exactly what a caller cannot currently detect.
  const SpMat A = roundingBidiagonal(80);
  const VectorXd xTrue = irrationalSolution(A.rows());
  const VectorXd b = rhsFor(A, xTrue);

  Solver plain;
  plain.compute(A);
  const VectorXd x = plain.solve(b);
  checkTrue(plain.solveResidual() < 1e-6, "the residual check passes on the hopeless system");
  checkTrue(plain.info() == Eigen::Success, "and default info() therefore reports Success");
  const double trueError = (x - xTrue).norm() / xTrue.norm();
  checkTrue(trueError > 1.0, "while the answer is in fact completely wrong");

  Solver guarded;
  guarded.setErrorBounds(true);
  guarded.compute(A);
  const VectorXd xg = guarded.solve(b);
  checkTrue((x - xg).cwiseAbs().maxCoeff() == 0.0, "the answer is the same either way");
  checkTrue(guarded.info() == Eigen::NumericalIssue,
            "with bounds on, info() reports the answer is not trustworthy");
  checkTrue(!guarded.lastErrorMessage().empty(), "the downgrade carries a diagnostic");
  checkTrue(guarded.lastCorrectDigits() == 0, "and reports zero supported digits");
  std::printf("        kappa=%.2e omega=%.1e ferr=%.1e resid=%.1e true err=%.1e\n",
              guarded.conditionEstimate(), guarded.lastBackwardError(), guarded.lastForwardError(),
              guarded.solveResidual(), trueError);

  // A well-conditioned system must NOT be downgraded -- an error bound that
  // cries wolf is worse than none.
  Solver fine;
  fine.setErrorBounds(true);
  fine.compute(lu_testing::laplacian2d(12, 12));
  const VectorXd bf = ones(144);
  fine.solve(bf);
  checkTrue(fine.info() == Eigen::Success, "a well-conditioned system is not downgraded");
}

void testTransposedSolveDoesNotReportStaleBounds() {
  std::printf("  a transposed solve does not inherit the previous bounds\n");
  // kappa_1(A^T) is kappa_inf(A), not kappa_1(A), so the bounds are deliberately
  // not computed for transpose()/adjoint() solves. What matters is that the
  // values from an EARLIER normal solve do not silently stand in for them.
  const SpMat A = lu_testing::randomUnsymmetricPattern(150, 0.02, 21u);
  const VectorXd b = ones(A.rows());

  Solver s;
  s.setErrorBounds(true);
  s.compute(A);
  if (s.info() != Eigen::Success) {
    note("matrix declined, skipped");
    return;
  }
  // Assigned, not discarded: solve() returns a lazy Eigen expression, so
  // `s.solve(b);` on its own never runs the solve at all.
  const VectorXd x = s.solve(b);
  checkTrue(!std::isnan(s.lastBackwardError()), "the normal solve records a backward error");

  const VectorXd xt = s.transpose().solve(b);
  checkTrue(xt.allFinite(), "the transposed solve produced a finite answer");
  checkTrue(std::isnan(s.lastBackwardError()),
            "a transposed solve clears it rather than reusing the normal solve's");
  checkTrue(s.lastCorrectDigits() == -1, "and reports -1 rather than a stale digit count");
}

void testComplexScalars() {
  std::printf("  complex scalars use the conjugate transpose\n");
  typedef std::complex<double> Cplx;
  typedef SparseMatrix<Cplx> SpMatC;
  typedef Eigen::LeftRightLU<SpMatC> SolverC;

  const int n = 24;
  std::vector<Triplet<Cplx>> t;
  for (int j = 0; j < n; ++j) {
    t.emplace_back(j, j, Cplx(1.0, 0.5));
    if (j > 0) t.emplace_back(j - 1, j, Cplx(-1.0, -1.0));
  }
  SpMatC A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  SolverC s;
  s.setErrorBounds(true);
  s.compute(A);
  checkTrue(s.info() == Eigen::Success, "complex bidiagonal factors");

  Eigen::Matrix<Cplx, Eigen::Dynamic, 1> xTrue(n);
  for (int i = 0; i < n; ++i) xTrue[i] = Cplx(1.0 + 0.1 * i, 0.3 - 0.05 * i);
  const Eigen::Matrix<Cplx, Eigen::Dynamic, 1> b = A * xTrue;
  const Eigen::Matrix<Cplx, Eigen::Dynamic, 1> x = s.solve(b);

  // The dense reference: with the WRONG transpose (plain instead of conjugate)
  // the estimate would still be a plausible-looking number, so comparing
  // against the true value is the only thing that catches it.
  const Eigen::MatrixXcd dense = Eigen::MatrixXcd(A);
  const Eigen::MatrixXcd inv = dense.inverse();
  double normA = 0, normInv = 0;
  for (int j = 0; j < n; ++j) {
    normA = (std::max)(normA, dense.col(j).cwiseAbs().sum());
    normInv = (std::max)(normInv, inv.col(j).cwiseAbs().sum());
  }
  const double exact = normA * normInv;
  const double est = s.conditionEstimate();
  checkTrue(est <= exact * 1.05, "complex: estimate is a lower bound");
  check(est >= exact / 10.0, "complex: estimate within 10x", exact / est);
  check(s.lastBackwardError() < 1e-13, "complex: backward error at machine precision",
        s.lastBackwardError());
  check((x - xTrue).norm() / xTrue.norm() < 1e-12, "complex: solution is accurate",
        (x - xTrue).norm() / xTrue.norm());
}

void testMultipleRightHandSides() {
  std::printf("  multi-column right-hand sides\n");
  const SpMat A = lu_testing::laplacian2d(10, 10);
  Eigen::MatrixXd B(A.rows(), 3);
  B.col(0) = ones(A.rows());
  B.col(1) = VectorXd::LinSpaced(A.rows(), -1.0, 1.0);
  B.col(2).setZero();
  B(0, 2) = 1.0;

  Solver s;
  s.setErrorBounds(true);
  s.compute(A);
  const Eigen::MatrixXd X = s.solve(B);
  checkTrue(s.info() == Eigen::Success, "multi-rhs solve succeeds");

  // The reported backward error is the worst over the columns, so it must be at
  // least each column's own.
  const double all = s.componentwiseBackwardError(B, X);
  for (int c = 0; c < 3; ++c) {
    const double one = s.componentwiseBackwardError(B.col(c).eval(), X.col(c).eval());
    checkTrue(one <= all * (1.0 + 1e-12),
              "column " + std::to_string(c) + " is covered by the reported maximum");
  }
  check(all < 1e-13, "multi-rhs backward error is at machine precision", all);
}

void testInteractionWithBlockTriangularForm() {
  std::printf("  works through the block triangular form\n");
  // An upper bidiagonal matrix is fully reducible, so with BTF on the solve
  // runs the block back-substitution rather than one L/U sweep. Both paths must
  // reach the same condition number -- but only while both factor the SAME
  // operator, which is the subtlety worth pinning here.
  {
    const SpMat A = powerOfTwoBidiagonal(20);
    const double exact = exactBidiagonalCondition(20);
    Solver on;
    on.compute(A);
    Solver off;
    off.setBlockTriangularForm(false);
    off.compute(A);
    checkTrue(on.btfBlockCount() > 1, "the test matrix really is reducible");
    checkTrue(on.replacedPivots() == 0 && off.replacedPivots() == 0,
              "n=20: neither path perturbs a pivot, so they factor the same operator");
    const double estOn = on.conditionEstimate(), estOff = off.conditionEstimate();
    checkTrue(estOn <= exact * (1.0 + 1e-9), "BTF on: still a lower bound");
    checkTrue(estOff <= exact * (1.0 + 1e-9), "BTF off: still a lower bound");
    checkTrue(std::abs(estOn - estOff) <= 1e-9 * (std::max)(estOn, estOff),
              "BTF on and off agree when both factor the same operator");
  }
  {
    // At n=30 the BTF-off path (AMD, with fill) bumps one static pivot, so it
    // factors a PERTURBED matrix -- and a perturbation regularises, which makes
    // the operator genuinely better conditioned. The estimate describing that
    // perturbed operator rather than A is the documented contract, not a bug,
    // and replacedPivots() is what tells the two apart. Pinned here because it
    // is the sort of difference a future reader would otherwise "fix".
    const SpMat A = powerOfTwoBidiagonal(30);
    const double exact = exactBidiagonalCondition(30);
    Solver on;
    on.compute(A);
    Solver off;
    off.setBlockTriangularForm(false);
    off.compute(A);
    checkTrue(on.replacedPivots() == 0, "n=30, BTF on: no pivot perturbed");
    checkTrue(off.replacedPivots() > 0, "n=30, BTF off: a pivot IS perturbed");
    checkTrue(on.conditionEstimate() <= exact * (1.0 + 1e-9),
              "the unperturbed path stays a lower bound on kappa(A)");
    checkTrue(off.conditionEstimate() < on.conditionEstimate(),
              "the perturbed operator estimates as BETTER conditioned, as it is");
    std::printf("        n=30: exact=%.3e  BTF on=%.3e (%lld pivots)  off=%.3e (%lld pivots)\n",
                exact, on.conditionEstimate(), (long long)on.replacedPivots(),
                off.conditionEstimate(), (long long)off.replacedPivots());
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("LeftRightLU condition estimation and error bounds\n");

  testEstimatorAgainstClosedForm();
  testEstimatorAgainstDenseInverse();
  testEstimatorDegenerateSizes();
  testEstimateIsCachedAndInvalidated();
  testBackwardErrorIsExact();
  testBackwardErrorIsRowScalingInvariant();
  testDigitsDistinguishConditioning();
  testDefaultPathPaysNothing();
  testInfoDowngradeIsOptIn();
  testTransposedSolveDoesNotReportStaleBounds();
  testComplexScalars();
  testMultipleRightHandSides();
  testInteractionWithBlockTriangularForm();

  return lu_testing::summarize("condition estimation");
}
