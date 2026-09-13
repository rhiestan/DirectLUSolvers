// PointBlockLU correctness: the unsymmetric-pattern core, the replay path, and
// the places where an unsymmetric LU legitimately behaves differently from its
// symmetric-pattern siblings.
//
// The replay path is the part worth testing hardest. A refactorization reuses a
// pattern and a pivot sequence chosen from DIFFERENT values, so the failure mode
// is not a crash: it is a quietly wrong answer on the second and later solves,
// which a suite that only ever factors once cannot see. Every replay assertion
// here therefore compares against a solver that factored the same values from
// scratch, not against a tolerance.
//
// Build + run (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/test_pointblock_lu

#include <Eigen/Dense>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "PointBlockLU.h"
#include "PointBlockOrdering.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
typedef SparseMatrix<double> SpMat;

namespace {

// Deterministic right-hand side, so a failure reproduces.
VectorXd rhsFor(const SpMat& A) {
  VectorXd x(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) x(i) = 1.0 + 0.5 * std::sin(static_cast<double>(i));
  return A * x;
}

double relativeResidual(const SpMat& A, const VectorXd& x, const VectorXd& b) {
  return (A * x - b).norm() / b.norm();
}

// Scale every stored value by a smooth, index-dependent factor: same pattern,
// genuinely different numbers, which is what a Newton step looks like.
SpMat perturbed(const SpMat& A, double amount) {
  SpMat B = A;
  int k = 0;
  for (int j = 0; j < B.outerSize(); ++j)
    for (SpMat::InnerIterator it(B, j); it; ++it, ++k)
      it.valueRef() *= 1.0 + amount * std::sin(static_cast<double>(k));
  return B;
}

template <typename Ordering>
void solvesCorrectly(const std::string& label, const SpMat& A, double tolerance = 1e-9) {
  const VectorXd b = rhsFor(A);
  Eigen::PointBlockLU<SpMat, Ordering> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    lu_testing::check(false, label + ": factorize -- " + solver.lastErrorMessage(), 0.0);
    return;
  }
  const VectorXd x = solver.solve(b);
  lu_testing::check(relativeResidual(A, x, b) < tolerance, label + ": residual",
                    relativeResidual(A, x, b));
}

// ---------------------------------------------------------------------------

void testOrderings() {
  std::printf("\n-- every ordering functor, and the permutation convention each uses --\n");
  const SpMat lap2d = lu_testing::laplacian2d(30, 30);
  const SpMat lap3d = lu_testing::laplacian3d(10, 10, 10);
  solvesCorrectly<Eigen::PointBlockOrdering<int>>("lap2d_30^2, PointBlock", lap2d);
  solvesCorrectly<Eigen::COLAMDOrdering<int>>("lap2d_30^2, COLAMD", lap2d);
  solvesCorrectly<Eigen::AMDOrdering<int>>("lap2d_30^2, AMD", lap2d);
  solvesCorrectly<Eigen::NaturalOrdering<int>>("lap2d_30^2, Natural", lap2d);
  solvesCorrectly<Eigen::PointBlockOrdering<int>>("lap3d_10^3, PointBlock", lap3d);

  // A wrong permutation direction leaves the residual at machine precision and
  // shows up only as fill, so assert on fill: COLAMD (which returns the direct
  // map, unlike AMD) must not be dramatically worse than AMD on a mesh.
  Eigen::PointBlockLU<SpMat, Eigen::COLAMDOrdering<int>> colamd;
  Eigen::PointBlockLU<SpMat, Eigen::AMDOrdering<int>> amd;
  colamd.compute(lap2d);
  amd.compute(lap2d);
  const double ratio = double(colamd.nnzL() + colamd.nnzU()) / double(amd.nnzL() + amd.nnzU());
  lu_testing::check(ratio < 3.0, "COLAMD fill within 3x of AMD (guards the ordering convention)", ratio);
}

void testUnsymmetricPattern() {
  std::printf("\n-- unsymmetric patterns: where keeping A unsymmetrized pays --\n");
  // An upwind grid: each interior node points only west and south, so the
  // pattern is maximally unsymmetric and symmetrizing it doubles the graph.
  const SpMat upwind = lu_testing::upwind2d(40, 40);
  solvesCorrectly<Eigen::COLAMDOrdering<int>>("upwind 40x40", upwind);

  Eigen::PointBlockLU<SpMat, Eigen::COLAMDOrdering<int>> pb;
  Eigen::LeftRightLU<SpMat, Eigen::COLAMDOrdering<int>> lr;
  // LeftRightLU's BTF is disabled here to keep the comparison about the one
  // thing this test is measuring: the cost of symmetrizing the pattern. An
  // upwind operator is fully triangular after matching, so with BTF on
  // LeftRightLU reduces this matrix to singleton blocks and wins by ~4x --
  // a real result, but one that says nothing about symmetrization, and it
  // would go away the moment PointBlockLU grew a BTF of its own.
  lr.setBlockTriangularForm(false);
  pb.compute(upwind);
  lr.compute(upwind);
  const double pbFill = double(pb.nnzL() + pb.nnzU());
  const double lrFill = double(lr.nnzL() + lr.nnzU());
  lu_testing::check(pbFill <= lrFill, "unsymmetric-pattern fill <= symmetric-pattern fill",
                    pbFill / lrFill);
}

void testReplay() {
  std::printf("\n-- refactorization: the replay must match a fresh factorization --\n");
  const SpMat A = lu_testing::laplacian2d(25, 25);

  Eigen::PointBlockLU<SpMat> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  lu_testing::check(solver.info() == Eigen::Success, "first factorize", 0.0);
  lu_testing::check(solver.refactorizations() == 0, "first factorize is not a replay",
                    double(solver.refactorizations()));

  const Eigen::Index fillFirst = solver.nnzL() + solver.nnzU();
  for (int step = 1; step <= 4; ++step) {
    const SpMat B = perturbed(A, 0.2 * step);
    const VectorXd b = rhsFor(B);
    solver.factorize(B);
    lu_testing::check(solver.info() == Eigen::Success, "replay " + std::to_string(step), 0.0);
    lu_testing::check(solver.refactorizations() == step,
                      "replay " + std::to_string(step) + " really replayed (no re-analysis)",
                      double(solver.refactorizations()));
    lu_testing::check(solver.nnzL() + solver.nnzU() == fillFirst,
                      "replay " + std::to_string(step) + ": pattern unchanged",
                      double(solver.nnzL() + solver.nnzU() - fillFirst));

    const VectorXd x = solver.solve(b);
    lu_testing::check(relativeResidual(B, x, b) < 1e-9,
                      "replay " + std::to_string(step) + ": residual", relativeResidual(B, x, b));

    // The assertion that matters: agreement with a solver that never saw A.
    Eigen::PointBlockLU<SpMat> fresh;
    fresh.compute(B);
    const VectorXd xf = fresh.solve(b);
    lu_testing::check((x - xf).norm() / xf.norm() < 1e-8,
                      "replay " + std::to_string(step) + ": agrees with a fresh factorization",
                      (x - xf).norm() / xf.norm());
  }

  // Forcing full factorizations must give the same answers, and must reset the
  // replay counter -- otherwise the option silently does nothing.
  Eigen::PointBlockLU<SpMat> forced;
  forced.setForceFullFactorization(true);
  forced.analyzePattern(A);
  forced.factorize(A);
  forced.factorize(A);
  lu_testing::check(forced.refactorizations() == 0, "setForceFullFactorization suppresses replay",
                    double(forced.refactorizations()));
}

void testReplayRejection() {
  std::printf("\n-- a replay whose recorded pivots no longer fit must be rejected --\n");
  // Build a matrix whose pivot ordering genuinely depends on the values: two
  // rows that swap dominance when a single entry changes sign and magnitude.
  SpMat A(4, 4);
  std::vector<Eigen::Triplet<double>> t = {{0, 0, 1.0},  {0, 1, 2.0}, {1, 0, 3.0},  {1, 1, 1.0},
                                           {1, 2, 1.0},  {2, 1, 1.0}, {2, 2, 4.0},  {2, 3, 1.0},
                                           {3, 2, 1.0},  {3, 3, 5.0}, {0, 3, 0.5}};
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  Eigen::PointBlockLU<SpMat> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  lu_testing::check(solver.info() == Eigen::Success, "rejection case: first factorize", 0.0);

  // Collapse the recorded pivots by many orders of magnitude. Whatever the
  // solver does about it, the answer must stay right.
  SpMat B = A;
  for (int j = 0; j < B.outerSize(); ++j)
    for (SpMat::InnerIterator it(B, j); it; ++it)
      if (it.row() == it.col()) it.valueRef() *= 1e-14;
  const VectorXd b = rhsFor(B);
  solver.factorize(B);
  if (solver.info() == Eigen::Success) {
    const VectorXd x = solver.solve(b);
    lu_testing::check(relativeResidual(B, x, b) < 1e-6,
                      "collapsed pivots: answer still correct", relativeResidual(B, x, b));
  } else {
    lu_testing::check(true, "collapsed pivots: declined rather than answering wrongly", 0.0);
  }
}

void testDegenerate() {
  std::printf("\n-- degenerate sizes and structurally singular input --\n");
  {
    SpMat A(0, 0);
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    lu_testing::check(s.info() == Eigen::Success, "n=0 factorizes", 0.0);
  }
  for (int n : {1, 2, 3}) {
    SpMat A(n, n);
    for (int i = 0; i < n; ++i) A.insert(i, i) = double(i + 2);
    A.makeCompressed();
    const VectorXd b = rhsFor(A);
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    const VectorXd x = s.solve(b);
    lu_testing::check(relativeResidual(A, x, b) < 1e-12, "n=" + std::to_string(n) + " diagonal",
                      relativeResidual(A, x, b));
  }
  {
    // A structurally singular matrix must be DECLINED, not silently perturbed:
    // an unsymmetric LU has no pivot for an empty column, and saying so is the
    // honest answer (Eigen::SparseLU declines the same input).
    SpMat A(3, 3);
    A.insert(0, 0) = 1.0;
    A.insert(1, 0) = 2.0;  // column 1 is empty
    A.insert(2, 2) = 3.0;
    A.makeCompressed();
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    lu_testing::check(s.info() != Eigen::Success, "empty column is declined", 0.0);
    lu_testing::check(!s.lastErrorMessage().empty(), "declining says why", 0.0);
  }
}

void testDeterminant() {
  std::printf("\n-- determinant against a dense reference --\n");
  for (int n : {3, 6, 9}) {
    SpMat A = lu_testing::randomSymmetricPattern(n, 0.4, 42u + unsigned(n));
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    if (s.info() != Eigen::Success) continue;
    const double ref = Eigen::MatrixXd(A).determinant();
    const double got = s.determinant();
    lu_testing::check(std::abs(got - ref) <= 1e-8 * std::abs(ref) + 1e-12,
                      "determinant n=" + std::to_string(n), std::abs(got - ref));
  }
}

// ---------------------------------------------------------------------------
//  Reliability: does the solver know how much of its own answer to believe?
// ---------------------------------------------------------------------------
//
// Finding a pivot in every column is the coarse half of that question, and the
// tests above cover it. This section covers the fine half: partial pivoting can
// succeed in every column and still return an answer with no correct digit in
// it, because the matrix was ill-conditioned rather than singular. The residual
// cannot see that case -- it is small either way -- which is exactly why the
// condition estimate and the forward error exist.

// Upper bidiagonal, 1 on the diagonal and `superdiagonal` above it. With -2 the
// inverse has entries 2^(j-i), so kappa_1 = 3 * (2^n - 1) in closed form: a
// yardstick dialled purely by n, with no rounding in the "true" value to argue
// about. With -1.7 the same shape rounds at every operation, which is what makes
// the backward error nonzero and lets kappa actually amplify it.
SpMat bidiagonal(int n, double superdiagonal) {
  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    t.emplace_back(j, j, 1.0);
    if (j > 0) t.emplace_back(j - 1, j, superdiagonal);
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// The true 1-norm condition number, densely. The O(n^3) reference, small n only.
double denseCondition1(const SpMat& A) {
  const Eigen::MatrixXd dense = Eigen::MatrixXd(A);
  const Eigen::MatrixXd inv = dense.inverse();
  double normA = 0, normInv = 0;
  for (Eigen::Index j = 0; j < dense.cols(); ++j) {
    normA = (std::max)(normA, dense.col(j).cwiseAbs().sum());
    normInv = (std::max)(normInv, inv.col(j).cwiseAbs().sum());
  }
  return normA * normInv;
}

VectorXd irrationalSolution(Eigen::Index n) {
  VectorXd x(n);
  for (Eigen::Index i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * std::sin(3.0 * double(i));
  return x;
}

// std::to_string renders 2e-14 as "0.000000", which is exactly the range every
// number in this section lives in.
std::string sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.3e", v);
  return std::string(buf);
}

void testConditionEstimate() {
  std::printf("\n-- condition estimate against a known kappa --\n");
  // Hager's algorithm is a LOWER bound, so `est <= exact` is the load-bearing
  // invariant: an estimator that overshoots reports a matrix as worse
  // conditioned than it is, which is a different and equally real bug.
  //
  // The lower gate is what tests the ADJOINT solve, and it is not slack. The
  // estimator's first probe never touches A^H and on this matrix returns roughly
  // exact/n on its own -- 2.5% of the true value at n=40. Only the A^H step
  // lifts it to the exact vertex, so anything above half the true value can only
  // have come from a correct adjoint.
  for (int n : {2, 5, 10, 20, 30, 40}) {
    const SpMat A = bidiagonal(n, -2.0);
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    const double exact = (n == 1) ? 1.0 : 3.0 * (std::pow(2.0, n) - 1.0);
    const double est = s.conditionEstimate();
    const std::string tag = "kappa n=" + std::to_string(n);
    lu_testing::check(est <= exact * (1.0 + 1e-9), tag + ": a lower bound", est / exact);
    lu_testing::check(est >= exact * 0.5, tag + ": tight, so A^H is right", est / exact);
    lu_testing::checkTrue(s.conditionEstimateSolves() > 0, tag + ": solves were actually spent");
  }
  for (int n : {8, 16, 32, 64}) {
    const SpMat A = lu_testing::randomUnsymmetricPattern(n, 0.3, 7u + unsigned(n));
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    if (s.info() != Eigen::Success) continue;
    const double exact = denseCondition1(A);
    lu_testing::check(s.conditionEstimate() <= exact * (1.0 + 1e-9),
                      "kappa vs dense inverse n=" + std::to_string(n) + ": a lower bound",
                      s.conditionEstimate() / exact);
  }
  {  // Cached until the next factorization, and honest about having none.
    const SpMat A = bidiagonal(16, -2.0);
    Eigen::PointBlockLU<SpMat> fresh;
    lu_testing::checkTrue(std::isinf(fresh.conditionEstimate()),
                          "un-factorized: kappa is infinite, not zero");
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    const double first = s.conditionEstimate();
    const Eigen::Index spent = s.conditionEstimateSolves();
    lu_testing::checkTrue(s.conditionEstimate() == first && s.conditionEstimateSolves() == spent,
                          "a second call is cached, not recomputed");
    s.factorize(A);
    lu_testing::checkTrue(s.conditionEstimateSolves() == 0,
                          "factorize() invalidates the cached estimate");
  }
}

void testErrorBoundsCatchWhatTheResidualCannot() {
  std::printf("\n-- an answer that is backward stable and still worthless --\n");
  // THE case this whole section exists for. At n=80 the bidiagonal's kappa is
  // ~1e19: the computed x is the exact solution of a system a rounding error
  // away from A, so its residual is ~1e-14 and every residual check on earth
  // passes it -- and it is wrong by a factor of 70.
  const int n = 80;
  const SpMat A = bidiagonal(n, -1.7);
  const VectorXd xTrue = irrationalSolution(n);
  const VectorXd b = A * xTrue;

  Eigen::PointBlockLU<SpMat> plain;
  plain.compute(A);
  const VectorXd x = plain.solve(b);
  const double trueError = (x - xTrue).norm() / xTrue.norm();
  lu_testing::check(trueError > 1.0, "the answer really is worthless", trueError);
  lu_testing::check(plain.solveResidual() < 1e-9, "and its residual is tiny anyway",
                    plain.solveResidual());
  lu_testing::checkTrue(plain.info() == Eigen::Success,
                        "so the residual check alone passes it -- as documented");

  Eigen::PointBlockLU<SpMat> bounded;
  bounded.setErrorBounds(true);
  bounded.compute(A);
  const VectorXd xb = bounded.solve(b);
  lu_testing::check(bounded.lastBackwardError() < 1e-12,
                    "error bounds agree it is backward stable", bounded.lastBackwardError());
  lu_testing::checkTrue(bounded.info() == Eigen::NumericalIssue,
                        "and STILL report NumericalIssue -- the gap is closed");
  lu_testing::checkTrue(bounded.lastCorrectDigits() == 0, "zero correct digits claimed");
  lu_testing::checkTrue(!bounded.lastErrorMessage().empty(), "and it says why");
  lu_testing::note("n=80: kappa=" + sci(bounded.conditionEstimate()) + " resid=" +
                   sci(plain.solveResidual()) + " true error=" + sci(trueError));
  (void)xb;

  // The other half of the contract: a well-conditioned system must NOT be
  // downgraded, or the check is just noise.
  const SpMat W = lu_testing::upwind2d(12, 12);
  const VectorXd wTrue = irrationalSolution(W.rows());
  Eigen::PointBlockLU<SpMat> good;
  good.setErrorBounds(true);
  good.compute(W);
  const VectorXd wx = good.solve(VectorXd(W * wTrue));
  lu_testing::checkTrue(good.info() == Eigen::Success, "a well-conditioned system is left alone");
  lu_testing::check(good.lastCorrectDigits() >= 10, "and claims double-precision digits",
                    double(good.lastCorrectDigits()));
  lu_testing::check((wx - wTrue).norm() / wTrue.norm() < 1e-12, "which it deserves",
                    (wx - wTrue).norm() / wTrue.norm());
}

void testDefaultPathPaysOnlyForTheResidual() {
  std::printf("\n-- what the default path does and does not spend --\n");
  const SpMat A = lu_testing::upwind2d(10, 10);
  const VectorXd xTrue = irrationalSolution(A.rows());
  const VectorXd b = A * xTrue;
  {
    Eigen::PointBlockLU<SpMat> s;
    s.compute(A);
    const VectorXd x = s.solve(b);
    lu_testing::checkTrue(s.conditionEstimateSolves() == 0,
                          "no triangular solve is spent on kappa unless asked");
    lu_testing::checkTrue(std::isnan(s.lastBackwardError()) && std::isnan(s.lastForwardError()),
                          "no error bounds are computed unless asked");
    lu_testing::checkTrue(s.lastCorrectDigits() == -1, "and none are claimed");
    lu_testing::check(std::isfinite(s.solveResidual()), "but the residual IS measured by default",
                      s.solveResidual());
    lu_testing::checkTrue(std::isnan(s.growthFactor()) || s.growthFactor() > 0,
                          "growthFactor() answers after a factorization");
    (void)x;
  }
  {  // Opting out must opt out of the work, not just the verdict.
    Eigen::PointBlockLU<SpMat> s;
    s.setSolveFailureThreshold(0.0);
    s.compute(A);
    const VectorXd x = s.solve(b);
    lu_testing::checkTrue(std::isnan(s.solveResidual()),
                          "threshold 0 skips the residual entirely");
    lu_testing::checkTrue(s.info() == Eigen::Success, "and leaves info() alone");
    lu_testing::check((x - xTrue).norm() / xTrue.norm() < 1e-12,
                      "the answer is unchanged either way", (x - xTrue).norm() / xTrue.norm());
  }
  {  // And the gate must actually be able to fire.
    Eigen::PointBlockLU<SpMat> s;
    s.setSolveFailureThreshold(1e-300);  // no honest solve can meet this
    s.compute(A);
    const VectorXd x = s.solve(b);
    lu_testing::checkTrue(s.info() == Eigen::NumericalIssue,
                          "a threshold nothing can meet does downgrade info()");
    lu_testing::checkTrue(s.lastErrorMessage().find("solveResidual") != std::string::npos,
                          "and the message points at the number that failed");
    (void)x;
  }
}

void testGrowthFactorTracksPivotThreshold() {
  std::printf("\n-- growth factor: what earns a relaxed pivot threshold --\n");
  // setPivotThreshold() below 1.0 buys less fill by allowing element growth.
  // That is a documented trade, and growthFactor() is what makes the price
  // visible instead of leaving it to be discovered as a wrong answer.
  const SpMat A = lu_testing::weakDiagonal(200, 11u);
  const VectorXd xTrue = VectorXd::Ones(A.rows());
  const VectorXd b = A * xTrue;
  double strictGrowth = 0, relaxedGrowth = 0;
  Eigen::Index strictFill = 0, relaxedFill = 0;
  for (double t : {1.0, 1e-8}) {
    Eigen::PointBlockLU<SpMat> s;
    s.setPivotThreshold(t);
    s.compute(A);
    if (s.info() != Eigen::Success) {
      lu_testing::fail("weakDiagonal declined at pivot threshold " + sci(t));
      return;
    }
    const VectorXd x = s.solve(b);
    (t == 1.0 ? strictGrowth : relaxedGrowth) = s.growthFactor();
    (t == 1.0 ? strictFill : relaxedFill) = s.nnzL() + s.nnzU();
    lu_testing::note("threshold " + sci(t) + ": growth=" + sci(s.growthFactor()) + " fill=" +
                     std::to_string((long long)(s.nnzL() + s.nnzU())) + " error=" +
                     sci((x - xTrue).norm() / xTrue.norm()));
  }
  lu_testing::check(strictGrowth < 10.0, "strict partial pivoting keeps growth near 1",
                    strictGrowth);
  lu_testing::check(relaxedGrowth > 100.0 * strictGrowth,
                    "relaxing the threshold shows up as growth", relaxedGrowth / strictGrowth);
  lu_testing::checkTrue(relaxedFill < strictFill, "which is what bought the lower fill");
}

void testReplayRefreshesTheResidualCheck() {
  std::printf("\n-- a replay measures the NEW matrix, not the recorded one --\n");
  // The residual check needs a copy of A, and the replay path refreshes that
  // copy by value memcpy rather than sparse assignment. If that refresh were
  // ever skipped, every replayed solve would be scored against the matrix from
  // the first factorization -- a wrong residual, and the one place this whole
  // feature could go quietly wrong.
  const SpMat A = lu_testing::randomUnsymmetricPattern(80, 0.25, 3u);
  const SpMat B = perturbed(A, 0.3);
  const double drift = (Eigen::MatrixXd(A) - Eigen::MatrixXd(B)).norm() / Eigen::MatrixXd(A).norm();
  lu_testing::check(drift > 0.05, "the two matrices really do differ", drift);

  const VectorXd xTrue = irrationalSolution(A.rows());
  Eigen::PointBlockLU<SpMat> s;
  s.analyzePattern(A);
  s.factorize(A);
  const VectorXd xa = s.solve(VectorXd(A * xTrue));
  lu_testing::check(s.solveResidual() < 1e-12, "first factorization: residual measured vs A",
                    s.solveResidual());
  lu_testing::check((xa - xTrue).norm() / xTrue.norm() < 1e-9, "and the answer is right",
                    (xa - xTrue).norm() / xTrue.norm());

  s.factorize(B);
  lu_testing::checkTrue(s.refactorizations() > 0, "the second factorize really replayed");
  const VectorXd xb = s.solve(VectorXd(B * xTrue));
  // Against a stale copy of A this residual would land near `drift`, not at eps.
  lu_testing::check(s.solveResidual() < 1e-12, "replay: residual measured vs B, not A",
                    s.solveResidual());
  lu_testing::checkTrue(s.info() == Eigen::Success, "so the replay is not falsely flagged");
  lu_testing::check((xb - xTrue).norm() / xTrue.norm() < 1e-9, "and its answer is right too",
                    (xb - xTrue).norm() / xTrue.norm());
}

void testComplexScalars() {
  std::printf("\n-- complex scalars (the adjoint conjugates, or kappa is wrong) --\n");
  typedef std::complex<double> Complex;
  typedef Eigen::SparseMatrix<Complex> SpCplx;
  const int n = 24;
  std::vector<Eigen::Triplet<Complex>> t;
  for (int j = 0; j < n; ++j) {
    t.emplace_back(j, j, Complex(1.0, 0.3));
    if (j > 0) t.emplace_back(j - 1, j, Complex(-1.4, 0.6));
    if (j + 1 < n) t.emplace_back(j + 1, j, Complex(0.2, -0.1));
  }
  SpCplx A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  const Eigen::MatrixXcd dense = Eigen::MatrixXcd(A), inv = dense.inverse();
  double normA = 0, normInv = 0;
  for (int j = 0; j < n; ++j) {
    normA = (std::max)(normA, dense.col(j).cwiseAbs().sum());
    normInv = (std::max)(normInv, inv.col(j).cwiseAbs().sum());
  }
  const double exact = normA * normInv;

  Eigen::PointBlockLU<SpCplx> s;
  s.setErrorBounds(true);
  s.compute(A);
  const Eigen::VectorXcd xTrue = Eigen::VectorXcd::Ones(n);
  const Eigen::VectorXcd x = s.solve(Eigen::VectorXcd(A * xTrue));
  lu_testing::check(s.conditionEstimate() <= exact * (1.0 + 1e-9), "complex kappa: a lower bound",
                    s.conditionEstimate() / exact);
  // A plain transpose instead of the adjoint lands far off here; half the true
  // value is only reachable with the conjugation in place.
  lu_testing::check(s.conditionEstimate() >= exact * 0.5, "complex kappa: tight, so A^H conjugates",
                    s.conditionEstimate() / exact);
  lu_testing::checkTrue(s.info() == Eigen::Success, "well-conditioned complex system passes");
  lu_testing::check((x - xTrue).norm() / xTrue.norm() < 1e-12, "complex answer is right",
                    (x - xTrue).norm() / xTrue.norm());
  lu_testing::check(s.growthFactor() < 10.0, "and complex growth is benign", s.growthFactor());
}

void testTestdata() {
  std::printf("\n-- testdata corpus (skipped when testdata/ is absent) --\n");
  int seen = 0;
  for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices()) {
    if (m.tier != lu_testing::Tier::Small) continue;
    SpMat A;
    try {
      A = lu_testing::loadMatrixMarket(lu_testing::testdataPath(m.relative));
    } catch (const std::exception&) {
      continue;
    }
    ++seen;
    const VectorXd b = rhsFor(A);
    Eigen::PointBlockLU<SpMat, Eigen::COLAMDOrdering<int>> s;
    s.compute(A);
    if (s.info() != Eigen::Success) {
      // Declining is a legitimate outcome; it must be reasoned, not silent.
      std::printf("  [INFO] %-13s declined: %s\n", m.label, s.lastErrorMessage().c_str());
      lu_testing::check(!s.lastErrorMessage().empty(), std::string(m.label) + ": decline explained", 0.0);
      continue;
    }
    const VectorXd x = s.solve(b);
    lu_testing::check(relativeResidual(A, x, b) < 1e-6, std::string(m.label) + ": residual",
                      relativeResidual(A, x, b));
  }
  if (seen == 0) std::printf("  (no testdata matrices found -- corpus checks skipped)\n");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("PointBlockLU correctness\n");
  testOrderings();
  testUnsymmetricPattern();
  testReplay();
  testReplayRejection();
  testDegenerate();
  testDeterminant();
  testConditionEstimate();
  testErrorBoundsCatchWhatTheResidualCannot();
  testDefaultPathPaysOnlyForTheResidual();
  testGrowthFactorTracksPivotThreshold();
  testReplayRefreshesTheResidualCheck();
  testComplexScalars();
  testTestdata();
  return lu_testing::summarize("test_pointblock_lu");
}
