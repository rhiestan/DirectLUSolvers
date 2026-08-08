// Edge cases and API contracts neither correctness suite covered.
//
//   ctest --test-dir build -R test_edge_cases --output-on-failure
//
// WHY THIS EXISTS
//
// test_supernodal_lu and test_leftright_lu both exercise the solvers on
// comfortable matrices: a few hundred rows, several supernodes, a well-behaved
// right-hand side. Three classes of input were therefore never tried:
//
//   * DEGENERATE SIZES -- n = 0, 1, 2; a purely diagonal matrix (every supernode
//     one column wide, no off-diagonal panels at all); a matrix dense enough to
//     collapse into a single supernode. These take different branches through
//     the blocking and panel code than anything else in the suite.
//   * THE REFACTORIZE WORKFLOW -- analyzePattern() once, factorize() repeatedly
//     with new values. The README documents it prominently; nothing tested that
//     the second factorization actually replaces the first.
//   * A ZERO RIGHT-HAND SIDE, where the honest-failure check computes
//     ||b - Ax|| / ||b|| and would divide by zero if written naively.
//
// It also runs a cross-solver differential: SupernodalLU and LeftRightLU share
// their entire symbolic pipeline and differ only in the numeric core, so a
// disagreement localizes a regression to one of the two cores immediately.

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;

namespace {

typedef SparseMatrix<double> SpMat;

VectorXd rampRhs(Eigen::Index n) {
  VectorXd v(n);
  for (Eigen::Index i = 0; i < n; ++i) v(i) = 1.0 + 0.25 * static_cast<double>(i % 5);
  return v;
}

// --------------------------------------------------------------------------
//  Degenerate sizes
// --------------------------------------------------------------------------

template <typename Solver>
void solveAndCheck(const SpMat& A, const std::string& tag, double tol = 1e-8) {
  const Eigen::Index n = A.rows();
  Solver s;
  s.compute(A);
  if (s.info() != Eigen::Success) {
    lu_testing::fail(tag + ": factorization failed: " + s.lastErrorMessage());
    return;
  }
  const VectorXd xTrue = rampRhs(n);
  const VectorXd b = A * xTrue;
  const VectorXd x = s.solve(b);
  if (n == 0) {
    checkTrue(x.size() == 0, tag + ": empty solve returns an empty vector");
    return;
  }
  const double err = (x - xTrue).norm() / xTrue.norm();
  check(err < tol, tag + ": solution error", err);
}

void testTinySizes() {
  // n = 0. Nothing in the contract says this must work, but it must not crash
  // or corrupt state, and an empty system trivially has an empty solution.
  {
    SpMat A(0, 0);
    A.makeCompressed();
    Eigen::SupernodalLU<SpMat> s;
    s.compute(A);
    checkTrue(s.info() == Eigen::Success, "n=0: compute() survives an empty matrix");
    const VectorXd x = s.solve(VectorXd(0));
    checkTrue(x.size() == 0, "n=0: solve() returns an empty vector");
  }

  for (int n : {1, 2, 3}) {
    SpMat A(n, n);
    for (int i = 0; i < n; ++i) {
      A.insert(i, i) = 3.0 + i;
      if (i + 1 < n) {
        A.insert(i, i + 1) = -1.0;
        A.insert(i + 1, i) = -0.5;
      }
    }
    A.makeCompressed();
    const std::string tag = "n=" + std::to_string(n);
    solveAndCheck<Eigen::SupernodalLU<SpMat>>(A, "SupernodalLU " + tag);
    solveAndCheck<Eigen::LeftRightLU<SpMat>>(A, "LeftRightLU " + tag);
  }
}

// A purely diagonal matrix: every supernode is one column wide and there are NO
// off-diagonal panels anywhere, so every panel TRSM and Schur update is skipped.
void testDiagonalOnly() {
  const int n = 64;
  SpMat A(n, n);
  for (int i = 0; i < n; ++i) A.insert(i, i) = 1.0 + 0.5 * i;
  A.makeCompressed();
  solveAndCheck<Eigen::SupernodalLU<SpMat>>(A, "SupernodalLU diagonal-only");
  solveAndCheck<Eigen::LeftRightLU<SpMat>>(A, "LeftRightLU diagonal-only");

  Eigen::SupernodalLU<SpMat> s;
  s.compute(A);
  // With no off-diagonal structure the factor is exactly the diagonal.
  checkTrue(s.nnzL() == n && s.nnzU() == n, "diagonal-only: factor has no fill");
}

// A fully dense small block: amalgamation should collapse this to one supernode,
// exercising the "single supernode, nothing to schedule" path.
void testSingleSupernode() {
  const int n = 40;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) t.emplace_back(i, j, i == j ? n : 0.5 / (1.0 + std::abs(i - j)));
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  solveAndCheck<Eigen::SupernodalLU<SpMat>>(A, "SupernodalLU dense block");
  solveAndCheck<Eigen::LeftRightLU<SpMat>>(A, "LeftRightLU dense block");

  Eigen::SupernodalLU<SpMat> s;
  s.compute(A);
  checkTrue(s.supernodeCount() >= 1 && s.levelCount() >= 1,
            "dense block: supernode/level counts are sane");
}

// A zero right-hand side. The honest-failure check divides by ||b||; done
// naively that is 0/0, which would make info() NumericalIssue (or NaN) on a
// perfectly valid solve.
void testZeroRhs() {
  const int n = 32;
  const SpMat A = lu_testing::laplacian2d(8, 4);
  Eigen::SupernodalLU<SpMat> s;
  s.compute(A);
  const VectorXd x = s.solve(VectorXd::Zero(n));
  checkTrue(x.allFinite(), "zero RHS: solution is finite");
  check(x.norm() < 1e-12, "zero RHS: solution is zero", x.norm());
  checkTrue(s.info() == Eigen::Success, "zero RHS: info() stays Success");
  checkTrue(std::isfinite(s.solveResidual()), "zero RHS: solveResidual() is finite");
}

// --------------------------------------------------------------------------
//  Refactorize workflow
// --------------------------------------------------------------------------

// The documented pattern: analyze once, then factorize repeatedly with new
// values. The trap is a second factorize() that silently keeps the first
// factorization -- every residual would then be computed against the wrong
// matrix, so the test deliberately makes the two systems have very different
// solutions.
template <typename Solver>
void testRefactorize(const char* who) {
  const std::string tag = std::string(who) + " refactorize";
  const int gx = 10, gy = 8;
  SpMat A1 = lu_testing::laplacian2d(gx, gy);
  SpMat A2 = A1;  // same pattern
  for (int j = 0; j < A2.outerSize(); ++j)
    for (SpMat::InnerIterator it(A2, j); it; ++it)
      it.valueRef() = it.row() == it.col() ? 17.0 + 0.5 * it.row() : -0.25;

  const Eigen::Index n = A1.rows();
  const VectorXd xTrue = rampRhs(n);
  const VectorXd b1 = A1 * xTrue;
  const VectorXd b2 = A2 * xTrue;

  Solver s;
  s.analyzePattern(A1);
  const long long predicted = s.predictedFactorNonzeros();

  s.factorize(A1);
  checkTrue(s.info() == Eigen::Success, tag + ": first factorize succeeds");
  const double e1 = (s.solve(b1) - xTrue).norm() / xTrue.norm();
  check(e1 < 1e-8, tag + ": solve after first factorize", e1);

  s.factorize(A2);  // same pattern, different values -- no re-analysis
  checkTrue(s.info() == Eigen::Success, tag + ": second factorize succeeds");
  const double e2 = (s.solve(b2) - xTrue).norm() / xTrue.norm();
  check(e2 < 1e-8, tag + ": solve after second factorize", e2);

  // The decisive check: the refreshed factors must NOT still solve the old
  // system. If factorize(A2) were a no-op, this would pass A1's system and fail
  // here, which is exactly the regression being guarded.
  const double stale = (A1 * s.solve(b2) - b2).norm() / b2.norm();
  checkTrue(stale > 1e-6, tag + ": second factorization really replaced the first");

  // Re-analysis must not change the symbolic prediction for the same pattern.
  s.analyzePattern(A2);
  checkTrue(s.predictedFactorNonzeros() == predicted,
            tag + ": symbolic prediction is stable for a fixed pattern");

  // compute() after the manual sequence must also work.
  s.compute(A1);
  const double e3 = (s.solve(b1) - xTrue).norm() / xTrue.norm();
  check(e3 < 1e-8, tag + ": compute() after manual analyze/factorize", e3);
}

// Repeated solves against one factorization must be stable and independent.
void testRepeatedSolves() {
  const SpMat A = lu_testing::laplacian2d(12, 10);
  const Eigen::Index n = A.rows();
  Eigen::SupernodalLU<SpMat> s;
  s.compute(A);

  const VectorXd xTrue = rampRhs(n);
  const VectorXd b = A * xTrue;
  const VectorXd first = s.solve(b);
  double worst = 0.0;
  for (int i = 0; i < 5; ++i) {
    // Interleave an unrelated solve; it must not perturb the repeat.
    (void)s.solve(VectorXd::Ones(n));
    worst = std::max(worst, (s.solve(b) - first).norm() / std::max(1e-300, first.norm()));
  }
  check(worst == 0.0, "repeated solves are bit-identical", worst);
}

// --------------------------------------------------------------------------
//  Cross-solver differential
// --------------------------------------------------------------------------

// The two solvers share matching, ordering, the elimination tree, supernode
// detection, the block symbolic factorization and the triangular solve. Only the
// numeric core differs. So they must agree on fill exactly, and on the solution
// to within refinement tolerance -- a divergence points at one numeric core.
void testCrossSolverDifferential() {
  struct Case {
    std::string label;
    SpMat A;
  };
  std::vector<Case> cases;
  cases.push_back({"laplacian2d(20,16)", lu_testing::laplacian2d(20, 16)});
  cases.push_back({"laplacian3d(8,8,8)", lu_testing::laplacian3d(8, 8, 8)});
  cases.push_back({"random(200,0.04)", lu_testing::randomSymmetricPattern(200, 0.04, 91)});
  cases.push_back({"weakDiagonal(300)", lu_testing::weakDiagonal(300, 13)});

  for (const Case& c : cases) {
    const Eigen::Index n = c.A.rows();
    const VectorXd xTrue = rampRhs(n);
    const VectorXd b = c.A * xTrue;

    Eigen::SupernodalLU<SpMat> a;
    Eigen::LeftRightLU<SpMat> d;
    // Make the comparison fair: LeftRightLU skips refinement by default when no
    // pivot was perturbed, which legitimately leaves a looser residual.
    d.setRefineOnlyIfPerturbed(false);
    a.compute(c.A);
    d.compute(c.A);
    if (a.info() != Eigen::Success || d.info() != Eigen::Success) {
      lu_testing::fail(c.label + ": one solver failed to factorize");
      continue;
    }

    checkTrue(a.nnzL() == d.nnzL() && a.nnzU() == d.nnzU(),
              c.label + ": solvers agree on fill exactly");
    checkTrue(a.supernodeCount() == d.supernodeCount(),
              c.label + ": solvers agree on the supernode partition");

    const VectorXd xa = a.solve(b), xd = d.solve(b);
    const double agree = (xa - xd).norm() / std::max(1e-300, xa.norm());
    check(agree < 1e-8, c.label + ": solutions agree", agree);

    // Determinants come from different pivoting schemes (row-only vs complete),
    // but must still describe the same matrix.
    const double da = a.determinant(), dd = d.determinant();
    if (std::isfinite(da) && std::isfinite(dd) && std::abs(da) > 1e-300) {
      const double drel = std::abs(da - dd) / std::abs(da);
      check(drel < 1e-6, c.label + ": determinants agree", drel);
    }
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("Edge cases and API contracts\n");

  std::printf("Degenerate sizes:\n");
  testTinySizes();
  testDiagonalOnly();
  testSingleSupernode();
  testZeroRhs();

  std::printf("Refactorize workflow:\n");
  testRefactorize<Eigen::SupernodalLU<SpMat>>("SupernodalLU");
  testRefactorize<Eigen::LeftRightLU<SpMat>>("LeftRightLU");
  testRepeatedSolves();

  std::printf("Cross-solver differential:\n");
  testCrossSolverDifferential();

  return lu_testing::summarize("Edge cases");
}
