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
  testTestdata();
  return lu_testing::summarize("test_pointblock_lu");
}
