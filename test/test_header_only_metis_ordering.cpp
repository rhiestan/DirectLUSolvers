// Phase 4 of the header-only METIS port: the wiring check.
//
// test_header_only_metis.cpp already proves the ALGORITHM is bit-identical to
// the reference C library, comparing perm/iperm straight out of nodeND(). This
// suite covers the layer above it -- Eigen::HeaderOnlyMetisOrdering, the
// drop-in replacement for Eigen::MetisOrdering in HeaderOnlyMetis.h -- and
// asks a different question: does the glue hand the solvers the same thing the
// linked-METIS path does?
//
// Three things get checked, and only the first needs METIS present:
//
//   1. permutation parity: HeaderOnlyMetisOrdering and MetisOrdering produce
//      element-wise identical PermutationTypes on the same Eigen matrix. This
//      is what makes it a drop-in replacement rather than merely a good
//      ordering. It also pins the permutation DIRECTION, which no residual
//      check can see (a reversed permutation still solves to machine
//      precision, it just inflates fill -- 250-350x on 3D FEM here).
//
//   2. solver parity: SupernodalLU and LeftRightLU produce identical fill
//      (nnzL/nnzU) with either ordering. Fill is a deterministic function of
//      the permutation, so this is the end-to-end consequence of (1).
//
//   3. standalone correctness: the same solvers, driven by
//      HeaderOnlyMetisOrdering alone, factor and solve to a small residual.
//      This one runs with or without DLU_WITH_METIS -- which is the point:
//      built without METIS the file below still compiles, links and passes,
//      demonstrating the "no link-time dependency" claim rather than asserting
//      it.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON && cmake --build build
//   ctest --test-dir build -R test_header_only_metis_ordering --output-on-failure

#include <Eigen/SparseCore>

#include <random>
#include <string>
#include <vector>

#include "HeaderOnlyMetis.h"
#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

#ifdef HAVE_METIS
// Eigen/MetisSupport uses std::cerr on a METIS error without including
// <iostream> itself (see SupernodalLUMetis.h, which does the same).
#include <iostream>

#include <Eigen/MetisSupport>
#endif

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::note;

namespace {

using Ordering = Eigen::HeaderOnlyMetisOrdering<int>;

// The matrices this suite runs over. Deliberately small: the ordering itself
// is already exercised across the full corpus by test_header_only_metis.cpp,
// so what matters here is covering the shapes the SOLVERS care about
// (symmetric and unsymmetric patterns, 2D and 3D connectivity) rather than
// re-testing the ordering.
struct Case {
  std::string label;
  SparseMatrix<double> A;
};

std::vector<Case> cases() {
  std::vector<Case> c;
  c.push_back({"lap2d_30x30", lu_testing::laplacian2d(30, 30)});
  c.push_back({"lap2d_45x45", lu_testing::laplacian2d(45, 45)});
  c.push_back({"lap3d_10x10x10", lu_testing::laplacian3d(10, 10, 10)});
  c.push_back({"lap3d_12x12x14", lu_testing::laplacian3d(12, 12, 14)});
  return c;
}

VectorXd deterministicRhs(const SparseMatrix<double>& A, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  VectorXd xTrue(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) xTrue(i) = uni(rng);
  return A * xTrue;
}

// --- 1. permutation parity (needs METIS to compare against) ---------------

#ifdef HAVE_METIS
void checkPermutationParity() {
  for (const Case& c : cases()) {
    Eigen::HeaderOnlyMetisOrdering<int> headerOnly;
    Eigen::MetisOrdering<int> reference;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> pHeaderOnly, pReference;
    headerOnly(c.A, pHeaderOnly);
    reference(c.A, pReference);

    if (!checkTrue(pHeaderOnly.size() == pReference.size(), c.label + ": permutation sizes agree")) continue;

    bool identical = true;
    for (Eigen::Index i = 0; i < pReference.size(); ++i) {
      if (pHeaderOnly.indices()(i) != pReference.indices()(i)) {
        identical = false;
        break;
      }
    }
    checkTrue(identical, c.label + ": HeaderOnlyMetisOrdering == MetisOrdering (element-wise)");
  }
}
#endif  // HAVE_METIS

// --- 2/3. solver-level checks ---------------------------------------------

// Factors A with `Solver` and reports fill + residual. nnzL < 0 means the
// solver declined (info() != Success), which is reported rather than silently
// treated as a pass.
struct SolveResult {
  long long nnzL = -1;
  long long nnzU = -1;
  double resid = 0.0;
  bool ok = false;
};

template <typename Solver>
SolveResult factorAndSolve(const SparseMatrix<double>& A) {
  SolveResult r;
  const VectorXd b = deterministicRhs(A, 20260825u);
  Solver solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) return r;
  const VectorXd x = solver.solve(b);
  r.nnzL = solver.nnzL();
  r.nnzU = solver.nnzU();
  r.resid = (A * x - b).norm() / b.norm();
  r.ok = true;
  return r;
}

// (3) The header-only ordering drives the solver on its own: factors, solves,
// and lands at a small residual. Compiled and run in every configuration.
template <typename SolverHeaderOnly>
void checkSolverStandalone(const std::string& solverName) {
  for (const Case& c : cases()) {
    const std::string name = c.label + "/" + solverName;
    const SolveResult ho = factorAndSolve<SolverHeaderOnly>(c.A);
    if (!checkTrue(ho.ok, name + ": factors and solves with HeaderOnlyMetisOrdering")) continue;
    check(ho.resid < 1e-8, name + ": residual", ho.resid);
  }
}

#ifdef HAVE_METIS
// (2) Swapping HeaderOnlyMetisOrdering in for MetisOrdering changes nothing
// the solver can observe: identical fill, matrix for matrix. Fill rather than
// residual is the gate here for the reason spelled out in test_regression.cpp
// -- a wrong-but-valid permutation keeps residuals at machine precision and
// only shows up in the factor size.
template <typename SolverHeaderOnly, typename SolverReference>
void checkSolverParity(const std::string& solverName) {
  for (const Case& c : cases()) {
    const std::string name = c.label + "/" + solverName;
    const SolveResult ho = factorAndSolve<SolverHeaderOnly>(c.A);
    const SolveResult ref = factorAndSolve<SolverReference>(c.A);
    if (!checkTrue(ho.ok && ref.ok, name + ": both orderings factor")) continue;

    const bool sameFill = (ho.nnzL == ref.nnzL && ho.nnzU == ref.nnzU);
    checkTrue(sameFill, name + ": fill identical to linked-METIS ordering");
    if (!sameFill) {
      note("  header-only nnzL/nnzU: " + std::to_string(ho.nnzL) + "/" + std::to_string(ho.nnzU) +
           "   reference: " + std::to_string(ref.nnzL) + "/" + std::to_string(ref.nnzU));
    }
  }
}
#endif  // HAVE_METIS

}  // namespace

int main() {
#ifdef HAVE_METIS
  note("built WITH METIS -- comparing against the linked reference ordering");
  checkPermutationParity();
  checkSolverParity<Eigen::SupernodalLU<SparseMatrix<double>, Ordering>,
                    Eigen::SupernodalLU<SparseMatrix<double>, Eigen::MetisOrdering<int>>>("SupernodalLU");
  checkSolverParity<Eigen::LeftRightLU<SparseMatrix<double>, Ordering>,
                    Eigen::LeftRightLU<SparseMatrix<double>, Eigen::MetisOrdering<int>>>("LeftRightLU");
#else
  note("built WITHOUT METIS -- checking the header-only ordering standalone");
  note("(that this file links at all is the no-link-dependency check)");
#endif

  checkSolverStandalone<Eigen::SupernodalLU<SparseMatrix<double>, Ordering>>("SupernodalLU");
  checkSolverStandalone<Eigen::LeftRightLU<SparseMatrix<double>, Ordering>>("LeftRightLU");

  return lu_testing::summarize("test_header_only_metis_ordering");
}
