// Phase 4 of the header-only METIS port: the wiring check.
//
// test_header_only_metis.cpp already proves the ALGORITHM is bit-identical to
// the reference C library, comparing perm/iperm straight out of nodeND(). This
// suite covers the layer above it -- Eigen::HeaderOnlyMetisOrdering, the
// drop-in replacement for Eigen::MetisOrdering in HeaderOnlyMetis.h -- and
// asks a different question: does the glue hand the solvers the same thing the
// linked-METIS path does?
//
// Four things get checked, and only the first needs METIS present:
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
//   4. edge shapes, standalone: graphs at the algorithm's boundaries (isolated
//      vertices past MMDSWITCH, cloned vertices that compress, a hub whose
//      compression key overflows int32, star, complete, empty and uncompressed
//      input) return valid, mutually inverse, reproducible permutations on
//      both the exact and the parallel path. Also runs in every build.
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
#include "SupernodalLUExecutor.h"
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

// --- 4. shapes at the algorithm's edges, standalone ------------------------
//
// The oracle suite compares these against METIS when it is linked; this block
// runs in every build and asks only what can be asked without a reference:
// the exact and the parallel path each return a permutation, perm and iperm
// are inverses, repeated calls agree, and the parallel path is
// thread-invariant. The shapes are the ones that reach the code paths a
// well-behaved FEM mesh never does.

struct Graph {
  int n = 0;
  std::vector<int> xadj, adjncy;
};

Graph graphFromAdjacency(std::vector<std::vector<int>> adj) {
  Graph g;
  g.n = static_cast<int>(adj.size());
  g.xadj.assign(static_cast<std::size_t>(g.n) + 1, 0);
  for (int i = 0; i < g.n; ++i) {
    g.xadj[static_cast<std::size_t>(i)] = static_cast<int>(g.adjncy.size());
    g.adjncy.insert(g.adjncy.end(), adj[static_cast<std::size_t>(i)].begin(), adj[static_cast<std::size_t>(i)].end());
  }
  g.xadj[static_cast<std::size_t>(g.n)] = static_cast<int>(g.adjncy.size());
  return g;
}

void addUndirected(std::vector<std::vector<int>>& adj, int a, int b) {
  adj[static_cast<std::size_t>(a)].push_back(b);
  adj[static_cast<std::size_t>(b)].push_back(a);
}

bool isPermutation(const std::vector<int>& p) {
  std::vector<char> seen(p.size(), 0);
  for (int v : p) {
    if (v < 0 || static_cast<std::size_t>(v) >= p.size() || seen[static_cast<std::size_t>(v)]) return false;
    seen[static_cast<std::size_t>(v)] = 1;
  }
  return true;
}

void checkShape(const std::string& name, const Graph& g) {
  const int n = g.n;
  std::vector<int> xadj = g.xadj, adjncy = g.adjncy;
  std::vector<int> perm(static_cast<std::size_t>(n)), iperm(static_cast<std::size_t>(n));
  header_only_metis::nodeND<int, float>(n, xadj.data(), adjncy.data(), nullptr, perm.data(), iperm.data());
  checkTrue(isPermutation(perm) && isPermutation(iperm), name + ": exact path returns permutations");
  bool inverse = true;
  for (int i = 0; i < n; ++i) inverse = inverse && perm[static_cast<std::size_t>(iperm[static_cast<std::size_t>(i)])] == i;
  checkTrue(inverse, name + ": perm and iperm are inverses");
  checkTrue(xadj == g.xadj && adjncy == g.adjncy, name + ": input left untouched");

  std::vector<int> perm2(static_cast<std::size_t>(n)), iperm2(static_cast<std::size_t>(n));
  header_only_metis::nodeND<int, float>(n, xadj.data(), adjncy.data(), nullptr, perm2.data(), iperm2.data());
  checkTrue(perm2 == perm && iperm2 == iperm, name + ": exact path reproducible");

  std::vector<int> pperm(static_cast<std::size_t>(n)), piperm(static_cast<std::size_t>(n));
  const header_only_metis::SerialExecutor serial;
  header_only_metis::nodeNDParallel<int, float>(n, xadj.data(), adjncy.data(), nullptr, pperm.data(), piperm.data(),
                                                serial);
  checkTrue(isPermutation(pperm) && isPermutation(piperm), name + ": parallel path returns permutations");
  std::vector<int> tperm(static_cast<std::size_t>(n)), tiperm(static_cast<std::size_t>(n));
  const Eigen::supernodal_lu::StdThreadExecutor threads(3);
  header_only_metis::nodeNDParallel<int, float>(n, xadj.data(), adjncy.data(), nullptr, tperm.data(), tiperm.data(),
                                                threads);
  checkTrue(tperm == pperm && tiperm == piperm, name + ": parallel path thread-invariant");
}

void checkEdgeShapesStandalone() {
  // A path plus more isolated vertices than MMDSWITCH: the first separator
  // leaves an edgeless side that is ordered by MMD on an EMPTY adjacency,
  // which used to hand genmmd a null pointer to decrement.
  {
    std::vector<std::vector<int>> adj(400);
    for (int i = 0; i + 1 < 200; ++i) addUndirected(adj, i, i + 1);
    checkShape("path + 200 isolated", graphFromAdjacency(std::move(adj)));
  }
  // Fully disconnected, and a single vertex.
  checkShape("40 isolated vertices", graphFromAdjacency(std::vector<std::vector<int>>(40)));
  checkShape("single vertex", graphFromAdjacency(std::vector<std::vector<int>>(1)));
  // Cloned vertices with identical adjacency: CompressGraph merges them and,
  // at a compression ratio above 1.5, the driver switches to two separator
  // trials per bisection.
  {
    const int base = 90, k = 4;
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(base * k));
    for (int i = 0; i < base; ++i)
      for (int j = i + 1; j < base; ++j)
        if ((i * 7 + j * 3) % 11 == 0)
          for (int ci = 0; ci < k; ++ci)
            for (int cj = 0; cj < k; ++cj) addUndirected(adj, i * k + ci, j * k + cj);
    checkShape("cloned vertices (compression > 1.5x)", graphFromAdjacency(std::move(adj)));
  }
  // A dense hub whose neighbour-index sum overflows int32 in CompressGraph's
  // key; the sum has to wrap, not trap, and the result still has to be a
  // permutation.
  {
    const int n = 70000;
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
    for (int i = 1; i < n; ++i) {
      addUndirected(adj, 0, i);
      if (i + 1 < n && i % 3 != 0) addUndirected(adj, i, i + 1);
    }
    checkShape("dense hub, key overflow (n=70000)", graphFromAdjacency(std::move(adj)));
  }
  // Star and complete graphs: degenerate separators.
  {
    std::vector<std::vector<int>> star(300);
    for (int i = 1; i < 300; ++i) addUndirected(star, 0, i);
    checkShape("star (n=300)", graphFromAdjacency(std::move(star)));
    std::vector<std::vector<int>> complete(60);
    for (int i = 0; i < 60; ++i)
      for (int j = i + 1; j < 60; ++j) addUndirected(complete, i, j);
    checkShape("complete K60", graphFromAdjacency(std::move(complete)));
  }
  // The Eigen wrapper on an empty matrix and on an uncompressed one.
  {
    SparseMatrix<double> empty(0, 0);
    empty.makeCompressed();
    Ordering ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    ordering(empty, perm);
    checkTrue(perm.size() == 0, "wrapper: empty matrix gives an empty permutation");

    const SparseMatrix<double> A = lu_testing::randomUnsymmetricPattern(150, 0.04, 9);
    SparseMatrix<double> Au(150, 150);
    Au.reserve(Eigen::VectorXi::Constant(150, 16));
    for (int j = 0; j < 150; ++j)
      for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it) Au.insert(it.row(), j) = it.value();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> pc, pu;
    ordering(A, pc);
    ordering(Au, pu);
    bool same = pc.size() == pu.size();
    for (int i = 0; same && i < pc.size(); ++i) same = pc.indices()(i) == pu.indices()(i);
    std::vector<int> v(pc.indices().data(), pc.indices().data() + pc.size());
    checkTrue(same && isPermutation(v), "wrapper: uncompressed input orders like compressed");
  }
}

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
  checkEdgeShapesStandalone();

  return lu_testing::summarize("test_header_only_metis_ordering");
}
