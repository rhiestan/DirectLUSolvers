// Oracle test harness for the header-only METIS nested-dissection port
// (planned under DirectLUSolvers/src/HeaderOnlyMetis*, not yet implemented).
//
// Phase 1 of that project's plan: build the comparison infrastructure BEFORE
// porting a single algorithm file. Right now (no port exists yet) this suite
// exercises the REFERENCE side only:
//   - determinism: METIS_NodeND(options=NULL) must return the same perm/iperm
//     on every call, now that GKlib is built with -DGKRAND=ON (MT19937-64
//     instead of the platform CRT's rand()). This generalizes the one-off
//     repro_check from the plan's Phase 0 across the whole corpus.
//   - validity: perm/iperm together must be a genuine permutation of [0,n).
// Once DirectLUSolvers/src/HeaderOnlyMetis.h exists (Phase 2), extend
// checkGraph() below to also call the port's nodeND() on the same xadj/adjncy
// and assert byte-for-byte equality against the reference -- that comparison
// is the actual bit-identical gate this suite exists to become.
//
// Build + run via CTest (from the DirectLUSolvers directory), requires METIS:
//   cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON && cmake --build build
//   ctest --test-dir build -R test_header_only_metis --output-on-failure

#include <Eigen/SparseCore>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/MetisGraph.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

#ifdef HAVE_METIS
#include <metis.h>
#endif

using Eigen::SparseMatrix;
using lu_testing::checkTrue;
using lu_testing::fail;
using lu_testing::note;

namespace {

#ifdef HAVE_METIS

// Runs METIS_NodeND(options=NULL, vwgt=NULL) on (xadj, adjncy) and checks:
//   1. it succeeds and returns a valid permutation of [0, nvtxs);
//   2. it is bit-for-bit deterministic across repeated calls.
// This is the "reference self-consistency" half of the oracle; the "compare
// to the header-only port" half is TODO until Phase 2 lands (see file header).
void checkGraph(const std::string& name, std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy) {
  idx_t nvtxs = static_cast<idx_t>(xadj.size()) - 1;
  if (nvtxs <= 0) {
    note(name + ": nvtxs=0, skipping (nothing for METIS_NodeND to order)");
    return;
  }

  std::vector<idx_t> perm1(static_cast<std::size_t>(nvtxs)), iperm1(static_cast<std::size_t>(nvtxs));
  std::vector<idx_t> perm2(static_cast<std::size_t>(nvtxs)), iperm2(static_cast<std::size_t>(nvtxs));

  const int rc1 = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr, nullptr, perm1.data(), iperm1.data());
  const int rc2 = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr, nullptr, perm2.data(), iperm2.data());

  if (!checkTrue(rc1 == METIS_OK && rc2 == METIS_OK, name + ": METIS_NodeND returns METIS_OK")) return;

  // perm/iperm must each be a permutation of [0, nvtxs): every value in
  // [0, nvtxs) appears exactly once. Checked independently of the
  // determinism comparison below, since two calls could agree with each
  // other while both being wrong in the same way.
  auto isPermutation = [&](const std::vector<idx_t>& p) {
    std::vector<char> seen(static_cast<std::size_t>(nvtxs), 0);
    for (idx_t v : p) {
      if (v < 0 || v >= nvtxs || seen[static_cast<std::size_t>(v)]) return false;
      seen[static_cast<std::size_t>(v)] = 1;
    }
    return true;
  };
  checkTrue(isPermutation(perm1), name + ": perm is a valid permutation");
  checkTrue(isPermutation(iperm1), name + ": iperm is a valid permutation");

  checkTrue(perm1 == perm2, name + ": perm deterministic across repeated calls");
  checkTrue(iperm1 == iperm2, name + ": iperm deterministic across repeated calls");

  // TODO(Phase 2): once DirectLUSolvers/src/HeaderOnlyMetis.h exists, call its
  // nodeND(nvtxs, xadj, adjncy, permPort, ipermPort) here and assert
  // permPort == perm1 && ipermPort == iperm1 -- the actual bit-identical gate.
}

void checkMatrix(const std::string& name, const SparseMatrix<double>& A) {
  if (A.rows() != A.cols()) {
    fail(name + ": not square, skipping");
    return;
  }
  lu_testing::SymmetrizedGraph<idx_t> g = lu_testing::buildSymmetrizedGraph<idx_t>(A);
  checkGraph(name, g.xadj, g.adjncy);
}

// Hand-built edge cases, straight in xadj/adjncy form (no matrix needed).
void checkHandBuiltGraphs() {
  // n=1, isolated vertex.
  {
    std::vector<idx_t> xadj = {0, 0};
    std::vector<idx_t> adjncy;
    checkGraph("edge: single isolated vertex", xadj, adjncy);
  }
  // Fully disconnected: 40 isolated vertices, no edges at all.
  {
    std::vector<idx_t> xadj(41, 0);
    std::vector<idx_t> adjncy;
    checkGraph("edge: fully disconnected (n=40)", xadj, adjncy);
  }
  // Path graph (chain): trivial separator, exercises the MMD fallback and the
  // multilevel path once long enough.
  {
    const idx_t n = 200;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    std::vector<idx_t> adjncy;
    adjncy.reserve(static_cast<std::size_t>(2 * (n - 1)));
    for (idx_t i = 0; i < n; ++i) {
      xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
      if (i > 0) adjncy.push_back(i - 1);
      if (i + 1 < n) adjncy.push_back(i + 1);
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
    checkGraph("edge: path graph (n=200)", xadj, adjncy);
  }
  // Star graph: one hub connected to every other vertex -- a degenerate
  // separator (the hub alone separates everything).
  {
    const idx_t n = 200;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    std::vector<idx_t> adjncy;
    adjncy.reserve(static_cast<std::size_t>(2 * (n - 1)));
    for (idx_t i = 0; i < n; ++i) {
      xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
      if (i == 0) {
        for (idx_t k = 1; k < n; ++k) adjncy.push_back(k);
      } else {
        adjncy.push_back(0);
      }
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
    checkGraph("edge: star graph (n=200)", xadj, adjncy);
  }
  // Complete graph: worst case for fill, no good separator at all.
  {
    const idx_t n = 60;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    std::vector<idx_t> adjncy;
    adjncy.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n - 1));
    for (idx_t i = 0; i < n; ++i) {
      xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
      for (idx_t k = 0; k < n; ++k)
        if (k != i) adjncy.push_back(k);
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
    checkGraph("edge: complete graph K60", xadj, adjncy);
  }
}

// Synthetic grids sized to straddle both sides of every size threshold in
// ometis.c's recursion (MMDSWITCH=120; the 1000/2000 cutoff in
// MlevelNodeBisectionMultiple; the 5000 cutoff in MlevelNodeBisectionL2), so
// every branch of that size-based dispatch gets exercised at least once.
void checkSyntheticGrids() {
  const int sizes2d[][2] = {{10, 10},    // 100   < 120  (pure MMD)
                            {11, 11},    // 121   > 120
                            {30, 30},    // 900
                            {32, 32},    // 1024  > 1000
                            {45, 45},    // 2025  > 2000
                            {70, 70},    // 4900  < 5000
                            {71, 71},    // 5041  > 5000
                            {100, 100}};  // 10000
  for (const auto& s : sizes2d) {
    const std::string name = "laplacian2d " + std::to_string(s[0]) + "x" + std::to_string(s[1]);
    checkMatrix(name, lu_testing::laplacian2d(s[0], s[1]));
  }

  const int sizes3d[][3] = {{10, 10, 10}, {12, 12, 14}, {20, 20, 5}};
  for (const auto& s : sizes3d) {
    const std::string name =
        "laplacian3d " + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "x" + std::to_string(s[2]);
    checkMatrix(name, lu_testing::laplacian3d(s[0], s[1], s[2]));
  }
}

void checkBenchmarkCorpus() {
  for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices()) {
    const std::string path = lu_testing::testdataPath(m.relative);
    try {
      SparseMatrix<double> A = lu_testing::loadMatrixMarket(path);
      checkMatrix(std::string("benchmark: ") + m.label, A);
    } catch (const std::exception& e) {
      fail(std::string("benchmark: ") + m.label + ": " + e.what());
    }
  }
}

void checkSuiteSparseCorpus() {
  std::vector<lu_testing::SuiteSparseMatrix> matrices = lu_testing::suitesparseMatrices();
  if (matrices.empty()) {
    note("SuiteSparse manifest not found or empty -- skipping");
    return;
  }
  int present = 0;
  for (const auto& m : matrices) present += m.available ? 1 : 0;
  note("SuiteSparse corpus: " + std::to_string(present) + " of " + std::to_string(matrices.size()) +
       " matrices downloaded (run fetch_suitesparse.py for the rest)");
  for (const auto& m : matrices) {
    if (!m.available) continue;
    try {
      SparseMatrix<double> A = lu_testing::loadMatrixMarket(m.path);
      checkMatrix(std::string("suitesparse: ") + m.label(), A);
    } catch (const std::exception& e) {
      fail(std::string("suitesparse: ") + m.label() + ": " + e.what());
    }
  }
}

#endif  // HAVE_METIS

}  // namespace

int main() {
#ifdef HAVE_METIS
  checkHandBuiltGraphs();
  checkSyntheticGrids();
  checkBenchmarkCorpus();
  checkSuiteSparseCorpus();
#else
  note("built without DLU_WITH_METIS -- nothing to check, pass by default");
#endif
  return lu_testing::summarize("test_header_only_metis");
}
