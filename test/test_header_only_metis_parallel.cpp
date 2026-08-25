// Determinism gate for the parallel nested-dissection path.
//
// The exact path has the C library as its oracle (test_header_only_metis.cpp).
// The parallel path deliberately produces a DIFFERENT ordering, so it cannot be
// checked that way -- but it is still fully specified, and this is what pins it:
//
//   1. thread-count invariance: nodeNDParallel with N threads must produce the
//      byte-identical permutation to nodeNDParallel run serially, for every N.
//      This is the race detector. A missed synchronization, a shared Ctrl, an
//      RNG stream leaking between subtrees -- all of them show up here as a
//      permutation that moves when the thread count does. It is the same trick
//      test_parallel_consistency.cpp uses for the solvers.
//
//   2. run-to-run reproducibility: repeated calls agree. Catches state carried
//      over between calls (the sort of bug the missing InitRandom once was).
//
//   3. validity: the result is a genuine permutation.
//
// Ordering QUALITY is not checked here. It used to be, as a ratio against the
// exact path's fill, and that guard was dropped: a relative check cannot see
// the two orderings degrading together, which is the failure mode most likely
// to matter since they share nearly all their code. Fill for this ordering is
// now pinned absolutely, per matrix, over the whole corpus, as the
// SupernodalLU+HOMetisPar rows of test/baselines/testdata.baseline (see
// test_regression.cpp) -- which is possible precisely because check 1 below
// establishes the ordering is deterministic.

#include <Eigen/SparseCore>

#include <string>
#include <vector>

#include "HeaderOnlyMetis.h"
#include "SupernodalLUExecutor.h"
#include "testing/Check.h"
#include "testing/MetisGraph.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using lu_testing::checkTrue;
using lu_testing::note;

namespace {

struct Case {
  std::string label;
  SparseMatrix<double> A;
};

std::vector<Case> cases() {
  std::vector<Case> c;
  // Sized so the dissection tree actually has levels to spread across: a graph
  // that bottoms out in MMD after one bisection would exercise no parallelism.
  c.push_back({"lap2d_60x60", lu_testing::laplacian2d(60, 60)});
  c.push_back({"lap2d_120x120", lu_testing::laplacian2d(120, 120)});
  c.push_back({"lap3d_16x16x16", lu_testing::laplacian3d(16, 16, 16)});
  c.push_back({"lap3d_24x24x24", lu_testing::laplacian3d(24, 24, 24)});
  return c;
}

std::vector<int> orderWith(const lu_testing::SymmetrizedGraph<int>& g, int threads) {
  const int n = static_cast<int>(g.xadj.size()) - 1;
  std::vector<int> perm(n), iperm(n);
  std::vector<int> xadj = g.xadj, adjncy = g.adjncy;
  if (threads <= 1) {
    const header_only_metis::SerialExecutor exec;
    header_only_metis::nodeNDParallel<int, float>(n, xadj.data(), adjncy.data(), nullptr, perm.data(),
                                                  iperm.data(), exec);
  } else {
    const Eigen::supernodal_lu::StdThreadExecutor exec(static_cast<unsigned>(threads));
    header_only_metis::nodeNDParallel<int, float>(n, xadj.data(), adjncy.data(), nullptr, perm.data(),
                                                  iperm.data(), exec);
  }
  return iperm;
}

bool isPermutation(const std::vector<int>& p) {
  std::vector<char> seen(p.size(), 0);
  for (int v : p) {
    if (v < 0 || static_cast<std::size_t>(v) >= p.size() || seen[static_cast<std::size_t>(v)]) return false;
    seen[static_cast<std::size_t>(v)] = 1;
  }
  return true;
}

// (1) + (2) + (3)
void checkThreadInvariance() {
  for (const Case& c : cases()) {
    lu_testing::SymmetrizedGraph<int> g = lu_testing::buildSymmetrizedGraph<int>(c.A);

    const std::vector<int> serial = orderWith(g, 1);
    checkTrue(isPermutation(serial), c.label + ": parallel ordering is a valid permutation");
    checkTrue(orderWith(g, 1) == serial, c.label + ": reproducible across repeated calls");

    // Includes the real hardware width: that is the configuration users
    // actually run, and the one most likely to expose a race.
    const int hw = Eigen::supernodal_lu::StdThreadExecutor().concurrency();
    for (int threads : {2, 4, 8, hw}) {
      const std::vector<int> parallel = orderWith(g, threads);
      checkTrue(parallel == serial,
                c.label + ": " + std::to_string(threads) + " threads == serial (byte-identical)");
    }
  }
}

}  // namespace

int main() {
  note("hardware concurrency: " + std::to_string(Eigen::supernodal_lu::StdThreadExecutor().concurrency()));
  checkThreadInvariance();
  return lu_testing::summarize("test_header_only_metis_parallel");
}
