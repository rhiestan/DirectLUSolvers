// Level-synchronous, parallel nested dissection.
//
// This is NOT a bit-identical path, and cannot be: see "Why the output differs"
// below. It is, however, fully DETERMINISTIC -- the ordering depends only on
// the input graph, not on the thread count or the scheduling -- which is what
// keeps fill-regression baselines meaningful and gives the path a testable
// oracle (its own serial run).
//
// SHAPE
// -----
// mlevelNestedDissection recurses depth-first. That cannot be parallelized by
// simply spawning the two subtrees, because the Executor backends this project
// ships are not re-entrant: StdThreadExecutor::dispatch keeps a single shared
// body pointer, so a parallelFor issued from inside a worker deadlocks (see
// Executor.h). Instead the recursion is turned inside out into an explicit
// frontier, and each level of the dissection tree is one flat dispatch --
// structurally the same trick as SupernodalLU's level-set schedule over the
// elimination tree.
//
// Everything inside a single bisection stays serial, and deliberately so:
// coarsening levels are sequentially dependent, matching is order-dependent,
// and FM refinement is inherently sequential (each move rewrites its
// neighbours' gains). All the available parallelism is across tree nodes.
//
// A SECOND PARALLEL AXIS WAS TRIED AND DID NOT PAY
// ------------------------------------------------
// MlevelNodeBisectionL2 runs five INDEPENDENT speculative bisection trials on
// big (>=5000 vertex) nodes, and the obvious next move is to flatten those
// into this dispatch -- (node, trial) pairs in one parallelFor -- so that the
// top of the tree, where the frontier is only 1, 2, 4 nodes wide, still fills
// the machine. It was implemented and measured, and it is NOT here because it
// bought nothing:
//
//   laoss_1, 32 threads, best of 3: 224ms flattened vs 225ms as written
//   full corpus, 8 threads:         579ms flattened vs 586ms as written
//
// both inside the run-to-run noise, against ~120 extra lines of scheduling,
// five clones of every big node's coarsened graph, and two extra barriers per
// level. The cost model said it should have helped: work is roughly flat per
// tree level (laoss_1, serial, ms per level: 96, 72, 89, 85, 97, 65, 71, 46,
// 40, 60, 21), and within a big node the split is coarsen ~39% / trials ~39%
// / uncoarsen ~23%, so compressing the trial slice five-way at level 0 should
// have cut ~11% overall.
//
// It did not, and the likely reason is that this path stops being lane-bound
// well before it runs out of parallelism: 16 -> 32 threads already only moved
// 233ms -> 228ms. Past roughly 8-16 threads the limit looks like memory
// bandwidth and pointer-chasing latency, not idle cores -- which is also why
// adding a whole new axis of concurrency changed nothing. Anyone revisiting
// this should measure that ceiling FIRST (e.g. VTune memory-access analysis on
// the parallel path) rather than adding more parallelism to a bound that is
// not lanes.
//
// WHY THE OUTPUT DIFFERS FROM METIS
// ---------------------------------
// The reference draws every random number of the whole recursion from one
// stream, so the ORDER in which tree nodes are visited is itself an input to
// the result. A frontier visits nodes breadth-first, so a shared stream would
// hand each node different numbers than the depth-first walk did -- and under
// threads the interleaving would not even be repeatable. Rather than pretend
// otherwise, each node here seeds its own generator from its POSITION in the
// tree (depth and lastvtx, both fixed by the input). A node's result is then a
// function of its subgraph and its position only, so it does not matter when,
// where, or in what order it runs.
//
// WHAT IS SAFE TO SHARE
// ---------------------
// order[] is written concurrently without synchronization, and that is sound:
// each subtree owns a disjoint half-open range of output positions AND a
// disjoint set of original vertex labels, so no two tasks address the same
// element (see bisectSplitAndOrder -- it writes order[label[...]]).
//
// MEMORY
// ------
// Breadth-first holds one whole level's subgraphs at once, where depth-first
// holds one root-to-leaf path. Peak is O(n) per level rather than O(n) total,
// so this trades memory for parallelism; on the largest matrices here that is
// a real increase, not a rounding error.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_NESTED_DISSECTION_PARALLEL_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_NESTED_DISSECTION_PARALLEL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "Ctrl.h"
#include "Executor.h"
#include "Graph.h"
#include "NestedDissection.h"
#include "Random.h"

namespace header_only_metis {

// A node's seed is a pure function of its position in the dissection tree.
// (depth, lastvtx) identifies a node uniquely: lastvtx is the end of the
// output range the node owns, and no two live nodes own the same range.
// splitmix64's finalizer decorrelates the two small integers, which matters
// because MT19937-64 seeded with near-identical values produces correlated
// early output -- sibling subtrees would otherwise make similar random
// choices.
inline std::uint64_t seedForSubtree(std::uint64_t depth, std::uint64_t lastvtx) {
  std::uint64_t z = 0x243F6A8885A308D3ULL + (depth * 0x9E3779B97F4A7C15ULL) + (lastvtx * 0xC2B2AE3D27D4EB4FULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

template <typename IndexT, typename RealT>
struct NdTask {
  std::unique_ptr<Graph<IndexT, RealT>> graph;
  IndexT lastvtx = 0;
  IndexT depth = 0;
};

// Orders `root` into order[], using `exec` across the nodes of each tree level.
// `proto` supplies the algorithm settings; its rng and workspace are not used.
template <typename IndexT, typename RealT, typename Executor>
void nestedDissectionFrontier(const Ctrl<IndexT, RealT>& proto,
                              std::unique_ptr<Graph<IndexT, RealT>> root, IndexT* order, IndexT lastvtx,
                              const Executor& exec) {
  std::vector<NdTask<IndexT, RealT>> frontier, next;
  frontier.push_back({std::move(root), lastvtx, IndexT(0)});

  std::vector<NdChildren<IndexT, RealT>> children;

  while (!frontier.empty()) {
    // Pre-sized so every task writes only its own slot: no shared container,
    // hence no locking and no order dependence.
    children.clear();
    children.resize(frontier.size());

    exec.parallelFor(0, static_cast<std::ptrdiff_t>(frontier.size()), [&](std::ptrdiff_t t) {
      // One Ctrl per worker thread, kept alive across tasks so its Workspace
      // buffers stay warm. Reusing it cannot perturb the result: every field
      // that affects the outcome is either refreshed from `proto` here or set
      // during the bisection itself, and the workspace is write-before-read by
      // construction (Workspace.h).
      thread_local Ctrl<IndexT, RealT> ctrl;
      ctrl.copySettingsFrom(proto);

      NdTask<IndexT, RealT>& task = frontier[static_cast<std::size_t>(t)];
      randSeed64(ctrl.rng, seedForSubtree(static_cast<std::uint64_t>(task.depth),
                                          static_cast<std::uint64_t>(task.lastvtx)));
      children[static_cast<std::size_t>(t)] =
          bisectSplitAndOrder(ctrl, std::move(task.graph), order, task.lastvtx);
    });

    // Rebuilt on one thread in a fixed order, so the next frontier -- and
    // therefore every subsequent seed -- is independent of how the level above
    // was scheduled.
    next.clear();
    for (std::size_t t = 0; t < children.size(); ++t) {
      const IndexT childDepth = frontier[t].depth + 1;
      for (int side = 0; side < 2; ++side) {
        if (!children[t].graph[side]) continue;  // leaf: already MMD-ordered
        next.push_back({std::move(children[t].graph[side]), children[t].lastvtx[side], childDepth});
      }
    }
    frontier.swap(next);
  }
}

// Parallel counterpart of nodeND(). Same signature plus an Executor; same
// compress/uncompress bookkeeping (shared via nodeNDWithDriver); different
// traversal and RNG seeding, hence a different -- but deterministic --
// ordering. With SerialExecutor this is the single-threaded reference the
// threaded runs are checked against.
template <typename IndexT, typename RealT, typename Executor>
int nodeNDParallel(IndexT nvtxs, const IndexT* xadj, const IndexT* adjncy, const IndexT* vwgt, IndexT* perm,
                   IndexT* iperm, const Executor& exec) {
  return nodeNDWithDriver<IndexT, RealT>(
      nvtxs, xadj, adjncy, vwgt, perm, iperm,
      [&exec](Ctrl<IndexT, RealT>& ctrl, std::unique_ptr<Graph<IndexT, RealT>> graph, IndexT* order,
              IndexT lastvtx) { nestedDissectionFrontier(ctrl, std::move(graph), order, lastvtx, exec); });
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_NESTED_DISSECTION_PARALLEL_H
