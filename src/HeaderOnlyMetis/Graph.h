// Minimal graph_t equivalent (libmetis/struct.h) for the header-only METIS
// port. Deliberately pared down to what the in-scope nested-dissection path
// (compress -> coarsen -> initial separator -> separator refinement -> node
// bisection driver, all under default options) actually touches:
//   - multi-constraint fields (ncon>1) are dropped -- Eigen's MetisOrdering
//     always calls with ncon=1, so vwgt/tvwgt/invtvwgt are scalars per
//     vertex/graph here, not per-constraint arrays.
//   - vsize/ckrinfo/vkrinfo (k-way volume/cut refinement) are dropped -- the
//     ND path always has objtype=METIS_OBJTYPE_NODE (options.c's OMETIS
//     defaults), so dovsize is always false and k-way refinement never runs.
//   - free_xadj/free_vwgt/... ownership-tracking flags are dropped: they
//     exist only to drive GKlib's manual gk_free() bookkeeping, which
//     std::vector's RAII replaces outright. This changes WHEN memory is
//     freed, never what any algorithm decides, so it carries none of the
//     bit-identical risk the Phase 2 plan reserves for Stage B.
//   - gID/ondisk/droppedewgt are dropped: out-of-core (graph_WriteToDisk) and
//     dropedges are both off by default (ctrl->ondisk=0, ctrl->dropedges=0 in
//     options.c) and are provably unreachable on the default-options path.
//
// Extend this struct as later modules need more of graph_t's fields --
// deliberately starting minimal rather than front-loading every field before
// any module has proven it needs one, per the module-by-module plan.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_GRAPH_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_GRAPH_H

#include <memory>
#include <vector>

namespace header_only_metis {

template <typename IndexT, typename RealT>
struct Graph {
  IndexT nvtxs = 0;
  IndexT nedges = 0;

  std::vector<IndexT> xadj;    // size nvtxs+1
  std::vector<IndexT> vwgt;    // size nvtxs (ncon=1)
  std::vector<IndexT> adjncy;  // size nedges
  std::vector<IndexT> adjwgt;  // size nedges

  IndexT tvwgt = 0;     // sum of vwgt (ncon=1)
  RealT invtvwgt = 0;   // 1/tvwgt (or 1 if tvwgt==0)

  std::vector<IndexT> cmap;   // coarsening map, size nvtxs (allocated by CoarsenGraph)
  std::vector<IndexT> label;  // original vertex labels, size nvtxs

  // Partition state (2-way bisection / node separator), populated by
  // InitialSeparator.h / SeparatorRefinement.h once they exist.
  IndexT mincut = 0;
  std::vector<IndexT> where;  // size nvtxs, values in {0,1,2} for a node separator
  std::vector<IndexT> pwgts;  // size nparts (3 for a node separator)
  IndexT nbnd = 0;
  std::vector<IndexT> bndptr;  // size nvtxs
  std::vector<IndexT> bndind;  // size nvtxs

  std::vector<IndexT> id;  // bisection (edge) refinement: internal degree
  std::vector<IndexT> ed;  // bisection (edge) refinement: external degree

  struct NRInfo {
    IndexT edegrees[2] = {0, 0};
  };
  std::vector<NRInfo> nrinfo;  // node refinement info, size nvtxs

  // Coarsening-level chain. coarser owns the next level; finer is a
  // non-owning back-reference, matching graph_t's doubly-linked coarser/
  // finer pointers but with RAII ownership instead of manual free.
  std::unique_ptr<Graph> coarser;
  Graph* finer = nullptr;

  // Matches SetupGraph_tvwgt (graph.c), specialized to ncon=1. The reference
  // computes `1.0/(tvwgt>0?tvwgt:1)` with `1.0` a double literal, so the
  // division happens in double precision and is narrowed to RealT (float, by
  // default) only on the final store -- not the other way around.
  void setupTvwgt() {
    tvwgt = IndexT(0);
    for (IndexT v : vwgt) tvwgt += v;
    invtvwgt = static_cast<RealT>(1.0 / (tvwgt > 0 ? static_cast<double>(tvwgt) : 1.0));
  }

  // Matches SetupGraph_label (graph.c): label[i] = i.
  void setupLabel() {
    label.resize(static_cast<std::size_t>(nvtxs));
    for (IndexT i = 0; i < nvtxs; ++i) label[static_cast<std::size_t>(i)] = i;
  }

  // Matches FreeRData (graph.c): drops the partition/refinement fields
  // (where/pwgts/id/ed/bndptr/bndind/nrinfo), leaving the graph's structural
  // fields (xadj/adjncy/vwgt/...) untouched. Every allocator downstream
  // (allocate2WayNodePartitionMemory, ...) unconditionally re-.assign()s
  // these on the next use, so this exists to match the reference's memory
  // lifecycle between repeated bisection trials on the same graph object
  // (MlevelNodeBisectionL2's 5-run loop) -- not because a stale value here is
  // ever read before being overwritten.
  void freeRData() {
    where.clear();
    pwgts.clear();
    bndptr.clear();
    bndind.clear();
    id.clear();
    ed.clear();
    nrinfo.clear();
  }
};

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_GRAPH_H
