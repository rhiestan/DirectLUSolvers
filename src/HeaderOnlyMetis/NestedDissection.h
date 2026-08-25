// Faithful port of libmetis/ometis.c: the top-level nested-dissection driver
// that ties every other module together into the actual METIS_NodeND
// equivalent -- MlevelNestedDissection, MlevelNodeBisectionMultiple/L2/L1,
// SplitGraphOrder, MMDOrder, and the nodeND() entry point matching
// METIS_NodeND's array signature.
//
// Not ported (confirmed unreachable/dead on the default-options path, as
// established across the whole project):
//   - numflag handling (Change2CNumbering/Change2FNumberingOrder): always 0,
//     Eigen never sets 1-based numbering.
//   - PruneGraph: ctrl->pfactor defaults to 0.
//   - MlevelNestedDissectionCC / SplitGraphOrderCC / FindSepInducedComponents:
//     ctrl->ccorder defaults to 0.
//   - FreeRData calls scattered through MlevelNodeBisectionMultiple/L2 (used
//     to discard a non-winning trial's refinement data before retrying):
//     std::vector fields just get overwritten by the next call, no explicit
//     free needed.
//
// Ownership model: graph_t's are raw-pointer, manually gk_free'd in the
// reference. This port uses std::unique_ptr<Graph> at every point the
// reference calls FreeGraph explicitly (MlevelNestedDissection after
// splitting, Project2WayNodePartition/project2WayNodePartition after
// uncoarsening past a level -- see SeparatorRefinement.h's
// `graph->coarser.reset()`), so destruction happens automatically via RAII
// at the equivalent point in the control flow instead of an explicit call.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_NESTED_DISSECTION_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_NESTED_DISSECTION_H

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "Coarsen.h"
#include "Compress.h"
#include "Ctrl.h"
#include "Graph.h"
#include "InitialSeparator.h"
#include "MinimumDegree.h"
#include "SeparatorRefinement.h"

namespace header_only_metis {

constexpr int kMMDSwitch = 120;      // libmetis/ometis.c: MMDSWITCH
constexpr int kLargeNiparts = 7;     // libmetis/defs.h: LARGENIPARTS

template <typename IndexT, typename RealT>
void mlevelNodeBisectionMultiple(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph);
template <typename IndexT, typename RealT>
void mlevelNodeBisectionL2(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niparts);
template <typename IndexT, typename RealT>
void mlevelNodeBisectionL1(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niparts);
template <typename IndexT, typename RealT>
void splitGraphOrder(Graph<IndexT, RealT>* graph, std::unique_ptr<Graph<IndexT, RealT>>& lgraphOut,
                     std::unique_ptr<Graph<IndexT, RealT>>& rgraphOut);
template <typename IndexT, typename RealT>
void mmdOrder(Graph<IndexT, RealT>* graph, IndexT* order, IndexT lastvtx);

// This is the driver for the recursive tri-section of a graph into the left,
// separator, and right partitions. The graphs corresponding to the left and
// right parts are further tri-sected in a recursive fashion. The nodes in
// the separator are ordered at the end of the left & right nodes.
//
// Takes ownership of `graph` (matching FreeGraph(&graph) after splitting).
template <typename IndexT, typename RealT>
void mlevelNestedDissection(Ctrl<IndexT, RealT>& ctrl, std::unique_ptr<Graph<IndexT, RealT>> graph,
                            IndexT* order, IndexT lastvtx) {
  mlevelNodeBisectionMultiple(ctrl, graph.get());

  /* Order the nodes in the separator */
  const IndexT nbnd = graph->nbnd;
  const std::vector<IndexT>& bndind = graph->bndind;
  const std::vector<IndexT>& label = graph->label;
  for (IndexT i = 0; i < nbnd; i++)
    order[static_cast<std::size_t>(label[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])])] =
        --lastvtx;

  std::unique_ptr<Graph<IndexT, RealT>> lgraph, rgraph;
  splitGraphOrder(graph.get(), lgraph, rgraph);

  /* Free the memory of the top level graph */
  graph.reset();

  // Recurse on lgraph first, as its lastvtx depends on rgraph->nvtxs, which
  // will not be defined once rgraph is moved-from below.
  const IndexT rgraphNvtxs = rgraph->nvtxs;
  if (lgraph->nvtxs > IndexT(kMMDSwitch) && lgraph->nedges > 0) {
    mlevelNestedDissection(ctrl, std::move(lgraph), order, lastvtx - rgraphNvtxs);
  } else {
    mmdOrder(lgraph.get(), order, lastvtx - rgraphNvtxs);
  }
  if (rgraph->nvtxs > IndexT(kMMDSwitch) && rgraph->nedges > 0) {
    mlevelNestedDissection(ctrl, std::move(rgraph), order, lastvtx);
  } else {
    mmdOrder(rgraph.get(), order, lastvtx);
  }
}

// This function performs multilevel node bisection (i.e., tri-section). It
// performs multiple bisections and selects the best.
template <typename IndexT, typename RealT>
void mlevelNodeBisectionMultiple(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph) {
  if (ctrl.nseps == 1 ||
      graph->nvtxs < (ctrl.compress ? IndexT(1000) : IndexT(2000))) {
    mlevelNodeBisectionL2(ctrl, graph, IndexT(kLargeNiparts));
    return;
  }

  std::vector<IndexT> bestwhere(static_cast<std::size_t>(graph->nvtxs));

  IndexT mincut = graph->tvwgt;
  for (IndexT i = 0; i < ctrl.nseps; i++) {
    mlevelNodeBisectionL2(ctrl, graph, IndexT(kLargeNiparts));

    if (i == 0 || graph->mincut < mincut) {
      mincut = graph->mincut;
      if (i < ctrl.nseps - 1) bestwhere = graph->where;
    }

    if (mincut == 0) break;

    if (i < ctrl.nseps - 1) graph->freeRData();
  }

  if (mincut != graph->mincut) {
    graph->where = bestwhere;
    compute2WayNodePartitionParams(graph);
  }
}

// This function performs multilevel node bisection (i.e., tri-section).
template <typename IndexT, typename RealT>
void mlevelNodeBisectionL2(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niparts) {
  if (graph->nvtxs < 5000) {
    mlevelNodeBisectionL1(ctrl, graph, niparts);
    return;
  }

  ctrl.CoarsenTo = std::max<IndexT>(100, graph->nvtxs / 30);

  Graph<IndexT, RealT>* cgraph = coarsenGraphNlevels(ctrl, graph, IndexT(4));

  std::vector<IndexT> bestwhere(static_cast<std::size_t>(cgraph->nvtxs));

  IndexT mincut = graph->tvwgt;
  constexpr int nruns = 5;
  for (int i = 0; i < nruns; i++) {
    mlevelNodeBisectionL1(ctrl, cgraph, static_cast<IndexT>(0.7 * static_cast<double>(niparts)));

    if (i == 0 || cgraph->mincut < mincut) {
      mincut = cgraph->mincut;
      if (i < nruns - 1) bestwhere = cgraph->where;
    }

    if (mincut == 0) break;

    if (i < nruns - 1) cgraph->freeRData();
  }

  if (mincut != cgraph->mincut) cgraph->where = bestwhere;

  refine2WayNode(ctrl, graph, cgraph);
}

// The top-level routine of the actual multilevel node bisection.
template <typename IndexT, typename RealT>
void mlevelNodeBisectionL1(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niparts) {
  ctrl.CoarsenTo = graph->nvtxs / 8;
  if (ctrl.CoarsenTo > 100)
    ctrl.CoarsenTo = 100;
  else if (ctrl.CoarsenTo < 40)
    ctrl.CoarsenTo = 40;

  Graph<IndexT, RealT>* cgraph = coarsenGraph(ctrl, graph);

  niparts = std::max<IndexT>(1, (cgraph->nvtxs <= ctrl.CoarsenTo ? niparts / 2 : niparts));
  initSeparator(ctrl, cgraph, niparts);

  refine2WayNode(ctrl, graph, cgraph);
}

// This function takes a graph and a tri-section (left, right, separator) and
// splits it into two graphs.
//
// This function relies on the fact that adjwgt is all equal to 1.
template <typename IndexT, typename RealT>
void splitGraphOrder(Graph<IndexT, RealT>* graph, std::unique_ptr<Graph<IndexT, RealT>>& lgraphOut,
                     std::unique_ptr<Graph<IndexT, RealT>>& rgraphOut) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& label = graph->label;
  const std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& bndptr = graph->bndptr;  // repurposed below as a scratch "near separator" marker
  const std::vector<IndexT>& bndind = graph->bndind;

  std::vector<IndexT> rename(static_cast<std::size_t>(nvtxs));

  IndexT snvtxs[3] = {0, 0, 0};
  IndexT snedges[3] = {0, 0, 0};
  for (IndexT i = 0; i < nvtxs; i++) {
    const IndexT k = where[static_cast<std::size_t>(i)];
    rename[static_cast<std::size_t>(i)] = snvtxs[k]++;
    snedges[k] += xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)];
  }

  auto lgraph = std::make_unique<Graph<IndexT, RealT>>();
  lgraph->nvtxs = snvtxs[0];
  lgraph->xadj.assign(static_cast<std::size_t>(snvtxs[0]) + 1, IndexT(0));
  lgraph->vwgt.assign(static_cast<std::size_t>(snvtxs[0]), IndexT(0));
  lgraph->label.assign(static_cast<std::size_t>(snvtxs[0]), IndexT(0));
  lgraph->adjncy.reserve(static_cast<std::size_t>(snedges[0]));

  auto rgraph = std::make_unique<Graph<IndexT, RealT>>();
  rgraph->nvtxs = snvtxs[1];
  rgraph->xadj.assign(static_cast<std::size_t>(snvtxs[1]) + 1, IndexT(0));
  rgraph->vwgt.assign(static_cast<std::size_t>(snvtxs[1]), IndexT(0));
  rgraph->label.assign(static_cast<std::size_t>(snvtxs[1]), IndexT(0));
  rgraph->adjncy.reserve(static_cast<std::size_t>(snedges[1]));

  Graph<IndexT, RealT>* sgraph[2] = {lgraph.get(), rgraph.get()};

  /* Go and use bndptr to also mark the boundary nodes in the two partitions */
  for (IndexT ii = 0; ii < graph->nbnd; ii++) {
    const IndexT i = bndind[static_cast<std::size_t>(ii)];
    for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++)
      bndptr[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])] = 1;
  }

  IndexT curNvtxs[2] = {0, 0};
  for (IndexT i = 0; i < nvtxs; i++) {
    const IndexT mypart = where[static_cast<std::size_t>(i)];
    if (mypart == 2) continue;

    const IndexT istart = xadj[static_cast<std::size_t>(i)];
    const IndexT iend = xadj[static_cast<std::size_t>(i) + 1];
    Graph<IndexT, RealT>* sg = sgraph[mypart];

    if (bndptr[static_cast<std::size_t>(i)] == -1) {
      /* This is an interior vertex: every neighbor is provably in the same
         partition (nothing marked it as touching the separator), so the
         reference's fast path copies the whole adjacency list unchecked.
         Appending unconditionally is exactly equivalent. */
      for (IndexT j = istart; j < iend; j++)
        sg->adjncy.push_back(rename[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])]);
    } else {
      for (IndexT j = istart; j < iend; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        if (where[static_cast<std::size_t>(k)] == mypart)
          sg->adjncy.push_back(rename[static_cast<std::size_t>(k)]);
      }
    }

    sg->vwgt[static_cast<std::size_t>(curNvtxs[mypart])] = vwgt[static_cast<std::size_t>(i)];
    sg->label[static_cast<std::size_t>(curNvtxs[mypart])] = label[static_cast<std::size_t>(i)];
    sg->xadj[static_cast<std::size_t>(++curNvtxs[mypart])] = static_cast<IndexT>(sg->adjncy.size());
  }

  /* adjwgt is all-1 throughout the ND path (see file header) */
  lgraph->adjwgt.assign(lgraph->adjncy.size(), IndexT(1));
  rgraph->adjwgt.assign(rgraph->adjncy.size(), IndexT(1));

  lgraph->nedges = static_cast<IndexT>(lgraph->adjncy.size());
  rgraph->nedges = static_cast<IndexT>(rgraph->adjncy.size());

  lgraph->setupTvwgt();
  rgraph->setupTvwgt();

  lgraphOut = std::move(lgraph);
  rgraphOut = std::move(rgraph);
}

// This function uses MMD to order the graph. The vertices are numbered from
// lastvtx downwards.
template <typename IndexT, typename RealT>
void mmdOrder(Graph<IndexT, RealT>* graph, IndexT* order, IndexT lastvtx) {
  const IndexT nvtxs = graph->nvtxs;
  std::vector<IndexT>& xadj = graph->xadj;
  std::vector<IndexT>& adjncy = graph->adjncy;

  /* Relabel the vertices so that it starts from 1 */
  const IndexT k = xadj[static_cast<std::size_t>(nvtxs)];
  for (IndexT i = 0; i < k; i++) adjncy[static_cast<std::size_t>(i)]++;
  for (IndexT i = 0; i <= nvtxs; i++) xadj[static_cast<std::size_t>(i)]++;

  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs) + 5);
  std::vector<IndexT> iperm(static_cast<std::size_t>(nvtxs) + 5);
  std::vector<IndexT> head(static_cast<std::size_t>(nvtxs) + 5);
  std::vector<IndexT> qsize(static_cast<std::size_t>(nvtxs) + 5);
  std::vector<IndexT> list(static_cast<std::size_t>(nvtxs) + 5);
  std::vector<IndexT> marker(static_cast<std::size_t>(nvtxs) + 5);
  IndexT nofsub = 0;

  genmmd<IndexT>(nvtxs, xadj.data(), adjncy.data(), iperm.data(), perm.data(), IndexT(1), head.data(),
                qsize.data(), list.data(), marker.data(), std::numeric_limits<IndexT>::max(), &nofsub);

  const std::vector<IndexT>& label = graph->label;
  const IndexT firstvtx = lastvtx - nvtxs;
  for (IndexT i = 0; i < nvtxs; i++)
    order[static_cast<std::size_t>(label[static_cast<std::size_t>(i)])] = firstvtx + iperm[static_cast<std::size_t>(i)] - 1;

  // No need to restore xadj/adjncy to 0-based: this graph is destroyed
  // immediately after mmdOrder returns (RAII, matching FreeGraph right after
  // MMDOrder in the reference).
}

// Top-level entry point, matching METIS_NodeND's array signature for the
// default-options path: numflag=0, vwgt optional (nullptr -> uniform 1,
// matching SetupGraph's own NULL-vwgt handling), adjwgt always uniform 1
// (Eigen's call convention -- see the project-wide scope notes). Returns 1
// (METIS_OK) on success; the reference's SIGERR/out-of-memory error paths
// are not modeled since they are unreachable for well-formed input in this
// port's scope.
template <typename IndexT, typename RealT>
int nodeND(IndexT nvtxs, const IndexT* xadj, const IndexT* adjncy, const IndexT* vwgt, IndexT* perm,
          IndexT* iperm) {
  Ctrl<IndexT, RealT> ctrl;

  // Matches SetupCtrl's InitRandom(ctrl->seed) (options.c), which METIS_NodeND
  // calls on EVERY invocation via SetupCtrl -- ctrl->seed defaults to -1
  // (options=NULL, Eigen's call convention), and InitRandom maps -1 to the
  // fixed constant 4321 (libmetis/util.c), not time-based. Without this, the
  // port's RNG stream carries over whatever state a PRIOR nodeND() call (or
  // any other test) left it in instead of resetting to the same known point
  // the reference resets to on every call -- individual sub-functions tested
  // in isolation (which always seed explicitly before calling) never expose
  // this, only the full top-level entry point does.
  randSeed<IndexT>(IndexT(4321));

  std::vector<IndexT> cptr, cind;
  std::unique_ptr<Graph<IndexT, RealT>> graph;
  IndexT nnvtxs = nvtxs;

  if (ctrl.compress) {
    cptr.assign(static_cast<std::size_t>(nvtxs) + 1, IndexT(0));
    cind.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
    graph = compressGraph<IndexT, RealT>(nvtxs, xadj, adjncy, vwgt, cptr.data(), cind.data());
    if (!graph) {
      ctrl.compress = false;
    } else {
      nnvtxs = graph->nvtxs;
      const double cfactor = 1.0 * static_cast<double>(nvtxs) / static_cast<double>(nnvtxs);
      if (cfactor > 1.5 && ctrl.nseps == 1) ctrl.nseps = 2;
    }
  }

  if (!ctrl.compress) {
    graph = std::make_unique<Graph<IndexT, RealT>>();
    graph->nvtxs = nvtxs;
    graph->xadj.assign(xadj, xadj + nvtxs + 1);
    const IndexT nedges = xadj[nvtxs];
    graph->adjncy.assign(adjncy, adjncy + nedges);
    graph->nedges = nedges;
    if (vwgt != nullptr)
      graph->vwgt.assign(vwgt, vwgt + nvtxs);
    else
      graph->vwgt.assign(static_cast<std::size_t>(nvtxs), IndexT(1));
    graph->adjwgt.assign(static_cast<std::size_t>(nedges), IndexT(1));
    graph->setupTvwgt();
    graph->setupLabel();
    nnvtxs = nvtxs;
  }

  mlevelNestedDissection(ctrl, std::move(graph), iperm, nnvtxs);

  if (ctrl.compress) {
    /* Uncompress the ordering. perm[] is used as scratch here, matching the
       reference exactly (ometis.c) -- it is fully overwritten by the final
       loop below regardless, so this is safe. */
    for (IndexT i = 0; i < nnvtxs; i++) perm[iperm[i]] = i;
    IndexT l = 0;
    for (IndexT ii = 0; ii < nnvtxs; ii++) {
      const IndexT i = perm[ii];
      for (IndexT j = cptr[static_cast<std::size_t>(i)]; j < cptr[static_cast<std::size_t>(i) + 1]; j++)
        iperm[cind[static_cast<std::size_t>(j)]] = l++;
    }
  }

  for (IndexT i = 0; i < nvtxs; i++) perm[iperm[i]] = i;

  return 1;  // METIS_OK
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_NESTED_DISSECTION_H
