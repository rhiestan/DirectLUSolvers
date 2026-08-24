// Faithful port of the coarsest-graph initial bisection + conversion to a
// node separator: libmetis/initpart.c (RandomBisection, GrowBisection,
// InitSeparator), libmetis/separator.c (ConstructSeparator),
// libmetis/refine.c (Allocate2WayPartitionMemory, Compute2WayPartitionParams
// -- the EDGE-cut variants, distinct from SeparatorRefinement.h's node-based
// namesakes), libmetis/balance.c (Balance2Way, Bnd2WayBalance,
// General2WayBalance), libmetis/fm.c (FM_2WayRefine/FM_2WayCutRefine),
// libmetis/options.c (Setup2WayBalMultipliers), libmetis/mcutil.c
// (ComputeLoadImbalanceDiff).
//
// Not ported (confirmed unreachable: InitSeparator's ctrl->iptype defaults
// to METIS_IPTYPE_EDGE, options.c): Init2WayPartition (a DIFFERENT function,
// used only by kmetis.c/pmetis.c, not ometis.c's InitSeparator -- their
// shared callee RandomBisection is still in scope), McRandomBisection/
// McGrowBisection (ncon>1), GrowBisectionNode/GrowBisectionNode2 (only
// reachable via InitSeparator's METIS_IPTYPE_NODE case), FM_Mc2WayCutRefine/
// McGeneral2WayBalance (ncon>1), SelectQueue (ncon>1 only caller).
//
// Depends on SeparatorRefinement.h for bndInsert/bndDelete and
// allocate2WayNodePartitionMemory/compute2WayNodePartitionParams/
// fm2WayNodeRefine2Sided/fm2WayNodeRefine1Sided, which ConstructSeparator
// calls directly to finish converting the edge bisection into a node
// separator.
//
// Two float-vs-double precision traps, both load-bearing for bit-identical
// output and easy to get backwards:
//   - zeromaxpwgt/onemaxpwgt = ubfactor*tvwgt*ntpwgt: no double literal
//     anywhere in the reference expression, so this stays in RealT (float,
//     by default) precision throughout.
//   - oneminpwgt = (1.0/ubfactor)*tvwgt*ntpwgt: the `1.0` IS a double
//     literal in the reference, so `1.0/ubfactor` promotes to double and the
//     WHOLE expression is double precision, not RealT -- opposite of the
//     other two despite looking almost identical.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_INITIAL_SEPARATOR_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_INITIAL_SEPARATOR_H

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "Ctrl.h"
#include "Graph.h"
#include "PQueue.h"
#include "Random.h"
#include "SeparatorRefinement.h"

namespace header_only_metis {

// Matches Setup2WayBalMultipliers (options.c), ncon=1: pijbm[i] =
// invtvwgt/tpwgts[i].
template <typename IndexT, typename RealT>
void setup2WayBalMultipliers(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT tpwgts[2]) {
  for (int i = 0; i < 2; i++) ctrl.pijbm[i] = graph->invtvwgt / tpwgts[i];
}

// Matches ComputeLoadImbalanceDiff (mcutil.c), ncon=1, nparts=2.
template <typename IndexT, typename RealT>
RealT computeLoadImbalanceDiff(Graph<IndexT, RealT>* graph, const RealT pijbm[2], RealT ubfactor) {
  RealT max = RealT(-1.0);
  for (int j = 0; j < 2; j++) {
    const RealT cur = static_cast<RealT>(graph->pwgts[static_cast<std::size_t>(j)]) * pijbm[j] - ubfactor;
    if (cur > max) max = cur;
  }
  return max;
}

// Matches Allocate2WayPartitionMemory (refine.c).
template <typename IndexT, typename RealT>
void allocate2WayPartitionMemory(Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  graph->pwgts.assign(2, IndexT(0));
  graph->where.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->bndptr.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->bndind.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->id.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->ed.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
}

// Matches Compute2WayPartitionParams (refine.c), ncon=1: the EDGE-cut
// variant (internal/external degree id/ed), distinct from
// SeparatorRefinement.h's compute2WayNodePartitionParams (edegrees/nrinfo).
template <typename IndexT, typename RealT>
void compute2WayPartitionParams(Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& adjwgt = graph->adjwgt;
  const std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& id = graph->id;
  std::vector<IndexT>& ed = graph->ed;

  graph->pwgts.assign(2, IndexT(0));
  std::vector<IndexT>& pwgts = graph->pwgts;
  graph->bndptr.assign(static_cast<std::size_t>(nvtxs), IndexT(-1));
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& bndind = graph->bndind;

  for (IndexT i = 0; i < nvtxs; i++) pwgts[static_cast<std::size_t>(where[static_cast<std::size_t>(i)])] += vwgt[static_cast<std::size_t>(i)];

  IndexT nbnd = 0;
  IndexT mincut = 0;
  for (IndexT i = 0; i < nvtxs; i++) {
    const IndexT istart = xadj[static_cast<std::size_t>(i)];
    const IndexT iend = xadj[static_cast<std::size_t>(i) + 1];
    const IndexT me = where[static_cast<std::size_t>(i)];
    IndexT tid = 0, ted = 0;

    for (IndexT j = istart; j < iend; j++) {
      if (me == where[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])])
        tid += adjwgt[static_cast<std::size_t>(j)];
      else
        ted += adjwgt[static_cast<std::size_t>(j)];
    }
    id[static_cast<std::size_t>(i)] = tid;
    ed[static_cast<std::size_t>(i)] = ted;

    if (ted > 0 || istart == iend) {
      bndInsert(nbnd, bndind, bndptr, i);
      mincut += ted;
    }
  }

  graph->mincut = mincut / 2;
  graph->nbnd = nbnd;
}

// This function balances two partitions by moving boundary nodes from the
// domain that is overweight to the one that is underweight.
template <typename IndexT, typename RealT>
void bnd2WayBalance(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2]) {
  (void)ctrl;
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& adjwgt = graph->adjwgt;
  std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& id = graph->id;
  std::vector<IndexT>& ed = graph->ed;
  std::vector<IndexT>& pwgts = graph->pwgts;
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& bndind = graph->bndind;

  std::vector<IndexT> moved(static_cast<std::size_t>(nvtxs), IndexT(-1));
  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));

  const IndexT tpwgts0 = static_cast<IndexT>(static_cast<RealT>(graph->tvwgt) * ntpwgts[0]);
  const IndexT tpwgts1 = graph->tvwgt - tpwgts0;
  const IndexT mindiff = std::abs(tpwgts0 - pwgts[0]);
  const IndexT from = (pwgts[0] < tpwgts0 ? 1 : 0);
  const IndexT to = (from + 1) % 2;
  const IndexT tpwgts[2] = {tpwgts0, tpwgts1};

  PQueue<IndexT, IndexT> queue(static_cast<std::size_t>(nvtxs));

  IndexT nbnd = graph->nbnd;
  randArrayPermute<IndexT>(nbnd, perm.data(), nbnd / 5, 1);
  for (IndexT ii = 0; ii < nbnd; ii++) {
    const IndexT i = perm[static_cast<std::size_t>(ii)];
    if (where[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])] == from &&
        vwgt[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])] <= mindiff)
      queue.insert(bndind[static_cast<std::size_t>(i)],
                   ed[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])] -
                       id[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])]);
  }

  IndexT mincut = graph->mincut;
  for (IndexT nswaps = 0; nswaps < nvtxs; nswaps++) {
    const IndexT higain = queue.getTop();
    if (higain == -1) break;

    if (pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(higain)] > tpwgts[to]) break;

    mincut -= (ed[static_cast<std::size_t>(higain)] - id[static_cast<std::size_t>(higain)]);
    pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
    pwgts[static_cast<std::size_t>(from)] -= vwgt[static_cast<std::size_t>(higain)];

    where[static_cast<std::size_t>(higain)] = to;
    moved[static_cast<std::size_t>(higain)] = nswaps;

    std::swap(id[static_cast<std::size_t>(higain)], ed[static_cast<std::size_t>(higain)]);
    if (ed[static_cast<std::size_t>(higain)] == 0 &&
        xadj[static_cast<std::size_t>(higain)] < xadj[static_cast<std::size_t>(higain) + 1])
      bndDelete(nbnd, bndind, bndptr, higain);

    for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
      const IndexT k = adjncy[static_cast<std::size_t>(j)];
      const IndexT kwgt = (to == where[static_cast<std::size_t>(k)] ? adjwgt[static_cast<std::size_t>(j)]
                                                                     : -adjwgt[static_cast<std::size_t>(j)]);
      id[static_cast<std::size_t>(k)] += kwgt;
      ed[static_cast<std::size_t>(k)] -= kwgt;

      if (bndptr[static_cast<std::size_t>(k)] != -1) { /* If k was a boundary vertex */
        if (ed[static_cast<std::size_t>(k)] == 0) {     /* Not a boundary vertex any more */
          bndDelete(nbnd, bndind, bndptr, k);
          if (moved[static_cast<std::size_t>(k)] == -1 && where[static_cast<std::size_t>(k)] == from &&
              vwgt[static_cast<std::size_t>(k)] <= mindiff)
            queue.erase(k);
        } else {
          if (moved[static_cast<std::size_t>(k)] == -1 && where[static_cast<std::size_t>(k)] == from &&
              vwgt[static_cast<std::size_t>(k)] <= mindiff)
            queue.update(k, ed[static_cast<std::size_t>(k)] - id[static_cast<std::size_t>(k)]);
        }
      } else {
        if (ed[static_cast<std::size_t>(k)] > 0) { /* It will now become a boundary vertex */
          bndInsert(nbnd, bndind, bndptr, k);
          if (moved[static_cast<std::size_t>(k)] == -1 && where[static_cast<std::size_t>(k)] == from &&
              vwgt[static_cast<std::size_t>(k)] <= mindiff)
            queue.insert(k, ed[static_cast<std::size_t>(k)] - id[static_cast<std::size_t>(k)]);
        }
      }
    }
  }

  graph->mincut = mincut;
  graph->nbnd = nbnd;
}

// This function balances two partitions by moving the highest gain
// (including negative gain) vertices to the other domain. Used only when
// there are no boundary vertices (non-contiguous subdomains).
template <typename IndexT, typename RealT>
void general2WayBalance(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2]) {
  (void)ctrl;
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& adjwgt = graph->adjwgt;
  std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& id = graph->id;
  std::vector<IndexT>& ed = graph->ed;
  std::vector<IndexT>& pwgts = graph->pwgts;
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& bndind = graph->bndind;

  std::vector<IndexT> moved(static_cast<std::size_t>(nvtxs), IndexT(-1));
  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));

  const IndexT tpwgts0 = static_cast<IndexT>(static_cast<RealT>(graph->tvwgt) * ntpwgts[0]);
  const IndexT tpwgts1 = graph->tvwgt - tpwgts0;
  const IndexT mindiff = std::abs(tpwgts0 - pwgts[0]);
  const IndexT from = (pwgts[0] < tpwgts0 ? 1 : 0);
  const IndexT to = (from + 1) % 2;
  const IndexT tpwgts[2] = {tpwgts0, tpwgts1};

  PQueue<IndexT, IndexT> queue(static_cast<std::size_t>(nvtxs));

  randArrayPermute<IndexT>(nvtxs, perm.data(), nvtxs / 5, 1);
  for (IndexT ii = 0; ii < nvtxs; ii++) {
    const IndexT i = perm[static_cast<std::size_t>(ii)];
    if (where[static_cast<std::size_t>(i)] == from && vwgt[static_cast<std::size_t>(i)] <= mindiff)
      queue.insert(i, ed[static_cast<std::size_t>(i)] - id[static_cast<std::size_t>(i)]);
  }

  IndexT mincut = graph->mincut;
  IndexT nbnd = graph->nbnd;
  for (IndexT nswaps = 0; nswaps < nvtxs; nswaps++) {
    const IndexT higain = queue.getTop();
    if (higain == -1) break;

    if (pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(higain)] > tpwgts[to]) break;

    mincut -= (ed[static_cast<std::size_t>(higain)] - id[static_cast<std::size_t>(higain)]);
    pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
    pwgts[static_cast<std::size_t>(from)] -= vwgt[static_cast<std::size_t>(higain)];

    where[static_cast<std::size_t>(higain)] = to;
    moved[static_cast<std::size_t>(higain)] = nswaps;

    std::swap(id[static_cast<std::size_t>(higain)], ed[static_cast<std::size_t>(higain)]);
    if (ed[static_cast<std::size_t>(higain)] == 0 && bndptr[static_cast<std::size_t>(higain)] != -1 &&
        xadj[static_cast<std::size_t>(higain)] < xadj[static_cast<std::size_t>(higain) + 1])
      bndDelete(nbnd, bndind, bndptr, higain);
    if (ed[static_cast<std::size_t>(higain)] > 0 && bndptr[static_cast<std::size_t>(higain)] == -1)
      bndInsert(nbnd, bndind, bndptr, higain);

    for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
      const IndexT k = adjncy[static_cast<std::size_t>(j)];
      const IndexT kwgt = (to == where[static_cast<std::size_t>(k)] ? adjwgt[static_cast<std::size_t>(j)]
                                                                     : -adjwgt[static_cast<std::size_t>(j)]);
      id[static_cast<std::size_t>(k)] += kwgt;
      ed[static_cast<std::size_t>(k)] -= kwgt;

      if (moved[static_cast<std::size_t>(k)] == -1 && where[static_cast<std::size_t>(k)] == from &&
          vwgt[static_cast<std::size_t>(k)] <= mindiff)
        queue.update(k, ed[static_cast<std::size_t>(k)] - id[static_cast<std::size_t>(k)]);

      if (ed[static_cast<std::size_t>(k)] == 0 && bndptr[static_cast<std::size_t>(k)] != -1)
        bndDelete(nbnd, bndind, bndptr, k);
      else if (ed[static_cast<std::size_t>(k)] > 0 && bndptr[static_cast<std::size_t>(k)] == -1)
        bndInsert(nbnd, bndind, bndptr, k);
    }
  }

  graph->mincut = mincut;
  graph->nbnd = nbnd;
}

// This function is the entry point of the bisection balancing algorithms.
template <typename IndexT, typename RealT>
void balance2Way(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2]) {
  if (computeLoadImbalanceDiff(graph, ctrl.pijbm, ctrl.ubfactor) <= 0) return;

  /* return right away if the balance is OK */
  if (std::abs(ntpwgts[0] * static_cast<RealT>(graph->tvwgt) - static_cast<RealT>(graph->pwgts[0])) <
      static_cast<RealT>(3 * graph->tvwgt / graph->nvtxs))
    return;

  if (graph->nbnd > 0)
    bnd2WayBalance(ctrl, graph, ntpwgts);
  else
    general2WayBalance(ctrl, graph, ntpwgts);
}

// This function performs a cut-focused FM refinement.
template <typename IndexT, typename RealT>
void fm2WayCutRefine(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2],
                     IndexT niter) {
  (void)ctrl;
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& adjwgt = graph->adjwgt;
  std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& id = graph->id;
  std::vector<IndexT>& ed = graph->ed;
  std::vector<IndexT>& pwgts = graph->pwgts;
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& bndind = graph->bndind;

  std::vector<IndexT> moved(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> swaps(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));

  const IndexT tpwgts0 = static_cast<IndexT>(static_cast<RealT>(graph->tvwgt) * ntpwgts[0]);
  const IndexT tpwgts1 = graph->tvwgt - tpwgts0;
  const IndexT tpwgts[2] = {tpwgts0, tpwgts1};

  const IndexT limit = static_cast<IndexT>(
      std::min(std::max(0.01 * static_cast<double>(nvtxs), 15.0), 100.0));
  const IndexT avgvwgt =
      std::min<IndexT>((pwgts[0] + pwgts[1]) / 20, 2 * (pwgts[0] + pwgts[1]) / nvtxs);

  PQueue<IndexT, IndexT> queues[2] = {PQueue<IndexT, IndexT>(static_cast<std::size_t>(nvtxs)),
                                      PQueue<IndexT, IndexT>(static_cast<std::size_t>(nvtxs))};

  const IndexT origdiff = std::abs(tpwgts[0] - pwgts[0]);
  std::fill(moved.begin(), moved.end(), IndexT(-1));
  for (IndexT pass = 0; pass < niter; pass++) {
    queues[0].reset();
    queues[1].reset();

    IndexT mincutorder = -1;
    IndexT initcut = graph->mincut;
    IndexT newcut = initcut, mincut = initcut;
    IndexT mindiff = std::abs(tpwgts[0] - pwgts[0]);

    IndexT nbnd = graph->nbnd;
    randArrayPermute<IndexT>(nbnd, perm.data(), nbnd, 1);
    for (IndexT ii = 0; ii < nbnd; ii++) {
      const IndexT i = perm[static_cast<std::size_t>(ii)];
      queues[static_cast<std::size_t>(where[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])])].insert(
          bndind[static_cast<std::size_t>(i)],
          ed[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])] -
              id[static_cast<std::size_t>(bndind[static_cast<std::size_t>(i)])]);
    }

    IndexT nswaps;
    for (nswaps = 0; nswaps < nvtxs; nswaps++) {
      const IndexT from = (tpwgts[0] - pwgts[0] < tpwgts[1] - pwgts[1] ? 0 : 1);
      const IndexT to = (from + 1) % 2;

      const IndexT higain = queues[static_cast<std::size_t>(from)].getTop();
      if (higain == -1) break;

      newcut -= (ed[static_cast<std::size_t>(higain)] - id[static_cast<std::size_t>(higain)]);
      pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
      pwgts[static_cast<std::size_t>(from)] -= vwgt[static_cast<std::size_t>(higain)];

      if ((newcut < mincut && std::abs(tpwgts[0] - pwgts[0]) <= origdiff + avgvwgt) ||
          (newcut == mincut && std::abs(tpwgts[0] - pwgts[0]) < mindiff)) {
        mincut = newcut;
        mindiff = std::abs(tpwgts[0] - pwgts[0]);
        mincutorder = nswaps;
      } else if (nswaps - mincutorder > limit) { /* We hit the limit, undo last move */
        newcut += (ed[static_cast<std::size_t>(higain)] - id[static_cast<std::size_t>(higain)]);
        pwgts[static_cast<std::size_t>(from)] += vwgt[static_cast<std::size_t>(higain)];
        pwgts[static_cast<std::size_t>(to)] -= vwgt[static_cast<std::size_t>(higain)];
        break;
      }

      where[static_cast<std::size_t>(higain)] = to;
      moved[static_cast<std::size_t>(higain)] = nswaps;
      swaps[static_cast<std::size_t>(nswaps)] = higain;

      std::swap(id[static_cast<std::size_t>(higain)], ed[static_cast<std::size_t>(higain)]);
      if (ed[static_cast<std::size_t>(higain)] == 0 &&
          xadj[static_cast<std::size_t>(higain)] < xadj[static_cast<std::size_t>(higain) + 1])
        bndDelete(nbnd, bndind, bndptr, higain);

      for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        const IndexT kwgt = (to == where[static_cast<std::size_t>(k)] ? adjwgt[static_cast<std::size_t>(j)]
                                                                       : -adjwgt[static_cast<std::size_t>(j)]);
        id[static_cast<std::size_t>(k)] += kwgt;
        ed[static_cast<std::size_t>(k)] -= kwgt;

        if (bndptr[static_cast<std::size_t>(k)] != -1) {
          if (ed[static_cast<std::size_t>(k)] == 0) {
            bndDelete(nbnd, bndind, bndptr, k);
            if (moved[static_cast<std::size_t>(k)] == -1) queues[static_cast<std::size_t>(where[static_cast<std::size_t>(k)])].erase(k);
          } else {
            if (moved[static_cast<std::size_t>(k)] == -1)
              queues[static_cast<std::size_t>(where[static_cast<std::size_t>(k)])].update(
                  k, ed[static_cast<std::size_t>(k)] - id[static_cast<std::size_t>(k)]);
          }
        } else {
          if (ed[static_cast<std::size_t>(k)] > 0) {
            bndInsert(nbnd, bndind, bndptr, k);
            if (moved[static_cast<std::size_t>(k)] == -1)
              queues[static_cast<std::size_t>(where[static_cast<std::size_t>(k)])].insert(
                  k, ed[static_cast<std::size_t>(k)] - id[static_cast<std::size_t>(k)]);
          }
        }
      }
    }

    /* Roll back computations */
    for (IndexT i = 0; i < nswaps; i++) moved[static_cast<std::size_t>(swaps[static_cast<std::size_t>(i)])] = -1;
    for (nswaps--; nswaps > mincutorder; nswaps--) {
      const IndexT higain = swaps[static_cast<std::size_t>(nswaps)];

      const IndexT to = where[static_cast<std::size_t>(higain)] = (where[static_cast<std::size_t>(higain)] + 1) % 2;
      std::swap(id[static_cast<std::size_t>(higain)], ed[static_cast<std::size_t>(higain)]);
      if (ed[static_cast<std::size_t>(higain)] == 0 && bndptr[static_cast<std::size_t>(higain)] != -1 &&
          xadj[static_cast<std::size_t>(higain)] < xadj[static_cast<std::size_t>(higain) + 1])
        bndDelete(nbnd, bndind, bndptr, higain);
      else if (ed[static_cast<std::size_t>(higain)] > 0 && bndptr[static_cast<std::size_t>(higain)] == -1)
        bndInsert(nbnd, bndind, bndptr, higain);

      pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
      pwgts[static_cast<std::size_t>((to + 1) % 2)] -= vwgt[static_cast<std::size_t>(higain)];
      for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        const IndexT kwgt = (to == where[static_cast<std::size_t>(k)] ? adjwgt[static_cast<std::size_t>(j)]
                                                                       : -adjwgt[static_cast<std::size_t>(j)]);
        id[static_cast<std::size_t>(k)] += kwgt;
        ed[static_cast<std::size_t>(k)] -= kwgt;

        if (bndptr[static_cast<std::size_t>(k)] != -1 && ed[static_cast<std::size_t>(k)] == 0)
          bndDelete(nbnd, bndind, bndptr, k);
        if (bndptr[static_cast<std::size_t>(k)] == -1 && ed[static_cast<std::size_t>(k)] > 0)
          bndInsert(nbnd, bndind, bndptr, k);
      }
    }

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder <= 0 || mincut == initcut) break;
  }
}

// FM_2WayRefine's ncon==1 dispatch (the only path in scope: Eigen's call
// convention hardcodes ncon=1, so FM_Mc2WayCutRefine is never reached).
template <typename IndexT, typename RealT>
void fm2WayRefine(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2], IndexT niter) {
  fm2WayCutRefine(ctrl, graph, ntpwgts, niter);
}

// This function computes a bisection of a graph by randomly assigning the
// vertices followed by a bisection refinement.
template <typename IndexT, typename RealT>
void randomBisection(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2],
                     IndexT niparts) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& vwgt = graph->vwgt;

  allocate2WayPartitionMemory(graph);
  std::vector<IndexT>& where = graph->where;

  std::vector<IndexT> bestwhere(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));

  const IndexT zeromaxpwgt = static_cast<IndexT>(ctrl.ubfactor * static_cast<RealT>(graph->tvwgt) * ntpwgts[0]);

  IndexT bestcut = 0;
  for (IndexT inbfs = 0; inbfs < niparts; inbfs++) {
    std::fill(where.begin(), where.end(), IndexT(1));

    if (inbfs > 0) {
      randArrayPermute<IndexT>(nvtxs, perm.data(), nvtxs / 2, 1);
      IndexT pwgts[2] = {0, graph->tvwgt};

      for (IndexT ii = 0; ii < nvtxs; ii++) {
        const IndexT i = perm[static_cast<std::size_t>(ii)];
        if (pwgts[0] + vwgt[static_cast<std::size_t>(i)] < zeromaxpwgt) {
          where[static_cast<std::size_t>(i)] = 0;
          pwgts[0] += vwgt[static_cast<std::size_t>(i)];
          pwgts[1] -= vwgt[static_cast<std::size_t>(i)];
          if (pwgts[0] > zeromaxpwgt) break;
        }
      }
    }

    /* Do some partition refinement  */
    compute2WayPartitionParams(graph);

    balance2Way(ctrl, graph, ntpwgts);

    fm2WayRefine(ctrl, graph, ntpwgts, IndexT(4));

    if (inbfs == 0 || bestcut > graph->mincut) {
      bestcut = graph->mincut;
      bestwhere = where;
      if (bestcut == 0) break;
    }
  }

  graph->mincut = bestcut;
  where = bestwhere;
}

// This function takes a graph and produces a bisection by using a region
// growing algorithm. The resulting bisection is refined using FM.
template <typename IndexT, typename RealT>
void growBisection(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, const RealT ntpwgts[2],
                   IndexT niparts) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;

  allocate2WayPartitionMemory(graph);
  std::vector<IndexT>& where = graph->where;

  std::vector<IndexT> bestwhere(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> queue(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> touched(static_cast<std::size_t>(nvtxs));

  const IndexT onemaxpwgt = static_cast<IndexT>(ctrl.ubfactor * static_cast<RealT>(graph->tvwgt) * ntpwgts[1]);
  const IndexT oneminpwgt = static_cast<IndexT>(
      (1.0 / static_cast<double>(ctrl.ubfactor)) * static_cast<double>(graph->tvwgt) * static_cast<double>(ntpwgts[1]));

  IndexT bestcut = 0;
  for (IndexT inbfs = 0; inbfs < niparts; inbfs++) {
    std::fill(where.begin(), where.end(), IndexT(1));
    std::fill(touched.begin(), touched.end(), IndexT(0));

    IndexT pwgts[2] = {0, graph->tvwgt};

    queue[0] = randInRange<IndexT>(nvtxs);
    touched[static_cast<std::size_t>(queue[0])] = 1;
    IndexT first = 0, last = 1;
    IndexT nleft = nvtxs - 1;
    IndexT drain = 0;

    /* Start the BFS from queue to get a partition */
    for (;;) {
      if (first == last) { /* Empty. Disconnected graph! */
        if (nleft == 0 || drain) break;

        IndexT k = randInRange<IndexT>(nleft);
        IndexT i = 0;
        for (; i < nvtxs; i++) {
          if (touched[static_cast<std::size_t>(i)] == 0) {
            if (k == 0)
              break;
            else
              k--;
          }
        }

        queue[0] = i;
        touched[static_cast<std::size_t>(i)] = 1;
        first = 0;
        last = 1;
        nleft--;
      }

      const IndexT i = queue[static_cast<std::size_t>(first++)];
      if (pwgts[0] > 0 && pwgts[1] - vwgt[static_cast<std::size_t>(i)] < oneminpwgt) {
        drain = 1;
        continue;
      }

      where[static_cast<std::size_t>(i)] = 0;
      pwgts[0] += vwgt[static_cast<std::size_t>(i)];
      pwgts[1] -= vwgt[static_cast<std::size_t>(i)];
      if (pwgts[1] <= onemaxpwgt) break;

      drain = 0;
      for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        if (touched[static_cast<std::size_t>(k)] == 0) {
          queue[static_cast<std::size_t>(last++)] = k;
          touched[static_cast<std::size_t>(k)] = 1;
          nleft--;
        }
      }
    }

    /* Check to see if we hit any bad limiting cases */
    if (pwgts[1] == 0) where[static_cast<std::size_t>(randInRange<IndexT>(nvtxs))] = 1;
    if (pwgts[0] == 0) where[static_cast<std::size_t>(randInRange<IndexT>(nvtxs))] = 0;

    /* Do some partition refinement */
    compute2WayPartitionParams(graph);

    balance2Way(ctrl, graph, ntpwgts);

    fm2WayRefine(ctrl, graph, ntpwgts, ctrl.niter);

    if (inbfs == 0 || bestcut > graph->mincut) {
      bestcut = graph->mincut;
      bestwhere = where;
      if (bestcut == 0) break;
    }
  }

  graph->mincut = bestcut;
  where = bestwhere;
}

// This function takes a bisection and constructs a minimum weight vertex
// separator out of it. It uses the node-based separator refinement for it.
template <typename IndexT, typename RealT>
void constructSeparator(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const IndexT nbnd = graph->nbnd;
  const std::vector<IndexT>& bndind = graph->bndind;

  std::vector<IndexT> where = graph->where;

  /* Put the nodes in the boundary into the separator */
  for (IndexT i = 0; i < nbnd; i++) {
    const IndexT j = bndind[static_cast<std::size_t>(i)];
    if (xadj[static_cast<std::size_t>(j) + 1] - xadj[static_cast<std::size_t>(j)] > 0) /* Ignore islands */
      where[static_cast<std::size_t>(j)] = 2;
  }

  allocate2WayNodePartitionMemory(graph);
  graph->where = where;

  compute2WayNodePartitionParams(graph);

  fm2WayNodeRefine2Sided(ctrl, graph, IndexT(1));
  fm2WayNodeRefine1Sided(ctrl, graph, IndexT(4));
}

// This function computes the initial separator of the coarsest graph.
template <typename IndexT, typename RealT>
void initSeparator(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niparts) {
  const RealT ntpwgts[2] = {RealT(0.5), RealT(0.5)};

  /* this is required for the cut-based part of the refinement */
  setup2WayBalMultipliers(ctrl, graph, ntpwgts);

  if (graph->nedges == 0)
    randomBisection(ctrl, graph, ntpwgts, niparts);
  else
    growBisection(ctrl, graph, ntpwgts, niparts);

  compute2WayPartitionParams(graph);
  constructSeparator(ctrl, graph);
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_INITIAL_SEPARATOR_H
