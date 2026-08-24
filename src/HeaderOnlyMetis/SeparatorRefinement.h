// Faithful port of the node-separator refinement + uncoarsening driver:
// libmetis/srefine.c (Refine2WayNode, Allocate2WayNodePartitionMemory,
// Compute2WayNodePartitionParams, Project2WayNodePartition) and
// libmetis/sfm.c (FM_2WayNodeRefine2Sided, FM_2WayNodeRefine1Sided,
// FM_2WayNodeBalance).
//
// Not ported: FreeRData (GKlib memory-management bookkeeping the port's
// std::vector-based Graph doesn't need -- fields are just reassigned
// directly); ASSERT-gated helpers (CheckNodeBnd, CheckNodePartitionParams,
// IsSeparable) since assertions compile out under NDEBUG, which is how the
// reference build used for comparison is built.
//
// BNDInsert/BNDDelete (macros.h) and INC_DEC (gk_macros.h) are inlined
// directly rather than kept as separate helpers, matching their trivial
// bodies: BNDInsert/BNDDelete maintain bndind/bndptr as a swap-remove index
// set (bndptr[v] = v's position in bndind, or -1 if absent); INC_DEC(a,b,val)
// is just `a+=val; b-=val;`.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_SEPARATOR_REFINEMENT_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_SEPARATOR_REFINEMENT_H

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "Ctrl.h"
#include "Graph.h"
#include "PQueue.h"
#include "Random.h"

namespace header_only_metis {

// Matches BNDInsert(nbnd, bndind, bndptr, vtx) exactly (macros.h's
// ListInsert): appends vtx to bndind and records its position in bndptr.
template <typename IndexT>
inline void bndInsert(IndexT& nbnd, std::vector<IndexT>& bndind, std::vector<IndexT>& bndptr, IndexT vtx) {
  bndind[static_cast<std::size_t>(nbnd)] = vtx;
  bndptr[static_cast<std::size_t>(vtx)] = nbnd++;
}

// Matches BNDDelete(nbnd, bndind, bndptr, vtx) exactly (macros.h's
// ListDelete): swap-removes vtx from bndind/bndptr.
template <typename IndexT>
inline void bndDelete(IndexT& nbnd, std::vector<IndexT>& bndind, std::vector<IndexT>& bndptr, IndexT vtx) {
  bndind[static_cast<std::size_t>(bndptr[static_cast<std::size_t>(vtx)])] =
      bndind[static_cast<std::size_t>(--nbnd)];
  bndptr[static_cast<std::size_t>(bndind[static_cast<std::size_t>(nbnd)])] = bndptr[static_cast<std::size_t>(vtx)];
  bndptr[static_cast<std::size_t>(vtx)] = -1;
}

// This function allocates memory for 2-way node-based refinement.
template <typename IndexT, typename RealT>
void allocate2WayNodePartitionMemory(Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  graph->pwgts.assign(3, IndexT(0));
  graph->where.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->bndptr.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->bndind.assign(static_cast<std::size_t>(nvtxs), IndexT(0));
  graph->nrinfo.assign(static_cast<std::size_t>(nvtxs), typename Graph<IndexT, RealT>::NRInfo());
}

// This function computes the edegrees[] to the left & right sides.
template <typename IndexT, typename RealT>
void compute2WayNodePartitionParams(Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& where = graph->where;
  auto& rinfo = graph->nrinfo;

  graph->pwgts.assign(3, IndexT(0));
  std::vector<IndexT>& pwgts = graph->pwgts;
  graph->bndptr.assign(static_cast<std::size_t>(nvtxs), IndexT(-1));
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& bndind = graph->bndind;

  IndexT nbnd = 0;
  for (IndexT i = 0; i < nvtxs; i++) {
    const IndexT me = where[static_cast<std::size_t>(i)];
    pwgts[static_cast<std::size_t>(me)] += vwgt[static_cast<std::size_t>(i)];

    if (me == 2) { /* If it is on the separator do some computations */
      bndInsert(nbnd, bndind, bndptr, i);

      IndexT* edegrees = rinfo[static_cast<std::size_t>(i)].edegrees;
      edegrees[0] = edegrees[1] = 0;

      for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++) {
        const IndexT other = where[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])];
        if (other != 2) edegrees[other] += vwgt[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])];
      }
    }
  }

  graph->mincut = pwgts[2];
  graph->nbnd = nbnd;
}

// This function projects the node-based bisection.
template <typename IndexT, typename RealT>
void project2WayNodePartition(Graph<IndexT, RealT>* graph) {
  Graph<IndexT, RealT>* cgraph = graph->coarser.get();
  const std::vector<IndexT>& cwhere = cgraph->where;

  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& cmap = graph->cmap;

  allocate2WayNodePartitionMemory(graph);
  std::vector<IndexT>& where = graph->where;

  for (IndexT i = 0; i < nvtxs; i++) where[static_cast<std::size_t>(i)] = cwhere[static_cast<std::size_t>(cmap[static_cast<std::size_t>(i)])];

  graph->coarser.reset();

  compute2WayNodePartitionParams(graph);
}

template <typename IndexT, typename RealT>
void fm2WayNodeRefine2Sided(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niter);
template <typename IndexT, typename RealT>
void fm2WayNodeRefine1Sided(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niter);
template <typename IndexT, typename RealT>
void fm2WayNodeBalance(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph);

// This function is the entry point of the separator refinement. It does not
// perform any refinement on graph, but it starts by first projecting it to
// the next level finer graph and proceeds from there.
template <typename IndexT, typename RealT>
void refine2WayNode(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* orggraph, Graph<IndexT, RealT>* graph) {
  if (graph == orggraph) {
    compute2WayNodePartitionParams(graph);
  } else {
    do {
      graph = graph->finer;

      project2WayNodePartition(graph);

      fm2WayNodeBalance(ctrl, graph);

      switch (ctrl.rtype) {
        case RType::SEP2SIDED:
          fm2WayNodeRefine2Sided(ctrl, graph, ctrl.niter);
          break;
        case RType::SEP1SIDED:
          fm2WayNodeRefine1Sided(ctrl, graph, ctrl.niter);
          break;
      }

    } while (graph != orggraph);
  }
}

// This function performs a node-based FM refinement.
template <typename IndexT, typename RealT>
void fm2WayNodeRefine2Sided(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niter) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& vwgt = graph->vwgt;

  std::vector<IndexT>& bndind = graph->bndind;
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& pwgts = graph->pwgts;
  auto& rinfo = graph->nrinfo;

  PQueue<IndexT, IndexT> queues[2] = {PQueue<IndexT, IndexT>(static_cast<std::size_t>(nvtxs)),
                                      PQueue<IndexT, IndexT>(static_cast<std::size_t>(nvtxs))};

  std::vector<IndexT> moved(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> swaps(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> mptr(static_cast<std::size_t>(nvtxs) + 1);
  std::vector<IndexT> mind(2 * static_cast<std::size_t>(nvtxs));

  // Computed ONCE from the pre-refinement pwgts and reused unchanged across
  // every pass -- the reference does not recompute it as pwgts evolve during
  // refinement (libmetis/sfm.c:55-56, outside the `for (pass...)` loop).
  const RealT mult = RealT(0.5) * ctrl.ubfactor;
  const IndexT badmaxpwgt =
      static_cast<IndexT>(mult * static_cast<RealT>(pwgts[0] + pwgts[1] + pwgts[2]));

  for (IndexT pass = 0; pass < niter; pass++) {
    std::fill(moved.begin(), moved.end(), IndexT(-1));
    queues[0].reset();
    queues[1].reset();

    IndexT mincutorder = -1;
    IndexT initcut = graph->mincut;
    IndexT mincut = initcut;
    IndexT nbnd = graph->nbnd;

    /* use the swaps array in place of the traditional perm array to save memory */
    randArrayPermute<IndexT>(nbnd, swaps.data(), nbnd, 1);
    for (IndexT ii = 0; ii < nbnd; ii++) {
      const IndexT i = bndind[static_cast<std::size_t>(swaps[static_cast<std::size_t>(ii)])];
      queues[0].insert(i, vwgt[static_cast<std::size_t>(i)] - rinfo[static_cast<std::size_t>(i)].edegrees[1]);
      queues[1].insert(i, vwgt[static_cast<std::size_t>(i)] - rinfo[static_cast<std::size_t>(i)].edegrees[0]);
    }

    const IndexT limit = ctrl.compress ? std::min<IndexT>(5 * nbnd, 400) : std::min<IndexT>(2 * nbnd, 300);

    IndexT nmind = 0;
    mptr[0] = 0;
    IndexT mindiff = std::abs(pwgts[0] - pwgts[1]);
    IndexT to = (pwgts[0] < pwgts[1] ? 0 : 1);

    IndexT nswaps = 0;
    for (; nswaps < nvtxs; nswaps++) {
      const IndexT u0 = queues[0].seeTopVal();
      const IndexT u1 = queues[1].seeTopVal();
      IndexT g0 = 0, g1 = 0;
      if (u0 != -1 && u1 != -1) {
        g0 = vwgt[static_cast<std::size_t>(u0)] - rinfo[static_cast<std::size_t>(u0)].edegrees[1];
        g1 = vwgt[static_cast<std::size_t>(u1)] - rinfo[static_cast<std::size_t>(u1)].edegrees[0];

        to = (g0 > g1 ? 0 : (g0 < g1 ? 1 : pass % 2));

        const IndexT uTo = (to == 0 ? u0 : u1);
        if (pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(uTo)] > badmaxpwgt)
          to = (to + 1) % 2;
      } else if (u0 == -1 && u1 == -1) {
        break;
      } else if (u0 != -1 && pwgts[0] + vwgt[static_cast<std::size_t>(u0)] <= badmaxpwgt) {
        to = 0;
      } else if (u1 != -1 && pwgts[1] + vwgt[static_cast<std::size_t>(u1)] <= badmaxpwgt) {
        to = 1;
      } else {
        break;
      }

      const IndexT other = (to + 1) % 2;

      const IndexT higain = queues[static_cast<std::size_t>(to)].getTop();
      if (moved[static_cast<std::size_t>(higain)] == -1) /* Delete if it was in the separator originally */
        queues[static_cast<std::size_t>(other)].erase(higain);

      /* The following check is to ensure we break out if there is a possibility
         of over-running the mind array.  */
      if (nmind + xadj[static_cast<std::size_t>(higain) + 1] - xadj[static_cast<std::size_t>(higain)] >=
          2 * nvtxs - 1)
        break;

      pwgts[2] -= (vwgt[static_cast<std::size_t>(higain)] - rinfo[static_cast<std::size_t>(higain)].edegrees[other]);

      const IndexT newdiff = std::abs(pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(higain)] -
                                      (pwgts[static_cast<std::size_t>(other)] -
                                       rinfo[static_cast<std::size_t>(higain)].edegrees[other]));
      if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) {
        mincut = pwgts[2];
        mincutorder = nswaps;
        mindiff = newdiff;
      } else {
        // `1.10` is a double literal in the reference (pwgts[2] > 1.10*mincut),
        // so this comparison happens in double regardless of RealT's width.
        if (nswaps - mincutorder > 2 * limit ||
            (nswaps - mincutorder > limit &&
             static_cast<double>(pwgts[2]) > 1.10 * static_cast<double>(mincut))) {
          pwgts[2] += (vwgt[static_cast<std::size_t>(higain)] - rinfo[static_cast<std::size_t>(higain)].edegrees[other]);
          break; /* No further improvement, break out */
        }
      }

      bndDelete(nbnd, bndind, bndptr, higain);
      pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
      where[static_cast<std::size_t>(higain)] = to;
      moved[static_cast<std::size_t>(higain)] = nswaps;
      swaps[static_cast<std::size_t>(nswaps)] = higain;

      for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        if (where[static_cast<std::size_t>(k)] == 2) { /* For the in-separator vertices modify their edegree[to] */
          const IndexT oldgain = vwgt[static_cast<std::size_t>(k)] - rinfo[static_cast<std::size_t>(k)].edegrees[to];
          rinfo[static_cast<std::size_t>(k)].edegrees[to] += vwgt[static_cast<std::size_t>(higain)];
          if (moved[static_cast<std::size_t>(k)] == -1 || moved[static_cast<std::size_t>(k)] == -(2 + other))
            queues[static_cast<std::size_t>(other)].update(k, oldgain - vwgt[static_cast<std::size_t>(higain)]);
        } else if (where[static_cast<std::size_t>(k)] == other) { /* This vertex is pulled into the separator */
          bndInsert(nbnd, bndind, bndptr, k);

          mind[static_cast<std::size_t>(nmind++)] = k; /* Keep track for rollback */
          where[static_cast<std::size_t>(k)] = 2;
          pwgts[static_cast<std::size_t>(other)] -= vwgt[static_cast<std::size_t>(k)];

          IndexT* edegrees = rinfo[static_cast<std::size_t>(k)].edegrees;
          edegrees[0] = edegrees[1] = 0;
          for (IndexT jj = xadj[static_cast<std::size_t>(k)]; jj < xadj[static_cast<std::size_t>(k) + 1]; jj++) {
            const IndexT kk = adjncy[static_cast<std::size_t>(jj)];
            if (where[static_cast<std::size_t>(kk)] != 2) {
              edegrees[where[static_cast<std::size_t>(kk)]] += vwgt[static_cast<std::size_t>(kk)];
            } else {
              const IndexT oldgain =
                  vwgt[static_cast<std::size_t>(kk)] - rinfo[static_cast<std::size_t>(kk)].edegrees[other];
              rinfo[static_cast<std::size_t>(kk)].edegrees[other] -= vwgt[static_cast<std::size_t>(k)];
              if (moved[static_cast<std::size_t>(kk)] == -1 || moved[static_cast<std::size_t>(kk)] == -(2 + to))
                queues[static_cast<std::size_t>(to)].update(kk, oldgain + vwgt[static_cast<std::size_t>(k)]);
            }
          }

          /* Insert the new vertex into the priority queue. Only one side! */
          if (moved[static_cast<std::size_t>(k)] == -1) {
            queues[static_cast<std::size_t>(to)].insert(k, vwgt[static_cast<std::size_t>(k)] - edegrees[other]);
            moved[static_cast<std::size_t>(k)] = -(2 + to);
          }
        }
      }
      mptr[static_cast<std::size_t>(nswaps) + 1] = nmind;
    }

    /* Roll back computation */
    for (nswaps--; nswaps > mincutorder; nswaps--) {
      const IndexT higain = swaps[static_cast<std::size_t>(nswaps)];

      const IndexT to = where[static_cast<std::size_t>(higain)];
      const IndexT other = (to + 1) % 2;
      pwgts[2] += vwgt[static_cast<std::size_t>(higain)];
      pwgts[static_cast<std::size_t>(to)] -= vwgt[static_cast<std::size_t>(higain)];
      where[static_cast<std::size_t>(higain)] = 2;
      bndInsert(nbnd, bndind, bndptr, higain);

      IndexT* edegrees = rinfo[static_cast<std::size_t>(higain)].edegrees;
      edegrees[0] = edegrees[1] = 0;
      for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        if (where[static_cast<std::size_t>(k)] == 2)
          rinfo[static_cast<std::size_t>(k)].edegrees[to] -= vwgt[static_cast<std::size_t>(higain)];
        else
          edegrees[where[static_cast<std::size_t>(k)]] += vwgt[static_cast<std::size_t>(k)];
      }

      /* Push nodes out of the separator */
      for (IndexT j = mptr[static_cast<std::size_t>(nswaps)]; j < mptr[static_cast<std::size_t>(nswaps) + 1]; j++) {
        const IndexT k = mind[static_cast<std::size_t>(j)];
        where[static_cast<std::size_t>(k)] = other;
        pwgts[static_cast<std::size_t>(other)] += vwgt[static_cast<std::size_t>(k)];
        pwgts[2] -= vwgt[static_cast<std::size_t>(k)];
        bndDelete(nbnd, bndind, bndptr, k);
        for (IndexT jj = xadj[static_cast<std::size_t>(k)]; jj < xadj[static_cast<std::size_t>(k) + 1]; jj++) {
          const IndexT kk = adjncy[static_cast<std::size_t>(jj)];
          if (where[static_cast<std::size_t>(kk)] == 2) rinfo[static_cast<std::size_t>(kk)].edegrees[other] += vwgt[static_cast<std::size_t>(k)];
        }
      }
    }

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder == -1 || mincut >= initcut) break;
  }
}

// This function performs a node-based FM refinement. Each refinement
// iteration is split into two sub-iterations. In each sub-iteration only
// moves to one of the left/right partitions is allowed; hence, it is
// one-sided.
template <typename IndexT, typename RealT>
void fm2WayNodeRefine1Sided(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph, IndexT niter) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& vwgt = graph->vwgt;

  std::vector<IndexT>& bndind = graph->bndind;
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& pwgts = graph->pwgts;
  auto& rinfo = graph->nrinfo;

  PQueue<IndexT, IndexT> queue(static_cast<std::size_t>(nvtxs));

  std::vector<IndexT> swaps(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> mptr(static_cast<std::size_t>(nvtxs) + 1);
  std::vector<IndexT> mind(2 * static_cast<std::size_t>(nvtxs));

  // Computed ONCE, same reasoning as fm2WayNodeRefine2Sided.
  const RealT mult = RealT(0.5) * ctrl.ubfactor;
  const IndexT badmaxpwgt =
      static_cast<IndexT>(mult * static_cast<RealT>(pwgts[0] + pwgts[1] + pwgts[2]));

  IndexT to = (pwgts[0] < pwgts[1] ? 1 : 0);
  for (IndexT pass = 0; pass < 2 * niter; pass++) { /* the 2*niter is for the two sides */
    const IndexT other = to;
    to = (to + 1) % 2;

    queue.reset();

    IndexT mincutorder = -1;
    const IndexT initcut = graph->mincut;
    IndexT mincut = initcut;
    IndexT nbnd = graph->nbnd;

    /* use the swaps array in place of the traditional perm array to save memory */
    randArrayPermute<IndexT>(nbnd, swaps.data(), nbnd, 1);
    for (IndexT ii = 0; ii < nbnd; ii++) {
      const IndexT i = bndind[static_cast<std::size_t>(swaps[static_cast<std::size_t>(ii)])];
      queue.insert(i, vwgt[static_cast<std::size_t>(i)] - rinfo[static_cast<std::size_t>(i)].edegrees[other]);
    }

    const IndexT limit = ctrl.compress ? std::min<IndexT>(5 * nbnd, 500) : std::min<IndexT>(3 * nbnd, 300);

    IndexT nmind = 0;
    mptr[0] = 0;
    IndexT mindiff = std::abs(pwgts[0] - pwgts[1]);

    IndexT nswaps = 0;
    for (; nswaps < nvtxs; nswaps++) {
      const IndexT higain = queue.getTop();
      if (higain == -1) break;

      /* The following check is to ensure we break out if there is a possibility
         of over-running the mind array.  */
      if (nmind + xadj[static_cast<std::size_t>(higain) + 1] - xadj[static_cast<std::size_t>(higain)] >=
          2 * nvtxs - 1)
        break;

      if (pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(higain)] > badmaxpwgt)
        break; /* No point going any further. Balance will be bad */

      pwgts[2] -= (vwgt[static_cast<std::size_t>(higain)] - rinfo[static_cast<std::size_t>(higain)].edegrees[other]);

      const IndexT newdiff = std::abs(pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(higain)] -
                                      (pwgts[static_cast<std::size_t>(other)] -
                                       rinfo[static_cast<std::size_t>(higain)].edegrees[other]));
      if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) {
        mincut = pwgts[2];
        mincutorder = nswaps;
        mindiff = newdiff;
      } else {
        if (nswaps - mincutorder > 3 * limit ||
            (nswaps - mincutorder > limit &&
             static_cast<double>(pwgts[2]) > 1.10 * static_cast<double>(mincut))) {
          pwgts[2] += (vwgt[static_cast<std::size_t>(higain)] - rinfo[static_cast<std::size_t>(higain)].edegrees[other]);
          break; /* No further improvement, break out */
        }
      }

      bndDelete(nbnd, bndind, bndptr, higain);
      pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
      where[static_cast<std::size_t>(higain)] = to;
      swaps[static_cast<std::size_t>(nswaps)] = higain;

      for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];

        if (where[static_cast<std::size_t>(k)] == 2) { /* For the in-separator vertices modify their edegree[to] */
          rinfo[static_cast<std::size_t>(k)].edegrees[to] += vwgt[static_cast<std::size_t>(higain)];
        } else if (where[static_cast<std::size_t>(k)] == other) { /* This vertex is pulled into the separator */
          bndInsert(nbnd, bndind, bndptr, k);

          mind[static_cast<std::size_t>(nmind++)] = k; /* Keep track for rollback */
          where[static_cast<std::size_t>(k)] = 2;
          pwgts[static_cast<std::size_t>(other)] -= vwgt[static_cast<std::size_t>(k)];

          IndexT* edegrees = rinfo[static_cast<std::size_t>(k)].edegrees;
          edegrees[0] = edegrees[1] = 0;
          const IndexT iend = xadj[static_cast<std::size_t>(k) + 1];
          for (IndexT jj = xadj[static_cast<std::size_t>(k)]; jj < iend; jj++) {
            const IndexT kk = adjncy[static_cast<std::size_t>(jj)];
            if (where[static_cast<std::size_t>(kk)] != 2) {
              edegrees[where[static_cast<std::size_t>(kk)]] += vwgt[static_cast<std::size_t>(kk)];
            } else {
              rinfo[static_cast<std::size_t>(kk)].edegrees[other] -= vwgt[static_cast<std::size_t>(k)];

              /* Since the moves are one-sided this vertex has not been moved yet */
              queue.update(kk, vwgt[static_cast<std::size_t>(kk)] - rinfo[static_cast<std::size_t>(kk)].edegrees[other]);
            }
          }

          /* Insert the new vertex into the priority queue. Safe due to one-sided moves */
          queue.insert(k, vwgt[static_cast<std::size_t>(k)] - edegrees[other]);
        }
      }
      mptr[static_cast<std::size_t>(nswaps) + 1] = nmind;
    }

    /* Roll back computation */
    for (nswaps--; nswaps > mincutorder; nswaps--) {
      const IndexT higain = swaps[static_cast<std::size_t>(nswaps)];

      pwgts[2] += vwgt[static_cast<std::size_t>(higain)];
      pwgts[static_cast<std::size_t>(to)] -= vwgt[static_cast<std::size_t>(higain)];
      where[static_cast<std::size_t>(higain)] = 2;
      bndInsert(nbnd, bndind, bndptr, higain);

      IndexT* edegrees = rinfo[static_cast<std::size_t>(higain)].edegrees;
      edegrees[0] = edegrees[1] = 0;
      for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
        const IndexT k = adjncy[static_cast<std::size_t>(j)];
        if (where[static_cast<std::size_t>(k)] == 2)
          rinfo[static_cast<std::size_t>(k)].edegrees[to] -= vwgt[static_cast<std::size_t>(higain)];
        else
          edegrees[where[static_cast<std::size_t>(k)]] += vwgt[static_cast<std::size_t>(k)];
      }

      /* Push nodes out of the separator */
      for (IndexT j = mptr[static_cast<std::size_t>(nswaps)]; j < mptr[static_cast<std::size_t>(nswaps) + 1]; j++) {
        const IndexT k = mind[static_cast<std::size_t>(j)];
        where[static_cast<std::size_t>(k)] = other;
        pwgts[static_cast<std::size_t>(other)] += vwgt[static_cast<std::size_t>(k)];
        pwgts[2] -= vwgt[static_cast<std::size_t>(k)];
        bndDelete(nbnd, bndind, bndptr, k);
        const IndexT iend = xadj[static_cast<std::size_t>(k) + 1];
        for (IndexT jj = xadj[static_cast<std::size_t>(k)]; jj < iend; jj++) {
          const IndexT kk = adjncy[static_cast<std::size_t>(jj)];
          if (where[static_cast<std::size_t>(kk)] == 2) rinfo[static_cast<std::size_t>(kk)].edegrees[other] += vwgt[static_cast<std::size_t>(k)];
        }
      }
    }

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (pass % 2 == 1 && (mincutorder == -1 || mincut >= initcut)) break;
  }
}

// This function balances the left/right partitions of a separator
// tri-section.
template <typename IndexT, typename RealT>
void fm2WayNodeBalance(Ctrl<IndexT, RealT>& ctrl, Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& vwgt = graph->vwgt;

  std::vector<IndexT>& bndind = graph->bndind;
  std::vector<IndexT>& bndptr = graph->bndptr;
  std::vector<IndexT>& where = graph->where;
  std::vector<IndexT>& pwgts = graph->pwgts;
  auto& rinfo = graph->nrinfo;

  const RealT mult = RealT(0.5) * ctrl.ubfactor;

  IndexT badmaxpwgt = static_cast<IndexT>(mult * static_cast<RealT>(pwgts[0] + pwgts[1]));
  if (std::max(pwgts[0], pwgts[1]) < badmaxpwgt) return;
  if (std::abs(pwgts[0] - pwgts[1]) < 3 * graph->tvwgt / nvtxs) return;

  const IndexT to = (pwgts[0] < pwgts[1] ? 0 : 1);
  const IndexT other = (to + 1) % 2;

  PQueue<IndexT, IndexT> queue(static_cast<std::size_t>(nvtxs));

  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> moved(static_cast<std::size_t>(nvtxs), IndexT(-1));

  const IndexT nbndInit = graph->nbnd;
  IndexT nbnd = nbndInit;
  randArrayPermute<IndexT>(nbnd, perm.data(), nbnd, 1);
  for (IndexT ii = 0; ii < nbnd; ii++) {
    const IndexT i = bndind[static_cast<std::size_t>(perm[static_cast<std::size_t>(ii)])];
    queue.insert(i, vwgt[static_cast<std::size_t>(i)] - rinfo[static_cast<std::size_t>(i)].edegrees[other]);
  }

  /* Get into the FM loop */
  for (IndexT nswaps = 0; nswaps < nvtxs; nswaps++) {
    const IndexT higain = queue.getTop();
    if (higain == -1) break;

    moved[static_cast<std::size_t>(higain)] = 1;

    const IndexT gain = vwgt[static_cast<std::size_t>(higain)] - rinfo[static_cast<std::size_t>(higain)].edegrees[other];
    badmaxpwgt = static_cast<IndexT>(mult * static_cast<RealT>(pwgts[0] + pwgts[1]));

    /* break if other is now underweight */
    if (pwgts[static_cast<std::size_t>(to)] > pwgts[static_cast<std::size_t>(other)]) break;

    /* break if balance is achieved and no +ve or zero gain */
    if (gain < 0 && pwgts[static_cast<std::size_t>(other)] < badmaxpwgt) break;

    /* skip this vertex if it will violate balance on the other side */
    if (pwgts[static_cast<std::size_t>(to)] + vwgt[static_cast<std::size_t>(higain)] > badmaxpwgt) continue;

    pwgts[2] -= gain;

    bndDelete(nbnd, bndind, bndptr, higain);
    pwgts[static_cast<std::size_t>(to)] += vwgt[static_cast<std::size_t>(higain)];
    where[static_cast<std::size_t>(higain)] = to;

    for (IndexT j = xadj[static_cast<std::size_t>(higain)]; j < xadj[static_cast<std::size_t>(higain) + 1]; j++) {
      const IndexT k = adjncy[static_cast<std::size_t>(j)];
      if (where[static_cast<std::size_t>(k)] == 2) { /* For the in-separator vertices modify their edegree[to] */
        rinfo[static_cast<std::size_t>(k)].edegrees[to] += vwgt[static_cast<std::size_t>(higain)];
      } else if (where[static_cast<std::size_t>(k)] == other) { /* This vertex is pulled into the separator */
        bndInsert(nbnd, bndind, bndptr, k);

        where[static_cast<std::size_t>(k)] = 2;
        pwgts[static_cast<std::size_t>(other)] -= vwgt[static_cast<std::size_t>(k)];

        IndexT* edegrees = rinfo[static_cast<std::size_t>(k)].edegrees;
        edegrees[0] = edegrees[1] = 0;
        for (IndexT jj = xadj[static_cast<std::size_t>(k)]; jj < xadj[static_cast<std::size_t>(k) + 1]; jj++) {
          const IndexT kk = adjncy[static_cast<std::size_t>(jj)];
          if (where[static_cast<std::size_t>(kk)] != 2) {
            edegrees[where[static_cast<std::size_t>(kk)]] += vwgt[static_cast<std::size_t>(kk)];
          } else {
            const IndexT oldgain =
                vwgt[static_cast<std::size_t>(kk)] - rinfo[static_cast<std::size_t>(kk)].edegrees[other];
            rinfo[static_cast<std::size_t>(kk)].edegrees[other] -= vwgt[static_cast<std::size_t>(k)];

            if (moved[static_cast<std::size_t>(kk)] == -1) queue.update(kk, oldgain + vwgt[static_cast<std::size_t>(k)]);
          }
        }

        /* Insert the new vertex into the priority queue */
        queue.insert(k, vwgt[static_cast<std::size_t>(k)] - edegrees[other]);
      }
    }
  }

  graph->mincut = pwgts[2];
  graph->nbnd = nbnd;
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_SEPARATOR_REFINEMENT_H
