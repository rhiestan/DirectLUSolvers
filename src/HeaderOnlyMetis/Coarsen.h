// Faithful port of libmetis/coarsen.c's matching/coarsening pipeline:
// CoarsenGraph, CoarsenGraphNlevels, Match_RM, Match_SHEM, Match_2Hop{,Any,All},
// CreateCoarseGraph, SetupCoarseGraph.
//
// NOT ported (confirmed unreachable on the default-options node-ND path):
//   - Match_JC: not wired to any METIS_CTYPE value (mctype_et only has RM and
//     SHEM), so CoarsenGraph's ctype switch can never reach it.
//   - the `dropedges` branches in CreateCoarseGraph: ctrl->dropedges defaults
//     to 0 (options.c).
//   - `vsize`/dovsize: dovsize is `ctrl->objtype==METIS_OBJTYPE_VOL`, and the
//     default-options OMETIS objtype is always METIS_OBJTYPE_NODE.
//   - multi-constraint (ncon>1) branches: Graph<IndexT,RealT> hardcodes
//     ncon=1, matching Eigen's MetisOrdering call convention.
//   - ReAdjustMemory: exists in the reference to shrink an over-allocated
//     upper-bound adjncy/adjwgt array back down; std::vector only ever holds
//     exactly what's appended to it, so there's nothing to shrink.
//
// Two arithmetic precision traps worth flagging up front, both replicated
// exactly here:
//   - Match_RM/Match_SHEM's avgdegree is computed as `4.0*(xadj[nvtxs]/nvtxs)`
//     -- the division happens in INTEGER arithmetic first (both idx_t), and
//     only the truncated quotient gets promoted to double and multiplied by
//     4.0, not the other way around.
//   - CreateCoarseGraph builds each contracted vertex's edge list into a
//     LOCAL, per-iteration window (the reference advances raw pointers
//     `cadjncy += nedges` through one big pre-allocated array; a htable/
//     dtable dedup pass writes into that window using 0-based *relative*
//     indices). This port uses an explicit local scratch buffer per iteration
//     instead, appended to the growing output vector afterward -- same
//     relative-indexing semantics, no pointer arithmetic on the persistent
//     vector's storage.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_COARSEN_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_COARSEN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "Ctrl.h"
#include "Graph.h"
#include "Random.h"
#include "Sorting.h"

namespace header_only_metis {

constexpr double kCoarsenFraction = 0.85;    // libmetis/defs.h: COARSEN_FRACTION
constexpr double kUnmatchedFor2Hop = 0.10;   // coarsen.c: UNMATCHEDFOR2HOP
constexpr int kHtLength = (1 << 13) - 1;     // libmetis/defs.h: HTLENGTH

template <typename IndexT>
constexpr IndexT kUnmatched = IndexT(-1);  // libmetis/defs.h: UNMATCHED

template <typename IndexT, typename RealT>
void createCoarseGraph(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph, IndexT cnvtxs,
                       const std::vector<IndexT>& match);
template <typename IndexT, typename RealT>
IndexT matchRM(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph);
template <typename IndexT, typename RealT>
IndexT matchSHEM(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph);

// This function takes a graph and creates a sequence of coarser graphs. It
// implements the coarsening phase of the multilevel paradigm.
//
// Returns a raw pointer to the coarsest graph reached, which lives inside
// the `coarser` ownership chain rooted at `graph` -- the caller must keep
// `graph` (or whichever object owns it) alive for as long as the returned
// pointer, or any graph in the chain, is used.
template <typename IndexT, typename RealT>
Graph<IndexT, RealT>* coarsenGraph(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph) {
  IndexT eqewgts = 1;
  for (IndexT i = 1; i < graph->nedges; i++) {
    if (graph->adjwgt[0] != graph->adjwgt[static_cast<std::size_t>(i)]) {
      eqewgts = 0;
      break;
    }
  }

  ctrl.maxvwgt =
      static_cast<IndexT>(1.5 * static_cast<double>(graph->tvwgt) / static_cast<double>(ctrl.CoarsenTo));

  do {
    if (graph->cmap.empty()) graph->cmap.assign(static_cast<std::size_t>(graph->nvtxs), IndexT(0));

    switch (ctrl.ctype) {
      case CType::RM:
        matchRM(ctrl, graph);
        break;
      case CType::SHEM:
        if (eqewgts || graph->nedges == 0)
          matchRM(ctrl, graph);
        else
          matchSHEM(ctrl, graph);
        break;
    }

    graph = graph->coarser.get();
    eqewgts = 0;

  } while (graph->nvtxs > ctrl.CoarsenTo &&
           static_cast<double>(graph->nvtxs) < kCoarsenFraction * static_cast<double>(graph->finer->nvtxs) &&
           graph->nedges > graph->nvtxs / 2);

  return graph;
}

// This function takes a graph and creates a sequence of nlevels coarser
// graphs, where nlevels is an input parameter.
template <typename IndexT, typename RealT>
Graph<IndexT, RealT>* coarsenGraphNlevels(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph,
                                          IndexT nlevels) {
  IndexT eqewgts = 1;
  for (IndexT i = 1; i < graph->nedges; i++) {
    if (graph->adjwgt[0] != graph->adjwgt[static_cast<std::size_t>(i)]) {
      eqewgts = 0;
      break;
    }
  }

  ctrl.maxvwgt =
      static_cast<IndexT>(1.5 * static_cast<double>(graph->tvwgt) / static_cast<double>(ctrl.CoarsenTo));

  for (IndexT level = 0; level < nlevels; level++) {
    if (graph->cmap.empty()) graph->cmap.assign(static_cast<std::size_t>(graph->nvtxs), IndexT(0));

    switch (ctrl.ctype) {
      case CType::RM:
        matchRM(ctrl, graph);
        break;
      case CType::SHEM:
        if (eqewgts || graph->nedges == 0)
          matchRM(ctrl, graph);
        else
          matchSHEM(ctrl, graph);
        break;
    }

    graph = graph->coarser.get();
    eqewgts = 0;

    if (graph->nvtxs < ctrl.CoarsenTo ||
        static_cast<double>(graph->nvtxs) > kCoarsenFraction * static_cast<double>(graph->finer->nvtxs) ||
        graph->nedges < graph->nvtxs / 2)
      break;
  }

  return graph;
}

template <typename IndexT, typename RealT>
IndexT match2HopAny(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph, const std::vector<IndexT>& perm,
                    std::vector<IndexT>& match, IndexT cnvtxs, std::size_t& nunmatched,
                    std::size_t maxdegree) {
  (void)ctrl;
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& adjncy = graph->adjncy;

  /* create the inverted index */
  std::vector<IndexT> colptr(static_cast<std::size_t>(nvtxs) + 1, IndexT(0));
  for (IndexT i = 0; i < nvtxs; i++) {
    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT> &&
        static_cast<std::size_t>(xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)]) <
            maxdegree) {
      for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++)
        colptr[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])]++;
    }
  }
  // MAKECSR(i, nvtxs, colptr)
  for (IndexT i = 1; i < nvtxs; i++)
    colptr[static_cast<std::size_t>(i)] += colptr[static_cast<std::size_t>(i) - 1];
  for (IndexT i = nvtxs; i > 0; i--) colptr[static_cast<std::size_t>(i)] = colptr[static_cast<std::size_t>(i) - 1];
  colptr[0] = 0;

  std::vector<IndexT> rowind(static_cast<std::size_t>(colptr[static_cast<std::size_t>(nvtxs)]));
  for (IndexT pi = 0; pi < nvtxs; pi++) {
    const IndexT i = perm[static_cast<std::size_t>(pi)];
    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT> &&
        static_cast<std::size_t>(xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)]) <
            maxdegree) {
      for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++)
        rowind[static_cast<std::size_t>(colptr[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])]++)] =
            i;
    }
  }
  // SHIFTCSR(i, nvtxs, colptr)
  for (IndexT i = nvtxs; i > 0; i--) colptr[static_cast<std::size_t>(i)] = colptr[static_cast<std::size_t>(i) - 1];
  colptr[0] = 0;

  /* compute matchings by going down the inverted index */
  for (IndexT pi = 0; pi < nvtxs; pi++) {
    const IndexT i = perm[static_cast<std::size_t>(pi)];
    if (colptr[static_cast<std::size_t>(i) + 1] - colptr[static_cast<std::size_t>(i)] < 2) continue;

    IndexT jj = colptr[static_cast<std::size_t>(i) + 1];
    for (IndexT j = colptr[static_cast<std::size_t>(i)]; j < jj; j++) {
      if (match[static_cast<std::size_t>(rowind[static_cast<std::size_t>(j)])] == kUnmatched<IndexT>) {
        for (jj--; jj > j; jj--) {
          if (match[static_cast<std::size_t>(rowind[static_cast<std::size_t>(jj)])] == kUnmatched<IndexT>) {
            cnvtxs++;
            match[static_cast<std::size_t>(rowind[static_cast<std::size_t>(j)])] = rowind[static_cast<std::size_t>(jj)];
            match[static_cast<std::size_t>(rowind[static_cast<std::size_t>(jj)])] = rowind[static_cast<std::size_t>(j)];
            nunmatched -= 2;
            break;
          }
        }
      }
    }
  }

  return cnvtxs;
}

template <typename IndexT, typename RealT>
IndexT match2HopAll(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph, const std::vector<IndexT>& perm,
                    std::vector<IndexT>& match, IndexT cnvtxs, std::size_t& nunmatched,
                    std::size_t maxdegree) {
  (void)ctrl;
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& adjncy = graph->adjncy;

  const IndexT mask = static_cast<IndexT>(static_cast<std::size_t>(std::numeric_limits<IndexT>::max()) / maxdegree);

  std::vector<IndexKeyValue<IndexT>> keys(nunmatched);
  std::size_t ncand = 0;
  for (IndexT pi = 0; pi < nvtxs; pi++) {
    const IndexT i = perm[static_cast<std::size_t>(pi)];
    const IndexT idegree = xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)];
    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT> && idegree > 1 &&
        static_cast<std::size_t>(idegree) < maxdegree) {
      IndexT k = 0;
      for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++)
        k += adjncy[static_cast<std::size_t>(j)] % mask;
      keys[ncand].val = i;
      keys[ncand].key = (k % mask) * static_cast<IndexT>(maxdegree) + idegree;
      ncand++;
    }
  }
  ikvSortInc<IndexT>(ncand, keys.data());

  std::vector<IndexT> mark(static_cast<std::size_t>(nvtxs), IndexT(0));
  for (std::size_t pi = 0; pi < ncand; pi++) {
    const IndexT i = keys[pi].val;
    if (match[static_cast<std::size_t>(i)] != kUnmatched<IndexT>) continue;

    for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++)
      mark[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])] = i;

    for (std::size_t pk = pi + 1; pk < ncand; pk++) {
      const IndexT k = keys[pk].val;
      if (match[static_cast<std::size_t>(k)] != kUnmatched<IndexT>) continue;

      if (keys[pi].key != keys[pk].key) break;
      if (xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)] !=
          xadj[static_cast<std::size_t>(k) + 1] - xadj[static_cast<std::size_t>(k)])
        break;

      IndexT jj = xadj[static_cast<std::size_t>(k)];
      for (; jj < xadj[static_cast<std::size_t>(k) + 1]; jj++) {
        if (mark[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(jj)])] != i) break;
      }
      if (jj == xadj[static_cast<std::size_t>(k) + 1]) {
        cnvtxs++;
        match[static_cast<std::size_t>(i)] = k;
        match[static_cast<std::size_t>(k)] = i;
        nunmatched -= 2;
        break;
      }
    }
  }

  return cnvtxs;
}

template <typename IndexT, typename RealT>
IndexT match2Hop(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph, const std::vector<IndexT>& perm,
                 std::vector<IndexT>& match, IndexT cnvtxs, std::size_t nunmatched) {
  cnvtxs = match2HopAny(ctrl, graph, perm, match, cnvtxs, nunmatched, std::size_t(2));
  cnvtxs = match2HopAll(ctrl, graph, perm, match, cnvtxs, nunmatched, std::size_t(64));
  if (static_cast<double>(nunmatched) > 1.5 * kUnmatchedFor2Hop * static_cast<double>(graph->nvtxs))
    cnvtxs = match2HopAny(ctrl, graph, perm, match, cnvtxs, nunmatched, std::size_t(3));
  if (static_cast<double>(nunmatched) > 2.0 * kUnmatchedFor2Hop * static_cast<double>(graph->nvtxs))
    cnvtxs = match2HopAny(ctrl, graph, perm, match, cnvtxs, nunmatched,
                          static_cast<std::size_t>(graph->nvtxs));
  return cnvtxs;
}

// This function finds a matching by randomly selecting one of the unmatched
// adjacent vertices.
template <typename IndexT, typename RealT>
IndexT matchRM(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  std::vector<IndexT>& cmap = graph->cmap;
  const IndexT maxvwgt = ctrl.maxvwgt;

  std::vector<IndexT> match(static_cast<std::size_t>(nvtxs), kUnmatched<IndexT>);
  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> tperm(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> degrees(static_cast<std::size_t>(nvtxs));
  std::size_t nunmatched = 0;

  /* Determine a "random" traversal order that is biased towards low-degree
     vertices */
  randArrayPermute<IndexT>(nvtxs, tperm.data(), nvtxs / 8, 1);

  const IndexT avgdegree = static_cast<IndexT>(4.0 * static_cast<double>(xadj[static_cast<std::size_t>(nvtxs)] / nvtxs));
  for (IndexT i = 0; i < nvtxs; i++) {
    const IndexT bnum = static_cast<IndexT>(std::sqrt(static_cast<double>(
        1 + xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)])));
    degrees[static_cast<std::size_t>(i)] = (bnum > avgdegree ? avgdegree : bnum);
  }
  bucketSortKeysInc<IndexT>(nvtxs, avgdegree, degrees.data(), tperm.data(), perm.data());

  /* Traverse the vertices and compute the matching */
  IndexT cnvtxs = 0;
  IndexT last_unmatched = 0;
  for (IndexT pi = 0; pi < nvtxs; pi++) {
    const IndexT i = perm[static_cast<std::size_t>(pi)];

    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT>) {
      IndexT maxidx = i;

      if (vwgt[static_cast<std::size_t>(i)] < maxvwgt) {
        /* Deal with island vertices. */
        if (xadj[static_cast<std::size_t>(i)] == xadj[static_cast<std::size_t>(i) + 1]) {
          last_unmatched = std::max(pi, last_unmatched) + 1;
          for (; last_unmatched < nvtxs; last_unmatched++) {
            const IndexT j = perm[static_cast<std::size_t>(last_unmatched)];
            if (match[static_cast<std::size_t>(j)] == kUnmatched<IndexT>) {
              maxidx = j;
              break;
            }
          }
        } else {
          /* Find a random matching, subject to maxvwgt constraints */
          for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++) {
            const IndexT k = adjncy[static_cast<std::size_t>(j)];
            if (match[static_cast<std::size_t>(k)] == kUnmatched<IndexT> &&
                vwgt[static_cast<std::size_t>(i)] + vwgt[static_cast<std::size_t>(k)] <= maxvwgt) {
              maxidx = k;
              break;
            }
          }

          /* If it did not match, record for a 2-hop matching. */
          if (maxidx == i && 2 * vwgt[static_cast<std::size_t>(i)] < maxvwgt) {
            nunmatched++;
            maxidx = kUnmatched<IndexT>;
          }
        }
      }

      if (maxidx != kUnmatched<IndexT>) {
        cnvtxs++;
        match[static_cast<std::size_t>(i)] = maxidx;
        match[static_cast<std::size_t>(maxidx)] = i;
      }
    }
  }

  /* see if a 2-hop matching is required/allowed */
  if (!ctrl.no2hop && static_cast<double>(nunmatched) > kUnmatchedFor2Hop * static_cast<double>(nvtxs))
    cnvtxs = match2Hop(ctrl, graph, perm, match, cnvtxs, nunmatched);

  /* match the final unmatched vertices with themselves and reorder the
     vertices of the coarse graph for memory-friendly contraction */
  cnvtxs = 0;
  for (IndexT i = 0; i < nvtxs; i++) {
    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT>) {
      match[static_cast<std::size_t>(i)] = i;
      cmap[static_cast<std::size_t>(i)] = cnvtxs++;
    } else {
      if (i <= match[static_cast<std::size_t>(i)])
        cmap[static_cast<std::size_t>(i)] = cmap[static_cast<std::size_t>(match[static_cast<std::size_t>(i)])] =
            cnvtxs++;
    }
  }

  createCoarseGraph(ctrl, graph, cnvtxs, match);

  return cnvtxs;
}

// This function finds a matching using the HEM heuristic. The vertices are
// visited based on increasing degree to ensure that all vertices are given a
// chance to match with something.
template <typename IndexT, typename RealT>
IndexT matchSHEM(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph) {
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& adjwgt = graph->adjwgt;
  std::vector<IndexT>& cmap = graph->cmap;
  const IndexT maxvwgt = ctrl.maxvwgt;

  std::vector<IndexT> match(static_cast<std::size_t>(nvtxs), kUnmatched<IndexT>);
  std::vector<IndexT> perm(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> tperm(static_cast<std::size_t>(nvtxs));
  std::vector<IndexT> degrees(static_cast<std::size_t>(nvtxs));
  std::size_t nunmatched = 0;

  randArrayPermute<IndexT>(nvtxs, tperm.data(), nvtxs / 8, 1);

  const IndexT avgdegree = static_cast<IndexT>(4.0 * static_cast<double>(xadj[static_cast<std::size_t>(nvtxs)] / nvtxs));
  for (IndexT i = 0; i < nvtxs; i++) {
    const IndexT bnum = static_cast<IndexT>(std::sqrt(static_cast<double>(
        1 + xadj[static_cast<std::size_t>(i) + 1] - xadj[static_cast<std::size_t>(i)])));
    degrees[static_cast<std::size_t>(i)] = (bnum > avgdegree ? avgdegree : bnum);
  }
  bucketSortKeysInc<IndexT>(nvtxs, avgdegree, degrees.data(), tperm.data(), perm.data());

  IndexT cnvtxs = 0;
  IndexT last_unmatched = 0;
  for (IndexT pi = 0; pi < nvtxs; pi++) {
    const IndexT i = perm[static_cast<std::size_t>(pi)];

    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT>) {
      IndexT maxidx = i;
      IndexT maxwgt = -1;

      if (vwgt[static_cast<std::size_t>(i)] < maxvwgt) {
        if (xadj[static_cast<std::size_t>(i)] == xadj[static_cast<std::size_t>(i) + 1]) {
          last_unmatched = std::max(pi, last_unmatched) + 1;
          for (; last_unmatched < nvtxs; last_unmatched++) {
            const IndexT j = perm[static_cast<std::size_t>(last_unmatched)];
            if (match[static_cast<std::size_t>(j)] == kUnmatched<IndexT>) {
              maxidx = j;
              break;
            }
          }
        } else {
          /* Find a heavy-edge matching, subject to maxvwgt constraints */
          for (IndexT j = xadj[static_cast<std::size_t>(i)]; j < xadj[static_cast<std::size_t>(i) + 1]; j++) {
            const IndexT k = adjncy[static_cast<std::size_t>(j)];
            if (maxwgt < adjwgt[static_cast<std::size_t>(j)] && match[static_cast<std::size_t>(k)] == kUnmatched<IndexT> &&
                vwgt[static_cast<std::size_t>(i)] + vwgt[static_cast<std::size_t>(k)] <= maxvwgt) {
              maxidx = k;
              maxwgt = adjwgt[static_cast<std::size_t>(j)];
            }
          }

          if (maxidx == i && 2 * vwgt[static_cast<std::size_t>(i)] < maxvwgt) {
            nunmatched++;
            maxidx = kUnmatched<IndexT>;
          }
        }
      }

      if (maxidx != kUnmatched<IndexT>) {
        cnvtxs++;
        match[static_cast<std::size_t>(i)] = maxidx;
        match[static_cast<std::size_t>(maxidx)] = i;
      }
    }
  }

  if (!ctrl.no2hop && static_cast<double>(nunmatched) > kUnmatchedFor2Hop * static_cast<double>(nvtxs))
    cnvtxs = match2Hop(ctrl, graph, perm, match, cnvtxs, nunmatched);

  cnvtxs = 0;
  for (IndexT i = 0; i < nvtxs; i++) {
    if (match[static_cast<std::size_t>(i)] == kUnmatched<IndexT>) {
      match[static_cast<std::size_t>(i)] = i;
      cmap[static_cast<std::size_t>(i)] = cnvtxs++;
    } else {
      if (i <= match[static_cast<std::size_t>(i)])
        cmap[static_cast<std::size_t>(i)] = cmap[static_cast<std::size_t>(match[static_cast<std::size_t>(i)])] =
            cnvtxs++;
    }
  }

  createCoarseGraph(ctrl, graph, cnvtxs, match);

  return cnvtxs;
}

// This function creates the coarser graph. Depending on the size of the
// candidate adjacency lists it either uses a hash table or an array to do
// duplicate detection.
template <typename IndexT, typename RealT>
void createCoarseGraph(Ctrl<IndexT>& ctrl, Graph<IndexT, RealT>* graph, IndexT cnvtxs,
                       const std::vector<IndexT>& match) {
  (void)ctrl;
  const IndexT nvtxs = graph->nvtxs;
  const std::vector<IndexT>& xadj = graph->xadj;
  const std::vector<IndexT>& vwgt = graph->vwgt;
  const std::vector<IndexT>& adjncy = graph->adjncy;
  const std::vector<IndexT>& adjwgt = graph->adjwgt;
  const std::vector<IndexT>& cmap = graph->cmap;

  const IndexT mask = kHtLength;

  auto cgraph = std::make_unique<Graph<IndexT, RealT>>();
  cgraph->nvtxs = cnvtxs;
  cgraph->xadj.assign(static_cast<std::size_t>(cnvtxs) + 1, IndexT(0));
  cgraph->vwgt.assign(static_cast<std::size_t>(cnvtxs), IndexT(0));
  cgraph->adjncy.reserve(static_cast<std::size_t>(graph->nedges));
  cgraph->adjwgt.reserve(static_cast<std::size_t>(graph->nedges));

  std::vector<IndexT> htable(static_cast<std::size_t>(mask) + 1, IndexT(-1));
  std::vector<IndexT> dtable(static_cast<std::size_t>(cnvtxs), IndexT(-1));

  std::vector<IndexT> localAdjncy;
  std::vector<IndexT> localAdjwgt;

  cgraph->xadj[0] = 0;
  IndexT runningCnvtxs = 0;
  IndexT cnedges = 0;
  for (IndexT v = 0; v < nvtxs; v++) {
    const IndexT u = match[static_cast<std::size_t>(v)];
    if (u < v) continue;

    /* take care of the vertices */
    cgraph->vwgt[static_cast<std::size_t>(runningCnvtxs)] = vwgt[static_cast<std::size_t>(v)];
    if (v != u) cgraph->vwgt[static_cast<std::size_t>(runningCnvtxs)] += vwgt[static_cast<std::size_t>(u)];

    /* take care of the edges */
    const IndexT degv = xadj[static_cast<std::size_t>(v) + 1] - xadj[static_cast<std::size_t>(v)];
    const IndexT degu = xadj[static_cast<std::size_t>(u) + 1] - xadj[static_cast<std::size_t>(u)];
    IndexT nedges;

    if ((degv + degu) < (mask >> 2)) {
      /* use hashtable. put the ID of the contracted node itself at the start
         so it can be removed easily. */
      localAdjncy.resize(static_cast<std::size_t>(degv + degu) + 1);
      localAdjwgt.resize(static_cast<std::size_t>(degv + degu) + 1);

      htable[static_cast<std::size_t>(runningCnvtxs & mask)] = 0;
      localAdjncy[0] = runningCnvtxs;
      nedges = 1;

      for (IndexT pass = 0; pass < (v != u ? 2 : 1); pass++) {
        const IndexT src = (pass == 0 ? v : u);
        const IndexT istart = xadj[static_cast<std::size_t>(src)];
        const IndexT iend = xadj[static_cast<std::size_t>(src) + 1];
        for (IndexT j = istart; j < iend; j++) {
          const IndexT k = cmap[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])];
          IndexT kk = k & mask;
          while (htable[static_cast<std::size_t>(kk)] != -1 &&
                localAdjncy[static_cast<std::size_t>(htable[static_cast<std::size_t>(kk)])] != k)
            kk = (kk + 1) & mask;
          const IndexT m = htable[static_cast<std::size_t>(kk)];
          if (m == -1) {
            localAdjncy[static_cast<std::size_t>(nedges)] = k;
            localAdjwgt[static_cast<std::size_t>(nedges)] = adjwgt[static_cast<std::size_t>(j)];
            htable[static_cast<std::size_t>(kk)] = nedges++;
          } else {
            localAdjwgt[static_cast<std::size_t>(m)] += adjwgt[static_cast<std::size_t>(j)];
          }
        }
      }

      /* reset the htable -- reverse order (LIFO) is critical to prevent
         localAdjncy[-1] indexing due to a remove of an earlier entry */
      for (IndexT j = nedges - 1; j >= 0; j--) {
        const IndexT k = localAdjncy[static_cast<std::size_t>(j)];
        IndexT kk = k & mask;
        while (localAdjncy[static_cast<std::size_t>(htable[static_cast<std::size_t>(kk)])] != k) kk = (kk + 1) & mask;
        htable[static_cast<std::size_t>(kk)] = -1;
      }

      /* remove the contracted vertex from the list */
      localAdjncy[0] = localAdjncy[static_cast<std::size_t>(--nedges)];
      localAdjwgt[0] = localAdjwgt[static_cast<std::size_t>(nedges)];
    } else {
      /* use direct table */
      localAdjncy.resize(static_cast<std::size_t>(degv + degu));
      localAdjwgt.resize(static_cast<std::size_t>(degv + degu));
      nedges = 0;

      IndexT istart = xadj[static_cast<std::size_t>(v)];
      IndexT iend = xadj[static_cast<std::size_t>(v) + 1];
      for (IndexT j = istart; j < iend; j++) {
        const IndexT k = cmap[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])];
        const IndexT m = dtable[static_cast<std::size_t>(k)];
        if (m == -1) {
          localAdjncy[static_cast<std::size_t>(nedges)] = k;
          localAdjwgt[static_cast<std::size_t>(nedges)] = adjwgt[static_cast<std::size_t>(j)];
          dtable[static_cast<std::size_t>(k)] = nedges++;
        } else {
          localAdjwgt[static_cast<std::size_t>(m)] += adjwgt[static_cast<std::size_t>(j)];
        }
      }

      if (v != u) {
        istart = xadj[static_cast<std::size_t>(u)];
        iend = xadj[static_cast<std::size_t>(u) + 1];
        for (IndexT j = istart; j < iend; j++) {
          const IndexT k = cmap[static_cast<std::size_t>(adjncy[static_cast<std::size_t>(j)])];
          const IndexT m = dtable[static_cast<std::size_t>(k)];
          if (m == -1) {
            localAdjncy[static_cast<std::size_t>(nedges)] = k;
            localAdjwgt[static_cast<std::size_t>(nedges)] = adjwgt[static_cast<std::size_t>(j)];
            dtable[static_cast<std::size_t>(k)] = nedges++;
          } else {
            localAdjwgt[static_cast<std::size_t>(m)] += adjwgt[static_cast<std::size_t>(j)];
          }
        }

        /* Remove the contracted self-loop, when present */
        const IndexT j = dtable[static_cast<std::size_t>(runningCnvtxs)];
        if (j != -1) {
          localAdjncy[static_cast<std::size_t>(j)] = localAdjncy[static_cast<std::size_t>(--nedges)];
          localAdjwgt[static_cast<std::size_t>(j)] = localAdjwgt[static_cast<std::size_t>(nedges)];
          dtable[static_cast<std::size_t>(runningCnvtxs)] = -1;
        }
      }

      /* Zero out the dtable */
      for (IndexT j = 0; j < nedges; j++) dtable[static_cast<std::size_t>(localAdjncy[static_cast<std::size_t>(j)])] = -1;
    }

    cgraph->adjncy.insert(cgraph->adjncy.end(), localAdjncy.begin(), localAdjncy.begin() + nedges);
    cgraph->adjwgt.insert(cgraph->adjwgt.end(), localAdjwgt.begin(), localAdjwgt.begin() + nedges);
    cnedges += nedges;
    cgraph->xadj[static_cast<std::size_t>(++runningCnvtxs)] = cnedges;
  }

  cgraph->nedges = cnedges;
  cgraph->setupTvwgt();

  cgraph->finer = graph;
  graph->coarser = std::move(cgraph);
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_COARSEN_H
