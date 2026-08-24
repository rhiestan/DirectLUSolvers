// Faithful port of libmetis/compress.c's CompressGraph: merges vertices with
// identical adjacency structure, used because ctrl->compress defaults to 1
// for METIS_NodeND (options.c) -- this runs on the default Eigen call path.
// PruneGraph (the other function in compress.c) is NOT ported: ctrl->pfactor
// defaults to 0, so it never runs on that path.
//
// The reference signature takes a ctrl_t* but only ever reads
// ctrl->dbglvl through IFSET(...) debug-print gates (grep-confirmed: no other
// ctrl field is touched anywhere in CompressGraph). Since this port's target
// scope has dbglvl==0 always (Eigen's call never sets debug options), those
// branches are provably dead code, so ctrl is dropped from the signature
// entirely rather than carried as an unused parameter.
//
// Depends on Sorting.h's ikvSortInc (GKlib's non-stable quicksort) for the
// exact tie-break behaviour of ikvsorti on the adjacency-sum key -- ties are
// common (many small graphs share `sum(adjncy)+i` collisions) and determine
// which vertex becomes a merged group's representative, so this cannot be
// swapped for std::sort/std::stable_sort without risking a different (still
// "correct," but not bit-identical) representative choice.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_COMPRESS_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_COMPRESS_H

#include <memory>
#include <vector>

#include "Graph.h"
#include "Sorting.h"

namespace header_only_metis {

constexpr double kCompressionFraction = 0.85;  // libmetis/defs.h: COMPRESSION_FRACTION

// This function compresses a graph by merging identical vertices. The
// compression should lead to at least 10% reduction.
//
// The compressed graph that is generated has its adjwgts set to 1.
//
// Returns nullptr if compression did not reduce the vertex count enough to
// be worthwhile (matches the reference returning NULL).
//
// vwgt may be null (matches vwgt==NULL: every vertex has implicit weight 1).
template <typename IndexT, typename RealT>
std::unique_ptr<Graph<IndexT, RealT>> compressGraph(IndexT nvtxs, const IndexT* xadj,
                                                     const IndexT* adjncy, const IndexT* vwgt,
                                                     IndexT* cptr, IndexT* cind) {
  IndexT i, ii, iii, j, jj, k, l, cnvtxs;

  std::vector<IndexT> mark(static_cast<std::size_t>(nvtxs), IndexT(-1));
  std::vector<IndexT> map(static_cast<std::size_t>(nvtxs), IndexT(-1));
  std::vector<IndexKeyValue<IndexT>> keys(static_cast<std::size_t>(nvtxs));

  /* Compute a key for each adjacency list */
  for (i = 0; i < nvtxs; i++) {
    k = 0;
    for (j = xadj[i]; j < xadj[i + 1]; j++) k += adjncy[j];
    keys[static_cast<std::size_t>(i)].key = k + i; /* Add the diagonal entry as well */
    keys[static_cast<std::size_t>(i)].val = i;
  }

  ikvSortInc<IndexT>(static_cast<std::size_t>(nvtxs), keys.data());

  l = cptr[0] = 0;
  for (cnvtxs = i = 0; i < nvtxs; i++) {
    ii = keys[static_cast<std::size_t>(i)].val;
    if (map[static_cast<std::size_t>(ii)] == -1) {
      mark[static_cast<std::size_t>(ii)] = i; /* Add the diagonal entry */
      for (j = xadj[ii]; j < xadj[ii + 1]; j++) mark[static_cast<std::size_t>(adjncy[j])] = i;

      map[static_cast<std::size_t>(ii)] = cnvtxs;
      cind[l++] = ii;

      for (j = i + 1; j < nvtxs; j++) {
        iii = keys[static_cast<std::size_t>(j)].val;

        if (keys[static_cast<std::size_t>(i)].key != keys[static_cast<std::size_t>(j)].key ||
            xadj[ii + 1] - xadj[ii] != xadj[iii + 1] - xadj[iii])
          break; /* Break if keys or degrees are different */

        if (map[static_cast<std::size_t>(iii)] == -1) { /* Do a comparison if iii has not been mapped */
          for (jj = xadj[iii]; jj < xadj[iii + 1]; jj++) {
            if (mark[static_cast<std::size_t>(adjncy[jj])] != i) break;
          }

          if (jj == xadj[iii + 1]) { /* Identical adjacency structure */
            map[static_cast<std::size_t>(iii)] = cnvtxs;
            cind[l++] = iii;
          }
        }
      }

      cptr[++cnvtxs] = l;

      if (static_cast<double>(cnvtxs) >= kCompressionFraction * static_cast<double>(nvtxs)) break;
    }
  }

  if (static_cast<double>(cnvtxs) < kCompressionFraction * static_cast<double>(nvtxs)) {
    /* Sufficient compression is possible, so go ahead and create the
       compressed graph */
    auto graph = std::make_unique<Graph<IndexT, RealT>>();

    IndexT cnedges = 0;
    for (i = 0; i < cnvtxs; i++) {
      ii = cind[cptr[i]];
      cnedges += xadj[ii + 1] - xadj[ii];
    }

    graph->xadj.assign(static_cast<std::size_t>(cnvtxs) + 1, IndexT(0));
    graph->vwgt.assign(static_cast<std::size_t>(cnvtxs), IndexT(0));
    graph->adjncy.assign(static_cast<std::size_t>(cnedges), IndexT(0));
    graph->adjwgt.assign(static_cast<std::size_t>(cnedges), IndexT(1));

    /* Now go and compress the graph */
    std::fill(mark.begin(), mark.end(), IndexT(-1));
    l = 0;
    graph->xadj[0] = 0;
    for (i = 0; i < cnvtxs; i++) {
      mark[static_cast<std::size_t>(i)] = i; /* Remove any diagonal entries in the compressed graph */
      for (j = cptr[i]; j < cptr[i + 1]; j++) {
        ii = cind[j];

        /* accumulate the vertex weights of the constituent vertices */
        graph->vwgt[static_cast<std::size_t>(i)] += (vwgt == nullptr ? IndexT(1) : vwgt[ii]);

        /* generate the combined adjacency list */
        for (jj = xadj[ii]; jj < xadj[ii + 1]; jj++) {
          k = map[static_cast<std::size_t>(adjncy[jj])];
          if (mark[static_cast<std::size_t>(k)] != i) {
            mark[static_cast<std::size_t>(k)] = i;
            graph->adjncy[static_cast<std::size_t>(l++)] = k;
          }
        }
      }
      graph->xadj[static_cast<std::size_t>(i) + 1] = l;
    }

    // adjncy/adjwgt were sized to an UPPER BOUND (the representative
    // vertices' original degree sum) since the exact post-dedup,
    // post-self-loop-removal count isn't known until the loop above
    // finishes. The reference leaves its malloc'd arrays at that upper bound
    // too and relies on every reader trusting nedges/xadj as the logical
    // length rather than the array's allocated size -- std::vector has no
    // such "logical vs. allocated" distinction, so this must be truncated to
    // the true edge count l here, or callers reading the full vector would
    // see stale trailing zeros past the real data.
    graph->adjncy.resize(static_cast<std::size_t>(l));
    graph->adjwgt.resize(static_cast<std::size_t>(l));

    graph->nvtxs = cnvtxs;
    graph->nedges = l;

    graph->setupTvwgt();
    graph->setupLabel();

    return graph;
  }

  return nullptr;
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_COMPRESS_H
