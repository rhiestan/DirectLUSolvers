// Symmetrized-graph construction shared by the METIS-comparison test suites.
//
// This replicates Eigen::MetisOrdering::get_symmetrized_graph
// (Eigen/src/MetisSupport/MetisSupport.h) exactly: the same two-pass
// visited-marker construction, the same "diagonal excluded" convention. That
// exactness matters here specifically -- this is the graph METIS_NodeND
// actually sees when called through Eigen, so a test comparing orderings must
// feed both sides the identical xadj/adjncy or a mismatch could just be a
// difference in this construction, not in the ordering algorithm being tested.
//
// Templated on the index type (IndexT) rather than on <metis.h>'s idx_t
// directly, so this header has no METIS dependency itself and can be reused
// by the header-only port's own tests later.

#ifndef DIRECTLUSOLVERS_TEST_TESTING_METISGRAPH_H
#define DIRECTLUSOLVERS_TEST_TESTING_METISGRAPH_H

#include <Eigen/SparseCore>

#include <algorithm>
#include <vector>

namespace lu_testing {

template <typename IndexT>
struct SymmetrizedGraph {
  std::vector<IndexT> xadj;
  std::vector<IndexT> adjncy;
  IndexT nvtxs = 0;
};

template <typename IndexT, typename MatrixType>
SymmetrizedGraph<IndexT> buildSymmetrizedGraph(const MatrixType& A) {
  using Eigen::Index;
  eigen_assert(A.rows() == A.cols() && "buildSymmetrizedGraph: matrix must be square");
  const IndexT m = static_cast<IndexT>(A.cols());
  MatrixType At = A.transpose();

  std::vector<IndexT> visited(static_cast<std::size_t>(m), IndexT(-1));
  Index totNz = 0;
  for (IndexT j = 0; j < m; ++j) {
    visited[static_cast<std::size_t>(j)] = j;  // diagonal excluded
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      const IndexT idx = static_cast<IndexT>(it.index());
      if (visited[static_cast<std::size_t>(idx)] != j) {
        visited[static_cast<std::size_t>(idx)] = j;
        ++totNz;
      }
    }
    for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
      const IndexT idx = static_cast<IndexT>(it.index());
      if (visited[static_cast<std::size_t>(idx)] != j) {
        visited[static_cast<std::size_t>(idx)] = j;
        ++totNz;
      }
    }
  }

  SymmetrizedGraph<IndexT> g;
  g.nvtxs = m;
  g.xadj.assign(static_cast<std::size_t>(m) + 1, IndexT(0));
  g.adjncy.assign(static_cast<std::size_t>(totNz), IndexT(0));

  std::fill(visited.begin(), visited.end(), IndexT(-1));
  IndexT curNz = 0;
  for (IndexT j = 0; j < m; ++j) {
    g.xadj[static_cast<std::size_t>(j)] = curNz;
    visited[static_cast<std::size_t>(j)] = j;
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      const IndexT idx = static_cast<IndexT>(it.index());
      if (visited[static_cast<std::size_t>(idx)] != j) {
        visited[static_cast<std::size_t>(idx)] = j;
        g.adjncy[static_cast<std::size_t>(curNz)] = idx;
        ++curNz;
      }
    }
    for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
      const IndexT idx = static_cast<IndexT>(it.index());
      if (visited[static_cast<std::size_t>(idx)] != j) {
        visited[static_cast<std::size_t>(idx)] = j;
        g.adjncy[static_cast<std::size_t>(curNz)] = idx;
        ++curNz;
      }
    }
  }
  g.xadj[static_cast<std::size_t>(m)] = curNz;
  return g;
}

}  // namespace lu_testing

#endif  // DIRECTLUSOLVERS_TEST_TESTING_METISGRAPH_H
