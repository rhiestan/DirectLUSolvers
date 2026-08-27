// Drop-in, dependency-free replacement for Eigen::MetisOrdering.
//
// Eigen::MetisOrdering (Eigen/src/MetisSupport/MetisSupport.h) calls exactly
// one METIS entry point -- METIS_NodeND -- and therefore forces every consumer
// to link the METIS library and its GKlib dependency.
// Eigen::HeaderOnlyMetisOrdering below mirrors that class's public contract
// exactly but calls header_only_metis::nodeND(), a templated reimplementation
// of the same algorithm living under HeaderOnlyMetis/. No link-time
// dependency, and the ordering is bit-identical to the C library's (see
// HeaderOnlyMetis/NestedDissection.h and test/test_header_only_metis.cpp).
//
// Usage -- anywhere MetisOrdering is accepted as the OrderingType_ parameter:
//
//     #include <HeaderOnlyMetis.h>        // DirectLUSolvers/src on include path
//     LeftRightLU<SparseMatrix<double>,
//                 HeaderOnlyMetisOrdering<int>> solver;
//
// It is the ordering functor alone -- deliberately not paired with any solver
// alias, so that LeftRightLU, SupernodalLU and PointBlockLU can all take it
// without this header dragging one solver's definition in behind it. The
// no-link equivalent of Eigen::SupernodalLUMetis is therefore spelled out at
// the use site:
//
//     SupernodalLU<SparseMatrix<double>, HeaderOnlyMetisOrdering<int>> solver;
//
// Bit-identity is defined against a reference METIS built with the SAME type
// widths (IDXTYPEWIDTH=32, REALTYPEWIDTH=32 -- the defaults) and with GKlib's
// GKRAND=ON, i.e. its portable MT19937-64 rather than the platform CRT's
// rand(). RealT is pinned to float below for that reason: it is the width of
// METIS's real_t in that build, and the balance thresholds this algorithm
// truncates to integers are sensitive to it.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_H

#include <Eigen/SparseCore>

#include "HeaderOnlyMetis/NestedDissection.h"
#include "HeaderOnlyMetis/NestedDissectionParallel.h"

namespace Eigen {

/** \brief Fill-reducing nested-dissection ordering, header-only.
 *
 * Mirrors Eigen::MetisOrdering's contract exactly, including the direction of
 * the permutation it hands back:
 *
 *   If A is the original matrix and Ap the permuted matrix, row (column) i of
 *   A is the matperm(i) row (column) of Ap. As computed by METIS this is the
 *   `iperm` vector, not `perm`.
 *
 * That direction matters to this project's solvers: they read it through
 * left_right_lu::OrderingConvention / point_block::OrderingConvention, whose
 * primary template already declares `returnsInverse = true` -- the convention
 * AMDOrdering and MetisOrdering use. Because this class deliberately matches
 * MetisOrdering, it needs no specialization of those traits, and must not
 * grow one. Reading the permutation the wrong way round stays a valid
 * permutation with a machine-precision residual and shows up only as fill
 * (measured elsewhere in this project at 250-350x on 3D FEM systems), so it
 * is not something a numerical check would catch.
 *
 * \tparam StorageIndex the matrix's index type (typically int).
 */
template <typename StorageIndex>
class HeaderOnlyMetisOrdering {
 public:
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;
  typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;

  /** Builds the adjacency structure of A + A^T (diagonal excluded) into
   * m_indexPtr / m_innerIndices.
   *
   * Reproduces Eigen::MetisOrdering::get_symmetrized_graph line for line --
   * the same two-pass visited-marker construction, in the same order. That
   * exactness is load-bearing rather than stylistic: nested dissection's
   * tie-breaking depends on the order neighbours appear in each adjacency
   * list, so a differently-ordered (but structurally identical) graph yields
   * a different, equally valid ordering -- which would defeat the point of
   * being a drop-in replacement.
   */
  template <typename MatrixType>
  void get_symmetrized_graph(const MatrixType& A) {
    Index m = A.cols();
    eigen_assert((A.rows() == A.cols()) && "ONLY FOR SQUARED MATRICES");
    // Get the transpose of the input matrix
    MatrixType At = A.transpose();
    // Get the number of nonzeros elements in each row/col of At+A
    Index TotNz = 0;
    IndexVector visited(m);
    visited.setConstant(-1);
    for (StorageIndex j = 0; j < m; j++) {
      // Compute the union structure of A(j,:) and At(j,:)
      visited(j) = j;  // Do not include the diagonal element
      // Get the nonzeros in row/column j of A
      for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
        Index idx = it.index();  // Get the row index (for column major) or column index (for row major)
        if (visited(idx) != j) {
          visited(idx) = j;
          ++TotNz;
        }
      }
      // Get the nonzeros in row/column j of At
      for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
        Index idx = it.index();
        if (visited(idx) != j) {
          visited(idx) = j;
          ++TotNz;
        }
      }
    }
    // Reserve place for A + At
    m_indexPtr.resize(m + 1);
    m_innerIndices.resize(TotNz);

    // Now compute the real adjacency list of each column/row
    visited.setConstant(-1);
    StorageIndex CurNz = 0;
    for (StorageIndex j = 0; j < m; j++) {
      m_indexPtr(j) = CurNz;

      visited(j) = j;  // Do not include the diagonal element
      // Add the pattern of row/column j of A to A+At
      for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
        StorageIndex idx = it.index();  // Get the row index (for column major) or column index (for row major)
        if (visited(idx) != j) {
          visited(idx) = j;
          m_innerIndices(CurNz) = idx;
          CurNz++;
        }
      }
      // Add the pattern of row/column j of At to A+At
      for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
        StorageIndex idx = it.index();
        if (visited(idx) != j) {
          visited(idx) = j;
          m_innerIndices(CurNz) = idx;
          ++CurNz;
        }
      }
    }
    m_indexPtr(m) = CurNz;
  }

  template <typename MatrixType>
  void operator()(const MatrixType& A, PermutationType& matperm) {
    StorageIndex m = internal::convert_index<StorageIndex>(A.cols());
    IndexVector perm(m), iperm(m);
    // First, symmetrize the matrix graph.
    get_symmetrized_graph(A);

    // An empty matrix has no adjacency arrays to hand over; METIS_NodeND is
    // not called for it either (m==0 makes the loop below a no-op), so just
    // return the empty permutation rather than passing null pointers on.
    if (m == 0) {
      matperm.resize(0);
      return;
    }

    // Call the fill-reducing routine -- the header-only port rather than the
    // linked C library. RealT is float: see the file header.
    header_only_metis::nodeND<StorageIndex, float>(m, m_indexPtr.data(), m_innerIndices.data(),
                                                   /*vwgt=*/nullptr, perm.data(), iperm.data());

    // Get the fill-reducing permutation
    // NOTE:  If Ap is the permuted matrix then perm and iperm vectors are defined as follows
    // Row (column) i of Ap is the perm(i) row(column) of A, and row (column) i of A is the iperm(i) row(column) of Ap
    matperm.resize(m);
    for (StorageIndex j = 0; j < m; j++) matperm.indices()(iperm(j)) = j;
  }

 protected:
  IndexVector m_indexPtr;      // Pointer to the adjacency list of each row/column
  IndexVector m_innerIndices;  // Adjacency list
};

/** \brief Parallel, deterministic nested-dissection ordering.
 *
 * Same interface as HeaderOnlyMetisOrdering, but the dissection tree is walked
 * level by level and dispatched through an Executor, and each subtree seeds its
 * own generator from its position in the tree.
 *
 * Its output is NOT the METIS ordering. It is a different, equally valid
 * fill-reducing permutation -- deliberately so: the reference's result depends
 * on the order a single shared random stream is consumed in, which a parallel
 * traversal cannot reproduce (see HeaderOnlyMetis/NestedDissectionParallel.h).
 * Use HeaderOnlyMetisOrdering when matching METIS matters; use this when
 * ordering time does.
 *
 * It IS deterministic: the permutation depends only on the matrix, not on the
 * executor, the thread count or the scheduling. Fill baselines recorded against
 * it therefore stay meaningful, and a threaded run can be checked against a
 * SerialExecutor run of the same class.
 *
 * \tparam StorageIndex the matrix's index type (typically int).
 * \tparam Executor     anything with parallelFor/concurrency; defaults to
 *                      serial. Eigen::supernodal_lu::StdThreadExecutor and the
 *                      OpenMP/TBB backends all qualify.
 */
template <typename StorageIndex, typename Executor = header_only_metis::SerialExecutor>
class HeaderOnlyMetisParallelOrdering : public HeaderOnlyMetisOrdering<StorageIndex> {
 public:
  typedef typename HeaderOnlyMetisOrdering<StorageIndex>::PermutationType PermutationType;
  typedef typename HeaderOnlyMetisOrdering<StorageIndex>::IndexVector IndexVector;

  HeaderOnlyMetisParallelOrdering() = default;
  explicit HeaderOnlyMetisParallelOrdering(const Executor& exec) : m_exec(&exec) {}

  /** The executor to dispatch through. Null means the built-in serial one. */
  void setExecutor(const Executor& exec) { m_exec = &exec; }

  template <typename MatrixType>
  void operator()(const MatrixType& A, PermutationType& matperm) {
    StorageIndex m = internal::convert_index<StorageIndex>(A.cols());
    IndexVector perm(m), iperm(m);
    this->get_symmetrized_graph(A);

    if (m == 0) {
      matperm.resize(0);
      return;
    }

    const Executor* exec = m_exec;
    if (exec != nullptr) {
      header_only_metis::nodeNDParallel<StorageIndex, float>(m, this->m_indexPtr.data(),
                                                             this->m_innerIndices.data(), /*vwgt=*/nullptr,
                                                             perm.data(), iperm.data(), *exec);
    } else {
      const header_only_metis::SerialExecutor serial;
      header_only_metis::nodeNDParallel<StorageIndex, float>(m, this->m_indexPtr.data(),
                                                             this->m_innerIndices.data(), /*vwgt=*/nullptr,
                                                             perm.data(), iperm.data(), serial);
    }

    matperm.resize(m);
    for (StorageIndex j = 0; j < m; j++) matperm.indices()(iperm(j)) = j;
  }

 private:
  const Executor* m_exec = nullptr;  // non-owning; the caller keeps the pool alive
};

}  // namespace Eigen

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_H
