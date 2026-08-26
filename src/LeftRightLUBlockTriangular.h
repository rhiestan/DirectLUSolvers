// Block triangular form (BTF) for LeftRightLU -- the second half of the
// Dulmage-Mendelsohn decomposition.
//
// BTF is a PERMUTATION, not a factorization. It comes in two phases, and this
// header is only the second one:
//
//   phase 1  a maximum transversal permutes rows so every diagonal entry is
//            structurally nonzero. LeftRightLU::analyzePattern already does
//            this (SupernodalLUMatching / SupernodalLUMC64), so it is an input
//            here, not something this header computes.
//
//   phase 2  view the matched matrix as a digraph and find its strongly
//            connected components. Ordering the components topologically
//            permutes the matrix to block upper triangular form:
//
//                        [ A00  *    *   ]
//                P A Q = [      A11  *   ]
//                        [           A22 ]
//
// Why the zero-free diagonal from phase 1 matters: it gives every vertex a
// self-loop, which makes the SCCs of that digraph exactly the IRREDUCIBLE
// blocks. That also yields a fact worth relying on -- the block structure is
// unique, independent of WHICH maximum transversal phase 1 picked. MC64's
// magnitude-maximising matching and a purely structural transversal produce the
// same blocks, only different values on the diagonal.
//
// What the caller does with it: factor the diagonal blocks and nothing else.
// The off-diagonal blocks are never eliminated -- they are applied once each
// during a block back-substitution, so they generate no fill and introduce no
// rounding error of their own. Running a normal LU over the whole block
// triangular matrix instead would fill U's off-diagonal blocks in with
// L00^-1 A01 and defeat the entire point.
//
// EDGE DIRECTION AND BLOCK NUMBERING. The digraph here has an edge j -> i for
// every nonzero A(i, j) -- that is, the successors of column j are the rows it
// occupies, which is exactly a column-major matrix's inner index list, so no
// transpose is ever formed. Under that convention Tarjan's natural pop order
// already numbers the components correctly: it completes a component only after
// every component reachable from it, so a cross-block nonzero A(i, j) always has
// component(i) < component(j) -- the definition of block UPPER triangular. No
// separate topological sort is needed, and none is done. (isBlockUpperTriangular
// below checks the property directly; the test suite asserts it rather than
// leaving it as a comment.)
//
// Cost: one O(n + nnz) sweep. Measured against the two steps analyzePattern
// already runs on the same graph, this is 5-100x cheaper than the matching that
// precedes it (e.g. n=251k: 14 ms SCC vs 122 ms matching vs 108 ms AMD), which
// is what makes it affordable to run unconditionally and simply discover
// nblocks == 1 on the irreducible matrices -- every PDE/FEM system, where a
// symmetric pattern collapses SCCs to connected components and BTF has nothing
// to offer.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef LEFT_RIGHT_LU_BLOCK_TRIANGULAR_H
#define LEFT_RIGHT_LU_BLOCK_TRIANGULAR_H

#include <Eigen/SparseCore>

#include <algorithm>
#include <vector>

namespace Eigen {
namespace left_right_lu {

/** Strongly connected components of the digraph with an edge j -> i for every
  * nonzero A(i, j), given as compressed column storage.
  *
  * \param n           number of vertices (columns).
  * \param outerStart  n+1 column starts.
  * \param innerIndex  row indices.
  * \param component   out: component[v], numbered in Tarjan pop order, which is
  *                    reverse topological order of the condensation -- see the
  *                    file comment for why that is the numbering BTF wants.
  * \returns the number of components.
  *
  * Iterative rather than recursive: Tarjan's recursion depth is O(n) and n
  * reaches the hundreds of thousands here, which overflows a default stack.
  * The explicit stack stores, per frame, the vertex and how far its adjacency
  * list has been scanned.
  */
template <typename StorageIndex>
StorageIndex stronglyConnectedComponents(StorageIndex n, const StorageIndex* outerStart,
                                         const StorageIndex* innerIndex,
                                         std::vector<StorageIndex>& component) {
  const StorageIndex kUnvisited = StorageIndex(-1);
  component.assign(static_cast<std::size_t>(n), kUnvisited);
  if (n <= 0) return 0;

  std::vector<StorageIndex> index(static_cast<std::size_t>(n), kUnvisited);
  std::vector<StorageIndex> low(static_cast<std::size_t>(n), 0);
  std::vector<char> onStack(static_cast<std::size_t>(n), 0);
  std::vector<StorageIndex> stack;
  std::vector<StorageIndex> frameNode, frameEdge;
  stack.reserve(static_cast<std::size_t>(n));
  frameNode.reserve(64);
  frameEdge.reserve(64);

  StorageIndex nextIndex = 0;
  StorageIndex numComponents = 0;

  for (StorageIndex root = 0; root < n; ++root) {
    if (index[root] != kUnvisited) continue;
    frameNode.push_back(root);
    frameEdge.push_back(0);

    while (!frameNode.empty()) {
      const StorageIndex v = frameNode.back();
      if (frameEdge.back() == 0) {
        index[v] = low[v] = nextIndex++;
        stack.push_back(v);
        onStack[v] = 1;
      }

      const StorageIndex begin = outerStart[v];
      const StorageIndex end = outerStart[v + 1];
      bool descended = false;
      for (StorageIndex k = begin + frameEdge.back(); k < end; ++k) {
        const StorageIndex w = innerIndex[k];
        frameEdge.back() = k - begin + 1;  // resume past w when this frame returns
        if (index[w] == kUnvisited) {
          frameNode.push_back(w);
          frameEdge.push_back(0);
          descended = true;
          break;
        }
        if (onStack[w]) low[v] = numext::mini(low[v], index[w]);
      }
      if (descended) continue;

      if (low[v] == index[v]) {
        while (true) {
          const StorageIndex w = stack.back();
          stack.pop_back();
          onStack[w] = 0;
          component[w] = numComponents;
          if (w == v) break;
        }
        ++numComponents;
      }

      frameNode.pop_back();
      frameEdge.pop_back();
      if (!frameNode.empty()) low[frameNode.back()] = numext::mini(low[frameNode.back()], low[v]);
    }
  }
  return numComponents;
}

/** Block triangular ordering of a matched (zero-free diagonal) matrix.
  *
  * \param n           matrix dimension.
  * \param outerStart  n+1 column starts (column-major, compressed).
  * \param innerIndex  row indices.
  * \param position    out: position[j] = the place original column j takes in
  *                    the block-contiguous ordering. A DIRECT map, matching the
  *                    convention analyzePattern's m_toInternal uses.
  * \param blockPtr    out: nblocks+1 boundaries, so block k occupies positions
  *                    [blockPtr[k], blockPtr[k+1]).
  * \returns the number of diagonal blocks; 1 means the matrix is irreducible
  *          and BTF has nothing to offer.
  *
  * Vertices keep their relative order inside a block. That is arbitrary -- the
  * caller re-orders within each block with a fill-reducing ordering anyway --
  * but it keeps the result deterministic.
  */
template <typename StorageIndex>
StorageIndex blockTriangularOrder(StorageIndex n, const StorageIndex* outerStart,
                                  const StorageIndex* innerIndex, std::vector<StorageIndex>& position,
                                  std::vector<StorageIndex>& blockPtr) {
  position.assign(static_cast<std::size_t>(n), 0);
  blockPtr.assign(1, 0);
  if (n <= 0) return 0;

  std::vector<StorageIndex> component;
  const StorageIndex nblocks = stronglyConnectedComponents(n, outerStart, innerIndex, component);

  // Counting sort by component id: stable, so within-block order is index order.
  blockPtr.assign(static_cast<std::size_t>(nblocks) + 1, 0);
  for (StorageIndex v = 0; v < n; ++v) blockPtr[static_cast<std::size_t>(component[v]) + 1]++;
  for (StorageIndex k = 0; k < nblocks; ++k) blockPtr[static_cast<std::size_t>(k) + 1] += blockPtr[static_cast<std::size_t>(k)];

  std::vector<StorageIndex> cursor(blockPtr.begin(), blockPtr.end() - 1);
  for (StorageIndex v = 0; v < n; ++v) position[v] = cursor[static_cast<std::size_t>(component[v])]++;
  return nblocks;
}

/** Convenience overload taking a compressed column-major sparse matrix. */
template <typename MatrixType>
typename MatrixType::StorageIndex blockTriangularOrder(
    const MatrixType& matrix, std::vector<typename MatrixType::StorageIndex>& position,
    std::vector<typename MatrixType::StorageIndex>& blockPtr) {
  using StorageIndex = typename MatrixType::StorageIndex;
  eigen_assert(matrix.isCompressed() && "blockTriangularOrder needs a compressed matrix");
  return blockTriangularOrder(static_cast<StorageIndex>(matrix.cols()), matrix.outerIndexPtr(),
                              matrix.innerIndexPtr(), position, blockPtr);
}

/** True when `position` really does permute `matrix` to block upper triangular
  * form with the given block boundaries -- i.e. every cross-block nonzero
  * A(i, j) satisfies block(i) < block(j).
  *
  * Not called by the solver: the property is a consequence of Tarjan's pop
  * order (see the file comment), and re-deriving it per factorization would
  * cost another O(nnz) sweep for nothing. It exists so the test suite can
  * assert the invariant on real matrices instead of trusting the argument.
  */
template <typename MatrixType>
bool isBlockUpperTriangular(const MatrixType& matrix,
                            const std::vector<typename MatrixType::StorageIndex>& position,
                            const std::vector<typename MatrixType::StorageIndex>& blockPtr) {
  using StorageIndex = typename MatrixType::StorageIndex;
  const StorageIndex n = static_cast<StorageIndex>(matrix.cols());
  if (static_cast<StorageIndex>(position.size()) != n) return false;

  // position -> block id, by binary search over the boundaries.
  std::vector<StorageIndex> blockOf(static_cast<std::size_t>(n), 0);
  const StorageIndex nblocks = static_cast<StorageIndex>(blockPtr.size()) - 1;
  for (StorageIndex k = 0; k < nblocks; ++k)
    for (StorageIndex p = blockPtr[static_cast<std::size_t>(k)]; p < blockPtr[static_cast<std::size_t>(k) + 1]; ++p)
      blockOf[static_cast<std::size_t>(p)] = k;

  for (StorageIndex j = 0; j < n; ++j) {
    const StorageIndex bj = blockOf[static_cast<std::size_t>(position[j])];
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex bi = blockOf[static_cast<std::size_t>(position[static_cast<StorageIndex>(it.index())])];
      if (bi != bj && !(bi < bj)) return false;
    }
  }
  return true;
}

}  // namespace left_right_lu
}  // namespace Eigen

#endif  // LEFT_RIGHT_LU_BLOCK_TRIANGULAR_H
