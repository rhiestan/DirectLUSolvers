// SupernodalLU - a sparse direct LU solver for Eigen using PaStiX-style
// supernodal block algorithms (see pastix_algorithms.md).
//
// Scope of this implementation:
//   * General (unsymmetric) values with a SYMMETRIC nonzero pattern.
//   * LU factorization with STATIC pivoting (no row interchanges); small pivots
//     are bumped to a threshold (automatic by default) to preserve the static
//     block structure. The accuracy lost to that perturbation is recovered by
//     iterative refinement, applied automatically during solve().
//   * Single-threaded, no MPI. Modeled on the public interface of
//     Eigen::SparseLU so it can be used as a drop-in solver.
//
// Pipeline (mirrors Eigen's analyzePattern / factorize split):
//   analyzePattern : ordering -> elimination tree -> postorder -> supernode
//                    detection -> symbolic block factorization (structure of L/U)
//   factorize      : scatter values -> left-looking supernodal numeric factorization
//   solve          : block forward (L) and backward (U) substitution
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_H
#define SUPERNODAL_LU_H

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "SupernodalLUSupport.h"

namespace Eigen {

/** \class SupernodalLU
 * \brief Supernodal sparse direct LU factorization for general matrices with a
 *        symmetric nonzero pattern.
 *
 * \tparam MatrixType_   A column-major Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>.
 * \tparam OrderingType_ A fill-reducing ordering functor (default AMDOrdering).
 *
 * Typical usage (identical to Eigen::SparseLU):
 * \code
 *   SupernodalLU<SparseMatrix<double>> solver;
 *   solver.compute(A);
 *   x = solver.solve(b);
 * \endcode
 */
template <typename MatrixType_, typename OrderingType_ = AMDOrdering<typename MatrixType_::StorageIndex>>
class SupernodalLU : public SparseSolverBase<SupernodalLU<MatrixType_, OrderingType_>> {
 protected:
  typedef SparseSolverBase<SupernodalLU<MatrixType_, OrderingType_>> Base;
  using Base::m_isInitialized;

 public:
  typedef MatrixType_ MatrixType;
  typedef OrderingType_ OrderingType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  using Base::_solve_impl;

  SupernodalLU() { init(); }
  explicit SupernodalLU(const MatrixType& matrix) {
    init();
    compute(matrix);
  }

  // --- main driver ----------------------------------------------------------

  void analyzePattern(const MatrixType& matrix);
  void factorize(const MatrixType& matrix);

  void compute(const MatrixType& matrix) {
    analyzePattern(matrix);
    factorize(matrix);
  }

  // --- solve (used by SparseSolverBase::solve) ------------------------------

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

  // --- queries --------------------------------------------------------------

  inline Index rows() const { return m_size; }
  inline Index cols() const { return m_size; }

  /** \returns Success if (numeric) factorization succeeded. */
  ComputationInfo info() const { return m_info; }

  const std::string& lastErrorMessage() const { return m_lastError; }

  /** Row permutation; equals the column permutation (symmetric pattern). */
  const PermutationType& rowsPermutation() const { return m_permutation; }
  /** Column permutation chosen by the ordering + postordering. */
  const PermutationType& colsPermutation() const { return m_permutation; }

  /** Magnitude below which a diagonal pivot is replaced (static pivoting).
   *  By default the threshold is chosen automatically at factorization time as
   *  sqrt(epsilon) * max|A_ij|, which steps over zero/tiny pivots while barely
   *  perturbing well-conditioned matrices. Pass any value (including 0, to
   *  disable replacement) to override the automatic choice. */
  void setStaticPivotThreshold(const RealScalar& threshold) {
    m_staticPivotThreshold = threshold;
    m_thresholdIsAuto = false;
  }
  /** Eigen::SparseLU-compatible alias. */
  void setPivotThreshold(const RealScalar& threshold) { setStaticPivotThreshold(threshold); }

  /** Number of pivots that were replaced during the last factorization. */
  Index replacedPivots() const { return m_replacedPivots; }

  /** Maximum number of iterative-refinement steps applied during solve().
   *  Refinement cheaply recovers the accuracy lost to static pivoting by
   *  reusing the stored factors; it stops early once the relative residual
   *  meets refinementTolerance() or stops improving. Default 5; set 0 to
   *  disable. */
  void setMaxIterativeRefinements(Index iters) { m_maxRefinementIterations = iters; }
  Index maxIterativeRefinements() const { return m_maxRefinementIterations; }

  /** Relative-residual target ||b - A x|| / ||b|| at which refinement stops.
   *  Default is the working precision (NumTraits epsilon). */
  void setRefinementTolerance(const RealScalar& tol) { m_refinementTolerance = tol; }
  RealScalar refinementTolerance() const { return m_refinementTolerance; }

  /** Number of refinement steps actually taken by the last solve(). */
  Index iterativeRefinements() const { return m_lastRefinementIterations; }

  /** Amalgamation (relaxed supernodes): merge fundamental supernodes along
   *  elimination-tree paths into larger dense panels for better BLAS-3
   *  efficiency, at the cost of a bounded number of explicit structural zeros.
   *  Merging is applied during analyzePattern().
   *
   *  A path-continuation boundary is merged when EITHER the supernode being
   *  closed is narrower than relaxedSize columns, OR the merge step introduces
   *  no more than maxZeroRows extra (logically zero) off-diagonal rows per
   *  column. Set relaxedSize = 1 and maxZeroRows = 0 to recover the pure
   *  fundamental-supernode partition (no amalgamation). */
  void setAmalgamation(Index relaxedSize, Index maxZeroRows) {
    m_relaxedSize = relaxedSize;
    m_maxAmalgamationZeroRows = maxZeroRows;
  }
  Index relaxedSize() const { return m_relaxedSize; }
  Index maxAmalgamationZeroRows() const { return m_maxAmalgamationZeroRows; }

  Index nnzL() const { return m_nnzL; }
  Index nnzU() const { return m_nnzU; }
  Index supernodeCount() const { return static_cast<Index>(m_supernodes.size()); }

  /** \returns the determinant of the factored matrix. */
  Scalar determinant() const {
    Scalar det(1);
    for (const auto& diag : m_diagonalFactor) {
      for (Index k = 0; k < diag.rows(); ++k) det *= diag(k, k);
    }
    return det;
  }

 private:
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrix;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> WorkMatrix;
  typedef supernodal_lu::Supernode<StorageIndex> Supernode;
  typedef supernodal_lu::RowBlock<StorageIndex> RowBlock;
  typedef supernodal_lu::UpdateSource<StorageIndex> UpdateSource;

  void init() {
    m_size = 0;
    m_analysisDone = false;
    m_info = InvalidInput;
    m_staticPivotThreshold = RealScalar(0);
    m_thresholdIsAuto = true;
    m_replacedPivots = 0;
    m_maxRefinementIterations = 5;
    m_refinementTolerance = NumTraits<RealScalar>::epsilon();
    m_lastRefinementIterations = 0;
    m_relaxedSize = 4;
    m_maxAmalgamationZeroRows = 4;
    m_nnzL = 0;
    m_nnzU = 0;
  }

  // analysis helpers
  void buildSymmetricAdjacency(const MatrixType& matrix, std::vector<std::vector<StorageIndex>>& adjacency) const;
  void computeEliminationTree(const std::vector<std::vector<StorageIndex>>& adjacency,
                              std::vector<StorageIndex>& parent) const;
  void computePostorder(const std::vector<StorageIndex>& parent, std::vector<StorageIndex>& postorder) const;
  void computeColumnStructures(const std::vector<std::vector<StorageIndex>>& adjacency,
                               const std::vector<StorageIndex>& parent,
                               std::vector<std::vector<StorageIndex>>& columnStructure) const;
  void detectSupernodesAndBlocks(const std::vector<StorageIndex>& parent,
                                 const std::vector<std::vector<StorageIndex>>& columnStructure);

  // numeric helper: subtract the Schur contribution of a source supernode.
  void applyUpdate(StorageIndex source, StorageIndex target, StorageIndex facingBlock);

  // one block forward/backward triangular solve (no refinement), in the
  // original numbering. Used both for the initial solve and the corrections.
  void solveTriangular(const DenseMatrix& rhs, DenseMatrix& solution) const;

  // state ---------------------------------------------------------------------
  StorageIndex m_size;
  OrderingType m_orderingFunctor;

  // m_toInternal[original] = internal (fill-reducing, postordered) index.
  std::vector<StorageIndex> m_toInternal;
  std::vector<StorageIndex> m_toOriginal;
  PermutationType m_permutation;  // public-facing form of m_toInternal

  std::vector<Supernode> m_supernodes;
  std::vector<RowBlock> m_rowBlocks;             // flattened, owned by supernodes
  std::vector<StorageIndex> m_supernodeOfColumn;  // internal column -> supernode id
  std::vector<std::vector<UpdateSource>> m_updateSources;  // per target supernode
  // m_rowPosition[s][internalRow] = position of that off-diagonal row in supernode s
  std::vector<std::unordered_map<StorageIndex, StorageIndex>> m_rowPosition;

  // numeric factors (one entry per supernode)
  std::vector<DenseMatrix> m_diagonalFactor;  // w x w  (packed LU)
  std::vector<DenseMatrix> m_lowerFactor;     // offDiag x w
  std::vector<DenseMatrix> m_upperFactor;     // w x offDiag

  // a copy of A in the original numbering, kept so solve() can form residuals
  // b - A x for iterative refinement.
  MatrixType m_originalMatrix;

  bool m_analysisDone;
  ComputationInfo m_info;
  std::string m_lastError;
  RealScalar m_staticPivotThreshold;
  bool m_thresholdIsAuto;
  Index m_replacedPivots;
  Index m_maxRefinementIterations;
  RealScalar m_refinementTolerance;
  mutable Index m_lastRefinementIterations;
  Index m_relaxedSize;                // amalgamation: force-merge below this width
  Index m_maxAmalgamationZeroRows;    // amalgamation: max extra zero rows per column
  Index m_nnzL;
  Index m_nnzU;
};

// ===========================================================================
//  Analysis (symbolic) phase
// ===========================================================================

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::buildSymmetricAdjacency(
    const MatrixType& matrix, std::vector<std::vector<StorageIndex>>& adjacency) const {
  const StorageIndex n = m_size;
  adjacency.assign(n, std::vector<StorageIndex>());

  // Symmetrize the pattern in the internal numbering: an edge (ii,jj) is added
  // for every off-diagonal nonzero of A (and its transpose).
  for (StorageIndex j = 0; j < n; ++j) {
    const StorageIndex jj = m_toInternal[j];
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex ii = m_toInternal[static_cast<StorageIndex>(it.index())];
      if (ii == jj) continue;
      adjacency[jj].push_back(ii);
      adjacency[ii].push_back(jj);
    }
  }
  for (StorageIndex j = 0; j < n; ++j) {
    auto& row = adjacency[j];
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::computeEliminationTree(
    const std::vector<std::vector<StorageIndex>>& adjacency, std::vector<StorageIndex>& parent) const {
  const StorageIndex n = m_size;
  parent.assign(n, StorageIndex(-1));
  std::vector<StorageIndex> ancestor(n, StorageIndex(-1));

  for (StorageIndex j = 0; j < n; ++j) {
    for (StorageIndex neighbor : adjacency[j]) {
      if (neighbor >= j) continue;  // only lower neighbors define the tree
      StorageIndex r = neighbor;
      while (ancestor[r] != StorageIndex(-1) && ancestor[r] != j) {
        StorageIndex next = ancestor[r];
        ancestor[r] = j;  // path compression
        r = next;
      }
      if (ancestor[r] == StorageIndex(-1)) {
        ancestor[r] = j;
        parent[r] = j;
      }
    }
  }
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::computePostorder(const std::vector<StorageIndex>& parent,
                                                              std::vector<StorageIndex>& postorder) const {
  const StorageIndex n = m_size;
  // children lists
  std::vector<StorageIndex> childHead(n, StorageIndex(-1));
  std::vector<StorageIndex> childNext(n, StorageIndex(-1));
  for (StorageIndex j = n - 1; j >= 0; --j) {
    if (parent[j] != StorageIndex(-1)) {
      childNext[j] = childHead[parent[j]];
      childHead[parent[j]] = j;
    }
    if (j == 0) break;  // avoid unsigned underflow if StorageIndex is unsigned
  }

  postorder.clear();
  postorder.reserve(n);
  std::vector<StorageIndex> stack;
  std::vector<StorageIndex> nextChild = childHead;  // iteration cursor per node

  for (StorageIndex root = 0; root < n; ++root) {
    if (parent[root] != StorageIndex(-1)) continue;
    stack.push_back(root);
    while (!stack.empty()) {
      StorageIndex node = stack.back();
      StorageIndex child = nextChild[node];
      if (child != StorageIndex(-1)) {
        nextChild[node] = childNext[child];
        stack.push_back(child);
      } else {
        postorder.push_back(node);
        stack.pop_back();
      }
    }
  }
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::computeColumnStructures(
    const std::vector<std::vector<StorageIndex>>& adjacency, const std::vector<StorageIndex>& parent,
    std::vector<std::vector<StorageIndex>>& columnStructure) const {
  const StorageIndex n = m_size;
  columnStructure.assign(n, std::vector<StorageIndex>());

  // children lists (parent[] is in postorder, so parent[j] > j)
  std::vector<std::vector<StorageIndex>> children(n);
  for (StorageIndex j = 0; j < n; ++j)
    if (parent[j] != StorageIndex(-1)) children[parent[j]].push_back(j);

  std::vector<StorageIndex> markedAt(n, StorageIndex(-1));
  std::vector<StorageIndex> scratch;

  for (StorageIndex j = 0; j < n; ++j) {
    scratch.clear();
    scratch.push_back(j);  // diagonal
    markedAt[j] = j;

    for (StorageIndex neighbor : adjacency[j]) {
      if (neighbor > j && markedAt[neighbor] != j) {
        markedAt[neighbor] = j;
        scratch.push_back(neighbor);
      }
    }
    for (StorageIndex c : children[j]) {
      for (StorageIndex r : columnStructure[c]) {
        if (r > j && markedAt[r] != j) {
          markedAt[r] = j;
          scratch.push_back(r);
        }
      }
    }
    std::sort(scratch.begin(), scratch.end());
    columnStructure[j] = scratch;
  }
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::detectSupernodesAndBlocks(
    const std::vector<StorageIndex>& parent, const std::vector<std::vector<StorageIndex>>& columnStructure) {
  const StorageIndex n = m_size;

  std::vector<StorageIndex> childCount(n, 0);
  for (StorageIndex j = 0; j < n; ++j)
    if (parent[j] != StorageIndex(-1)) ++childCount[parent[j]];

  // Number of entries in a sorted column structure that lie strictly beyond a
  // given column (i.e. its off-diagonal rows relative to that column).
  auto rowsBeyond = [](const std::vector<StorageIndex>& structure, StorageIndex col) -> StorageIndex {
    return static_cast<StorageIndex>(
        structure.end() - std::upper_bound(structure.begin(), structure.end(), col));
  };

  // Supernode partition. A boundary at column j is MANDATORY when parent[j-1]!=j
  // (j does not continue the elimination-tree path of j-1); merging across it
  // would violate the "last column has the largest structure" invariant. When
  // the path does continue, a fundamental boundary (childCount[j]!=1, a branch
  // point) may be AMALGAMATED into the running supernode: we accept the merge
  // when the supernode being closed is narrow, or the step adds few explicit
  // zeros. This trades a bounded number of structural zeros for larger dense
  // panels (better BLAS-3) without changing the factorization mathematically.
  std::vector<bool> startsSupernode(n, false);
  if (n > 0) startsSupernode[0] = true;
  StorageIndex currentStart = 0;
  for (StorageIndex j = 1; j < n; ++j) {
    bool start;
    if (parent[j - 1] != j) {
      start = true;  // mandatory structural boundary
    } else if (childCount[j] == 1) {
      start = false;  // fundamental supernode: no extra fill
    } else {
      // path-continuation branch point: amalgamate if cheap enough.
      const StorageIndex childWidth = j - currentStart;
      const StorageIndex deltaRows =
          rowsBeyond(columnStructure[j], j) - rowsBeyond(columnStructure[j - 1], j);
      const bool merge = (static_cast<Index>(childWidth) < m_relaxedSize) ||
                         (static_cast<Index>(deltaRows) <= m_maxAmalgamationZeroRows);
      start = !merge;
    }
    startsSupernode[j] = start;
    if (start) currentStart = j;
  }

  m_supernodes.clear();
  m_rowBlocks.clear();
  m_supernodeOfColumn.assign(n, 0);

  for (StorageIndex j = 0; j < n; ++j) {
    if (startsSupernode[j]) {
      Supernode s;
      s.firstColumn = j;
      s.lastColumn = j;
      m_supernodes.push_back(s);
    } else {
      m_supernodes.back().lastColumn = j;
    }
    m_supernodeOfColumn[j] = static_cast<StorageIndex>(m_supernodes.size() - 1);
  }

  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());

  // Build off-diagonal row blocks for each supernode from the structure of its
  // first column. Rows are split at non-contiguities and at facing-supernode
  // boundaries so every block faces exactly one supernode.
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    Supernode& sn = m_supernodes[s];
    // The last column of a supernode has the largest structure in the chain;
    // its rows below the supernode are exactly the shared off-diagonal rows.
    const std::vector<StorageIndex>& structure = columnStructure[sn.lastColumn];
    sn.firstRowBlock = static_cast<StorageIndex>(m_rowBlocks.size());
    sn.rowBlockCount = 0;
    sn.offDiagonalRowCount = 0;

    StorageIndex offset = 0;
    bool inBlock = false;
    RowBlock current;
    for (StorageIndex r : structure) {
      if (r <= sn.lastColumn) continue;  // belongs to the diagonal block
      const StorageIndex facing = m_supernodeOfColumn[r];
      if (inBlock && r == current.lastRow + 1 && facing == current.facingSupernode) {
        current.lastRow = r;
      } else {
        if (inBlock) {
          m_rowBlocks.push_back(current);
          ++sn.rowBlockCount;
          offset += current.height();
        }
        current.firstRow = r;
        current.lastRow = r;
        current.facingSupernode = facing;
        current.panelOffset = offset;
        inBlock = true;
      }
    }
    if (inBlock) {
      m_rowBlocks.push_back(current);
      ++sn.rowBlockCount;
      offset += current.height();
    }
    sn.offDiagonalRowCount = offset;
  }

  // Build, for each supernode, the map from off-diagonal internal row -> panel
  // position, and the list of contributing sources for the left-looking sweep.
  m_rowPosition.assign(supernodeNbr, std::unordered_map<StorageIndex, StorageIndex>());
  m_updateSources.assign(supernodeNbr, std::vector<UpdateSource>());

  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    auto& position = m_rowPosition[s];
    position.reserve(static_cast<std::size_t>(sn.offDiagonalRowCount));
    StorageIndex previousFacing = StorageIndex(-1);
    for (StorageIndex b = 0; b < sn.rowBlockCount; ++b) {
      const StorageIndex blockIndex = sn.firstRowBlock + b;
      const RowBlock& block = m_rowBlocks[blockIndex];
      for (StorageIndex r = block.firstRow; r <= block.lastRow; ++r)
        position[r] = block.panelOffset + (r - block.firstRow);
      // Register one update per (source, facing supernode); a source's rows
      // inside one target's columns are consecutive in the sorted panel, so all
      // blocks facing the same target are consecutive. Record only the first.
      if (block.facingSupernode != previousFacing) {
        UpdateSource src;
        src.sourceSupernode = s;
        src.facingRowBlock = blockIndex;  // first block facing this target
        m_updateSources[block.facingSupernode].push_back(src);
        previousFacing = block.facingSupernode;
      }
    }
  }
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::analyzePattern(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "SupernodalLU requires a square matrix");
  m_size = static_cast<StorageIndex>(matrix.rows());
  const StorageIndex n = m_size;

  // 1) fill-reducing ordering (orders the pattern of A + A^T).
  PermutationType orderingPerm;
  m_orderingFunctor(matrix, orderingPerm);

  m_toInternal.resize(n);
  if (orderingPerm.size() == 0) {  // NaturalOrdering returns empty
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = i;
  } else {
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = orderingPerm.indices()(i);
  }

  // 2) elimination tree of the ordered symmetric pattern.
  std::vector<std::vector<StorageIndex>> adjacency;
  buildSymmetricAdjacency(matrix, adjacency);
  std::vector<StorageIndex> parent;
  computeEliminationTree(adjacency, parent);

  // 3) postorder the tree and fold it into the internal numbering so that
  //    supernodes are contiguous (does not change fill).
  std::vector<StorageIndex> postorder;
  computePostorder(parent, postorder);
  std::vector<StorageIndex> relabel(n);
  for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
  for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = relabel[m_toInternal[i]];

  // 4) recompute adjacency + tree in the final numbering (clean and cheap).
  buildSymmetricAdjacency(matrix, adjacency);
  computeEliminationTree(adjacency, parent);

  // 5) symbolic factorization: per-column structure of the factor.
  std::vector<std::vector<StorageIndex>> columnStructure;
  computeColumnStructures(adjacency, parent, columnStructure);

  // 6) supernode partition + block structure + update lists.
  detectSupernodesAndBlocks(parent, columnStructure);

  // inverse permutation + public permutation object.
  m_toOriginal.resize(n);
  for (StorageIndex i = 0; i < n; ++i) m_toOriginal[m_toInternal[i]] = i;
  m_permutation.resize(n);
  for (StorageIndex i = 0; i < n; ++i) m_permutation.indices()(i) = m_toInternal[i];

  m_analysisDone = true;
  m_info = Success;
  m_isInitialized = true;
}

// ===========================================================================
//  Numeric factorization phase
// ===========================================================================

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::applyUpdate(StorageIndex source, StorageIndex target,
                                                         StorageIndex firstFacingBlock) {
  const Supernode& src = m_supernodes[source];
  const StorageIndex wSrc = src.width();
  const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
  const StorageIndex targetFirstColumn = m_supernodes[target].firstColumn;

  const DenseMatrix& srcLower = m_lowerFactor[source];
  const DenseMatrix& srcUpper = m_upperFactor[source];
  DenseMatrix& targetDiag = m_diagonalFactor[target];
  DenseMatrix& targetLower = m_lowerFactor[target];
  DenseMatrix& targetUpper = m_upperFactor[target];
  const auto& targetPosition = m_rowPosition[target];

  // The blocks of the source facing the target are consecutive: [firstFacing, lastFacing).
  StorageIndex lastFacing = firstFacingBlock;
  while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;

  // Each facing block contributes a contiguous set of the target's columns.
  for (StorageIndex cb = firstFacingBlock; cb < lastFacing; ++cb) {
    const RowBlock& colBlock = m_rowBlocks[cb];
    const StorageIndex columnCount = colBlock.height();
    const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;
    const auto facingUpper = srcUpper.block(0, colBlock.panelOffset, wSrc, columnCount);  // wSrc x cc
    const auto facingLower = srcLower.block(colBlock.panelOffset, 0, columnCount, wSrc);   // cc x wSrc

    // L-side: every source block from the first facing block onward contributes
    // to these target columns (diagonal rows if facing, lower panel if below).
    for (StorageIndex rb = firstFacingBlock; rb < lastBlock; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex rowCount = rowBlock.height();
      const auto lower = srcLower.block(rowBlock.panelOffset, 0, rowCount, wSrc);  // rc x wSrc
      if (rb < lastFacing) {  // facing rows -> target diagonal block
        const StorageIndex destRow = rowBlock.firstRow - targetFirstColumn;
        targetDiag.block(destRow, targetColStart, rowCount, columnCount).noalias() -= lower * facingUpper;
      } else {  // below rows -> target lower panel
        const StorageIndex destRow = targetPosition.at(rowBlock.firstRow);
        targetLower.block(destRow, targetColStart, rowCount, columnCount).noalias() -= lower * facingUpper;
      }
    }

    // U-side: only source blocks below the facing range contribute to the
    // target's upper panel (its off-diagonal columns).
    for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex rowCount = rowBlock.height();
      const StorageIndex destCol = targetPosition.at(rowBlock.firstRow);
      const auto upper = srcUpper.block(0, rowBlock.panelOffset, wSrc, rowCount);  // wSrc x rc
      targetUpper.block(targetColStart, destCol, columnCount, rowCount).noalias() -= facingLower * upper;
    }
  }
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analysisDone && "analyzePattern must be called before factorize");
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_replacedPivots = 0;
  m_info = Success;

  // keep A for iterative refinement during solve().
  m_originalMatrix = matrix;

  // resolve the effective static-pivot threshold. When automatic, scale it to
  // the matrix magnitude (sqrt(eps) * max|A_ij|, SuperLU_DIST style).
  RealScalar staticPivot = m_staticPivotThreshold;
  if (m_thresholdIsAuto) {
    RealScalar maxAbs(0);
    for (StorageIndex j = 0; j < m_size; ++j)
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it)
        maxAbs = numext::maxi(maxAbs, numext::abs(it.value()));
    staticPivot = numext::sqrt(NumTraits<RealScalar>::epsilon()) * maxAbs;
  }

  // 1) allocate and zero the dense panels.
  m_diagonalFactor.resize(supernodeNbr);
  m_lowerFactor.resize(supernodeNbr);
  m_upperFactor.resize(supernodeNbr);
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    const StorageIndex r = sn.offDiagonalRowCount;
    m_diagonalFactor[s] = DenseMatrix::Zero(w, w);
    m_lowerFactor[s] = DenseMatrix::Zero(r, w);
    m_upperFactor[s] = DenseMatrix::Zero(w, r);
  }

  // 2) scatter the (permuted) values of A into the panels.
  for (StorageIndex j = 0; j < m_size; ++j) {
    const StorageIndex jj = m_toInternal[j];
    const StorageIndex columnSupernode = m_supernodeOfColumn[jj];
    const Supernode& cs = m_supernodes[columnSupernode];
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex ii = m_toInternal[static_cast<StorageIndex>(it.index())];
      const Scalar value = it.value();
      if (m_supernodeOfColumn[ii] == columnSupernode) {
        m_diagonalFactor[columnSupernode](ii - cs.firstColumn, jj - cs.firstColumn) += value;
      } else if (ii > jj) {  // below-diagonal: owned by the column's supernode
#ifdef SNLU_DEBUG
        if (!m_rowPosition[columnSupernode].count(ii))
          std::fprintf(stderr, "SCATTER lower miss: ii=%d jj=%d colSn=%d [%d..%d]\n", (int)ii, (int)jj,
                       (int)columnSupernode, (int)cs.firstColumn, (int)cs.lastColumn);
#endif
        const StorageIndex pos = m_rowPosition[columnSupernode].at(ii);
        m_lowerFactor[columnSupernode](pos, jj - cs.firstColumn) += value;
      } else {  // above-diagonal: owned by the row's supernode
        const StorageIndex rowSupernode = m_supernodeOfColumn[ii];
        const Supernode& rs = m_supernodes[rowSupernode];
#ifdef SNLU_DEBUG
        if (!m_rowPosition[rowSupernode].count(jj))
          std::fprintf(stderr, "SCATTER upper miss: ii=%d jj=%d rowSn=%d [%d..%d]\n", (int)ii, (int)jj,
                       (int)rowSupernode, (int)rs.firstColumn, (int)rs.lastColumn);
#endif
        const StorageIndex pos = m_rowPosition[rowSupernode].at(jj);
        m_upperFactor[rowSupernode](ii - rs.firstColumn, pos) += value;
      }
    }
  }

  // 3) left-looking supernodal factorization.
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    // pull contributions from already-factored supernodes.
    for (const UpdateSource& source : m_updateSources[s])
      applyUpdate(source.sourceSupernode, s, source.facingRowBlock);

    DenseMatrix& diag = m_diagonalFactor[s];
    const StorageIndex w = m_supernodes[s].width();

    // unpivoted (static) LU of the diagonal block: packed L (unit) and U.
    for (StorageIndex k = 0; k < w; ++k) {
      Scalar pivot = diag(k, k);
      if (staticPivot > RealScalar(0) && std::abs(pivot) < staticPivot) {
        pivot = (std::abs(pivot) == RealScalar(0)) ? Scalar(staticPivot)
                                                   : pivot * (staticPivot / std::abs(pivot));
        diag(k, k) = pivot;
        ++m_replacedPivots;
      }
      if (pivot == Scalar(0)) {
        m_info = NumericalIssue;
        m_lastError = "SupernodalLU: zero pivot encountered (matrix is singular).";
        return;
      }
      const StorageIndex tail = w - k - 1;
      if (tail > 0) {
        diag.col(k).tail(tail) /= pivot;
        diag.bottomRightCorner(tail, tail).noalias() -= diag.col(k).tail(tail) * diag.row(k).tail(tail);
      }
    }

    // solve the off-diagonal panels against the diagonal factors.
    if (m_supernodes[s].offDiagonalRowCount > 0) {
      // lowerFactor := lowerFactor * U_kk^{-1}
      diag.template triangularView<Upper>().template solveInPlace<OnTheRight>(m_lowerFactor[s]);
      // upperFactor := L_kk^{-1} * upperFactor
      diag.template triangularView<UnitLower>().solveInPlace(m_upperFactor[s]);
    }
  }

  // 4) book-keeping: nonzero counts.
  m_nnzL = 0;
  m_nnzU = 0;
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const StorageIndex w = m_supernodes[s].width();
    const StorageIndex r = m_supernodes[s].offDiagonalRowCount;
    m_nnzL += Index(w) * Index(w + 1) / 2 + Index(r) * Index(w);  // unit-lower diag + lower panel
    m_nnzU += Index(w) * Index(w + 1) / 2 + Index(w) * Index(r);  // upper diag (incl) + upper panel
  }

  m_isInitialized = true;
}

// ===========================================================================
//  Solve phase
// ===========================================================================

template <typename MatrixType, typename OrderingType>
template <typename Rhs, typename Dest>
void SupernodalLU<MatrixType, OrderingType>::_solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const {
  eigen_assert(m_info == Success && "the matrix must be factorized first");
  const Index nrhs = b.cols();

  const DenseMatrix rhs = b;            // materialize the right-hand side
  DenseMatrix solution(m_size, nrhs);
  solveTriangular(rhs, solution);

  // iterative refinement: solution <- solution + (LU)^{-1} (b - A solution).
  // Static pivoting perturbs the factors; this cheaply restores accuracy. We
  // keep the best iterate seen and stop on convergence, divergence, or the
  // iteration cap, so an unrecoverable matrix cannot blow the answer up.
  m_lastRefinementIterations = 0;
  const RealScalar rhsNorm = rhs.norm();
  if (m_maxRefinementIterations > 0 && rhsNorm > RealScalar(0)) {
    DenseMatrix best = solution;
    RealScalar bestNorm = NumTraits<RealScalar>::highest();
    RealScalar prevNorm = NumTraits<RealScalar>::highest();
    for (Index it = 0; ; ++it) {
      const DenseMatrix residual = rhs - m_originalMatrix * solution;
      const RealScalar resNorm = residual.norm();
      if (resNorm < bestNorm) {
        bestNorm = resNorm;
        best = solution;
      }
      if (resNorm <= m_refinementTolerance * rhsNorm) break;  // converged
      if (resNorm > prevNorm) break;                          // diverging: keep best
      if (it >= m_maxRefinementIterations) break;             // cap reached
      prevNorm = resNorm;
      DenseMatrix correction(m_size, nrhs);
      solveTriangular(residual, correction);
      solution += correction;
      ++m_lastRefinementIterations;
    }
    solution = best;
  }

  x = solution;
}

template <typename MatrixType, typename OrderingType>
void SupernodalLU<MatrixType, OrderingType>::solveTriangular(const DenseMatrix& rhs,
                                                             DenseMatrix& x) const {
  const StorageIndex n = m_size;
  const Index nrhs = rhs.cols();

  // permute right-hand side into the internal numbering.
  DenseMatrix y(n, nrhs);
  for (StorageIndex i = 0; i < n; ++i) y.row(m_toInternal[i]) = rhs.row(i);

  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());

  // forward substitution: L (unit lower) y = Pb
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    m_diagonalFactor[s].template triangularView<UnitLower>().solveInPlace(head);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      y.middleRows(block.firstRow, hb).noalias() -= m_lowerFactor[s].middleRows(block.panelOffset, hb) * head;
    }
  }

  // backward substitution: U x = y
  for (StorageIndex s = supernodeNbr - 1; s >= 0; --s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      head.noalias() -= m_upperFactor[s].middleCols(block.panelOffset, hb) * y.middleRows(block.firstRow, hb);
    }
    m_diagonalFactor[s].template triangularView<Upper>().solveInPlace(head);
    if (s == 0) break;  // guard against unsigned underflow
  }

  // scatter the solution back to the original numbering.
  for (StorageIndex i = 0; i < n; ++i) x.row(i) = y.row(m_toInternal[i]);
}

}  // namespace Eigen

#endif  // SUPERNODAL_LU_H
