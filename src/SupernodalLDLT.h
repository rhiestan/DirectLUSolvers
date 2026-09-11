// SupernodalLDLT -- a supernodal LDL^T factorization for symmetric positive
// definite matrices.
//
// WHAT IT IS FOR
//
// SupernodalLU and LeftRightLU factor a general matrix as L U. Handed a
// SYMMETRIC one they still compute both factors, and U is the transpose of what
// L already holds. This solver keeps only one of them: A = L D L^T with L unit
// lower triangular and D diagonal, so both the stored factor and the flops that
// build it are roughly half. The saving is structural rather than a tuning win
// -- it is the redundant half of an LU of a symmetric matrix, not overhead.
//
// It is built on the SAME symbolic analysis as its two siblings
// (SupernodalLUSymbolic.h: elimination tree, postorder, supernode partition with
// amalgamation, block structure, update lists, scheduling levels). Only the
// numeric core and the solve differ. That is also why the analysis is worth
// reading there rather than here.
//
// SCOPE -- READ THIS FIRST
//
// The matrix must be symmetric (Hermitian for complex scalars) AND POSITIVE
// DEFINITE. Only one triangle is read, selected by the UpLo template parameter;
// whatever is stored in the other one is ignored, so a caller who has only half
// the matrix pays nothing to use it.
//
// An indefinite matrix is DECLINED, not approximated. LDL^T with 1x1 pivots is
// unstable on an indefinite matrix -- the stable factorization needs 2x2 pivot
// blocks (Bunch-Kaufman), which this solver does not implement. factorize()
// therefore reports NumericalIssue naming the column whose pivot was not
// positive, rather than returning a factor that happens to exist and does not
// mean anything. Saddle-point and KKT systems are the common case: they are
// symmetric and indefinite, so they belong in LeftRightLU until the 2x2 path
// exists.
//
// Positive definiteness is not something the caller has to assert up front. The
// pivot test IS the check, it costs nothing, and it is exact: a matrix that
// factors here was positive definite (in the scaled arithmetic actually used).
//
// WHAT IS NOT HERE YET, stated so it is not mistaken for an oversight:
//
//   * 2x2 pivots (symmetric indefinite / Bunch-Kaufman), and with them the
//     inertia that would fall out of the pivot signs for free.
//   * Intra-supernode parallelism. Factorization is dispatched over elimination-
//     tree levels only, so the few enormous root separators run on one lane.
//     SupernodalLU's measurements say that is where most of its parallel speedup
//     comes from, so expect this solver to scale worse than that one until the
//     chunking is ported, even though its serial work is smaller.
//   * matrixL() / vectorD() factor accessors.
//
// NO MATCHING, AND THAT IS NOT A GAP. The siblings permute rows to put large
// entries on the diagonal (MC64/transversal). That is an UNSYMMETRIC row
// permutation: applying it to a symmetric matrix destroys the symmetry this
// solver exists to exploit, and an SPD matrix does not need it -- its diagonal
// is already the largest entry in its row and column, and no pivoting is
// required for stability. Symmetric weighted matching (Duff-Pralet) is the thing
// an indefinite path would need, alongside the 2x2 pivots.
//
// Usage:
//   #include <SupernodalLDLT.h>
//   Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>> solver;   // Lower by default
//   solver.compute(A);
//   if (solver.info() != Eigen::Success) std::cerr << solver.lastErrorMessage();
//   Eigen::VectorXd x = solver.solve(b);
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LDLT_H
#define SUPERNODAL_LDLT_H

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "SupernodalLUSupport.h"
#include "SupernodalLUSymbolic.h"
#include "SupernodalLUExecutor.h"

namespace Eigen {

/** \class SupernodalLDLT
 * \brief Supernodal LDL^T factorization of a sparse symmetric positive definite
 *        matrix, sharing its symbolic analysis with SupernodalLU/LeftRightLU.
 *
 * \tparam MatrixType_   A column-major Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>.
 * \tparam UpLo_         Which triangle of the input to read: Lower (default) or Upper.
 *                       The other triangle is never touched.
 * \tparam OrderingType_ A fill-reducing ordering functor (default AMDOrdering).
 *                       Eigen's AMD/METIS functors symmetrize their input, so a
 *                       single triangle is a valid argument to them.
 * \tparam Executor_     A parallel-execution backend (default SerialExecutor);
 *                       see SupernodalLUExecutor.h.
 */
template <typename MatrixType_, int UpLo_ = Lower,
          typename OrderingType_ = AMDOrdering<typename MatrixType_::StorageIndex>,
          typename Executor_ = supernodal_lu::SerialExecutor>
class SupernodalLDLT : public SparseSolverBase<SupernodalLDLT<MatrixType_, UpLo_, OrderingType_, Executor_>> {
 protected:
  typedef SparseSolverBase<SupernodalLDLT<MatrixType_, UpLo_, OrderingType_, Executor_>> Base;
  using Base::m_isInitialized;

 public:
  typedef MatrixType_ MatrixType;
  typedef OrderingType_ OrderingType;
  typedef Executor_ Executor;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrix;
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

  enum {
    UpLo = UpLo_,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  using Base::_solve_impl;

  SupernodalLDLT() { init(); }
  explicit SupernodalLDLT(const MatrixType& matrix) {
    init();
    compute(matrix);
  }

  // --- main driver ----------------------------------------------------------

  /** Symbolic analysis: ordering, elimination tree, supernode partition, block
   *  structure. Depends on the sparsity pattern only, so a later factorize()
   *  with new values but the same pattern can skip it. */
  void analyzePattern(const MatrixType& matrix);

  /** Numeric factorization. Reports NumericalIssue if the matrix turns out not
   *  to be positive definite (see notPositiveDefiniteColumn()). */
  void factorize(const MatrixType& matrix);

  void compute(const MatrixType& matrix) {
    analyzePattern(matrix);
    if (m_info == Success) factorize(matrix);
  }

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

  // --- diagnostics ----------------------------------------------------------

  inline Index rows() const { return m_size; }
  inline Index cols() const { return m_size; }

  ComputationInfo info() const { return m_info; }
  /** True once a numeric factorization has succeeded. Unlike info(), unaffected
   *  by a subsequently failed solve() -- use it to ask whether the factors are
   *  still usable. */
  bool isFactorized() const { return m_factorized; }
  const std::string& lastErrorMessage() const { return m_lastError; }

  /** Internal column index whose pivot was not positive, or -1 if the last
   *  factorization did not fail that way. Internal numbering: pass it through
   *  permutation() to get back to the caller's column. */
  Index notPositiveDefiniteColumn() const { return m_failColumn; }

  /** Stored scalars in L, counting its implicit unit diagonal and the explicit
   *  structural zeros amalgamation introduces. D adds n more. */
  Index nnzL() const { return m_nnzL; }

  /** Scalars the factor arena will occupy, known after analyzePattern() and
   *  before factorize() allocates (memory ~ this x sizeof(Scalar)). */
  Index predictedFactorNonzeros() const {
    Index total = 0;
    for (const Supernode& sn : m_supernodes) {
      const Index w = Index(sn.width()), r = Index(sn.offDiagonalRowCount);
      total += (w + r) * w;
    }
    return total;
  }

  Index supernodeCount() const { return static_cast<Index>(m_supernodes.size()); }
  Index levelCount() const { return static_cast<Index>(m_levelGroups.size()); }

  /** The fill-reducing permutation: indices()(i) is the internal index of
   *  original column i. */
  const PermutationType& permutation() const { return m_permutation; }

  /** Relative residual ||b - Ax|| / ||b|| measured by the last solve(). */
  RealScalar solveResidual() const { return m_lastSolveRelativeResidual; }
  /** Refinement steps taken by the last solve(). */
  Index iterativeRefinements() const { return m_lastRefinementIterations; }

  /** det(A), with the equilibration scaling divided back out. Overflows on
   *  moderately sized systems -- prefer logAbsDeterminant(). */
  Scalar determinant() const {
    return Scalar(numext::exp(logAbsDeterminant()));  // D > 0, so det(A) > 0
  }

  /** log|det(A)| accumulated as a sum of logs, so it stays finite where
   *  determinant() would overflow. det(A) = det(D) / det(Dscale)^2, and every
   *  d is positive here, so there is no sign to carry. */
  RealScalar logAbsDeterminant() const {
    eigen_assert(m_factorized && "logAbsDeterminant() before a successful factorize()");
    RealScalar acc(0);
    for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
      const ConstStridedPanel diag = diagBlock(s);
      for (StorageIndex k = 0; k < m_supernodes[s].width(); ++k)
        acc += numext::log(numext::real(diag(k, k)));
    }
    // A~ = S A S with a symmetric scaling S, so log|det A| = log|det A~| - 2 sum log S.
    for (StorageIndex i = 0; i < m_size; ++i) acc -= RealScalar(2) * numext::log(m_scale[i]);
    return acc;
  }

  // --- options --------------------------------------------------------------

  /** Symmetric Ruiz equilibration A~ = S A S, on by default. Symmetric by
   *  construction -- one scaling applied on both sides -- because a two-sided
   *  Dr A Dc would not preserve symmetry. Fully transparent to solve() and the
   *  determinant. */
  void setEquilibration(bool on) { m_equilibrate = on; }
  bool equilibration() const { return m_equilibrate; }

  /** Amalgamation: merge adjacent supernodes into wider dense panels at the cost
   *  of a bounded number of explicit zeros. See SupernodalLU's documentation --
   *  the rules, and the banded-matrix caveat, are the same because the partition
   *  pass is literally the same code. */
  void setAmalgamation(Index relaxedSize, Index maxZeroRows) {
    m_relaxedSize = relaxedSize;
    m_maxAmalgamationZeroRows = maxZeroRows;
  }
  void setAmalgamationFillFraction(double fraction) { m_amalgamationFillFraction = fraction; }
  /** Cap supernode width. Adds no fill; keeps dense panels cache-friendly. */
  void setMaxBlockSize(Index maxBlockSize) { m_maxBlockSize = maxBlockSize; }
  Index maxBlockSize() const { return m_maxBlockSize; }

  /** Fail-fast fill guard: factorize() aborts before allocating if the symbolic
   *  structure predicts a factor above this many scalars. 0 (default) = off. */
  void setMaxFactorNonzeros(Index limit) { m_maxFactorNonzeros = limit; }
  Index maxFactorNonzeros() const { return m_maxFactorNonzeros; }

  /** Stationary iterative refinement steps, x += M^-1 (b - Ax). DEFAULT 0 -- off.
   *
   *  Unlike the LU siblings there is nothing here for refinement to repair: with
   *  no pivoting and no static-pivot perturbation, an SPD factorization is
   *  backward stable, so the first solve is already as good as the arithmetic
   *  allows. Turn it on for an ill-conditioned system where the extra accuracy
   *  is worth an O(nnz) matvec plus a triangular solve per step.
   *
   *  Note that stationary refinement is the right tool here only for small
   *  corrections; the natural Krylov method for an SPD operator is conjugate
   *  gradients, not the BiCGStab the unsymmetric siblings default to. */
  void setMaxIterativeRefinements(Index iters) { m_maxRefinementIterations = iters; }
  Index maxIterativeRefinements() const { return m_maxRefinementIterations; }
  void setRefinementTolerance(const RealScalar& tol) { m_refinementTolerance = tol; }

  /** Relative-residual ceiling above which solve() downgrades info() to
   *  NumericalIssue instead of returning a bad answer silently. Default 1e-6. */
  void setSolveFailureThreshold(const RealScalar& tol) { m_solveFailureThreshold = tol; }
  RealScalar solveFailureThreshold() const { return m_solveFailureThreshold; }

  /** Dispatch the triangular sweeps across the executor, over elimination-tree
   *  levels. Results are unchanged (each supernode writes only its own rows). */
  void setParallelSolve(bool on) { m_parallelSolve = on; }
  bool parallelSolve() const { return m_parallelSolve; }

  Executor& executor() { return m_executor; }
  const Executor& executor() const { return m_executor; }
  OrderingType& orderingFunctor() { return m_orderingFunctor; }

 private:
  typedef supernodal_lu::Supernode<StorageIndex> Supernode;
  typedef supernodal_lu::RowBlock<StorageIndex> RowBlock;
  typedef supernodal_lu::UpdateSource<StorageIndex> UpdateSource;

  // One contiguous panel per supernode, column-major, (width + offDiag) rows by
  // width columns, so the diagonal block and the off-diagonal panel share a
  // leading dimension. The top width x width block holds L_kk strictly below its
  // diagonal and D ON the diagonal (L_kk's unit diagonal is implicit); the rows
  // below hold the off-diagonal panel L_Rk. Only the lower triangle of the
  // diagonal block is ever read or written.
  //
  // This is the whole storage saving over an LU of the same matrix, which needs
  // a second width x offDiag arena for U on top of this one.
  typedef Map<DenseMatrix, 0, OuterStride<>> StridedPanel;
  typedef Map<const DenseMatrix, 0, OuterStride<>> ConstStridedPanel;

  StridedPanel diagBlock(StorageIndex s) {
    const Supernode& sn = m_supernodes[s];
    const Index stride = Index(sn.width()) + Index(sn.offDiagonalRowCount);
    return StridedPanel(m_lStorage.data() + m_lOffset[s], sn.width(), sn.width(), OuterStride<>(stride));
  }
  ConstStridedPanel diagBlock(StorageIndex s) const {
    const Supernode& sn = m_supernodes[s];
    const Index stride = Index(sn.width()) + Index(sn.offDiagonalRowCount);
    return ConstStridedPanel(m_lStorage.data() + m_lOffset[s], sn.width(), sn.width(), OuterStride<>(stride));
  }
  StridedPanel lowerPanel(StorageIndex s) {
    const Supernode& sn = m_supernodes[s];
    const Index w = sn.width(), r = sn.offDiagonalRowCount;
    return StridedPanel(m_lStorage.data() + m_lOffset[s] + w, r, w, OuterStride<>(w + r));
  }
  ConstStridedPanel lowerPanel(StorageIndex s) const {
    const Supernode& sn = m_supernodes[s];
    const Index w = sn.width(), r = sn.offDiagonalRowCount;
    return ConstStridedPanel(m_lStorage.data() + m_lOffset[s] + w, r, w, OuterStride<>(w + r));
  }

  void init() {
    m_size = 0;
    m_info = InvalidInput;
    m_analysisDone = false;
    m_factorized = false;
    m_failColumn = -1;
    m_equilibrate = true;
    m_relaxedSize = 4;
    m_maxAmalgamationZeroRows = 4;
    m_amalgamationFillFraction = 0.3;
    m_maxBlockSize = 128;
    m_maxFactorNonzeros = 0;
    m_maxRefinementIterations = 0;
    m_refinementTolerance = NumTraits<RealScalar>::epsilon();
    m_solveFailureThreshold = RealScalar(1e-6);
    m_parallelSolve = true;
    m_lastSolveRelativeResidual = RealScalar(0);
    m_lastRefinementIterations = 0;
    m_nnzL = 0;
    m_isInitialized = false;
  }

  supernodal_lu::symbolic::PartitionOptions partitionOptions() const {
    supernodal_lu::symbolic::PartitionOptions options;
    options.relaxedSize = m_relaxedSize;
    options.maxZeroRows = m_maxAmalgamationZeroRows;
    options.fillFraction = m_amalgamationFillFraction;
    options.maxBlockSize = m_maxBlockSize;
    return options;
  }

  /** Whether entry (row i, column j) lies in the triangle this solver reads.
   *
   *  Every pass over the input is filtered through this, which means a caller
   *  may equally well hand over a FULLY populated symmetric matrix: the
   *  redundant half is ignored rather than counted twice. Only the selected
   *  triangle needs to be present, not only it. */
  static bool inputTriangleHolds(StorageIndex i, StorageIndex j) {
    return (int(UpLo) & int(Lower)) ? (i >= j) : (i <= j);
  }

  // Symmetric adjacency in the internal numbering. The input holds one triangle,
  // and every stored off-diagonal entry contributes BOTH directions, so this is
  // the full elimination graph whichever triangle UpLo selects.
  void buildSymmetricAdjacency(const MatrixType& matrix,
                               std::vector<std::vector<StorageIndex>>& adjacency) const;

  // Symmetric Ruiz equilibration: one scaling vector applied on both sides.
  void computeEquilibration(const MatrixType& matrix);

  // `destRowScratch` is caller-owned rather than a member: factorizeSupernode
  // runs concurrently across a level, so a shared scratch buffer would race.
  void applyUpdate(StorageIndex source, StorageIndex target, StorageIndex firstFacingBlock,
                   std::vector<StorageIndex>& destRowScratch);
  void factorizeSupernode(StorageIndex s, bool& notPositiveDefinite, StorageIndex& failColumn);
  void factorizeDiagonalBlock(StridedPanel diag, bool& notPositiveDefinite,
                              StorageIndex& failColumn) const;

  void solveTriangular(const DenseMatrix& rhs, DenseMatrix& x) const;
  template <typename Dest>
  void applyInverseL(Dest& y) const;
  template <typename Dest>
  void applyInverseD(Dest& y) const;
  template <typename Dest>
  void applyInverseLTransposed(Dest& y) const;
  template <typename Dest>
  void forwardSolveSupernode(StorageIndex t, Dest& y) const;
  template <typename Dest>
  void backwardSolveSupernode(StorageIndex s, Dest& y) const;
  bool solveInParallel(Index nrhs) const;

  // A x for the residual check and refinement. The stored matrix holds one
  // triangle, so the product must go through the self-adjoint view.
  template <typename In, typename Out>
  void applyA(const In& in, Out& out) const {
    out.noalias() = m_originalMatrix.template selfadjointView<UpLo>() * in;
  }

  // Panel position of off-diagonal internal row `r` within supernode `s`, by
  // binary search over its sorted row blocks (the same offset arithmetic
  // SupernodalLU uses instead of a per-supernode hash map).
  StorageIndex rowPanelPosition(StorageIndex s, StorageIndex r) const {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex first = sn.firstRowBlock;
    StorageIndex lo = 0, hi = sn.rowBlockCount;
    while (lo < hi) {
      const StorageIndex mid = lo + ((hi - lo) >> 1);
      if (m_rowBlocks[first + mid].lastRow < r)
        lo = mid + 1;
      else
        hi = mid;
    }
    const RowBlock& block = m_rowBlocks[first + lo];
    eigen_assert(lo < sn.rowBlockCount && block.firstRow <= r && r <= block.lastRow &&
                 "rowPanelPosition: row is not an off-diagonal row of this supernode");
    return block.panelOffset + (r - block.firstRow);
  }

  // Below this much solve work (rows x right-hand sides) a per-level fork-join
  // dispatch costs more than the substitution it parallelizes.
  static constexpr Index kMinParallelSolveWork = 200000;
  // BLAS-3 panel width inside a dense diagonal block.
  static constexpr StorageIndex kDiagBlockSize = 64;

  OrderingType m_orderingFunctor;
  Executor m_executor;

  StorageIndex m_size;
  mutable ComputationInfo m_info;
  mutable std::string m_lastError;
  bool m_analysisDone, m_factorized;
  Index m_failColumn;

  std::vector<StorageIndex> m_toInternal;
  PermutationType m_permutation;

  std::vector<Supernode> m_supernodes;
  std::vector<StorageIndex> m_supernodeOfColumn;
  std::vector<RowBlock> m_rowBlocks;
  std::vector<std::vector<UpdateSource>> m_updateSources;
  std::vector<std::vector<StorageIndex>> m_levelGroups;

  std::vector<Scalar> m_lStorage;
  std::vector<std::size_t> m_lOffset;

  std::vector<RealScalar> m_scale;  // symmetric equilibration, original numbering
  MatrixType m_originalMatrix;

  bool m_equilibrate, m_parallelSolve;
  Index m_relaxedSize, m_maxAmalgamationZeroRows;
  double m_amalgamationFillFraction;
  Index m_maxBlockSize, m_maxFactorNonzeros;
  Index m_maxRefinementIterations;
  RealScalar m_refinementTolerance, m_solveFailureThreshold;
  mutable RealScalar m_lastSolveRelativeResidual;
  mutable Index m_lastRefinementIterations;
  Index m_nnzL;
};

// ===========================================================================
//  Analysis (symbolic) phase
// ===========================================================================

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::buildSymmetricAdjacency(
    const MatrixType& matrix, std::vector<std::vector<StorageIndex>>& adjacency) const {
  const StorageIndex n = m_size;
  adjacency.assign(n, std::vector<StorageIndex>());

  for (StorageIndex j = 0; j < n; ++j) {
    const StorageIndex jj = m_toInternal[j];
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex i = static_cast<StorageIndex>(it.index());
      if (!inputTriangleHolds(i, j)) continue;
      const StorageIndex ii = m_toInternal[i];
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

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::analyzePattern(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "SupernodalLDLT requires a square matrix");
  m_size = static_cast<StorageIndex>(matrix.rows());
  const StorageIndex n = m_size;
  m_analysisDone = false;
  m_factorized = false;
  m_failColumn = -1;
  m_lastError.clear();
  m_info = Success;

  m_toInternal.assign(n, 0);
  m_permutation.resize(n);
  if (n == 0) {
    m_supernodes.clear();
    m_levelGroups.clear();
    m_analysisDone = true;
    m_isInitialized = true;
    return;
  }

  // 1) fill-reducing ordering. Eigen's AMD/METIS functors symmetrize whatever
  //    they are given, so passing a single triangle orders the same graph the
  //    factorization will eliminate.
  PermutationType orderingPerm;
  m_orderingFunctor(matrix, orderingPerm);
  if (orderingPerm.size() == 0) {  // NaturalOrdering reports the identity as empty
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = i;
  } else {
    // indices()(k) is the ORIGINAL column placed at new position k, so invert to
    // get "the internal index OF original column i". Getting this backwards
    // leaves residuals at machine precision and shows up only as fill -- by
    // orders of magnitude on strongly directional 3D matrices.
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[orderingPerm.indices()(i)] = i;
  }

  namespace symbolic = supernodal_lu::symbolic;

  // 2) elimination tree of the ordered pattern.
  std::vector<std::vector<StorageIndex>> adjacency;
  buildSymmetricAdjacency(matrix, adjacency);
  std::vector<StorageIndex> parent;
  symbolic::computeEliminationTree(n, adjacency, parent);

  // 3) postorder and fold into the numbering so supernodes are contiguous.
  std::vector<StorageIndex> postorder;
  symbolic::computePostorder(n, parent, postorder);
  std::vector<StorageIndex> relabel(n);
  for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
  for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = relabel[m_toInternal[i]];

  // 4) recompute adjacency + tree in the final numbering.
  buildSymmetricAdjacency(matrix, adjacency);
  symbolic::computeEliminationTree(n, adjacency, parent);

  // 5) symbolic factorization + supernode partition (streaming).
  std::vector<std::vector<StorageIndex>> supernodeOffDiagRows;
  symbolic::computeSupernodePartition(n, adjacency, parent, partitionOptions(), m_supernodes,
                                      m_supernodeOfColumn, supernodeOffDiagRows);

  // 6) block structure + update-source lists.
  symbolic::buildRowBlocksAndUpdateSources(supernodeOffDiagRows, m_supernodeOfColumn, m_supernodes,
                                           m_rowBlocks, m_updateSources);

  // 7) elimination-tree levels for scheduling.
  symbolic::computeSupernodeLevels(static_cast<StorageIndex>(m_supernodes.size()), m_updateSources,
                                   m_levelGroups);

  for (StorageIndex i = 0; i < n; ++i) m_permutation.indices()(i) = m_toInternal[i];

  m_analysisDone = true;
  m_isInitialized = true;
}

// ===========================================================================
//  Numeric factorization phase
// ===========================================================================

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::computeEquilibration(
    const MatrixType& matrix) {
  const StorageIndex n = m_size;
  m_scale.assign(n, RealScalar(1));
  if (!m_equilibrate) return;

  // Symmetric Ruiz: scale by 1/sqrt(inf-norm) on BOTH sides at once, so the
  // scaled matrix S A S is still symmetric. A two-sided Dr A Dc, which the
  // unsymmetric siblings use, would not be.
  std::vector<RealScalar> rowMax(n);
  const int maxIters = 5;
  for (int iter = 0; iter < maxIters; ++iter) {
    std::fill(rowMax.begin(), rowMax.end(), RealScalar(0));
    for (StorageIndex j = 0; j < n; ++j) {
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
        const StorageIndex i = static_cast<StorageIndex>(it.index());
        if (!inputTriangleHolds(i, j)) continue;
        // Only one triangle is stored, so a stored off-diagonal entry bounds the
        // norms of BOTH its row and its column.
        const RealScalar scaled = numext::abs(it.value()) * m_scale[i] * m_scale[j];
        rowMax[i] = numext::maxi(rowMax[i], scaled);
        if (i != j) rowMax[j] = numext::maxi(rowMax[j], scaled);
      }
    }
    RealScalar deviation(0);
    for (StorageIndex i = 0; i < n; ++i)
      if (rowMax[i] > RealScalar(0)) {
        m_scale[i] /= numext::sqrt(rowMax[i]);
        deviation = numext::maxi(deviation, numext::abs(RealScalar(1) - rowMax[i]));
      }
    if (deviation < RealScalar(0.1)) break;
  }
}

// Dense LDL^T of one supernode's diagonal block, lower triangle only, blocked so
// the trailing update is a BLAS-3 symmetric product rather than a rank-1 sweep.
// No pivoting: for a positive definite block the diagonal is already the largest
// entry of its row and column, so none is needed for stability -- and a
// non-positive pivot means the matrix was not positive definite, which is
// reported rather than stepped over.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorizeDiagonalBlock(
    StridedPanel diag, bool& notPositiveDefinite, StorageIndex& failColumn) const {
  const StorageIndex w = static_cast<StorageIndex>(diag.rows());

  for (StorageIndex j0 = 0; j0 < w; j0 += kDiagBlockSize) {
    const StorageIndex jb = std::min<StorageIndex>(kDiagBlockSize, w - j0);
    const StorageIndex panelEnd = j0 + jb;

    // (1) unblocked LDL^T over the panel's columns, updating all rows below so
    //     that L21 is complete when the trailing update needs it.
    for (StorageIndex k = j0; k < panelEnd; ++k) {
      const RealScalar d = numext::real(diag(k, k));
      if (!(d > RealScalar(0))) {
        notPositiveDefinite = true;
        failColumn = k;
        return;
      }
      diag(k, k) = Scalar(d);  // D is real even for a Hermitian input
      const StorageIndex below = w - k - 1;
      if (below == 0) continue;
      diag.col(k).segment(k + 1, below) /= Scalar(d);
      const StorageIndex panelTail = panelEnd - (k + 1);
      if (panelTail > 0)
        diag.block(k + 1, k + 1, below, panelTail).noalias() -=
            (diag.col(k).segment(k + 1, below) * Scalar(d)) *
            diag.col(k).segment(k + 1, panelTail).adjoint();
    }

    // (2) trailing block update A22 -= L21 * D11 * L21^H. Forming D11 * L21^H
    //     once (jb x trailing, small) keeps this a single GEMM with no diagonal
    //     scaling in its inner loop.
    const StorageIndex trailing = w - panelEnd;
    if (trailing > 0) {
      DenseMatrix dl = diag.block(panelEnd, j0, trailing, jb).adjoint();  // jb x trailing
      for (StorageIndex q = 0; q < jb; ++q) dl.row(q) *= diag(j0 + q, j0 + q);
      diag.block(panelEnd, panelEnd, trailing, trailing).template triangularView<Lower>() -=
          diag.block(panelEnd, j0, trailing, jb) * dl;
    }
  }
}

// Subtract one source supernode's Schur contribution from a target.
//
// The LU form of this walks BOTH an L-side and a U-side; here the source sends
// only -L_R D L_R^H, so there is one side, and within the target's diagonal block
// only the lower triangle is computed. That is the whole factor-of-two in the
// numeric phase.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::applyUpdate(
    StorageIndex source, StorageIndex target, StorageIndex firstFacingBlock,
    std::vector<StorageIndex>& destRowScratch) {
  const Supernode& src = m_supernodes[source];
  const StorageIndex wSrc = src.width();
  const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
  const StorageIndex targetFirstColumn = m_supernodes[target].firstColumn;

  const StridedPanel srcLower = lowerPanel(source);
  const StridedPanel srcDiag = diagBlock(source);
  StridedPanel targetDiag = diagBlock(target);
  StridedPanel targetLower = lowerPanel(target);

  // Blocks of the source facing the target are consecutive: [firstFacing, lastFacing).
  StorageIndex lastFacing = firstFacingBlock;
  while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;

  // Hoist the panel-position search for the below-facing blocks: it depends on
  // the row block alone, and the column-block loop below would otherwise repeat
  // it once per facing column block.
  destRowScratch.clear();
  for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb)
    destRowScratch.push_back(rowPanelPosition(target, m_rowBlocks[rb].firstRow));

  // COLUMN BLOCK OUTER: D_src * L_cb^H is formed once per column block and every
  // row block reuses it, so the diagonal scaling never enters a GEMM inner loop.
  for (StorageIndex cb = firstFacingBlock; cb < lastFacing; ++cb) {
    const RowBlock& colBlock = m_rowBlocks[cb];
    const StorageIndex cc = colBlock.height();
    const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;

    DenseMatrix dl = srcLower.block(colBlock.panelOffset, 0, cc, wSrc).adjoint();  // wSrc x cc
    for (StorageIndex q = 0; q < wSrc; ++q) dl.row(q) *= srcDiag(q, q);

    // facing rows -> the target's diagonal block. rb starts at cb: the pairs with
    // rb < cb would land strictly above the diagonal, which is not stored.
    for (StorageIndex rb = cb; rb < lastFacing; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex rc = rowBlock.height();
      const StorageIndex destRow = rowBlock.firstRow - targetFirstColumn;
      const auto lower = srcLower.block(rowBlock.panelOffset, 0, rc, wSrc);
      if (rb == cb)  // straddles the diagonal: triangular output
        targetDiag.block(destRow, targetColStart, rc, cc).template triangularView<Lower>() -=
            lower * dl;
      else  // entirely below it: a plain GEMM
        targetDiag.block(destRow, targetColStart, rc, cc).noalias() -= lower * dl;
    }

    // rows below the target's columns -> the target's off-diagonal panel.
    for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex rc = rowBlock.height();
      const StorageIndex destRow = destRowScratch[static_cast<std::size_t>(rb - lastFacing)];
      targetLower.block(destRow, targetColStart, rc, cc).noalias() -=
          srcLower.block(rowBlock.panelOffset, 0, rc, wSrc) * dl;
    }
  }
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorizeSupernode(
    StorageIndex s, bool& notPositiveDefinite, StorageIndex& failColumn) {
  std::vector<StorageIndex> destRowScratch;  // per-call: this runs concurrently
  for (const UpdateSource& u : m_updateSources[s])
    applyUpdate(u.sourceSupernode, s, u.facingRowBlock, destRowScratch);

  StridedPanel diag = diagBlock(s);
  factorizeDiagonalBlock(diag, notPositiveDefinite, failColumn);
  if (notPositiveDefinite) {
    failColumn += m_supernodes[s].firstColumn;  // block-local -> internal index
    return;
  }

  // A_Rk = L_Rk D_kk L_kk^H, so L_Rk = A_Rk L_kk^-H D_kk^-1: one right-solve
  // against the unit upper triangle, then a column scaling by D.
  const StorageIndex w = m_supernodes[s].width();
  if (m_supernodes[s].offDiagonalRowCount > 0) {
    StridedPanel lower = lowerPanel(s);
    diag.template triangularView<UnitLower>().adjoint().template solveInPlace<OnTheRight>(lower);
    for (StorageIndex q = 0; q < w; ++q) lower.col(q) /= diag(q, q);
  }
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analysisDone && "analyzePattern must be called before factorize");
  eigen_assert(matrix.rows() == m_size && matrix.cols() == m_size &&
               "factorize(): matrix size differs from analyzePattern()");
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_factorized = false;
  m_failColumn = -1;
  m_lastError.clear();
  m_info = Success;
  if (m_size == 0) {
    m_factorized = true;
    m_isInitialized = true;
    return;
  }

  if (m_maxFactorNonzeros > 0) {
    const Index predicted = predictedFactorNonzeros();
    if (predicted > m_maxFactorNonzeros) {
      char buf[512];
      std::snprintf(buf, sizeof(buf),
                    "SupernodalLDLT: predicted factor size %lld scalars (~%.1f GB) exceeds the "
                    "configured limit of %lld (setMaxFactorNonzeros). The matrix likely lacks good "
                    "vertex separators; an iterative solver (e.g. CG with an incomplete Cholesky "
                    "preconditioner) is the right tool. Raise or clear the limit to force it.",
                    static_cast<long long>(predicted),
                    static_cast<double>(predicted) * static_cast<double>(sizeof(Scalar)) / 1e9,
                    static_cast<long long>(m_maxFactorNonzeros));
      m_lastError = buf;
      m_info = NumericalIssue;
      return;  // nothing allocated
    }
  }

  m_originalMatrix = matrix;  // kept for the residual check and refinement
  computeEquilibration(matrix);

  // 1) lay out and zero the single panel arena.
  m_lOffset.assign(supernodeNbr, 0);
  std::size_t total = 0;
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const std::size_t w = static_cast<std::size_t>(sn.width());
    const std::size_t r = static_cast<std::size_t>(sn.offDiagonalRowCount);
    m_lOffset[s] = total;
    total += (w + r) * w;
  }
  m_lStorage.assign(total, Scalar(0));

  // 2) scatter the scaled values of the stored triangle into the panels. Each
  //    entry is normalized to the LOWER triangle of the INTERNAL numbering --
  //    the ordering can send an originally-lower entry above the diagonal, and
  //    A is symmetric, so (max, min) is always the slot that exists.
  for (StorageIndex j = 0; j < m_size; ++j) {
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex i = static_cast<StorageIndex>(it.index());
      if (!inputTriangleHolds(i, j)) continue;
      const Scalar value = it.value() * m_scale[i] * m_scale[j];  // A~ = S A S
      const StorageIndex ii = m_toInternal[i], jj = m_toInternal[j];
      const StorageIndex r = numext::maxi(ii, jj), c = numext::mini(ii, jj);
      // An off-diagonal entry whose two internal indices swapped order is the
      // transposed element, so it is the conjugate that belongs in the slot.
      const Scalar placed = (ii < jj) ? numext::conj(value) : value;

      const StorageIndex cs = m_supernodeOfColumn[c];
      const Supernode& sn = m_supernodes[cs];
      const std::size_t width = static_cast<std::size_t>(sn.width());
      const std::size_t stride = width + static_cast<std::size_t>(sn.offDiagonalRowCount);
      const std::size_t col = static_cast<std::size_t>(c - sn.firstColumn);
      if (m_supernodeOfColumn[r] == cs) {  // inside the diagonal block
        const std::size_t row = static_cast<std::size_t>(r - sn.firstColumn);
        m_lStorage[m_lOffset[cs] + col * stride + row] += placed;
      } else {  // below it, in the off-diagonal panel
        const std::size_t pos = static_cast<std::size_t>(rowPanelPosition(cs, r));
        m_lStorage[m_lOffset[cs] + col * stride + width + pos] += placed;
      }
    }
  }

  // 3) left-looking supernodal factorization over elimination-tree levels: every
  //    supernode in a level writes only its own panel, so a level runs
  //    concurrently and levels run in order.
  std::vector<char> badPerSupernode(supernodeNbr, 0);
  std::vector<StorageIndex> failPerSupernode(supernodeNbr, 0);
  bool bad = false;
  for (const std::vector<StorageIndex>& group : m_levelGroups) {
    const Index groupSize = static_cast<Index>(group.size());
    m_executor.parallelFor(Index(0), groupSize, [&](Index k) {
      const StorageIndex s = group[static_cast<std::size_t>(k)];
      bool localBad = false;
      StorageIndex localFail = 0;
      factorizeSupernode(s, localBad, localFail);
      badPerSupernode[s] = localBad ? 1 : 0;
      failPerSupernode[s] = localFail;
    });
    for (StorageIndex s : group)
      if (badPerSupernode[s]) bad = true;
    if (bad) break;  // the factor is unusable; stop launching further levels
  }

  if (bad) {
    StorageIndex failed = 0;
    for (StorageIndex s = 0; s < supernodeNbr; ++s)
      if (badPerSupernode[s]) {
        failed = failPerSupernode[s];
        break;
      }
    m_failColumn = Index(failed);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "SupernodalLDLT: the matrix is not positive definite -- the pivot at internal "
                  "column %lld is not positive. An indefinite symmetric matrix (a saddle-point or "
                  "KKT system, say) needs 2x2 pivot blocks, which this solver does not implement; "
                  "use LeftRightLU for it.",
                  static_cast<long long>(failed));
    m_lastError = buf;
    m_info = NumericalIssue;
    return;
  }

  m_nnzL = 0;
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Index w = Index(m_supernodes[s].width());
    const Index r = Index(m_supernodes[s].offDiagonalRowCount);
    m_nnzL += w * (w + 1) / 2 + r * w;  // unit-lower diagonal block + lower panel
  }

  m_factorized = true;
  m_isInitialized = true;
}

// ===========================================================================
//  Solve phase
// ===========================================================================

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
bool SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::solveInParallel(Index nrhs) const {
  if (!m_parallelSolve || m_levelGroups.empty()) return false;
  if (m_executor.concurrency() <= 1) return false;
  return Index(m_size) * nrhs >= kMinParallelSolveWork;
}

// One supernode's share of the forward sweep, in GATHER form: pull every
// finished source's contribution into this supernode's own rows, then solve its
// diagonal block. Writes only inside [firstColumn, +w), so a level is race-free.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::forwardSolveSupernode(StorageIndex t,
                                                                                     Dest& y) const {
  const Supernode& tn = m_supernodes[t];
  for (const UpdateSource& u : m_updateSources[t]) {
    const Supernode& src = m_supernodes[u.sourceSupernode];
    const ConstStridedPanel lower = lowerPanel(u.sourceSupernode);
    const auto srcHead = y.middleRows(src.firstColumn, src.width());
    const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
    for (StorageIndex b = u.facingRowBlock; b < lastBlock; ++b) {
      const RowBlock& block = m_rowBlocks[b];
      if (block.facingSupernode != t) break;  // blocks facing t are consecutive
      const StorageIndex hb = block.height();
      y.middleRows(block.firstRow, hb).noalias() -= lower.middleRows(block.panelOffset, hb) * srcHead;
    }
  }
  auto head = y.middleRows(tn.firstColumn, tn.width());
  diagBlock(t).template triangularView<UnitLower>().solveInPlace(head);
}

// One supernode's share of the backward sweep, L^H x = x. Already a gather: it
// reads rows owned by higher-numbered supernodes and writes only its own head.
// The off-diagonal panel is the same L the forward sweep used, read transposed --
// there is no second factor to store.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::backwardSolveSupernode(StorageIndex s,
                                                                                      Dest& y) const {
  const Supernode& sn = m_supernodes[s];
  auto head = y.middleRows(sn.firstColumn, sn.width());
  const ConstStridedPanel lower = lowerPanel(s);
  for (StorageIndex b = 0; b < sn.rowBlockCount; ++b) {
    const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b];
    const StorageIndex hb = block.height();
    head.noalias() -= lower.middleRows(block.panelOffset, hb).adjoint() * y.middleRows(block.firstRow, hb);
  }
  diagBlock(s).template triangularView<UnitLower>().adjoint().solveInPlace(head);
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::applyInverseL(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  if (solveInParallel(y.cols())) {
    for (const std::vector<StorageIndex>& group : m_levelGroups) {
      const Index groupSize = static_cast<Index>(group.size());
      if (groupSize == 1) {
        forwardSolveSupernode(group[0], y);
        continue;
      }
      m_executor.parallelFor(Index(0), groupSize,
                             [&](Index k) { forwardSolveSupernode(group[static_cast<std::size_t>(k)], y); });
    }
    return;
  }
  // Serial path: SCATTER. Each supernode solves its head and pushes it into its
  // ancestors' rows -- equivalent to the gather above, and cheaper when there is
  // no level structure to exploit.
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    auto head = y.middleRows(sn.firstColumn, sn.width());
    diagBlock(s).template triangularView<UnitLower>().solveInPlace(head);
    const ConstStridedPanel lower = lowerPanel(s);
    for (StorageIndex b = 0; b < sn.rowBlockCount; ++b) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b];
      const StorageIndex hb = block.height();
      y.middleRows(block.firstRow, hb).noalias() -= lower.middleRows(block.panelOffset, hb) * head;
    }
  }
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::applyInverseD(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const ConstStridedPanel diag = diagBlock(s);
    for (StorageIndex k = 0; k < sn.width(); ++k) y.row(sn.firstColumn + k) /= diag(k, k);
  }
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::applyInverseLTransposed(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  if (solveInParallel(y.cols())) {
    for (std::size_t lv = m_levelGroups.size(); lv-- > 0;) {
      const std::vector<StorageIndex>& group = m_levelGroups[lv];
      const Index groupSize = static_cast<Index>(group.size());
      if (groupSize == 1) {
        backwardSolveSupernode(group[0], y);
        continue;
      }
      m_executor.parallelFor(Index(0), groupSize,
                             [&](Index k) { backwardSolveSupernode(group[static_cast<std::size_t>(k)], y); });
    }
    return;
  }
  for (StorageIndex s = supernodeNbr - 1; s >= 0; --s) {
    backwardSolveSupernode(s, y);
    if (s == 0) break;  // StorageIndex may be unsigned
  }
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::solveTriangular(const DenseMatrix& rhs,
                                                                               DenseMatrix& x) const {
  const StorageIndex n = m_size;
  // S A S = P^T (L D L^H) P, so the scaling goes on symmetrically at both ends.
  DenseMatrix y(n, rhs.cols());
  for (StorageIndex i = 0; i < n; ++i) y.row(m_toInternal[i]) = m_scale[i] * rhs.row(i);
  applyInverseL(y);
  applyInverseD(y);
  applyInverseLTransposed(y);
  for (StorageIndex i = 0; i < n; ++i) x.row(i) = m_scale[i] * y.row(m_toInternal[i]);
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
template <typename Rhs, typename Dest>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::_solve_impl(const MatrixBase<Rhs>& b,
                                                                          MatrixBase<Dest>& x) const {
  eigen_assert(m_factorized && "the matrix must be factorized first");
  const DenseMatrix rhs = b;
  DenseMatrix solution(m_size, rhs.cols());
  solveTriangular(rhs, solution);

  // Optional stationary refinement. Off by default: with no pivoting and no
  // perturbation there is nothing here to repair (see setMaxIterativeRefinements).
  m_lastRefinementIterations = 0;
  if (m_maxRefinementIterations > 0 && m_size > 0) {
    const RealScalar rhsNorm = rhs.norm();
    DenseMatrix product(rhs.rows(), rhs.cols()), residual, correction(m_size, rhs.cols());
    RealScalar bestNorm = NumTraits<RealScalar>::highest();
    for (Index iter = 0; iter < m_maxRefinementIterations; ++iter) {
      applyA(solution, product);
      residual = rhs - product;
      const RealScalar resNorm = residual.norm();
      const RealScalar rel = (rhsNorm > RealScalar(0)) ? resNorm / rhsNorm : resNorm;
      if (!(numext::isfinite)(rel) || rel <= m_refinementTolerance) break;
      if (rel >= bestNorm) break;  // stalled or diverging: keep what we have
      bestNorm = rel;
      solveTriangular(residual, correction);
      solution += correction;
      ++m_lastRefinementIterations;
    }
  }

  // Honest check: measure the true residual against the original operator and
  // downgrade info() rather than return a bad answer silently.
  {
    const RealScalar rhsNorm = rhs.norm();
    DenseMatrix product(rhs.rows(), rhs.cols());
    applyA(solution, product);
    const RealScalar resNorm = (rhs - product).norm();
    const RealScalar rel = (rhsNorm > RealScalar(0)) ? resNorm / rhsNorm : resNorm;
    m_lastSolveRelativeResidual = rel;
    const bool usable =
        solution.allFinite() && (numext::isfinite)(rel) && rel <= m_solveFailureThreshold;
    if (usable) {
      m_info = Success;
    } else {
      m_info = NumericalIssue;
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "SupernodalLDLT: solve() residual %.3e exceeds the failure threshold %.3e.",
                    static_cast<double>(rel), static_cast<double>(m_solveFailureThreshold));
      m_lastError = buf;
    }
  }
  x = solution;
}

}  // namespace Eigen

#endif  // SUPERNODAL_LDLT_H
