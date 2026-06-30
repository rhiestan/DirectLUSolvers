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
#include "SupernodalLUExecutor.h"
#include "SupernodalLUMatching.h"

namespace Eigen {
namespace supernodal_lu {
// Method used by solve() to refine the direct factor solve.
//   None                 - return the raw factor solve.
//   IterativeRefinement  - stationary refinement x += M^{-1}(b - A x).
//   BiCGStab             - Krylov (BiCGStab) preconditioned by the LU factors;
//                          robust where stationary refinement stalls/diverges.
enum class Refinement { None, IterativeRefinement, BiCGStab };
}  // namespace supernodal_lu

/** \class SupernodalLUTransposeView
 * \brief Solve expression for the transpose (Conjugate=false) or adjoint
 *        (Conjugate=true) of a factored SupernodalLU, returned by
 *        SupernodalLU::transpose() / adjoint(). Mirrors Eigen::SparseLU's
 *        SparseLUTransposeView. The only supported operation is solve(). */
template <bool Conjugate, typename SolverType>
class SupernodalLUTransposeView
    : public SparseSolverBase<SupernodalLUTransposeView<Conjugate, SolverType>> {
  typedef SparseSolverBase<SupernodalLUTransposeView<Conjugate, SolverType>> APIBase;
  using APIBase::m_isInitialized;

 public:
  typedef typename SolverType::Scalar Scalar;
  typedef typename SolverType::StorageIndex StorageIndex;
  typedef typename SolverType::MatrixType MatrixType;
  typedef typename SolverType::OrderingType OrderingType;

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  using APIBase::_solve_impl;

  SupernodalLUTransposeView() = default;

  void setSolver(SolverType* solver) { m_solver = solver; }
  void setIsInitialized(bool isInitialized) { this->m_isInitialized = isInitialized; }

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const {
    eigen_assert(m_solver && m_solver->isFactorized() &&
                 "the matrix must be factorized first");
    m_solver->template _solve_transposed_impl<Conjugate>(b, x);
  }

  inline Index rows() const { return m_solver->rows(); }
  inline Index cols() const { return m_solver->cols(); }

 private:
  SolverType* m_solver = nullptr;
};

/** \class SupernodalLU
 * \brief Supernodal sparse direct LU factorization for general matrices with a
 *        symmetric nonzero pattern.
 *
 * \tparam MatrixType_   A column-major Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>.
 * \tparam OrderingType_ A fill-reducing ordering functor (default AMDOrdering).
 * \tparam Executor_     A parallel-execution backend (default SerialExecutor);
 *                       see SupernodalLUExecutor.h. Controls how the numeric
 *                       factorization is parallelized over the elimination tree.
 *
 * Typical usage (identical to Eigen::SparseLU):
 * \code
 *   SupernodalLU<SparseMatrix<double>> solver;
 *   solver.compute(A);
 *   x = solver.solve(b);
 * \endcode
 *
 * To factor in parallel with the bundled std::thread pool:
 * \code
 *   SupernodalLU<SparseMatrix<double>, AMDOrdering<int>,
 *                supernodal_lu::StdThreadExecutor> solver;
 * \endcode
 */
template <typename MatrixType_, typename OrderingType_ = AMDOrdering<typename MatrixType_::StorageIndex>,
          typename Executor_ = supernodal_lu::SerialExecutor>
class SupernodalLU : public SparseSolverBase<SupernodalLU<MatrixType_, OrderingType_, Executor_>> {
 protected:
  typedef SparseSolverBase<SupernodalLU<MatrixType_, OrderingType_, Executor_>> Base;
  using Base::m_isInitialized;

 public:
  typedef MatrixType_ MatrixType;
  typedef OrderingType_ OrderingType;
  typedef Executor_ Executor;
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

  /** \returns the status of the last operation. After compute()/factorize() this
   *  is Success unless the factorization broke down (singular). After solve() it
   *  is downgraded to NumericalIssue if the computed solution failed the honesty
   *  check (non-finite, or relative residual above solveFailureThreshold()) — so
   *  a usable-looking result is never silently returned for an unsolvable system.
   *  A subsequent solve() with a good right-hand side restores Success. */
  ComputationInfo info() const { return m_info; }

  /** \returns true once a numeric factorization has completed successfully. This
   *  is the guard for solve(); unlike info() it is NOT affected by a failed
   *  solve, so the factors stay usable for further right-hand sides. */
  bool isFactorized() const { return m_factorized; }

  const std::string& lastErrorMessage() const { return m_lastError; }

  /** Row permutation mapping original rows to the internal (factored) numbering.
   *  This folds the maximum-transversal matching together with the fill-reducing
   *  ordering + postordering, so it differs from the column permutation whenever
   *  matching() reorders rows (it equals colsPermutation() when matching is off
   *  or is the identity). */
  const PermutationType& rowsPermutation() const { return m_rowPermutation; }
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

  /** Refinement method used by solve(). Default BiCGStab (LU-preconditioned
   *  Krylov), which is robust where stationary refinement diverges. */
  void setRefinementMethod(supernodal_lu::Refinement method) { m_refinementMethod = method; }
  supernodal_lu::Refinement refinementMethod() const { return m_refinementMethod; }

  /** Number of refinement steps actually taken by the last solve(). */
  Index iterativeRefinements() const { return m_lastRefinementIterations; }

  /** Relative-residual ceiling above which solve() reports failure (NumericalIssue).
   *  After every solve the true relative residual ||b - A x|| / ||b|| is measured
   *  against the original A; if it exceeds this threshold (or the solution is
   *  non-finite) info() becomes NumericalIssue instead of silently returning a
   *  bad answer. The default (1e-6) flags genuinely broken solves — e.g. a matrix
   *  whose diagonal is unusable so every static pivot is replaced — while leaving
   *  ordinary, well-converged solves reported as Success. */
  void setSolveFailureThreshold(const RealScalar& tol) { m_solveFailureThreshold = tol; }
  RealScalar solveFailureThreshold() const { return m_solveFailureThreshold; }

  /** The relative residual ||b - A x|| / ||b|| measured by the last solve(),
   *  against the original (unscaled) matrix. */
  RealScalar solveResidual() const { return m_lastSolveRelativeResidual; }

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

  /** Relative amalgamation tolerance: a path-continuation boundary is also merged
   *  when the extra (logically zero) off-diagonal rows it introduces are at most
   *  this fraction of the rows the supernode already carries. This coarsens the
   *  supernode partition of dense-ish factorizations — where a wide panel already
   *  has many off-diagonal rows, so a few more are negligible fill but the larger
   *  panels sharply reduce per-supernode bookkeeping/dispatch overhead. It barely
   *  affects sparse matrices (few rows -> the absolute rule already governs).
   *  Default 0.3; set 0 to use only the absolute (relaxedSize/maxZeroRows) rule. */
  void setAmalgamationFillFraction(double fraction) { m_amalgamationFillFraction = fraction; }
  double amalgamationFillFraction() const { return m_amalgamationFillFraction; }

  /** Supernode splitting: cap supernode width at maxBlockSize columns by forcing
   *  extra boundaries (PaStiX MAX_BLOCKSIZE). Splitting adds NO fill (it only
   *  relocates inter-block entries) and keeps dense panels at a cache-friendly
   *  size; it also improves parallel load balance (finer, more uniform tasks).
   *  Default 128; 0 = unlimited (no splitting). Applied during analyzePattern(). */
  void setMaxBlockSize(Index maxBlockSize) { m_maxBlockSize = maxBlockSize; }
  Index maxBlockSize() const { return m_maxBlockSize; }

  /** Row/column equilibration (Ruiz). When on (default), factorize() scales the
   *  matrix to A~ = Dr*A*Dc so rows/columns have comparable magnitude, which
   *  improves conditioning, reduces bumped static pivots, and improves backward
   *  stability. Scaling is folded transparently into solve()/transpose()/
   *  determinant(), which all operate on the original A. */
  void setEquilibration(bool on) { m_equilibrate = on; }
  bool equilibration() const { return m_equilibrate; }

  /** Maximum-transversal matching (MC64-style). When on (default), analyzePattern()
   *  permutes large-magnitude entries onto the diagonal BEFORE the symbolic
   *  factorization (à la SuperLU_DIST / MUMPS), giving a zero-free, well-scaled
   *  diagonal so the static-pivot threshold rarely fires. This is the key
   *  robustness ingredient for matrices with a numerically weak or structurally
   *  zero diagonal (e.g. unsymmetric circuit matrices); it is a pure row
   *  permutation and does not change the static block structure. The matched
   *  pattern is symmetrized internally. */
  void setMatching(bool on) { m_useMatching = on; }
  bool matching() const { return m_useMatching; }
  /** True if the last matching produced a full zero-free diagonal (perfect
   *  transversal). False indicates a structurally singular matrix. */
  bool matchingIsPerfect() const { return m_matchingIsPerfect; }

  /** Restricted (in-block) pivoting. When on (default), the dense diagonal block
   *  of each supernode is factored with partial pivoting CONFINED to that block
   *  (row swaps stay within the already-allocated dense block, so the global
   *  symbolic structure — and BLAS-3 / tree parallelism — are preserved). This
   *  makes the diagonal block numerically robust without the dynamic structure
   *  growth of true partial pivoting. Combined with matching() it removes the
   *  dead-diagonal failure mode that pure static pivoting cannot handle. */
  void setDiagonalPivoting(bool on) { m_diagonalPivoting = on; }
  bool diagonalPivoting() const { return m_diagonalPivoting; }
  /** The computed scaling vectors (original numbering), valid after factorize().
   *  A~ = diag(rowScaling()) * A * diag(colScaling()). */
  const std::vector<RealScalar>& rowScaling() const { return m_rowScale; }
  const std::vector<RealScalar>& colScaling() const { return m_colScale; }

  /** Access the parallel-execution backend (e.g. to configure its thread count
   *  for a stateful Executor). The factorization is dispatched through it. */
  Executor& executor() { return m_executor; }
  const Executor& executor() const { return m_executor; }

  Index nnzL() const { return m_nnzL; }
  Index nnzU() const { return m_nnzU; }
  Index supernodeCount() const { return static_cast<Index>(m_supernodes.size()); }

  /** Number of elimination-tree levels (parallel scheduling steps). */
  Index levelCount() const { return static_cast<Index>(m_levelGroups.size()); }
  /** Supernodes in the widest level — an upper bound on usable concurrency. */
  Index widestLevel() const {
    Index w = 0;
    for (const auto& g : m_levelGroups) w = std::max<Index>(w, static_cast<Index>(g.size()));
    return w;
  }

  /** \returns the determinant of the original matrix A. The factored matrix is
   *  the equilibrated A~ = Dr*A*Dc, so det(A) = det(A~) / (prod Dr * prod Dc). */
  Scalar determinant() const {
    Scalar det(1);
    for (const auto& diag : m_diagonalFactor) {
      for (Index k = 0; k < diag.rows(); ++k) det *= diag(k, k);
    }
    // m_factorizationSign folds in the parity of the matching row permutation and
    // of all in-block pivot swaps, so the sign of det(A) is correct.
    return m_factorizationSign * det / m_scalingDeterminant;
  }

  // --- factor accessors (Eigen::SparseLU-compatible) ------------------------

  /** Expression of the unit-lower factor L. The supported operation is the
   *  in-place triangular solve, in the solver's internal numbering:
   *  \code
   *    VectorXd y = solver.rowsPermutation() * b;   // permute into internal order
   *    solver.matrixL().solveInPlace(y);            // L y = (P b)
   *    solver.matrixU().solveInPlace(y);            // U y = ...
   *    VectorXd x = solver.colsPermutation().transpose() * y;
   *  \endcode
   *  with P A P^T = L U. */
  struct SupernodalLUMatrixLReturnType {
    explicit SupernodalLUMatrixLReturnType(const SupernodalLU& solver) : m_solver(solver) {}
    Index rows() const { return m_solver.rows(); }
    Index cols() const { return m_solver.cols(); }
    template <typename Dest>
    void solveInPlace(MatrixBase<Dest>& x) const;
    const SupernodalLU& m_solver;
  };

  /** Expression of the upper factor U; see SupernodalLUMatrixLReturnType. */
  struct SupernodalLUMatrixUReturnType {
    explicit SupernodalLUMatrixUReturnType(const SupernodalLU& solver) : m_solver(solver) {}
    Index rows() const { return m_solver.rows(); }
    Index cols() const { return m_solver.cols(); }
    template <typename Dest>
    void solveInPlace(MatrixBase<Dest>& x) const;
    const SupernodalLU& m_solver;
  };

  SupernodalLUMatrixLReturnType matrixL() const { return SupernodalLUMatrixLReturnType(*this); }
  SupernodalLUMatrixUReturnType matrixU() const { return SupernodalLUMatrixUReturnType(*this); }

  // --- transpose / adjoint solve views (Eigen::SparseLU-compatible) ---------

  /** \returns a view solving the transposed system: x = solver.transpose().solve(b)
   *  solves A^T x = b, reusing the existing factorization. */
  SupernodalLUTransposeView<false, SupernodalLU> transpose() {
    SupernodalLUTransposeView<false, SupernodalLU> view;
    view.setSolver(this);
    view.setIsInitialized(this->m_isInitialized);
    return view;
  }

  /** \returns a view solving the adjoint system: x = solver.adjoint().solve(b)
   *  solves A^H x = b. For real scalar types this equals transpose(). */
  SupernodalLUTransposeView<true, SupernodalLU> adjoint() {
    SupernodalLUTransposeView<true, SupernodalLU> view;
    view.setSolver(this);
    view.setIsInitialized(this->m_isInitialized);
    return view;
  }

  /** Internal: solve A^T x = b (Conjugate=false) or A^H x = b (Conjugate=true).
   *  Used by the transpose()/adjoint() views; prefer those. */
  template <bool Conjugate, typename Rhs, typename Dest>
  void _solve_transposed_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

 private:
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrix;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> WorkMatrix;
  typedef supernodal_lu::Supernode<StorageIndex> Supernode;
  typedef supernodal_lu::RowBlock<StorageIndex> RowBlock;
  typedef supernodal_lu::UpdateSource<StorageIndex> UpdateSource;

  void init() {
    m_size = 0;
    m_analysisDone = false;
    m_factorized = false;
    m_info = InvalidInput;
    m_solveFailureThreshold = RealScalar(1e-6);
    m_lastSolveRelativeResidual = RealScalar(0);
    m_staticPivotThreshold = RealScalar(0);
    m_thresholdIsAuto = true;
    m_replacedPivots = 0;
    m_maxRefinementIterations = 5;
    m_refinementTolerance = NumTraits<RealScalar>::epsilon();
    m_refinementMethod = supernodal_lu::Refinement::BiCGStab;
    m_lastRefinementIterations = 0;
    m_relaxedSize = 4;
    m_maxAmalgamationZeroRows = 4;
    m_amalgamationFillFraction = 0.3;
    m_maxBlockSize = 128;  // split wide supernodes (cf. PaStiX MAX_BLOCKSIZE ~120)
    m_equilibrate = true;
    m_useMatching = true;
    m_diagonalPivoting = true;
    m_matchingIsPerfect = true;
    m_matchSign = 1;
    m_factorizationSign = Scalar(1);
    m_scalingDeterminant = RealScalar(1);
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

  // factor one supernode in place: pull its Schur updates, do the unpivoted LU
  // of its diagonal block, then solve its off-diagonal panels. Writes only this
  // supernode's panels, so supernodes in the same elimination-tree level run
  // concurrently. Reports its local replaced-pivot count and singularity.
  void factorizeSupernode(StorageIndex s, const RealScalar& staticPivot, Index& replacedCount,
                          bool& singular, int& sign);

  // right-looking BLOCKED LU of a dense diagonal block, with static pivoting and
  // optional restricted (in-block) partial pivoting. Panels of width
  // kDiagBlockSize keep the trailing update a BLAS-3 GEMM (vs. an unblocked
  // BLAS-2 rank-1 sweep) for wide supernodes. When restrictedPivot is true, row
  // swaps confined to the block are recorded in rowPerm (rowPerm[k] = local row
  // now sitting at position k; left empty when no swap occurred) and sign carries
  // the permutation parity for the determinant.
  void factorizeDiagonalBlock(DenseMatrix& diag, const RealScalar& staticPivot, Index& replacedCount,
                              bool& singular, std::vector<StorageIndex>& rowPerm, int& sign,
                              bool restrictedPivot) const;

  // group supernodes into elimination-tree levels for parallel scheduling: a
  // supernode's level is 1 + the max level of its update sources.
  void computeSupernodeLevels();

  // Ruiz row/column equilibration: fill m_rowScale/m_colScale so that the scaled
  // matrix diag(Dr)*A*diag(Dc) has rows and columns of comparable magnitude.
  void computeEquilibration(const MatrixType& matrix);

  // one block forward/backward triangular solve (no refinement), in the
  // original numbering. Used both for the initial solve and the corrections.
  void solveTriangular(const DenseMatrix& rhs, DenseMatrix& solution) const;

  // honesty check shared by the forward and transposed solves: measure the true
  // relative residual ||rhs - op*solution|| / ||rhs|| (op via applyA) and set
  // m_info to Success, or NumericalIssue if the solution is non-finite or the
  // residual exceeds m_solveFailureThreshold. Records m_lastSolveRelativeResidual.
  template <typename ApplyA>
  void recordSolveStatus(const DenseMatrix& rhs, const DenseMatrix& solution, ApplyA applyA) const;

  // in-place triangular solves in the internal numbering: L y = y (unit lower)
  // and U y = y. Shared by solveTriangular and the matrixL()/matrixU() proxies.
  template <typename Dest>
  void applyInverseL(Dest& y) const;
  template <typename Dest>
  void applyInverseU(Dest& y) const;

  // transposed in-place triangular solves (internal numbering): L^T/L^H y = y
  // and U^T/U^H y = y, for the transpose()/adjoint() solve.
  template <bool Conjugate, typename Dest>
  void applyInverseLTransposed(Dest& y) const;
  template <bool Conjugate, typename Dest>
  void applyInverseUTransposed(Dest& y) const;
  template <bool Conjugate>
  void solveTriangularTransposed(const DenseMatrix& rhs, DenseMatrix& solution) const;

  // refine an initial solution in place against the operator applyA (A, A^T or
  // A^H) preconditioned by applyPrec (the matching LU solve), using the
  // configured refinement method. applyA(in,out): out = op*in; applyPrec(in,out):
  // out ~= op^{-1}*in. Both operate on pre-sized dense (multi-)columns.
  template <typename ApplyA, typename ApplyPrec>
  void refineSolution(const DenseMatrix& rhs, DenseMatrix& solution, ApplyA applyA,
                      ApplyPrec applyPrec) const;
  // single right-hand-side preconditioned BiCGStab; returns iterations taken.
  template <typename ApplyA, typename ApplyPrec>
  Index bicgstabColumn(ApplyA applyA, ApplyPrec applyPrec, const DenseMatrix& b,
                       DenseMatrix& x) const;

  // state ---------------------------------------------------------------------
  StorageIndex m_size;
  OrderingType m_orderingFunctor;

  // m_toInternal[original] = internal (fill-reducing, postordered) index. This is
  // the COLUMN map (columns are not touched by the matching).
  std::vector<StorageIndex> m_toInternal;
  std::vector<StorageIndex> m_toOriginal;
  PermutationType m_permutation;  // public-facing form of m_toInternal (columns)

  // Maximum-transversal matching (MC64-style). m_matchRow[j] = original row placed
  // on the diagonal of column j; m_matchRowInv is its inverse. m_rowToInternal
  // composes the matching with the symbolic ordering: it maps an original row to
  // its internal index (= m_toInternal[m_matchRowInv[row]]). Identity when
  // matching is off. m_rowPermutation is the public form of m_rowToInternal.
  std::vector<StorageIndex> m_matchRow;
  std::vector<StorageIndex> m_matchRowInv;
  std::vector<StorageIndex> m_rowToInternal;
  PermutationType m_rowPermutation;
  bool m_matchingIsPerfect;
  int m_matchSign;  // parity of the matching row permutation (for det sign)

  // Per-supernode restricted-pivoting row permutation (internal local order).
  // m_diagPivot[s][k] = local diagonal-block row sitting at position k after
  // in-block pivoting; an empty entry means no swap (identity) for that supernode.
  std::vector<std::vector<StorageIndex>> m_diagPivot;
  Scalar m_factorizationSign;  // sign(matching) * prod sign(in-block pivots)

  std::vector<Supernode> m_supernodes;
  std::vector<RowBlock> m_rowBlocks;             // flattened, owned by supernodes
  std::vector<StorageIndex> m_supernodeOfColumn;  // internal column -> supernode id
  std::vector<std::vector<UpdateSource>> m_updateSources;  // per target supernode
  // m_rowPosition[s][internalRow] = position of that off-diagonal row in supernode s
  std::vector<std::unordered_map<StorageIndex, StorageIndex>> m_rowPosition;
  // m_levelGroups[level] = supernodes that can be factored concurrently.
  std::vector<std::vector<StorageIndex>> m_levelGroups;

  Executor m_executor;  // parallel-execution backend (default serial)

  // numeric factors (one entry per supernode)
  std::vector<DenseMatrix> m_diagonalFactor;  // w x w  (packed LU)
  std::vector<DenseMatrix> m_lowerFactor;     // offDiag x w
  std::vector<DenseMatrix> m_upperFactor;     // w x offDiag

  // a copy of A in the original numbering, kept so solve() can form residuals
  // b - A x for iterative refinement.
  MatrixType m_originalMatrix;

  // equilibration: A~ = diag(m_rowScale) * A * diag(m_colScale), both in the
  // original numbering. m_scalingDeterminant = prod(m_rowScale) * prod(m_colScale).
  std::vector<RealScalar> m_rowScale;
  std::vector<RealScalar> m_colScale;
  RealScalar m_scalingDeterminant;

  bool m_analysisDone;
  bool m_factorized;            // a successful numeric factorization is available
  mutable ComputationInfo m_info;     // status of the last operation (solve may set it)
  mutable std::string m_lastError;
  RealScalar m_solveFailureThreshold;               // relative-residual ceiling for solve()
  mutable RealScalar m_lastSolveRelativeResidual;   // residual measured by the last solve()
  RealScalar m_staticPivotThreshold;
  bool m_thresholdIsAuto;
  Index m_replacedPivots;
  Index m_maxRefinementIterations;
  RealScalar m_refinementTolerance;
  supernodal_lu::Refinement m_refinementMethod;
  mutable Index m_lastRefinementIterations;
  Index m_relaxedSize;                // amalgamation: force-merge below this width
  Index m_maxAmalgamationZeroRows;    // amalgamation: max extra zero rows per column
  double m_amalgamationFillFraction;  // amalgamation: max extra rows as fraction of existing
  Index m_maxBlockSize;               // splitting: cap supernode width (0 = unlimited)
  bool m_equilibrate;                 // apply Ruiz row/column scaling
  bool m_useMatching;                 // apply MC64-style maximum-transversal matching
  bool m_diagonalPivoting;            // restricted (in-block) partial pivoting
  Index m_nnzL;
  Index m_nnzU;
};

// ===========================================================================
//  Analysis (symbolic) phase
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::buildSymmetricAdjacency(
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

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::computeEliminationTree(
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

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::computePostorder(const std::vector<StorageIndex>& parent,
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

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::computeColumnStructures(
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

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::detectSupernodesAndBlocks(
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
      const StorageIndex existingRows = rowsBeyond(columnStructure[j - 1], j);
      const StorageIndex deltaRows = rowsBeyond(columnStructure[j], j) - existingRows;
      // Absolute rule (governs sparse matrices: few rows, so the fraction term is
      // tiny) OR a RELATIVE rule: accept the merge when the extra zero rows are a
      // small fraction of the rows the supernode already carries. The relative
      // rule matters for dense-ish factorizations (e.g. gemat11), where a wide
      // panel already has hundreds of off-diagonal rows so a few more are
      // negligible fill but coarser supernodes sharply cut per-supernode overhead.
      const bool merge =
          (static_cast<Index>(childWidth) < m_relaxedSize) ||
          (static_cast<Index>(deltaRows) <= m_maxAmalgamationZeroRows) ||
          (static_cast<double>(deltaRows) <= m_amalgamationFillFraction * static_cast<double>(existingRows));
      start = !merge;
    }
    // Splitting: force a boundary once the running supernode hits maxBlockSize.
    // This adds no fill (inter-block entries just move to off-diagonal panels)
    // and caps dense-panel width for cache- and task-friendly BLAS.
    if (m_maxBlockSize > 0 && static_cast<Index>(j - currentStart) >= m_maxBlockSize)
      start = true;
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

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::computeSupernodeLevels() {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  std::vector<StorageIndex> level(supernodeNbr, 0);
  StorageIndex maxLevel = 0;
  // Update sources of supernode s are tree descendants, so they have smaller
  // ids; processing in increasing id order means level[source] is already final.
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    StorageIndex lv = 0;
    for (const UpdateSource& src : m_updateSources[s])
      lv = std::max<StorageIndex>(lv, static_cast<StorageIndex>(level[src.sourceSupernode] + 1));
    level[s] = lv;
    maxLevel = std::max(maxLevel, lv);
  }
  m_levelGroups.assign(supernodeNbr == 0 ? 0 : (maxLevel + 1), std::vector<StorageIndex>());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) m_levelGroups[level[s]].push_back(s);
}

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::analyzePattern(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "SupernodalLU requires a square matrix");
  m_size = static_cast<StorageIndex>(matrix.rows());
  const StorageIndex n = m_size;

  // 0) maximum-transversal matching: choose a row permutation that places large
  //    entries on the diagonal (MC64-style). All of the symbolic analysis below
  //    runs on the MATCHED matrix B = Pmatch * A, whose diagonal is zero-free, so
  //    the elimination tree / supernodes are built around a strong diagonal and
  //    static-pivot bumping rarely fires. Columns are untouched by the matching.
  m_matchRow.assign(n, 0);
  m_matchRowInv.assign(n, 0);
  m_matchingIsPerfect = true;
  if (m_useMatching && n > 0) {
    m_matchingIsPerfect = supernodal_lu::maximumWeightMatching(matrix, m_matchRow);
  } else {
    for (StorageIndex i = 0; i < n; ++i) m_matchRow[i] = i;
  }
  for (StorageIndex j = 0; j < n; ++j) m_matchRowInv[m_matchRow[j]] = j;
  m_matchSign = supernodal_lu::permutationSign(m_matchRow);

  // Build B = Pmatch * A (original row r -> position m_matchRowInv[r]). When
  // matching is the identity this is a copy of A. B is generally pattern-
  // unsymmetric; the helpers below symmetrize its pattern (structural zeros).
  MatrixType matched;
  if (m_useMatching) {
    std::vector<Triplet<Scalar, StorageIndex>> trips;
    trips.reserve(static_cast<std::size_t>(matrix.nonZeros()));
    for (StorageIndex j = 0; j < n; ++j)
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it)
        trips.emplace_back(m_matchRowInv[static_cast<StorageIndex>(it.index())], j, it.value());
    matched.resize(n, n);
    matched.setFromTriplets(trips.begin(), trips.end());
    matched.makeCompressed();
  }
  const MatrixType& B = m_useMatching ? matched : matrix;

  // 1) fill-reducing ordering (orders the pattern of B + B^T).
  PermutationType orderingPerm;
  m_orderingFunctor(B, orderingPerm);

  m_toInternal.resize(n);
  if (orderingPerm.size() == 0) {  // NaturalOrdering returns empty
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = i;
  } else {
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = orderingPerm.indices()(i);
  }

  // 2) elimination tree of the ordered symmetric pattern.
  std::vector<std::vector<StorageIndex>> adjacency;
  buildSymmetricAdjacency(B, adjacency);
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
  buildSymmetricAdjacency(B, adjacency);
  computeEliminationTree(adjacency, parent);

  // 5) symbolic factorization: per-column structure of the factor.
  std::vector<std::vector<StorageIndex>> columnStructure;
  computeColumnStructures(adjacency, parent, columnStructure);

  // 6) supernode partition + block structure + update lists.
  detectSupernodesAndBlocks(parent, columnStructure);

  // 7) elimination-tree levels for parallel factorization scheduling.
  computeSupernodeLevels();

  // inverse permutation + public permutation objects. m_rowToInternal composes
  // the matching with the symbolic ordering: original row -> internal index.
  m_toOriginal.resize(n);
  for (StorageIndex i = 0; i < n; ++i) m_toOriginal[m_toInternal[i]] = i;
  m_rowToInternal.resize(n);
  for (StorageIndex r = 0; r < n; ++r) m_rowToInternal[r] = m_toInternal[m_matchRowInv[r]];
  m_permutation.resize(n);
  m_rowPermutation.resize(n);
  for (StorageIndex i = 0; i < n; ++i) {
    m_permutation.indices()(i) = m_toInternal[i];
    m_rowPermutation.indices()(i) = m_rowToInternal[i];
  }

  m_analysisDone = true;
  m_info = Success;
  m_isInitialized = true;
}

// ===========================================================================
//  Numeric factorization phase
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::applyUpdate(StorageIndex source, StorageIndex target,
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

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::computeEquilibration(const MatrixType& matrix) {
  const StorageIndex n = m_size;
  m_rowScale.assign(n, RealScalar(1));
  m_colScale.assign(n, RealScalar(1));
  if (!m_equilibrate) return;

  // Ruiz equilibration: alternately scale rows and columns by 1/sqrt(inf-norm)
  // until every scaled row/column inf-norm is ~1. Converges geometrically.
  std::vector<RealScalar> rowMax(n), colMax(n);
  const int maxIters = 5;
  for (int iter = 0; iter < maxIters; ++iter) {
    std::fill(rowMax.begin(), rowMax.end(), RealScalar(0));
    std::fill(colMax.begin(), colMax.end(), RealScalar(0));
    for (StorageIndex j = 0; j < n; ++j) {
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
        const StorageIndex i = static_cast<StorageIndex>(it.index());
        const RealScalar scaled = numext::abs(it.value()) * m_rowScale[i] * m_colScale[j];
        rowMax[i] = numext::maxi(rowMax[i], scaled);
        colMax[j] = numext::maxi(colMax[j], scaled);
      }
    }
    RealScalar deviation(0);  // how far the current inf-norms are from 1
    for (StorageIndex i = 0; i < n; ++i)
      if (rowMax[i] > RealScalar(0)) {
        m_rowScale[i] /= numext::sqrt(rowMax[i]);
        deviation = numext::maxi(deviation, numext::abs(RealScalar(1) - rowMax[i]));
      }
    for (StorageIndex j = 0; j < n; ++j)
      if (colMax[j] > RealScalar(0)) {
        m_colScale[j] /= numext::sqrt(colMax[j]);
        deviation = numext::maxi(deviation, numext::abs(RealScalar(1) - colMax[j]));
      }
    if (deviation < RealScalar(0.1)) break;
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::factorizeDiagonalBlock(
    DenseMatrix& diag, const RealScalar& staticPivot, Index& replacedCount, bool& singular,
    std::vector<StorageIndex>& rowPerm, int& sign, bool restrictedPivot) const {
  const StorageIndex w = static_cast<StorageIndex>(diag.rows());
  const StorageIndex kDiagBlockSize = 64;  // BLAS-3 panel width (cf. PaStiX MAXSIZEOFBLOCKS)

  // Restricted (in-block) partial pivoting: track the row permutation applied to
  // the dense diagonal block. perm[k] = local block row now sitting at position
  // k. Swaps are confined to the block, so the global symbolic structure (and
  // thus L21 / U12 / parallelism) is untouched. Left as identity when off.
  sign = 1;
  rowPerm.clear();
  std::vector<StorageIndex> perm;
  bool anySwap = false;
  if (restrictedPivot) {
    perm.resize(static_cast<std::size_t>(w));
    for (StorageIndex i = 0; i < w; ++i) perm[i] = i;
  }

  for (StorageIndex j0 = 0; j0 < w; j0 += kDiagBlockSize) {
    const StorageIndex jb = std::min<StorageIndex>(kDiagBlockSize, w - j0);
    const StorageIndex panelEnd = j0 + jb;

    // (1) unblocked LU of the current panel: columns [j0, panelEnd), all rows
    //     below (so L21 is formed too). Rank-1 updates stay inside the panel.
    for (StorageIndex k = j0; k < panelEnd; ++k) {
      // restricted partial pivoting: bring the largest entry in column k from
      // within the diagonal block (rows [k, w)) onto the pivot, swapping FULL
      // rows of the block so the already-factored columns stay consistent.
      if (restrictedPivot) {
        Index relMax = 0;
        diag.col(k).segment(k, w - k).cwiseAbs().maxCoeff(&relMax);
        const StorageIndex p = k + static_cast<StorageIndex>(relMax);
        if (p != k) {
          diag.row(k).swap(diag.row(p));
          std::swap(perm[static_cast<std::size_t>(k)], perm[static_cast<std::size_t>(p)]);
          sign = -sign;
          anySwap = true;
        }
      }
      Scalar pivot = diag(k, k);
      if (staticPivot > RealScalar(0) && std::abs(pivot) < staticPivot) {
        pivot = (std::abs(pivot) == RealScalar(0)) ? Scalar(staticPivot)
                                                   : pivot * (staticPivot / std::abs(pivot));
        diag(k, k) = pivot;
        ++replacedCount;
      }
      if (pivot == Scalar(0)) {
        singular = true;
        return;
      }
      const StorageIndex below = w - k - 1;
      if (below > 0) {
        diag.col(k).segment(k + 1, below) /= pivot;
        const StorageIndex panelTail = panelEnd - (k + 1);
        if (panelTail > 0)
          diag.block(k + 1, k + 1, below, panelTail).noalias() -=
              diag.col(k).segment(k + 1, below) * diag.row(k).segment(k + 1, panelTail);
      }
    }

    // (2) U12 = L11^{-1} A12 (TRSM) and (3) A22 -= L21 * U12 (GEMM) — BLAS-3.
    const StorageIndex trailing = w - panelEnd;
    if (trailing > 0) {
      auto upperRight = diag.block(j0, panelEnd, jb, trailing);  // A12, becomes U12
      diag.block(j0, j0, jb, jb).template triangularView<UnitLower>().solveInPlace(upperRight);
      diag.block(panelEnd, panelEnd, trailing, trailing).noalias() -=
          diag.block(panelEnd, j0, trailing, jb) * upperRight;  // L21 * U12
    }
  }

  if (anySwap) rowPerm = std::move(perm);  // empty => identity (no swap)
}

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::factorizeSupernode(StorageIndex s,
                                                                          const RealScalar& staticPivot,
                                                                          Index& replacedCount,
                                                                          bool& singular, int& sign) {
  // pull contributions from already-factored sources (lower levels).
  for (const UpdateSource& source : m_updateSources[s])
    applyUpdate(source.sourceSupernode, s, source.facingRowBlock);

  DenseMatrix& diag = m_diagonalFactor[s];

  // (static + optional restricted in-block) LU of the diagonal block: packed L
  // (unit) and U. m_diagPivot[s] receives the in-block row permutation (empty if
  // no swap occurred). The Schur complement a source sends to its targets is
  // invariant to that local permutation (the P_s cancels), so applyUpdate and the
  // off-diagonal panel structure are unaffected.
  std::vector<StorageIndex>& rowPerm = m_diagPivot[s];
  factorizeDiagonalBlock(diag, staticPivot, replacedCount, singular, rowPerm, sign,
                         m_diagonalPivoting);
  if (singular) return;

  // solve the off-diagonal panels against the diagonal factors.
  if (m_supernodes[s].offDiagonalRowCount > 0) {
    // U12's rows are the same equations as the diagonal block, so apply the
    // in-block pivot permutation to upperFactor's rows before the L_kk solve.
    // (lowerFactor's rows are other equations, below the block, and are not
    // permuted — its values use the pivoted U_kk via the right solve below.)
    if (!rowPerm.empty()) {
      const StorageIndex w = m_supernodes[s].width();
      DenseMatrix permuted(w, m_upperFactor[s].cols());
      for (StorageIndex k = 0; k < w; ++k) permuted.row(k) = m_upperFactor[s].row(rowPerm[k]);
      m_upperFactor[s] = std::move(permuted);
    }
    // lowerFactor := lowerFactor * U_kk^{-1}
    diag.template triangularView<Upper>().template solveInPlace<OnTheRight>(m_lowerFactor[s]);
    // upperFactor := L_kk^{-1} * upperFactor
    diag.template triangularView<UnitLower>().solveInPlace(m_upperFactor[s]);
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analysisDone && "analyzePattern must be called before factorize");
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_replacedPivots = 0;
  m_factorized = false;
  m_info = Success;

  // keep A for iterative refinement during solve().
  m_originalMatrix = matrix;

  // row/column equilibration: factor the scaled matrix A~ = Dr*A*Dc.
  computeEquilibration(matrix);
  m_scalingDeterminant = RealScalar(1);
  for (StorageIndex i = 0; i < m_size; ++i)
    m_scalingDeterminant *= m_rowScale[i] * m_colScale[i];

  // resolve the effective static-pivot threshold. When automatic, scale it to
  // the SCALED matrix magnitude (sqrt(eps) * max|A~_ij|, SuperLU_DIST style).
  RealScalar staticPivot = m_staticPivotThreshold;
  if (m_thresholdIsAuto) {
    RealScalar maxAbs(0);
    for (StorageIndex j = 0; j < m_size; ++j)
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
        const StorageIndex i = static_cast<StorageIndex>(it.index());
        maxAbs = numext::maxi(maxAbs, numext::abs(it.value()) * m_rowScale[i] * m_colScale[j]);
      }
    staticPivot = numext::sqrt(NumTraits<RealScalar>::epsilon()) * maxAbs;
  }

  // 1) allocate and zero the dense panels.
  m_diagonalFactor.resize(supernodeNbr);
  m_lowerFactor.resize(supernodeNbr);
  m_upperFactor.resize(supernodeNbr);
  m_diagPivot.assign(supernodeNbr, std::vector<StorageIndex>());
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
      const StorageIndex origRow = static_cast<StorageIndex>(it.index());
      // m_rowToInternal folds the matching row permutation into the ordering, so
      // the matched entry of each column lands on the internal diagonal block.
      const StorageIndex ii = m_rowToInternal[origRow];
      const Scalar value = it.value() * m_rowScale[origRow] * m_colScale[j];  // A~ = Dr A Dc
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

  // 3) left-looking supernodal factorization, scheduled by elimination-tree
  //    levels: all supernodes within a level are independent (each writes only
  //    its own panels) and are dispatched concurrently; levels run in order so
  //    every update source is already factored. Per-supernode outputs are kept
  //    in disjoint slots to avoid shared writes during the parallel region.
  std::vector<Index> replacedPerSupernode(supernodeNbr, 0);
  std::vector<char> singularPerSupernode(supernodeNbr, 0);
  std::vector<int> signPerSupernode(supernodeNbr, 1);  // parity of each in-block pivot perm
  bool singular = false;
  for (const std::vector<StorageIndex>& group : m_levelGroups) {
    const Index groupSize = static_cast<Index>(group.size());
    m_executor.parallelFor(Index(0), groupSize, [&](Index k) {
      const StorageIndex s = group[static_cast<std::size_t>(k)];
      bool localSingular = false;
      factorizeSupernode(s, staticPivot, replacedPerSupernode[s], localSingular, signPerSupernode[s]);
      singularPerSupernode[s] = localSingular ? 1 : 0;
    });
    for (StorageIndex s : group)
      if (singularPerSupernode[s]) singular = true;
    if (singular) break;  // factor is unusable; stop launching further levels
  }

  m_replacedPivots = 0;
  for (Index r : replacedPerSupernode) m_replacedPivots += r;
  // determinant sign = parity of the matching row permutation times the parity of
  // every supernode's in-block pivot permutation.
  int totalSign = m_matchSign;
  for (int sg : signPerSupernode) totalSign *= sg;
  m_factorizationSign = Scalar(totalSign);
  if (singular) {
    m_info = NumericalIssue;
    m_lastError = "SupernodalLU: zero pivot encountered (matrix is singular).";
    return;
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

  m_factorized = true;
  m_isInitialized = true;
}

// ===========================================================================
//  Solve phase
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Rhs, typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::_solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const {
  eigen_assert(m_factorized && "the matrix must be factorized first");
  const Index nrhs = b.cols();

  const DenseMatrix rhs = b;            // materialize the right-hand side
  DenseMatrix solution(m_size, nrhs);
  solveTriangular(rhs, solution);       // initial direct factor solve

  // refine against A, preconditioned by the LU factors (M^{-1} = solveTriangular).
  auto applyA = [this](const DenseMatrix& in, DenseMatrix& out) { out.noalias() = m_originalMatrix * in; };
  refineSolution(rhs, solution, applyA,
                 [this](const DenseMatrix& in, DenseMatrix& out) { solveTriangular(in, out); });

  recordSolveStatus(rhs, solution, applyA);  // honest pass/fail on the final residual
  x = solution;
}

// Forward substitution L y = y (unit-lower), in place, on an internal-numbered
// right-hand side. Shared by the full solve and matrixL().solveInPlace().
template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::applyInverseL(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    // apply the in-block pivot permutation: head := P_s head (newhead[k]=head[piv[k]]).
    const std::vector<StorageIndex>& piv = m_diagPivot[s];
    if (!piv.empty()) {
      DenseMatrix tmp = head;
      for (StorageIndex k = 0; k < w; ++k) head.row(k) = tmp.row(piv[k]);
    }
    m_diagonalFactor[s].template triangularView<UnitLower>().solveInPlace(head);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      y.middleRows(block.firstRow, hb).noalias() -= m_lowerFactor[s].middleRows(block.panelOffset, hb) * head;
    }
  }
}

// Backward substitution U y = y (upper), in place, on an internal-numbered
// right-hand side. Shared by the full solve and matrixU().solveInPlace().
template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::applyInverseU(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
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
}

template <typename MatrixType, typename OrderingType, typename Executor>
void SupernodalLU<MatrixType, OrderingType, Executor>::solveTriangular(const DenseMatrix& rhs,
                                                             DenseMatrix& x) const {
  const StorageIndex n = m_size;
  const Index nrhs = rhs.cols();

  // Solve A x = rhs via the factors of the matched, scaled matrix
  //   Aeff = Pmatch * Dr * A * Dc,  Pdiag * P * Aeff * P^T = L U.
  // Rows permute in through the matching+ordering (m_rowToInternal) with row
  // scaling; columns permute out through the ordering (m_toInternal) with column
  // scaling. The in-block pivot permutation Pdiag is folded inside applyInverseL.
  DenseMatrix y(n, nrhs);
  for (StorageIndex i = 0; i < n; ++i) y.row(m_rowToInternal[i]) = m_rowScale[i] * rhs.row(i);
  applyInverseL(y);
  applyInverseU(y);
  for (StorageIndex i = 0; i < n; ++i) x.row(i) = m_colScale[i] * y.row(m_toInternal[i]);
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA>
void SupernodalLU<MatrixType, OrderingType, Executor>::recordSolveStatus(const DenseMatrix& rhs,
                                                                         const DenseMatrix& solution,
                                                                         ApplyA applyA) const {
  const RealScalar rhsNorm = rhs.norm();
  DenseMatrix product(rhs.rows(), rhs.cols());
  applyA(solution, product);
  const RealScalar resNorm = (rhs - product).norm();
  // Relative residual; for a zero right-hand side fall back to the absolute one.
  const RealScalar relResid = (rhsNorm > RealScalar(0)) ? resNorm / rhsNorm : resNorm;
  m_lastSolveRelativeResidual = relResid;

  const bool usable = solution.allFinite() && (numext::isfinite)(relResid) &&
                      relResid <= m_solveFailureThreshold;
  if (usable) {
    m_info = Success;
  } else {
    m_info = NumericalIssue;
    m_lastError =
        "SupernodalLU: solve produced a large residual (see solveResidual()); the matrix is "
        "likely too ill-conditioned for static pivoting and may require true row pivoting.";
  }
}

// ===========================================================================
//  matrixL() / matrixU() factor accessors
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::SupernodalLUMatrixLReturnType::solveInPlace(
    MatrixBase<Dest>& x) const {
  m_solver.applyInverseL(x.derived());
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::SupernodalLUMatrixUReturnType::solveInPlace(
    MatrixBase<Dest>& x) const {
  m_solver.applyInverseU(x.derived());
}

// ===========================================================================
//  transpose() / adjoint() solve
// ===========================================================================

// Solve U^T y = y (Conjugate=false) or U^H y = y, in place, internal numbering.
// U^T is lower-triangular, so this is a forward sweep mirroring applyInverseL:
// diagonal solve, then push the result down to higher-index rows via U's panel.
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::applyInverseUTransposed(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    if (Conjugate)
      m_diagonalFactor[s].template triangularView<Upper>().adjoint().solveInPlace(head);
    else
      m_diagonalFactor[s].template triangularView<Upper>().transpose().solveInPlace(head);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      const auto panel = m_upperFactor[s].middleCols(block.panelOffset, hb);  // w x hb
      if (Conjugate)
        y.middleRows(block.firstRow, hb).noalias() -= panel.adjoint() * head;
      else
        y.middleRows(block.firstRow, hb).noalias() -= panel.transpose() * head;
    }
  }
}

// Solve L^T z = z (Conjugate=false) or L^H z = z, in place, internal numbering.
// L^T is unit-upper-triangular, so this is a backward sweep mirroring
// applyInverseU: pull contributions from higher-index rows via L's panel, then
// the (unit) diagonal solve.
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::applyInverseLTransposed(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = supernodeNbr - 1; s >= 0; --s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      const auto panel = m_lowerFactor[s].middleRows(block.panelOffset, hb);  // hb x w
      if (Conjugate)
        head.noalias() -= panel.adjoint() * y.middleRows(block.firstRow, hb);
      else
        head.noalias() -= panel.transpose() * y.middleRows(block.firstRow, hb);
    }
    if (Conjugate)
      m_diagonalFactor[s].template triangularView<UnitLower>().adjoint().solveInPlace(head);
    else
      m_diagonalFactor[s].template triangularView<UnitLower>().transpose().solveInPlace(head);
    // transpose of applyInverseL applies the INVERSE in-block pivot permutation
    // here: head := P_s^{-1} head (newhead[piv[k]] = head[k]).
    const std::vector<StorageIndex>& piv = m_diagPivot[s];
    if (!piv.empty()) {
      DenseMatrix tmp = head;
      for (StorageIndex k = 0; k < w; ++k) head.row(piv[k]) = tmp.row(k);
    }
    if (s == 0) break;  // guard against unsigned underflow
  }
}

// A^T = Dc^{-1} A~^T Dr^{-1}, so solve A^T x = rhs as x = Dr * A~^{-T} * (Dc * rhs)
// (the row/column scalings swap roles vs. the forward solve).
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate>
void SupernodalLU<MatrixType, OrderingType, Executor>::solveTriangularTransposed(const DenseMatrix& rhs,
                                                                       DenseMatrix& x) const {
  const StorageIndex n = m_size;
  const Index nrhs = rhs.cols();
  DenseMatrix y(n, nrhs);
  for (StorageIndex i = 0; i < n; ++i) y.row(m_toInternal[i]) = m_colScale[i] * rhs.row(i);
  applyInverseUTransposed<Conjugate>(y);
  applyInverseLTransposed<Conjugate>(y);  // folds in the inverse in-block pivot perm
  for (StorageIndex i = 0; i < n; ++i) x.row(i) = m_rowScale[i] * y.row(m_rowToInternal[i]);
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Rhs, typename Dest>
void SupernodalLU<MatrixType, OrderingType, Executor>::_solve_transposed_impl(const MatrixBase<Rhs>& b,
                                                                    MatrixBase<Dest>& x) const {
  eigen_assert(m_factorized && "the matrix must be factorized first");
  const Index nrhs = b.cols();

  const DenseMatrix rhs = b;
  DenseMatrix solution(m_size, nrhs);
  solveTriangularTransposed<Conjugate>(rhs, solution);

  // refine against A^T (or A^H), preconditioned by the transposed LU solve.
  auto applyA = [this](const DenseMatrix& in, DenseMatrix& out) {
    if (Conjugate)
      out.noalias() = m_originalMatrix.adjoint() * in;
    else
      out.noalias() = m_originalMatrix.transpose() * in;
  };
  refineSolution(rhs, solution, applyA, [this](const DenseMatrix& in, DenseMatrix& out) {
    this->template solveTriangularTransposed<Conjugate>(in, out);
  });

  recordSolveStatus(rhs, solution, applyA);  // honest pass/fail on the final residual
  x = solution;
}

// ===========================================================================
//  Iterative / Krylov refinement
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA, typename ApplyPrec>
void SupernodalLU<MatrixType, OrderingType, Executor>::refineSolution(const DenseMatrix& rhs,
                                                                      DenseMatrix& solution,
                                                                      ApplyA applyA,
                                                                      ApplyPrec applyPrec) const {
  m_lastRefinementIterations = 0;
  const RealScalar rhsNorm = rhs.norm();
  if (m_maxRefinementIterations <= 0 || rhsNorm == RealScalar(0) ||
      m_refinementMethod == supernodal_lu::Refinement::None)
    return;

  const Index n = rhs.rows();
  const Index nrhs = rhs.cols();

  if (m_refinementMethod == supernodal_lu::Refinement::IterativeRefinement) {
    // stationary refinement (all columns at once): keep the best iterate and
    // stop on convergence, divergence, or the iteration cap.
    DenseMatrix best = solution;
    RealScalar bestNorm = NumTraits<RealScalar>::highest();
    RealScalar prevNorm = NumTraits<RealScalar>::highest();
    DenseMatrix product(n, nrhs), correction(n, nrhs);
    for (Index it = 0;; ++it) {
      applyA(solution, product);
      const DenseMatrix residual = rhs - product;
      const RealScalar resNorm = residual.norm();
      if (resNorm < bestNorm) {
        bestNorm = resNorm;
        best = solution;
      }
      if (resNorm <= m_refinementTolerance * rhsNorm) break;
      if (resNorm > prevNorm) break;
      if (it >= m_maxRefinementIterations) break;
      prevNorm = resNorm;
      applyPrec(residual, correction);
      solution += correction;
      ++m_lastRefinementIterations;
    }
    solution = best;
    return;
  }

  // BiCGStab, one right-hand-side column at a time.
  DenseMatrix bcol(n, 1), xcol(n, 1);
  for (Index c = 0; c < nrhs; ++c) {
    bcol.col(0) = rhs.col(c);
    xcol.col(0) = solution.col(c);
    const Index iters = bicgstabColumn(applyA, applyPrec, bcol, xcol);
    solution.col(c) = xcol.col(0);
    m_lastRefinementIterations = numext::maxi(m_lastRefinementIterations, iters);
  }
}

// Preconditioned BiCGStab for one column: solve (op) x = b with preconditioner
// applyPrec ~= op^{-1}, starting from the given x. Returns the best iterate seen
// (so it never makes the direct solve worse) and the iteration count.
template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA, typename ApplyPrec>
Index SupernodalLU<MatrixType, OrderingType, Executor>::bicgstabColumn(ApplyA applyA,
                                                                       ApplyPrec applyPrec,
                                                                       const DenseMatrix& b,
                                                                       DenseMatrix& x) const {
  const Index n = b.rows();
  const RealScalar bnorm = b.col(0).norm();
  const RealScalar threshold = m_refinementTolerance * bnorm;

  DenseMatrix tmp(n, 1);
  applyA(x, tmp);
  DenseMatrix r = b - tmp;
  DenseMatrix best = x;
  RealScalar bestNorm = r.col(0).norm();
  if (bestNorm <= threshold) return 0;

  const DenseMatrix rhat = r;  // shadow residual
  Scalar rho(1), alpha(1), omega(1);
  DenseMatrix v = DenseMatrix::Zero(n, 1), p = DenseMatrix::Zero(n, 1);
  DenseMatrix y(n, 1), s(n, 1), z(n, 1), t(n, 1);

  Index iters = 0;
  for (Index it = 0; it < m_maxRefinementIterations; ++it) {
    const Scalar rhoNew = rhat.col(0).dot(r.col(0));
    if (numext::abs(rhoNew) == RealScalar(0)) break;  // breakdown
    if (it == 0) {
      p = r;
    } else {
      const Scalar beta = (rhoNew / rho) * (alpha / omega);
      p = r + beta * (p - omega * v);
    }
    applyPrec(p, y);  // y = M^{-1} p
    applyA(y, v);     // v = op y
    const Scalar denom = rhat.col(0).dot(v.col(0));
    if (numext::abs(denom) == RealScalar(0)) break;
    alpha = rhoNew / denom;
    s = r - alpha * v;
    ++iters;
    const RealScalar snorm = s.col(0).norm();
    if (snorm <= threshold) {
      x += alpha * y;
      best = x;
      break;
    }
    applyPrec(s, z);  // z = M^{-1} s
    applyA(z, t);     // t = op z
    const Scalar tt = t.col(0).dot(t.col(0));
    if (numext::abs(tt) == RealScalar(0)) break;
    omega = t.col(0).dot(s.col(0)) / tt;
    x += alpha * y + omega * z;
    r = s - omega * t;
    rho = rhoNew;
    const RealScalar rnorm = r.col(0).norm();
    if (rnorm < bestNorm) {
      bestNorm = rnorm;
      best = x;
    }
    if (rnorm <= threshold) break;
    if (numext::abs(omega) == RealScalar(0)) break;  // breakdown
  }
  x = best;
  return iters;
}

}  // namespace Eigen

#endif  // SUPERNODAL_LU_H
