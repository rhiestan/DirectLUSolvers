// LeftRightLU - a sparse direct LU solver for Eigen using the algorithmic
// design of PARDISO (see pardiso_algorithms.md), as a sibling of SupernodalLU.
//
// What LeftRightLU shares with SupernodalLU (and reuses verbatim from the
// supernodal_lu:: support/matching/executor headers):
//   * General (unsymmetric) values with a SYMMETRIC nonzero pattern.
//   * The whole symbolic pipeline: MC64-style maximum-transversal matching,
//     fill-reducing ordering, elimination tree + postorder, supernode detection
//     with amalgamation/splitting, and block symbolic factorization.
//   * Ruiz equilibration, static-pivot fallback, and the block triangular solve
//     with iterative / Krylov refinement.
//
// What is DIFFERENT here -- the PARDISO-distinctive numeric core:
//   1. LEFT-RIGHT-LOOKING, BARRIER-FREE DYNAMIC SCHEDULING. Instead of
//      SupernodalLU's bulk-synchronous elimination-tree levels (one parallelFor
//      per level with a hard barrier between levels), each supernode carries an
//      atomic count of unfinished update sources (its in-degree in the assembly
//      DAG). A worker takes a READY supernode (count 0), GATHERS its updates from
//      the already-finished sources (left-looking), factors it, then PUSHES
//      readiness to its consumers (right-looking): it decrements each consumer's
//      counter and enqueues any that reach zero. No worker ever blocks on a
//      dependency -- it always takes the next ready node -- so the big root
//      separators no longer serialize a whole tree level. A LIFO ready-stack
//      gives depth-first subtree affinity (PARDISO's cooperative subtree
//      ownership, minus the NUMA placement, which is out of scope). This is the
//      "two-level dynamic scheduler" idea reduced to what shared memory needs.
//   2. IN-BLOCK COMPLETE PIVOTING (PARDISO DGETC2, unsymmetric MTYPE=11/13):
//      the dense diagonal block of each supernode is factored with BOTH row and
//      column interchanges confined to that block, giving per-supernode row (P_s)
//      and column (Q_s) permutations. Because the search never leaves the block,
//      the precomputed symbolic structure is never invalidated. Selectable
//      None / Partial (row-only) / Complete (default); the column permutation is
//      folded transparently through solve()/transpose()/determinant().
//   3. Refinement GATED on whether perturbation occurred (PARDISO IPARM(8)=0):
//      by default solve() only runs refinement when static pivots were bumped.
//   4. logAbsDeterminant() (PARDISO IPARM(33) log-determinant).
//
// Excluded by design (per project scope): NUMA-aware data placement, out-of-core
// factorization, MPI. Bit-reproducibility across thread counts is not a goal
// (the dynamic scheduler reassociates floating-point updates across runs; the
// true residual is unaffected and refinement cleans up the rest).
//
// Pipeline (mirrors Eigen's analyzePattern / factorize split):
//   analyzePattern : matching -> ordering -> elimination tree -> postorder ->
//                    supernode detection -> symbolic block factorization ->
//                    assembly-DAG consumer lists
//   factorize      : scatter values -> left-right-looking dynamic factorization
//   solve          : block forward (L) and backward (U) substitution + refinement
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef LEFT_RIGHT_LU_H
#define LEFT_RIGHT_LU_H

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <vector>

#include "SupernodalLUSupport.h"
#include "SupernodalLUExecutor.h"
#include "SupernodalLUMatching.h"

namespace Eigen {
namespace left_right_lu {
// Method used by solve() to refine the direct factor solve.
//   None                 - return the raw factor solve.
//   IterativeRefinement  - stationary refinement x += M^{-1}(b - A x).
//   BiCGStab             - Krylov (BiCGStab) preconditioned by the LU factors;
//                          robust where stationary refinement stalls/diverges.
enum class Refinement { None, IterativeRefinement, BiCGStab };

// In-block pivoting strategy for each supernode's dense diagonal block.
//   None      - no interchanges; rely on matching + static-pivot bumping only.
//   Partial   - row-only partial pivoting confined to the block (as SupernodalLU).
//   Complete  - PARDISO DGETC2-style row AND column pivoting confined to the
//               block: most robust, and still leaves the symbolic structure fixed.
enum class Pivoting { None, Partial, Complete };
}  // namespace left_right_lu

/** \class LeftRightLUTransposeView
 * \brief Solve expression for the transpose (Conjugate=false) or adjoint
 *        (Conjugate=true) of a factored LeftRightLU, returned by
 *        LeftRightLU::transpose() / adjoint(). Mirrors Eigen::SparseLU's
 *        SparseLUTransposeView. The only supported operation is solve(). */
template <bool Conjugate, typename SolverType>
class LeftRightLUTransposeView
    : public SparseSolverBase<LeftRightLUTransposeView<Conjugate, SolverType>> {
  typedef SparseSolverBase<LeftRightLUTransposeView<Conjugate, SolverType>> APIBase;
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

  LeftRightLUTransposeView() = default;

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

/** \class LeftRightLU
 * \brief PARDISO-style supernodal sparse direct LU factorization for general
 *        matrices with a symmetric nonzero pattern: left-right-looking numeric
 *        factorization driven by a barrier-free dynamic (assembly-DAG) scheduler,
 *        with in-block complete pivoting.
 *
 * \tparam MatrixType_   A column-major Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>.
 * \tparam OrderingType_ A fill-reducing ordering functor (default AMDOrdering).
 * \tparam Executor_     A parallel-execution backend (default SerialExecutor);
 *                       see SupernodalLUExecutor.h. Supplies the worker count for
 *                       the dynamic scheduler (its concurrency()); the DAG
 *                       scheduling itself is driven by LeftRightLU, not by the
 *                       executor's fork-join parallelFor.
 *
 * Typical usage (identical to Eigen::SparseLU):
 * \code
 *   LeftRightLU<SparseMatrix<double>> solver;
 *   solver.compute(A);
 *   x = solver.solve(b);
 * \endcode
 */
template <typename MatrixType_, typename OrderingType_ = AMDOrdering<typename MatrixType_::StorageIndex>,
          typename Executor_ = supernodal_lu::SerialExecutor>
class LeftRightLU : public SparseSolverBase<LeftRightLU<MatrixType_, OrderingType_, Executor_>> {
 protected:
  typedef SparseSolverBase<LeftRightLU<MatrixType_, OrderingType_, Executor_>> Base;
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

  LeftRightLU() { init(); }
  explicit LeftRightLU(const MatrixType& matrix) {
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
   *  check (non-finite, or relative residual above solveFailureThreshold()). */
  ComputationInfo info() const { return m_info; }

  /** \returns true once a numeric factorization has completed successfully. */
  bool isFactorized() const { return m_factorized; }

  const std::string& lastErrorMessage() const { return m_lastError; }

  /** Row permutation mapping original rows to the internal (factored) numbering
   *  (matching composed with ordering + postordering). Differs from
   *  colsPermutation() whenever matching() reorders rows. */
  const PermutationType& rowsPermutation() const { return m_rowPermutation; }
  /** Column permutation chosen by the ordering + postordering. Note: in-block
   *  complete pivoting adds a further per-supernode column permutation that is
   *  folded internally into solve()/determinant() and is NOT reflected here. */
  const PermutationType& colsPermutation() const { return m_permutation; }

  /** Magnitude below which a diagonal pivot is replaced (static pivoting), used
   *  only as a LAST-RESORT fallback when in-block pivoting still cannot find an
   *  acceptable pivot (PARDISO IPARM(10)). Automatic by default:
   *  sqrt(epsilon) * max|A~_ij| of the equilibrated matrix. Pass 0 to disable. */
  void setStaticPivotThreshold(const RealScalar& threshold) {
    m_staticPivotThreshold = threshold;
    m_thresholdIsAuto = false;
  }
  /** Eigen::SparseLU-compatible alias. */
  void setPivotThreshold(const RealScalar& threshold) { setStaticPivotThreshold(threshold); }

  /** Number of pivots that were replaced during the last factorization. */
  Index replacedPivots() const { return m_replacedPivots; }

  /** Maximum number of iterative-refinement steps applied during solve(). */
  void setMaxIterativeRefinements(Index iters) { m_maxRefinementIterations = iters; }
  Index maxIterativeRefinements() const { return m_maxRefinementIterations; }

  /** Relative-residual target ||b - A x|| / ||b|| at which refinement stops. */
  void setRefinementTolerance(const RealScalar& tol) { m_refinementTolerance = tol; }
  RealScalar refinementTolerance() const { return m_refinementTolerance; }

  /** Refinement method used by solve(). Default BiCGStab (LU-preconditioned
   *  Krylov), which is robust where stationary refinement diverges. */
  void setRefinementMethod(left_right_lu::Refinement method) { m_refinementMethod = method; }
  left_right_lu::Refinement refinementMethod() const { return m_refinementMethod; }

  /** Gate refinement on perturbation (PARDISO IPARM(8)=0 behavior). When on
   *  (default), solve() runs refinement only if the factorization actually bumped
   *  a static pivot (replacedPivots() > 0); an un-perturbed factorization is
   *  already backward-stable, so refinement is skipped. Turn off to always refine. */
  void setRefineOnlyIfPerturbed(bool on) { m_refineOnlyIfPerturbed = on; }
  bool refineOnlyIfPerturbed() const { return m_refineOnlyIfPerturbed; }

  /** Number of refinement steps actually taken by the last solve(). */
  Index iterativeRefinements() const { return m_lastRefinementIterations; }

  /** Relative-residual ceiling above which solve() reports failure. Default 1e-6. */
  void setSolveFailureThreshold(const RealScalar& tol) { m_solveFailureThreshold = tol; }
  RealScalar solveFailureThreshold() const { return m_solveFailureThreshold; }

  /** The relative residual ||b - A x|| / ||b|| measured by the last solve(). */
  RealScalar solveResidual() const { return m_lastSolveRelativeResidual; }

  /** Amalgamation (relaxed supernodes): merge fundamental supernodes into larger
   *  dense panels for better BLAS-3 efficiency at the cost of bounded fill.
   *  (1,0) recovers the pure fundamental-supernode partition. See SupernodalLU. */
  void setAmalgamation(Index relaxedSize, Index maxZeroRows) {
    m_relaxedSize = relaxedSize;
    m_maxAmalgamationZeroRows = maxZeroRows;
  }
  Index relaxedSize() const { return m_relaxedSize; }
  Index maxAmalgamationZeroRows() const { return m_maxAmalgamationZeroRows; }

  /** Relative amalgamation tolerance (extra zero rows as a fraction of existing).
   *  Default 0.3; 0 uses only the absolute rule. */
  void setAmalgamationFillFraction(double fraction) { m_amalgamationFillFraction = fraction; }
  double amalgamationFillFraction() const { return m_amalgamationFillFraction; }

  /** Supernode splitting: cap supernode width (PARDISO's ~80-column panels /
   *  PaStiX MAX_BLOCKSIZE). Adds no fill; caps the dense diagonal block so the
   *  complete-pivoting search stays cheap. Default 128; 0 = unlimited. */
  void setMaxBlockSize(Index maxBlockSize) { m_maxBlockSize = maxBlockSize; }
  Index maxBlockSize() const { return m_maxBlockSize; }

  /** Fail-fast fill guard. A symmetric-pattern factorization can require far more
   *  fill than an unsymmetric solver on matrices that lack good vertex separators
   *  (e.g. some 3D FEM systems): the predicted L/U factor can reach hundreds of
   *  GB where Eigen::SparseLU / PARDISO stay in the sub-GB range. When `limit`
   *  (in scalars; memory is limit*sizeof(Scalar)) is > 0, factorize() compares it
   *  against predictedFactorNonzeros() -- computed from the symbolic structure
   *  alone, in milliseconds -- and, if exceeded, ABORTS BEFORE allocating the
   *  arenas, setting info()==NumericalIssue and a descriptive lastErrorMessage()
   *  instead of attempting a doomed multi-hundred-GB allocation. Default 0 (off):
   *  behavior is unchanged unless you set a limit. Recommended when factoring
   *  matrices of unknown structure; pair with predictedFactorNonzeros() to size
   *  the limit to your machine's memory. */
  void setMaxFactorNonzeros(Index limit) { m_maxFactorNonzeros = limit; }
  Index maxFactorNonzeros() const { return m_maxFactorNonzeros; }

  /** Row/column equilibration (Ruiz). On by default; folded transparently into
   *  solve()/transpose()/determinant(). */
  void setEquilibration(bool on) { m_equilibrate = on; }
  bool equilibration() const { return m_equilibrate; }

  /** Maximum-transversal matching (MC64-style). On by default; permutes
   *  large-magnitude entries onto the diagonal before symbolic analysis. */
  void setMatching(bool on) { m_useMatching = on; }
  bool matching() const { return m_useMatching; }
  /** True if the last matching produced a full zero-free diagonal. */
  bool matchingIsPerfect() const { return m_matchingIsPerfect; }

  /** In-block pivoting strategy (PARDISO restricted/complete pivoting). Default
   *  Complete (row + column interchanges confined to each diagonal block). */
  void setPivoting(left_right_lu::Pivoting mode) { m_pivoting = mode; }
  left_right_lu::Pivoting pivoting() const { return m_pivoting; }
  /** SupernodalLU-compatible convenience: true -> Complete, false -> None. */
  void setDiagonalPivoting(bool on) {
    m_pivoting = on ? left_right_lu::Pivoting::Complete : left_right_lu::Pivoting::None;
  }
  bool diagonalPivoting() const { return m_pivoting != left_right_lu::Pivoting::None; }

  /** The computed scaling vectors (original numbering), valid after factorize().
   *  A~ = diag(rowScaling()) * A * diag(colScaling()). */
  const std::vector<RealScalar>& rowScaling() const { return m_rowScale; }
  const std::vector<RealScalar>& colScaling() const { return m_colScale; }

  /** Access the parallel-execution backend. Its concurrency() sets the number of
   *  dynamic-scheduler workers used by factorize(). */
  Executor& executor() { return m_executor; }
  const Executor& executor() const { return m_executor; }

  Index nnzL() const { return m_nnzL; }
  Index nnzU() const { return m_nnzU; }
  Index supernodeCount() const { return static_cast<Index>(m_supernodes.size()); }

  /** Predicted total scalars in the L and U factor arenas, computable right
   *  after analyzePattern() and BEFORE factorize() allocates them. Each L panel
   *  is (width+offDiag)*width (packed diagonal block + lower panel) and each U
   *  panel is width*offDiag; the sum, times sizeof(Scalar), is the factor memory
   *  factorize() is about to request. Use it to gauge memory or bail out on an
   *  infeasible fill without attempting the allocation. Valid after
   *  analyzePattern(). (nnzL()/nnzU() report the same structure post-factorize.) */
  Index predictedFactorNonzeros() const {
    Index total = 0;
    for (const Supernode& sn : m_supernodes) {
      const Index w = sn.width(), r = sn.offDiagonalRowCount;
      total += (w + r) * w + w * r;
    }
    return total;
  }

  /** Number of assembly-tree levels (a diagnostic; the dynamic scheduler does not
   *  use levels to schedule -- there are no level barriers). */
  Index levelCount() const { return static_cast<Index>(m_levelGroups.size()); }
  /** Supernodes in the widest level -- an upper bound on tree-level concurrency. */
  Index widestLevel() const {
    Index w = 0;
    for (const auto& g : m_levelGroups) w = std::max<Index>(w, static_cast<Index>(g.size()));
    return w;
  }

  /** \returns the determinant of the original matrix A. */
  Scalar determinant() const {
    Scalar det(1);
    for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
      const ConstStridedPanel diag = diagBlock(s);
      for (Index k = 0; k < diag.rows(); ++k) det *= diag(k, k);
    }
    // m_factorizationSign folds in the parity of the matching row permutation and
    // of every in-block row AND column pivot swap, so the sign of det(A) is right.
    return m_factorizationSign * det / m_scalingDeterminant;
  }

  /** \returns log|det(A)| (PARDISO IPARM(33) log-determinant), accumulated as a
   *  sum of logs so it stays finite for very large/small determinants where
   *  determinant() would overflow/underflow. Combine with determinantSign() for
   *  the full value. */
  RealScalar logAbsDeterminant() const {
    RealScalar acc(0);
    for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
      const ConstStridedPanel diag = diagBlock(s);
      for (Index k = 0; k < diag.rows(); ++k) acc += numext::log(numext::abs(diag(k, k)));
    }
    // divide out the (positive) equilibration scaling: log|det A| = log|det A~|
    //   - sum log(Dr) - sum log(Dc).
    for (StorageIndex i = 0; i < m_size; ++i)
      acc -= numext::log(m_rowScale[i]) + numext::log(m_colScale[i]);
    return acc;
  }

  /** \returns the sign of det(A) for real scalar types (+/-1, or 0 if a zero
   *  pivot was produced). Pairs with logAbsDeterminant(). */
  Scalar determinantSign() const {
    Scalar s = m_factorizationSign;
    for (StorageIndex sn = 0; sn < static_cast<StorageIndex>(m_supernodes.size()); ++sn) {
      const ConstStridedPanel diag = diagBlock(sn);
      for (Index k = 0; k < diag.rows(); ++k) {
        const Scalar d = diag(k, k);
        if (d == Scalar(0)) return Scalar(0);
        s *= d / numext::abs(d);
      }
    }
    return s;
  }

  // --- factor accessors (Eigen::SparseLU-compatible) ------------------------

  /** Expression of the unit-lower factor L; the supported operation is the
   *  in-place triangular solve in the solver's internal numbering. */
  struct LeftRightLUMatrixLReturnType {
    explicit LeftRightLUMatrixLReturnType(const LeftRightLU& solver) : m_solver(solver) {}
    Index rows() const { return m_solver.rows(); }
    Index cols() const { return m_solver.cols(); }
    template <typename Dest>
    void solveInPlace(MatrixBase<Dest>& x) const;
    const LeftRightLU& m_solver;
  };

  /** Expression of the upper factor U; the in-place solve folds in the
   *  per-supernode column permutation from complete pivoting. */
  struct LeftRightLUMatrixUReturnType {
    explicit LeftRightLUMatrixUReturnType(const LeftRightLU& solver) : m_solver(solver) {}
    Index rows() const { return m_solver.rows(); }
    Index cols() const { return m_solver.cols(); }
    template <typename Dest>
    void solveInPlace(MatrixBase<Dest>& x) const;
    const LeftRightLU& m_solver;
  };

  LeftRightLUMatrixLReturnType matrixL() const { return LeftRightLUMatrixLReturnType(*this); }
  LeftRightLUMatrixUReturnType matrixU() const { return LeftRightLUMatrixUReturnType(*this); }

  // --- transpose / adjoint solve views (Eigen::SparseLU-compatible) ---------

  LeftRightLUTransposeView<false, LeftRightLU> transpose() {
    LeftRightLUTransposeView<false, LeftRightLU> view;
    view.setSolver(this);
    view.setIsInitialized(this->m_isInitialized);
    return view;
  }

  LeftRightLUTransposeView<true, LeftRightLU> adjoint() {
    LeftRightLUTransposeView<true, LeftRightLU> view;
    view.setSolver(this);
    view.setIsInitialized(this->m_isInitialized);
    return view;
  }

  template <bool Conjugate, typename Rhs, typename Dest>
  void _solve_transposed_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

 private:
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrix;
  typedef supernodal_lu::Supernode<StorageIndex> Supernode;
  typedef supernodal_lu::RowBlock<StorageIndex> RowBlock;
  typedef supernodal_lu::UpdateSource<StorageIndex> UpdateSource;

  // Views into the contiguous panel storage (PaStiX coeftab/ucoeftab with
  // stride). Each supernode's L panel is (width + offDiag) x width column-major:
  // its top `width` rows are the packed-LU diagonal block, the rows below are the
  // L off-diagonal panel (shared leading dimension = width + offDiag). The U
  // off-diagonal panel is width x offDiag, stored contiguously.
  typedef Map<DenseMatrix, 0, OuterStride<>> StridedPanel;
  typedef Map<const DenseMatrix, 0, OuterStride<>> ConstStridedPanel;
  typedef Map<DenseMatrix> ContiguousPanel;
  typedef Map<const DenseMatrix> ConstContiguousPanel;

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
    const StorageIndex w = sn.width(), r = sn.offDiagonalRowCount;
    return StridedPanel(m_lStorage.data() + m_lOffset[s] + w, r, w, OuterStride<>(w + r));
  }
  ConstStridedPanel lowerPanel(StorageIndex s) const {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width(), r = sn.offDiagonalRowCount;
    return ConstStridedPanel(m_lStorage.data() + m_lOffset[s] + w, r, w, OuterStride<>(w + r));
  }
  ContiguousPanel upperPanel(StorageIndex s) {
    const Supernode& sn = m_supernodes[s];
    return ContiguousPanel(m_uStorage.data() + m_uOffset[s], sn.width(), sn.offDiagonalRowCount);
  }
  ConstContiguousPanel upperPanel(StorageIndex s) const {
    const Supernode& sn = m_supernodes[s];
    return ConstContiguousPanel(m_uStorage.data() + m_uOffset[s], sn.width(), sn.offDiagonalRowCount);
  }

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
    m_refinementMethod = left_right_lu::Refinement::BiCGStab;
    m_refineOnlyIfPerturbed = true;
    m_lastRefinementIterations = 0;
    m_relaxedSize = 4;
    m_maxAmalgamationZeroRows = 4;
    m_amalgamationFillFraction = 0.3;
    m_maxBlockSize = 128;
    m_maxFactorNonzeros = 0;  // fail-fast fill guard off by default
    m_equilibrate = true;
    m_useMatching = true;
    m_pivoting = left_right_lu::Pivoting::Complete;
    m_matchingIsPerfect = true;
    m_matchSign = 1;
    m_factorizationSign = Scalar(1);
    m_scalingDeterminant = RealScalar(1);
    m_nnzL = 0;
    m_nnzU = 0;
  }

  // analysis helpers (shared design with SupernodalLU)
  void buildSymmetricAdjacency(const MatrixType& matrix, std::vector<std::vector<StorageIndex>>& adjacency) const;
  void computeEliminationTree(const std::vector<std::vector<StorageIndex>>& adjacency,
                              std::vector<StorageIndex>& parent) const;
  void computePostorder(const std::vector<StorageIndex>& parent, std::vector<StorageIndex>& postorder) const;
  void computeSupernodePartition(const std::vector<std::vector<StorageIndex>>& adjacency,
                                 const std::vector<StorageIndex>& parent,
                                 std::vector<std::vector<StorageIndex>>& supernodeOffDiagRows);
  void buildRowBlocksAndUpdateSources(const std::vector<std::vector<StorageIndex>>& supernodeOffDiagRows);
  void computeSupernodeLevels();
  // build the reverse assembly-DAG edges: m_consumers[s] = supernodes that s
  // updates (targets t with s in m_updateSources[t]). Drives the right-looking
  // readiness push in the dynamic scheduler.
  void buildConsumerLists();

  // Panel position of off-diagonal internal row `r` within supernode `s`.
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

  // numeric: subtract the Schur contribution of a source supernode (left-looking
  // gather). Identical block math to SupernodalLU::applyUpdate.
  void applyUpdate(StorageIndex source, StorageIndex target, StorageIndex facingBlock);

  // factor one supernode: gather all its sources' updates (left-looking), do the
  // pivoted LU of its diagonal block, then solve its off-diagonal panels. Writes
  // only this supernode's own panels.
  void factorizeSupernode(StorageIndex s, const RealScalar& staticPivot, Index& replacedCount,
                          bool& singular, int& sign);

  // Unblocked LU of a dense diagonal block with the selected in-block pivoting
  // (none / row-only partial / row+col complete) and static-pivot fallback. On
  // return, rowPerm[k] / colPerm[k] give the local block row / column now sitting
  // at position k (left empty when the corresponding permutation is the identity);
  // sign carries the combined parity of all row AND column swaps. The block is
  // <= maxBlockSize wide, so an unblocked BLAS-2 factorization is cheap and lets
  // complete pivoting scan the whole trailing block.
  void factorizeDiagonalBlock(StridedPanel diag, const RealScalar& staticPivot, Index& replacedCount,
                              bool& singular, std::vector<StorageIndex>& rowPerm,
                              std::vector<StorageIndex>& colPerm, int& sign) const;

  void computeEquilibration(const MatrixType& matrix);

  // solve helpers (shared design with SupernodalLU, extended for the column
  // permutation Q_s produced by complete pivoting).
  void solveTriangular(const DenseMatrix& rhs, DenseMatrix& solution) const;
  template <typename ApplyA>
  void recordSolveStatus(const DenseMatrix& rhs, const DenseMatrix& solution, ApplyA applyA) const;
  template <typename Dest>
  void applyInverseL(Dest& y) const;
  template <typename Dest>
  void applyInverseU(Dest& y) const;
  template <bool Conjugate, typename Dest>
  void applyInverseLTransposed(Dest& y) const;
  template <bool Conjugate, typename Dest>
  void applyInverseUTransposed(Dest& y) const;
  template <bool Conjugate>
  void solveTriangularTransposed(const DenseMatrix& rhs, DenseMatrix& solution) const;
  template <typename ApplyA, typename ApplyPrec>
  void refineSolution(const DenseMatrix& rhs, DenseMatrix& solution, ApplyA applyA,
                      ApplyPrec applyPrec) const;
  template <typename ApplyA, typename ApplyPrec>
  Index bicgstabColumn(ApplyA applyA, ApplyPrec applyPrec, const DenseMatrix& b,
                       DenseMatrix& x) const;

  // state ---------------------------------------------------------------------
  StorageIndex m_size;
  OrderingType m_orderingFunctor;

  std::vector<StorageIndex> m_toInternal;   // original column -> internal index
  std::vector<StorageIndex> m_toOriginal;
  PermutationType m_permutation;            // public form of m_toInternal

  std::vector<StorageIndex> m_matchRow;     // matching: col j -> original row on diagonal
  std::vector<StorageIndex> m_matchRowInv;
  std::vector<StorageIndex> m_rowToInternal;  // original row -> internal index
  PermutationType m_rowPermutation;
  bool m_matchingIsPerfect;
  int m_matchSign;

  // Per-supernode in-block pivot permutations (internal local order). *Row*
  // permutation P_s from partial/complete pivoting; *column* permutation Q_s from
  // complete pivoting. m_rowPivot[s][k] / m_colPivot[s][k] = local block row /
  // column at position k after pivoting; empty => identity.
  std::vector<std::vector<StorageIndex>> m_rowPivot;
  std::vector<std::vector<StorageIndex>> m_colPivot;
  Scalar m_factorizationSign;  // sign(matching) * prod sign(in-block row & col pivots)

  std::vector<Supernode> m_supernodes;
  std::vector<RowBlock> m_rowBlocks;
  std::vector<StorageIndex> m_supernodeOfColumn;
  std::vector<std::vector<UpdateSource>> m_updateSources;  // per target: its sources
  std::vector<std::vector<StorageIndex>> m_consumers;      // per source: its targets
  std::vector<std::vector<StorageIndex>> m_levelGroups;    // diagnostics only

  Executor m_executor;

  // contiguous factor storage (PaStiX coeftab/ucoeftab).
  std::vector<Scalar> m_lStorage;
  std::vector<Scalar> m_uStorage;
  std::vector<std::size_t> m_lOffset;
  std::vector<std::size_t> m_uOffset;

  MatrixType m_originalMatrix;  // kept for residual formation in refinement

  std::vector<RealScalar> m_rowScale;
  std::vector<RealScalar> m_colScale;
  RealScalar m_scalingDeterminant;

  bool m_analysisDone;
  bool m_factorized;
  mutable ComputationInfo m_info;
  mutable std::string m_lastError;
  RealScalar m_solveFailureThreshold;
  mutable RealScalar m_lastSolveRelativeResidual;
  RealScalar m_staticPivotThreshold;
  bool m_thresholdIsAuto;
  Index m_replacedPivots;
  Index m_maxRefinementIterations;
  RealScalar m_refinementTolerance;
  left_right_lu::Refinement m_refinementMethod;
  bool m_refineOnlyIfPerturbed;
  mutable Index m_lastRefinementIterations;
  Index m_relaxedSize;
  Index m_maxAmalgamationZeroRows;
  double m_amalgamationFillFraction;
  Index m_maxBlockSize;
  Index m_maxFactorNonzeros;  // fail-fast fill guard (0 = off)
  bool m_equilibrate;
  bool m_useMatching;
  left_right_lu::Pivoting m_pivoting;
  Index m_nnzL;
  Index m_nnzU;
};

// ===========================================================================
//  Analysis (symbolic) phase  -- shared design with SupernodalLU
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::buildSymmetricAdjacency(
    const MatrixType& matrix, std::vector<std::vector<StorageIndex>>& adjacency) const {
  const StorageIndex n = m_size;
  adjacency.assign(n, std::vector<StorageIndex>());
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
void LeftRightLU<MatrixType, OrderingType, Executor>::computeEliminationTree(
    const std::vector<std::vector<StorageIndex>>& adjacency, std::vector<StorageIndex>& parent) const {
  const StorageIndex n = m_size;
  parent.assign(n, StorageIndex(-1));
  std::vector<StorageIndex> ancestor(n, StorageIndex(-1));
  for (StorageIndex j = 0; j < n; ++j) {
    for (StorageIndex neighbor : adjacency[j]) {
      if (neighbor >= j) continue;
      StorageIndex r = neighbor;
      while (ancestor[r] != StorageIndex(-1) && ancestor[r] != j) {
        StorageIndex next = ancestor[r];
        ancestor[r] = j;
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
void LeftRightLU<MatrixType, OrderingType, Executor>::computePostorder(
    const std::vector<StorageIndex>& parent, std::vector<StorageIndex>& postorder) const {
  const StorageIndex n = m_size;
  std::vector<StorageIndex> childHead(n, StorageIndex(-1));
  std::vector<StorageIndex> childNext(n, StorageIndex(-1));
  for (StorageIndex j = n - 1; j >= 0; --j) {
    if (parent[j] != StorageIndex(-1)) {
      childNext[j] = childHead[parent[j]];
      childHead[parent[j]] = j;
    }
    if (j == 0) break;
  }
  postorder.clear();
  postorder.reserve(n);
  std::vector<StorageIndex> stack;
  std::vector<StorageIndex> nextChild = childHead;
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
void LeftRightLU<MatrixType, OrderingType, Executor>::computeSupernodePartition(
    const std::vector<std::vector<StorageIndex>>& adjacency, const std::vector<StorageIndex>& parent,
    std::vector<std::vector<StorageIndex>>& supernodeOffDiagRows) {
  const StorageIndex n = m_size;
  m_supernodes.clear();
  m_supernodeOfColumn.assign(n, 0);
  supernodeOffDiagRows.clear();
  if (n == 0) return;

  std::vector<std::vector<StorageIndex>> children(n);
  for (StorageIndex j = 0; j < n; ++j)
    if (parent[j] != StorageIndex(-1)) children[parent[j]].push_back(j);

  // Per-column symbolic fill via Liu's children-merge (sorted set-unions), with
  // each column's list freed the moment its elimination-tree parent consumes it.
  std::vector<std::vector<StorageIndex>> columnStructure(n);
  std::vector<StorageIndex> scratch, scratch2;

  auto rowsBeyond = [](const std::vector<StorageIndex>& structure, StorageIndex col) -> StorageIndex {
    return static_cast<StorageIndex>(structure.end() - std::upper_bound(structure.begin(), structure.end(), col));
  };
  auto closeSupernode = [&](StorageIndex lastColumn, const std::vector<StorageIndex>& structure) {
    auto tailBegin = std::upper_bound(structure.begin(), structure.end(), lastColumn);
    supernodeOffDiagRows.emplace_back(tailBegin, structure.end());
  };

  Supernode s0;
  s0.firstColumn = 0;
  s0.lastColumn = 0;
  m_supernodes.push_back(s0);
  StorageIndex currentStart = 0;

  for (StorageIndex j = 0; j < n; ++j) {
    const auto& adjJ = adjacency[j];
    auto adjBeyond = std::upper_bound(adjJ.begin(), adjJ.end(), j);
    scratch.assign(adjBeyond, adjJ.end());
    for (StorageIndex c : children[j]) {
      const std::vector<StorageIndex>& cs = columnStructure[c];
      auto childBeyond = std::upper_bound(cs.begin(), cs.end(), j);
      if (childBeyond == cs.end()) continue;
      scratch2.clear();
      scratch2.reserve(scratch.size() + static_cast<std::size_t>(cs.end() - childBeyond));
      std::set_union(scratch.begin(), scratch.end(), childBeyond, cs.end(), std::back_inserter(scratch2));
      scratch.swap(scratch2);
    }
    columnStructure[j].clear();
    columnStructure[j].reserve(scratch.size() + 1);
    columnStructure[j].push_back(j);
    columnStructure[j].insert(columnStructure[j].end(), scratch.begin(), scratch.end());

    if (j > 0) {
      const std::vector<StorageIndex>& structPrev = columnStructure[j - 1];
      const std::vector<StorageIndex>& structJ = columnStructure[j];
      bool start;
      if (parent[j - 1] != j) {
        start = true;  // mandatory structural boundary
      } else if (children[j].size() == 1) {
        start = false;  // fundamental supernode: no extra fill
      } else {
        const StorageIndex childWidth = j - currentStart;
        const StorageIndex existingRows = rowsBeyond(structPrev, j);
        const StorageIndex deltaRows = rowsBeyond(structJ, j) - existingRows;
        const bool merge =
            (static_cast<Index>(childWidth) < m_relaxedSize) ||
            (static_cast<Index>(deltaRows) <= m_maxAmalgamationZeroRows) ||
            (static_cast<double>(deltaRows) <= m_amalgamationFillFraction * static_cast<double>(existingRows));
        start = !merge;
      }
      if (m_maxBlockSize > 0 && static_cast<Index>(j - currentStart) >= m_maxBlockSize) start = true;

      if (start) {
        closeSupernode(static_cast<StorageIndex>(j - 1), structPrev);
        if (parent[j - 1] == StorageIndex(-1)) std::vector<StorageIndex>().swap(columnStructure[j - 1]);
        Supernode s;
        s.firstColumn = j;
        s.lastColumn = j;
        m_supernodes.push_back(s);
        currentStart = j;
      } else {
        m_supernodes.back().lastColumn = j;
      }
    }
    m_supernodeOfColumn[j] = static_cast<StorageIndex>(m_supernodes.size() - 1);
    for (StorageIndex c : children[j]) std::vector<StorageIndex>().swap(columnStructure[c]);
  }
  closeSupernode(static_cast<StorageIndex>(n - 1), columnStructure[n - 1]);
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::buildRowBlocksAndUpdateSources(
    const std::vector<std::vector<StorageIndex>>& supernodeOffDiagRows) {
  m_rowBlocks.clear();
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());

  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    Supernode& sn = m_supernodes[s];
    const std::vector<StorageIndex>& structure = supernodeOffDiagRows[s];
    sn.firstRowBlock = static_cast<StorageIndex>(m_rowBlocks.size());
    sn.rowBlockCount = 0;
    sn.offDiagonalRowCount = 0;

    StorageIndex offset = 0;
    bool inBlock = false;
    RowBlock current;
    for (StorageIndex r : structure) {
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

  m_updateSources.assign(supernodeNbr, std::vector<UpdateSource>());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    StorageIndex previousFacing = StorageIndex(-1);
    for (StorageIndex b = 0; b < sn.rowBlockCount; ++b) {
      const StorageIndex blockIndex = sn.firstRowBlock + b;
      const RowBlock& block = m_rowBlocks[blockIndex];
      if (block.facingSupernode != previousFacing) {
        UpdateSource src;
        src.sourceSupernode = s;
        src.facingRowBlock = blockIndex;
        m_updateSources[block.facingSupernode].push_back(src);
        previousFacing = block.facingSupernode;
      }
    }
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::buildConsumerLists() {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_consumers.assign(supernodeNbr, std::vector<StorageIndex>());
  for (StorageIndex t = 0; t < supernodeNbr; ++t)
    for (const UpdateSource& src : m_updateSources[t])
      m_consumers[src.sourceSupernode].push_back(t);
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::computeSupernodeLevels() {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  std::vector<StorageIndex> level(supernodeNbr, 0);
  StorageIndex maxLevel = 0;
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
void LeftRightLU<MatrixType, OrderingType, Executor>::analyzePattern(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "LeftRightLU requires a square matrix");
  m_size = static_cast<StorageIndex>(matrix.rows());
  const StorageIndex n = m_size;

  // 0) maximum-transversal matching (MC64-style): place large entries on the
  //    diagonal. All symbolic analysis runs on the matched matrix.
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

  // 1) fill-reducing ordering of the pattern of B + B^T.
  PermutationType orderingPerm;
  m_orderingFunctor(B, orderingPerm);
  m_toInternal.resize(n);
  if (orderingPerm.size() == 0) {
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = i;
  } else {
    // m_toInternal[i] must be the INTERNAL (elimination) index of original column
    // i -- i.e. the fill-reducing "new index of i" (separators eliminated last).
    // Eigen's ordering functors (AMDOrdering/MetisOrdering) return that mapping
    // as the INVERSE in indices(): orderingPerm.indices()(k) is the original
    // column placed at new position k. So we must invert it here, not copy it
    // directly. Getting this backwards eliminates the top separators FIRST and
    // inflates fill enormously on strongly directional matrices (e.g. 3D FEM:
    // 250-300x more fill), while being nearly invisible on near-symmetric
    // orderings -- which is exactly why the bug hid until the laoss matrices.
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[orderingPerm.indices()(i)] = i;
  }

  // 2) elimination tree of the ordered symmetric pattern.
  std::vector<std::vector<StorageIndex>> adjacency;
  buildSymmetricAdjacency(B, adjacency);
  std::vector<StorageIndex> parent;
  computeEliminationTree(adjacency, parent);

  // 3) postorder + fold into the internal numbering (contiguous supernodes).
  std::vector<StorageIndex> postorder;
  computePostorder(parent, postorder);
  std::vector<StorageIndex> relabel(n);
  for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
  for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = relabel[m_toInternal[i]];

  // 4) recompute adjacency + tree in the final numbering.
  buildSymmetricAdjacency(B, adjacency);
  computeEliminationTree(adjacency, parent);

  // 5) symbolic factorization + supernode partition (streaming).
  std::vector<std::vector<StorageIndex>> supernodeOffDiagRows;
  computeSupernodePartition(adjacency, parent, supernodeOffDiagRows);

  // 6) block structure + update-source lists.
  buildRowBlocksAndUpdateSources(supernodeOffDiagRows);

  // 7) reverse assembly-DAG edges (consumer lists) for the dynamic scheduler,
  //    and tree levels for diagnostics.
  buildConsumerLists();
  computeSupernodeLevels();

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
void LeftRightLU<MatrixType, OrderingType, Executor>::applyUpdate(StorageIndex source, StorageIndex target,
                                                                  StorageIndex firstFacingBlock) {
  const Supernode& src = m_supernodes[source];
  const StorageIndex wSrc = src.width();
  const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
  const StorageIndex targetFirstColumn = m_supernodes[target].firstColumn;

  const StridedPanel srcLower = lowerPanel(source);
  const ContiguousPanel srcUpper = upperPanel(source);
  StridedPanel targetDiag = diagBlock(target);
  StridedPanel targetLower = lowerPanel(target);
  ContiguousPanel targetUpper = upperPanel(target);

  StorageIndex lastFacing = firstFacingBlock;
  while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;

  for (StorageIndex cb = firstFacingBlock; cb < lastFacing; ++cb) {
    const RowBlock& colBlock = m_rowBlocks[cb];
    const StorageIndex columnCount = colBlock.height();
    const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;
    const auto facingUpper = srcUpper.block(0, colBlock.panelOffset, wSrc, columnCount);
    const auto facingLower = srcLower.block(colBlock.panelOffset, 0, columnCount, wSrc);

    for (StorageIndex rb = firstFacingBlock; rb < lastBlock; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex rowCount = rowBlock.height();
      const auto lower = srcLower.block(rowBlock.panelOffset, 0, rowCount, wSrc);
      if (rb < lastFacing) {
        const StorageIndex destRow = rowBlock.firstRow - targetFirstColumn;
        targetDiag.block(destRow, targetColStart, rowCount, columnCount).noalias() -= lower * facingUpper;
      } else {
        const StorageIndex destRow = rowPanelPosition(target, rowBlock.firstRow);
        targetLower.block(destRow, targetColStart, rowCount, columnCount).noalias() -= lower * facingUpper;
      }
    }

    for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex rowCount = rowBlock.height();
      const StorageIndex destCol = rowPanelPosition(target, rowBlock.firstRow);
      const auto upper = srcUpper.block(0, rowBlock.panelOffset, wSrc, rowCount);
      targetUpper.block(targetColStart, destCol, columnCount, rowCount).noalias() -= facingLower * upper;
    }
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::computeEquilibration(const MatrixType& matrix) {
  const StorageIndex n = m_size;
  m_rowScale.assign(n, RealScalar(1));
  m_colScale.assign(n, RealScalar(1));
  if (!m_equilibrate) return;

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
    RealScalar deviation(0);
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
void LeftRightLU<MatrixType, OrderingType, Executor>::factorizeDiagonalBlock(
    StridedPanel diag, const RealScalar& staticPivot, Index& replacedCount, bool& singular,
    std::vector<StorageIndex>& rowPerm, std::vector<StorageIndex>& colPerm, int& sign) const {
  const StorageIndex w = static_cast<StorageIndex>(diag.rows());
  const bool rowPivot = (m_pivoting == left_right_lu::Pivoting::Partial ||
                         m_pivoting == left_right_lu::Pivoting::Complete);
  const bool colPivot = (m_pivoting == left_right_lu::Pivoting::Complete);

  sign = 1;
  rowPerm.clear();
  colPerm.clear();
  std::vector<StorageIndex> rperm, cperm;
  bool anyRowSwap = false, anyColSwap = false;
  if (rowPivot) {
    rperm.resize(static_cast<std::size_t>(w));
    for (StorageIndex i = 0; i < w; ++i) rperm[i] = i;
  }
  if (colPivot) {
    cperm.resize(static_cast<std::size_t>(w));
    for (StorageIndex i = 0; i < w; ++i) cperm[i] = i;
  }

  // Unblocked right-looking LU with in-block pivoting. w <= maxBlockSize, so the
  // trailing rank-1 (BLAS-2) update is cheap and complete pivoting can scan the
  // whole remaining block.
  for (StorageIndex k = 0; k < w; ++k) {
    // choose the pivot within the trailing block [k,w) x [k,w).
    StorageIndex pr = k, pc = k;
    if (colPivot) {
      // complete pivoting: largest-magnitude entry of the whole trailing block.
      RealScalar best = numext::abs(diag(k, k));
      for (StorageIndex jj = k; jj < w; ++jj) {
        Index rel = 0;
        const RealScalar m = diag.col(jj).segment(k, w - k).cwiseAbs().maxCoeff(&rel);
        if (m > best) {
          best = m;
          pr = k + static_cast<StorageIndex>(rel);
          pc = jj;
        }
      }
    } else if (rowPivot) {
      // partial pivoting: largest-magnitude entry of column k, rows [k,w).
      Index rel = 0;
      diag.col(k).segment(k, w - k).cwiseAbs().maxCoeff(&rel);
      pr = k + static_cast<StorageIndex>(rel);
    }
    if (rowPivot && pr != k) {
      diag.row(k).swap(diag.row(pr));
      std::swap(rperm[static_cast<std::size_t>(k)], rperm[static_cast<std::size_t>(pr)]);
      sign = -sign;
      anyRowSwap = true;
    }
    if (colPivot && pc != k) {
      diag.col(k).swap(diag.col(pc));
      std::swap(cperm[static_cast<std::size_t>(k)], cperm[static_cast<std::size_t>(pc)]);
      sign = -sign;
      anyColSwap = true;
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
      diag.block(k + 1, k + 1, below, below).noalias() -=
          diag.col(k).segment(k + 1, below) * diag.row(k).segment(k + 1, below);
    }
  }

  if (anyRowSwap) rowPerm = std::move(rperm);
  if (anyColSwap) colPerm = std::move(cperm);
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::factorizeSupernode(StorageIndex s,
                                                                         const RealScalar& staticPivot,
                                                                         Index& replacedCount,
                                                                         bool& singular, int& sign) {
  // LEFT-LOOKING gather: pull contributions from all (already-finished) sources.
  for (const UpdateSource& source : m_updateSources[s])
    applyUpdate(source.sourceSupernode, s, source.facingRowBlock);

  StridedPanel diag = diagBlock(s);

  // pivoted LU of the diagonal block: P_s A11 Q_s = L11 U11.
  std::vector<StorageIndex>& rowPerm = m_rowPivot[s];
  std::vector<StorageIndex>& colPerm = m_colPivot[s];
  factorizeDiagonalBlock(diag, staticPivot, replacedCount, singular, rowPerm, colPerm, sign);
  if (singular) return;

  // off-diagonal panels. The Schur update L21 * U12 sent to consumers is
  // invariant under (P_s, Q_s) -- the local permutations cancel (see header) --
  // so applyUpdate and the off-diagonal structure are unchanged; we only need to
  // form L21 = A21 Q_s U11^{-1} and U12 = L11^{-1} P_s A12 consistently.
  if (m_supernodes[s].offDiagonalRowCount > 0) {
    const StorageIndex w = m_supernodes[s].width();
    ContiguousPanel upper = upperPanel(s);
    StridedPanel lower = lowerPanel(s);

    // U12: apply the row permutation P_s (its rows are the block's equations),
    // then the unit-lower L11 solve.
    if (!rowPerm.empty()) {
      DenseMatrix permuted(w, upper.cols());
      for (StorageIndex k = 0; k < w; ++k) permuted.row(k) = upper.row(rowPerm[k]);
      upper = permuted;
    }
    // L21: apply the column permutation Q_s (its columns are the block's
    // columns), then the upper U11 right-solve.
    if (!colPerm.empty()) {
      DenseMatrix permuted(lower.rows(), w);
      for (StorageIndex k = 0; k < w; ++k) permuted.col(k) = lower.col(colPerm[k]);
      lower = permuted;
    }
    diag.template triangularView<Upper>().template solveInPlace<OnTheRight>(lower);
    diag.template triangularView<UnitLower>().solveInPlace(upper);
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analysisDone && "analyzePattern must be called before factorize");
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_replacedPivots = 0;
  m_factorized = false;
  m_info = Success;

  // Fail-fast fill guard: abort BEFORE allocating the factor arenas if the
  // symbolic structure predicts a factor larger than the configured limit. This
  // turns an (unrecoverable) multi-hundred-GB bad_alloc / OOM kill into a clean
  // NumericalIssue with a diagnostic -- the symmetric-pattern fill of some
  // matrices (e.g. FEM systems without good separators) is orders of magnitude
  // larger than an unsymmetric solver would need. Off by default (limit 0).
  if (m_maxFactorNonzeros > 0) {
    const Index predicted = predictedFactorNonzeros();
    if (predicted > m_maxFactorNonzeros) {
      char buf[512];
      std::snprintf(buf, sizeof(buf),
                    "LeftRightLU: predicted factor size %lld scalars (~%.1f GB) exceeds the "
                    "configured limit of %lld (setMaxFactorNonzeros). This symmetric-pattern "
                    "fill is very large -- the matrix likely lacks good vertex separators; an "
                    "iterative (e.g. BiCGSTAB+ILUT) or unsymmetric (SparseLU/PARDISO) solver is "
                    "the right tool. Raise or clear the limit to force the factorization.",
                    static_cast<long long>(predicted),
                    static_cast<double>(predicted) * static_cast<double>(sizeof(Scalar)) / 1e9,
                    static_cast<long long>(m_maxFactorNonzeros));
      m_lastError = buf;
      m_info = NumericalIssue;
      return;  // nothing allocated; factors remain absent (isFactorized() stays false)
    }
  }

  m_originalMatrix = matrix;

  computeEquilibration(matrix);
  m_scalingDeterminant = RealScalar(1);
  for (StorageIndex i = 0; i < m_size; ++i) m_scalingDeterminant *= m_rowScale[i] * m_colScale[i];

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

  // 1) lay out and zero the two contiguous panel arenas.
  m_lOffset.assign(supernodeNbr, 0);
  m_uOffset.assign(supernodeNbr, 0);
  std::size_t lTotal = 0, uTotal = 0;
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const std::size_t w = static_cast<std::size_t>(sn.width());
    const std::size_t r = static_cast<std::size_t>(sn.offDiagonalRowCount);
    m_lOffset[s] = lTotal;
    m_uOffset[s] = uTotal;
    lTotal += (w + r) * w;
    uTotal += w * r;
  }
  m_lStorage.assign(lTotal, Scalar(0));
  m_uStorage.assign(uTotal, Scalar(0));
  m_rowPivot.assign(supernodeNbr, std::vector<StorageIndex>());
  m_colPivot.assign(supernodeNbr, std::vector<StorageIndex>());

  // 2) scatter the (permuted, scaled) values of A into the panels.
  for (StorageIndex j = 0; j < m_size; ++j) {
    const StorageIndex jj = m_toInternal[j];
    const StorageIndex columnSupernode = m_supernodeOfColumn[jj];
    const Supernode& cs = m_supernodes[columnSupernode];
    const std::size_t csFirst = static_cast<std::size_t>(cs.firstColumn);
    const std::size_t csWidth = static_cast<std::size_t>(cs.width());
    const std::size_t csStride = csWidth + static_cast<std::size_t>(cs.offDiagonalRowCount);
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex origRow = static_cast<StorageIndex>(it.index());
      const StorageIndex ii = m_rowToInternal[origRow];
      const Scalar value = it.value() * m_rowScale[origRow] * m_colScale[j];
      if (m_supernodeOfColumn[ii] == columnSupernode) {
        const std::size_t col = static_cast<std::size_t>(jj) - csFirst;
        const std::size_t row = static_cast<std::size_t>(ii) - csFirst;
        m_lStorage[m_lOffset[columnSupernode] + col * csStride + row] += value;
      } else if (ii > jj) {
        const std::size_t col = static_cast<std::size_t>(jj) - csFirst;
        const std::size_t pos = static_cast<std::size_t>(rowPanelPosition(columnSupernode, ii));
        m_lStorage[m_lOffset[columnSupernode] + col * csStride + csWidth + pos] += value;
      } else {
        const StorageIndex rowSupernode = m_supernodeOfColumn[ii];
        const Supernode& rs = m_supernodes[rowSupernode];
        const std::size_t rsWidth = static_cast<std::size_t>(rs.width());
        const std::size_t pos = static_cast<std::size_t>(rowPanelPosition(rowSupernode, jj));
        const std::size_t row = static_cast<std::size_t>(ii) - static_cast<std::size_t>(rs.firstColumn);
        m_uStorage[m_uOffset[rowSupernode] + pos * rsWidth + row] += value;
      }
    }
  }

  // 3) LEFT-RIGHT-LOOKING numeric factorization driven by a barrier-free dynamic
  //    scheduler. Each supernode holds an atomic count of unfinished update
  //    sources; a worker takes a ready (count 0) supernode, gathers + factors it,
  //    then decrements each consumer's counter (right-looking push), enqueuing
  //    any that hit zero. Supernode ids are in postorder, so every update source
  //    of s has a smaller id -- the serial path can simply sweep id order.
  std::vector<Index> replacedPerSupernode(supernodeNbr, 0);
  std::vector<char> singularPerSupernode(supernodeNbr, 0);
  std::vector<int> signPerSupernode(supernodeNbr, 1);
  bool singular = false;
  const int lanes = m_executor.concurrency();

  if (lanes <= 1 || supernodeNbr <= 1) {
    // Serial left-looking sweep in id (topological) order.
    for (StorageIndex s = 0; s < supernodeNbr; ++s) {
      bool localSingular = false;
      factorizeSupernode(s, staticPivot, replacedPerSupernode[s], localSingular, signPerSupernode[s]);
      if (localSingular) {
        singular = true;
        break;
      }
    }
  } else {
    // Parallel dynamic DAG scheduler. remaining[s] = number of unfinished sources.
    std::vector<std::atomic<int>> remaining(static_cast<std::size_t>(supernodeNbr));
    for (StorageIndex s = 0; s < supernodeNbr; ++s)
      remaining[s].store(static_cast<int>(m_updateSources[s].size()), std::memory_order_relaxed);

    std::mutex qmutex;
    std::condition_variable qcv;
    std::vector<StorageIndex> readyStack;  // LIFO: depth-first subtree affinity
    readyStack.reserve(static_cast<std::size_t>(supernodeNbr));
    for (StorageIndex s = 0; s < supernodeNbr; ++s)
      if (remaining[s].load(std::memory_order_relaxed) == 0) readyStack.push_back(s);
    std::atomic<StorageIndex> completed{0};
    std::atomic<bool> failed{false};

    auto worker = [&](Index) {
      std::vector<StorageIndex> newReady;
      for (;;) {
        StorageIndex s = StorageIndex(-1);
        {
          std::unique_lock<std::mutex> lk(qmutex);
          qcv.wait(lk, [&] {
            return !readyStack.empty() || completed.load(std::memory_order_acquire) == supernodeNbr ||
                   failed.load(std::memory_order_acquire);
          });
          if (failed.load(std::memory_order_acquire) ||
              completed.load(std::memory_order_acquire) == supernodeNbr)
            return;
          if (readyStack.empty()) continue;
          s = readyStack.back();
          readyStack.pop_back();
        }

        bool localSingular = false;
        factorizeSupernode(s, staticPivot, replacedPerSupernode[s], localSingular, signPerSupernode[s]);
        if (localSingular) {
          singularPerSupernode[s] = 1;
          failed.store(true, std::memory_order_release);
          qcv.notify_all();
          return;
        }

        // RIGHT-LOOKING push: notify consumers, collect the newly-ready ones.
        newReady.clear();
        for (StorageIndex t : m_consumers[s])
          if (remaining[t].fetch_sub(1, std::memory_order_acq_rel) == 1) newReady.push_back(t);
        const StorageIndex done = completed.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (!newReady.empty()) {
          std::lock_guard<std::mutex> lk(qmutex);
          for (StorageIndex t : newReady) readyStack.push_back(t);
        }
        if (!newReady.empty() || done == supernodeNbr) qcv.notify_all();
      }
    };
    m_executor.parallelFor(Index(0), static_cast<Index>(lanes), worker);

    if (failed.load(std::memory_order_acquire)) singular = true;
  }

  m_replacedPivots = 0;
  for (Index r : replacedPerSupernode) m_replacedPivots += r;
  int totalSign = m_matchSign;
  for (int sg : signPerSupernode) totalSign *= sg;
  m_factorizationSign = Scalar(totalSign);
  if (singular) {
    m_info = NumericalIssue;
    m_lastError = "LeftRightLU: zero pivot encountered (matrix is singular).";
    return;
  }

  // 4) nonzero counts.
  m_nnzL = 0;
  m_nnzU = 0;
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const StorageIndex w = m_supernodes[s].width();
    const StorageIndex r = m_supernodes[s].offDiagonalRowCount;
    m_nnzL += Index(w) * Index(w + 1) / 2 + Index(r) * Index(w);
    m_nnzU += Index(w) * Index(w + 1) / 2 + Index(w) * Index(r);
  }

  m_factorized = true;
  m_isInitialized = true;
}

// ===========================================================================
//  Solve phase
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Rhs, typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::_solve_impl(const MatrixBase<Rhs>& b,
                                                                  MatrixBase<Dest>& x) const {
  eigen_assert(m_factorized && "the matrix must be factorized first");
  const Index nrhs = b.cols();
  const DenseMatrix rhs = b;
  DenseMatrix solution(m_size, nrhs);
  solveTriangular(rhs, solution);

  auto applyA = [this](const DenseMatrix& in, DenseMatrix& out) { out.noalias() = m_originalMatrix * in; };
  // PARDISO IPARM(8)=0: refine only if the factorization perturbed a pivot.
  if (!(m_refineOnlyIfPerturbed && m_replacedPivots == 0))
    refineSolution(rhs, solution, applyA,
                   [this](const DenseMatrix& in, DenseMatrix& out) { solveTriangular(in, out); });
  else
    m_lastRefinementIterations = 0;

  recordSolveStatus(rhs, solution, applyA);
  x = solution;
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseL(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    // head := P_s head  (row permutation from pivoting: newhead[k] = head[piv[k]]).
    const std::vector<StorageIndex>& piv = m_rowPivot[s];
    if (!piv.empty()) {
      DenseMatrix tmp = head;
      for (StorageIndex k = 0; k < w; ++k) head.row(k) = tmp.row(piv[k]);
    }
    diagBlock(s).template triangularView<UnitLower>().solveInPlace(head);
    const ConstStridedPanel lower = lowerPanel(s);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      y.middleRows(block.firstRow, hb).noalias() -= lower.middleRows(block.panelOffset, hb) * head;
    }
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseU(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = supernodeNbr - 1; s >= 0; --s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    const ConstContiguousPanel upper = upperPanel(s);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      head.noalias() -= upper.middleCols(block.panelOffset, hb) * y.middleRows(block.firstRow, hb);
    }
    diagBlock(s).template triangularView<Upper>().solveInPlace(head);
    // x_block = Q_s z  (column permutation from complete pivoting: the solved
    // vector z is in pivoted-column order, scatter it to the real unknowns:
    // x[colPerm[k]] = z[k]).
    const std::vector<StorageIndex>& cpiv = m_colPivot[s];
    if (!cpiv.empty()) {
      DenseMatrix tmp = head;
      for (StorageIndex k = 0; k < w; ++k) head.row(cpiv[k]) = tmp.row(k);
    }
    if (s == 0) break;
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::solveTriangular(const DenseMatrix& rhs,
                                                                      DenseMatrix& x) const {
  const StorageIndex n = m_size;
  const Index nrhs = rhs.cols();
  DenseMatrix y(n, nrhs);
  for (StorageIndex i = 0; i < n; ++i) y.row(m_rowToInternal[i]) = m_rowScale[i] * rhs.row(i);
  applyInverseL(y);
  applyInverseU(y);
  for (StorageIndex i = 0; i < n; ++i) x.row(i) = m_colScale[i] * y.row(m_toInternal[i]);
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA>
void LeftRightLU<MatrixType, OrderingType, Executor>::recordSolveStatus(const DenseMatrix& rhs,
                                                                        const DenseMatrix& solution,
                                                                        ApplyA applyA) const {
  const RealScalar rhsNorm = rhs.norm();
  DenseMatrix product(rhs.rows(), rhs.cols());
  applyA(solution, product);
  const RealScalar resNorm = (rhs - product).norm();
  const RealScalar relResid = (rhsNorm > RealScalar(0)) ? resNorm / rhsNorm : resNorm;
  m_lastSolveRelativeResidual = relResid;
  const bool usable = solution.allFinite() && (numext::isfinite)(relResid) &&
                      relResid <= m_solveFailureThreshold;
  if (usable) {
    m_info = Success;
  } else {
    m_info = NumericalIssue;
    m_lastError =
        "LeftRightLU: solve produced a large residual (see solveResidual()); the matrix is "
        "likely too ill-conditioned for this factorization.";
  }
}

// ===========================================================================
//  matrixL() / matrixU() factor accessors
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::LeftRightLUMatrixLReturnType::solveInPlace(
    MatrixBase<Dest>& x) const {
  m_solver.applyInverseL(x.derived());
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::LeftRightLUMatrixUReturnType::solveInPlace(
    MatrixBase<Dest>& x) const {
  m_solver.applyInverseU(x.derived());
}

// ===========================================================================
//  transpose() / adjoint() solve
// ===========================================================================

// Solve U^T y = y (Conjugate=false) or U^H y = y, in place, internal numbering.
// U^T is lower-triangular -> forward sweep. The column permutation Q_s becomes a
// leading row operation here (inverse of the trailing scatter in applyInverseU).
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseUTransposed(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    // undo x_block = Q_s z: recover z from the transposed rhs (z[k] = head[colPerm[k]]).
    const std::vector<StorageIndex>& cpiv = m_colPivot[s];
    if (!cpiv.empty()) {
      DenseMatrix tmp = head;
      for (StorageIndex k = 0; k < w; ++k) head.row(k) = tmp.row(cpiv[k]);
    }
    if (Conjugate)
      diagBlock(s).template triangularView<Upper>().adjoint().solveInPlace(head);
    else
      diagBlock(s).template triangularView<Upper>().transpose().solveInPlace(head);
    const ConstContiguousPanel U = upperPanel(s);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      const auto panel = U.middleCols(block.panelOffset, hb);
      if (Conjugate)
        y.middleRows(block.firstRow, hb).noalias() -= panel.adjoint() * head;
      else
        y.middleRows(block.firstRow, hb).noalias() -= panel.transpose() * head;
    }
  }
}

// Solve L^T z = z (Conjugate=false) or L^H z = z, in place, internal numbering.
// L^T is unit-upper-triangular -> backward sweep. The row permutation P_s is
// applied (inverse) at the end (mirror of applyInverseL's leading gather).
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseLTransposed(Dest& y) const {
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  for (StorageIndex s = supernodeNbr - 1; s >= 0; --s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width();
    auto head = y.middleRows(sn.firstColumn, w);
    const ConstStridedPanel L = lowerPanel(s);
    for (StorageIndex b2 = 0; b2 < sn.rowBlockCount; ++b2) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b2];
      const StorageIndex hb = block.height();
      const auto panel = L.middleRows(block.panelOffset, hb);
      if (Conjugate)
        head.noalias() -= panel.adjoint() * y.middleRows(block.firstRow, hb);
      else
        head.noalias() -= panel.transpose() * y.middleRows(block.firstRow, hb);
    }
    if (Conjugate)
      diagBlock(s).template triangularView<UnitLower>().adjoint().solveInPlace(head);
    else
      diagBlock(s).template triangularView<UnitLower>().transpose().solveInPlace(head);
    // inverse row permutation: newhead[piv[k]] = head[k].
    const std::vector<StorageIndex>& piv = m_rowPivot[s];
    if (!piv.empty()) {
      DenseMatrix tmp = head;
      for (StorageIndex k = 0; k < w; ++k) head.row(piv[k]) = tmp.row(k);
    }
    if (s == 0) break;
  }
}

// A^T = Dc^{-1} A~^T Dr^{-1}, so solve A^T x = rhs as x = Dr * A~^{-T} * (Dc * rhs).
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate>
void LeftRightLU<MatrixType, OrderingType, Executor>::solveTriangularTransposed(const DenseMatrix& rhs,
                                                                                DenseMatrix& x) const {
  const StorageIndex n = m_size;
  const Index nrhs = rhs.cols();
  DenseMatrix y(n, nrhs);
  for (StorageIndex i = 0; i < n; ++i) y.row(m_toInternal[i]) = m_colScale[i] * rhs.row(i);
  applyInverseUTransposed<Conjugate>(y);
  applyInverseLTransposed<Conjugate>(y);
  for (StorageIndex i = 0; i < n; ++i) x.row(i) = m_rowScale[i] * y.row(m_rowToInternal[i]);
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Rhs, typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::_solve_transposed_impl(const MatrixBase<Rhs>& b,
                                                                             MatrixBase<Dest>& x) const {
  eigen_assert(m_factorized && "the matrix must be factorized first");
  const Index nrhs = b.cols();
  const DenseMatrix rhs = b;
  DenseMatrix solution(m_size, nrhs);
  solveTriangularTransposed<Conjugate>(rhs, solution);

  auto applyA = [this](const DenseMatrix& in, DenseMatrix& out) {
    if (Conjugate)
      out.noalias() = m_originalMatrix.adjoint() * in;
    else
      out.noalias() = m_originalMatrix.transpose() * in;
  };
  if (!(m_refineOnlyIfPerturbed && m_replacedPivots == 0))
    refineSolution(rhs, solution, applyA, [this](const DenseMatrix& in, DenseMatrix& out) {
      this->template solveTriangularTransposed<Conjugate>(in, out);
    });
  else
    m_lastRefinementIterations = 0;

  recordSolveStatus(rhs, solution, applyA);
  x = solution;
}

// ===========================================================================
//  Iterative / Krylov refinement
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA, typename ApplyPrec>
void LeftRightLU<MatrixType, OrderingType, Executor>::refineSolution(const DenseMatrix& rhs,
                                                                     DenseMatrix& solution,
                                                                     ApplyA applyA,
                                                                     ApplyPrec applyPrec) const {
  m_lastRefinementIterations = 0;
  const RealScalar rhsNorm = rhs.norm();
  if (m_maxRefinementIterations <= 0 || rhsNorm == RealScalar(0) ||
      m_refinementMethod == left_right_lu::Refinement::None)
    return;

  const Index n = rhs.rows();
  const Index nrhs = rhs.cols();

  if (m_refinementMethod == left_right_lu::Refinement::IterativeRefinement) {
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

  DenseMatrix bcol(n, 1), xcol(n, 1);
  for (Index c = 0; c < nrhs; ++c) {
    bcol.col(0) = rhs.col(c);
    xcol.col(0) = solution.col(c);
    const Index iters = bicgstabColumn(applyA, applyPrec, bcol, xcol);
    solution.col(c) = xcol.col(0);
    m_lastRefinementIterations = numext::maxi(m_lastRefinementIterations, iters);
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA, typename ApplyPrec>
Index LeftRightLU<MatrixType, OrderingType, Executor>::bicgstabColumn(ApplyA applyA, ApplyPrec applyPrec,
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

  const DenseMatrix rhat = r;
  Scalar rho(1), alpha(1), omega(1);
  DenseMatrix v = DenseMatrix::Zero(n, 1), p = DenseMatrix::Zero(n, 1);
  DenseMatrix y(n, 1), s(n, 1), z(n, 1), t(n, 1);

  Index iters = 0;
  for (Index it = 0; it < m_maxRefinementIterations; ++it) {
    const Scalar rhoNew = rhat.col(0).dot(r.col(0));
    if (numext::abs(rhoNew) == RealScalar(0)) break;
    if (it == 0) {
      p = r;
    } else {
      const Scalar beta = (rhoNew / rho) * (alpha / omega);
      p = r + beta * (p - omega * v);
    }
    applyPrec(p, y);
    applyA(y, v);
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
    applyPrec(s, z);
    applyA(z, t);
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
    if (numext::abs(omega) == RealScalar(0)) break;
  }
  x = best;
  return iters;
}

}  // namespace Eigen

#endif  // LEFT_RIGHT_LU_H
