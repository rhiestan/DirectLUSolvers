// LeftRightLU - a sparse direct LU solver for Eigen using the algorithmic
// design of PARDISO (see pardiso_algorithms.md), as a sibling of SupernodalLU.
//
// Accepts any square sparse matrix: general (unsymmetric) values AND a general
// (unsymmetric) nonzero pattern. The symbolic phase factors the pattern of
// B + B^T (B = the matched matrix), which is a valid superset of the structure
// of L and U because the factorization never interchanges rows or columns
// outside a diagonal block -- so an unsymmetric pattern needs no preprocessing
// from the caller. See "Unsymmetric nonzero patterns" on the class below for
// what it costs and how to keep that cost down.
//
// What LeftRightLU shares with SupernodalLU (and reuses verbatim from the
// supernodal_lu:: support/matching/executor headers):
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
//      separators never serialize a whole tree level. Per-worker LIFO deques
//      with work stealing give depth-first subtree affinity (PARDISO's
//      cooperative subtree ownership, minus the NUMA placement, which is out of
//      scope). This is the "two-level dynamic scheduler" idea reduced to what
//      shared memory needs.
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
//   analyzePattern : matching -> block triangular form -> per-block ordering ->
//                    elimination tree -> postorder -> supernode detection ->
//                    symbolic block factorization -> assembly-DAG consumer lists
//   factorize      : scatter values -> left-right-looking dynamic factorization
//   solve          : BTF block back-substitution, each block a forward (L) and
//                    backward (U) substitution + refinement
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef LEFT_RIGHT_LU_H
#define LEFT_RIGHT_LU_H

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iterator>
#include <limits>
#include <mutex>
#include <vector>

#include "LeftRightLUBlockTriangular.h"
#include "LeftRightLUConditionEstimate.h"
#include "LeftRightLUExtendedResidual.h"
#include "SupernodalLUSupport.h"
#include "SupernodalLUExecutor.h"
#include "SupernodalLUMatching.h"
#include "SupernodalLUMC64.h"

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

// Direction convention of the permutation a fill-reducing ordering functor
// returns. Eigen's own functors do NOT agree on it, and nothing in the type
// says which one you got:
//
//   AMDOrdering, MetisOrdering  ->  perm.indices()(k) = the ORIGINAL column
//                                   placed at new position k   (INVERSE map)
//   COLAMDOrdering              ->  perm.indices()(j) = the NEW position of
//                                   original column j          (DIRECT map)
//
// analyzePattern needs the direct map. Reading a permutation the wrong way
// round still yields a valid permutation and a machine-precision residual --
// it shows up only as fill, and then by orders of magnitude (measured: 4.6x on
// a 900-column upwind grid with COLAMD, 250-300x on 3D FEM systems with AMD).
// Nothing in the numbers themselves distinguishes the two directions, so this
// is a compile-time trait rather than something the solver tries to detect.
//
// Specialize it for a custom ordering functor that returns the direct map:
//
//   namespace Eigen { namespace left_right_lu {
//   template <> struct OrderingConvention<MyOrdering> {
//     static constexpr bool returnsInverse = false;
//   };
//   }}  // namespace Eigen::left_right_lu
template <typename OrderingType>
struct OrderingConvention {
  static constexpr bool returnsInverse = true;
};

template <typename StorageIndex>
struct OrderingConvention<COLAMDOrdering<StorageIndex>> {
  static constexpr bool returnsInverse = false;
};
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
 *        square matrices: left-right-looking numeric factorization driven by a
 *        barrier-free dynamic (assembly-DAG) scheduler, with in-block complete
 *        pivoting.
 *
 * \tparam MatrixType_   A column-major Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>.
 * \tparam OrderingType_ A fill-reducing ordering functor (default AMDOrdering).
 *                       A functor that returns the direct (rather than the
 *                       inverse) permutation needs an OrderingConvention
 *                       specialization; Eigen's own AMD/METIS/COLAMD/Natural
 *                       functors are all handled already.
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
 *
 * \section lrlu_unsym Unsymmetric nonzero patterns
 *
 * Both the values and the *pattern* of A may be unsymmetric; nothing has to be
 * done to A first.
 *
 * In particular, do NOT pad the pattern with explicit structural zeros to
 * symmetrize it. That is the documented workaround for SupernodalLU, and here it
 * is not merely redundant but actively harmful, because *when* the pattern is
 * symmetrized decides how much structure results. This solver symmetrizes AFTER
 * the matching row permutation, on `P*A`; a caller who symmetrizes up front does
 * it before, so the analysis then symmetrizes a second time and eliminates
 * `P*(A u A^T) u (A u A^T)*P^T` instead of `P*A u A^T*P^T`. Measured on gemat11
 * that is 102x the fill (51,728 -> 5,281,259 nnzL) and 546x the factorization
 * time, for the same answer. A pattern that is *already* symmetric is unaffected
 * either way -- padding only matters when there is something to pad.
 *
 * How it works: the symbolic phase runs on the pattern of `B + B^T`, where B is
 * A with the matching row permutation applied. Because the numeric phase never
 * moves a row or column outside its own diagonal block (that is the whole point
 * of the static-structure design), the fill pattern of `chol(B + B^T)` is a
 * valid superset of the structure of both L and U, and every entry of A lands
 * in a slot that already exists. The result is exact for any pattern -- an
 * unsymmetric pattern costs *fill*, never correctness. This is also what PARDISO
 * does for its unsymmetric matrix types (MTYPE=11/13).
 *
 * What it costs, and what to do about it:
 *
 *  - patternSymmetry() reports how much symmetrizing added: 1 means the pattern
 *    was already symmetric and nothing was added, 0 means no nonzero had a
 *    mirror and the symbolic pattern has twice A's off-diagonal entries. Low
 *    values are worth knowing about because the extra structure compounds
 *    through fill, not linearly. It measures `B`, the graph actually eliminated,
 *    so a symmetric-pattern A can still report below 1: matching permutes rows.
 *  - The default AMDOrdering is a reasonable starting point -- it minimizes
 *    degree in exactly the `A + A^T` graph this factorization eliminates -- but
 *    on an unsymmetric pattern it is worth trying the others, because the
 *    symmetrized graph AMD sees can be much denser than the one it was designed
 *    for. Measured on setfos_2 (n=3048, 238 nnz/row, symmetry 0.44), fill and
 *    factor time for the same system:
 *        AMD     3,933,570   201 ms
 *        METIS   1,609,832   127 ms
 *        COLAMD  2,360,714   116 ms
 *    Note COLAMD wins on *time* while losing on fill: it left 39 wide supernodes
 *    against METIS's 321 narrow ones, and the fatter dense blocks pay for the
 *    extra flops in BLAS-3 efficiency. Fill is the right first-order proxy for
 *    cost, but on a matrix this dense it is not the whole story -- measure.
 *  - On a strongly unsymmetric pattern with no good vertex separators, the fill
 *    can exceed what a partial-pivoting solver would need by orders of
 *    magnitude. setMaxFactorNonzeros() turns that into a clean, immediate error
 *    instead of an out-of-memory kill; predictedFactorNonzeros() prices it after
 *    analyzePattern() and before any factor memory is touched.
 *  - Unsymmetric patterns are also where structural singularity actually
 *    happens (an empty column, or one whose rows are all claimed elsewhere).
 *    matchingIsPerfect() reports it right after analyzePattern().
 *
 * \section lrlu_btf Block triangular form
 *
 * Many unsymmetric matrices -- circuit, chemical-process, economic and web
 * graphs especially -- are *reducible*: a symmetric permutation puts them in
 * block upper triangular form. Only the diagonal blocks then need factoring.
 * analyzePattern() finds that form (Dulmage-Mendelsohn: the maximum transversal
 * it already computes, followed by strongly connected components -- see
 * LeftRightLUBlockTriangular.h) and factors each diagonal block independently,
 * with its own fill-reducing ordering, elimination tree and supernodes. The
 * off-diagonal blocks are never eliminated: they are applied once each during a
 * block back-substitution, so they generate no fill and contribute exactly.
 *
 * On by default, and self-cancelling when it cannot help: an irreducible matrix
 * yields one block and takes exactly the path it would have taken anyway, for
 * the cost of one O(n + nnz) sweep -- 5-100x cheaper than the matching that
 * precedes it. A structurally symmetric pattern is always irreducible in this
 * sense (its SCCs are just connected components), so PDE and FEM systems pay
 * only that sweep. setBlockTriangularForm(false) disables it; btfBlockCount()
 * and largestBtfBlock() report what was found.
 *
 * The gain where it does apply is not proportional to the block sizes, because
 * fill is superlinear in them. Measured on this project's corpus: Simon/raefsky5
 * and Simon/raefsky6 reduce to blocks of size 1 -- fully triangular after
 * matching, so the "factorization" is exact and free -- and SNAP/wiki-RfA
 * (largest block 24.9% of n) and Mallya/lhr10c (34.3%) shrink by far more than
 * those fractions suggest.
 *
 * Two consequences worth knowing:
 *
 *  - Pivoting is confined to a diagonal block. Without BTF, partial pivoting on
 *    a column could legally take a row from an earlier block, which would
 *    destroy the structure; BTF forbids it, so the pivot pool is genuinely
 *    smaller. In practice this is a net stability gain -- far fewer operations
 *    means far less accumulated rounding, and the off-diagonal blocks are used
 *    exactly -- but it is a real constraint, not a free lunch.
 *  - Singularity is localized. A is singular exactly when some diagonal block
 *    is, which is a much more useful diagnostic than a column index.
 *  - Fill is confined to the diagonal blocks, but that is a structural bound,
 *    not a monotone guarantee: each block gets its own ordering, and a
 *    fill-reducing ordering is a heuristic, so a matrix that barely splits can
 *    come out marginally worse than it would under one global ordering. On this
 *    project's corpus the worst case is +1.3% (Bai/rw5151 under COLAMD, which
 *    splits into 6 blocks with the largest still 99.9% of n) against reductions
 *    of 100x and more where the matrix genuinely reduces.
 *
 * BTF needs the zero-free diagonal a maximum transversal provides, so it is
 * skipped when setMatching(false) removes it. matrixL()/matrixU() then expose
 * the BLOCK-DIAGONAL factors, whose product is not A -- determinant(),
 * logAbsDeterminant() and solve() account for the off-diagonal blocks, direct
 * use of the factor accessors does not.
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

  // --- condition estimation & error bounds ----------------------------------
  //
  // A residual says the answer solves a NEARBY system; it does not say how far
  // that answer is from the solution of the system you asked about. These are
  // what close the gap, and none of them runs unless you ask: solve() is
  // byte-for-byte the same work it was before, and the accessors below cost
  // what they cost only when called. See LeftRightLUConditionEstimate.h.

  /** 
eturns a Hager-Higham estimate of kappa_1(A) = ||A||_1 ||A^{-1}||_1,
    *  computed from the existing factors and cached until the next factorize().
    *
    *  Costs 4-5 triangular solves the FIRST time it is called after a
    *  factorization (measured across this project's corpus; 10 is the
    *  algorithm's ceiling) and nothing thereafter. It is a lower bound (see the header),
    *  and it describes the operator this factorization inverts -- which under
    *  static pivoting is a perturbed A, so read it together with
    *  replacedPivots(). Returns infinity if the factorization is singular. */
  RealScalar conditionEstimate() const;

  /** Triangular solves spent on the cached condition estimate, 0 if never asked. */
  Index conditionEstimateSolves() const { return m_conditionSolves; }

  /** 
eturns the Oettli-Prager componentwise relative backward error of  x
    *  as a solution of A x =  b -- EXACT, not an estimate, and O(nnz).
    *
    *  Near machine epsilon means the solver was backward stable on this system
    *  and no method could have done better in this precision; any inaccuracy
    *  left in x belongs to the matrix, not to the solver. */
  template <typename RhsT, typename SolT>
  RealScalar componentwiseBackwardError(const RhsT& b, const SolT& x) const {
    return left_right_lu::componentwiseBackwardError(m_originalMatrix, b, x);
  }

  /** 
eturns kappa * omega, clamped to 1: a first-order bound on
    *  ||x - x_exact|| / ||x_exact||. Meaningless once it approaches 1, which is
    *  exactly the case where it is telling you not to trust the answer. */
  template <typename RhsT, typename SolT>
  RealScalar estimatedForwardError(const RhsT& b, const SolT& x) const {
    return left_right_lu::estimateForwardError(conditionEstimate(),
                                               componentwiseBackwardError(b, x));
  }

  /** 
eturns how many decimal digits of  x the forward-error estimate
    *  supports, floored at 0 -- the "13 digits or 2 digits?" question a bare
    *  residual cannot answer. */
  template <typename RhsT, typename SolT>
  int estimatedCorrectDigits(const RhsT& b, const SolT& x) const {
    return digitsFromForwardError(estimatedForwardError(b, x));
  }

  /** Form refinement's residual r = b - A x in DOUBLE-DOUBLE arithmetic rather
    *  than working precision. OFF by default.
    *
    *  This is the knob that decides what refinement can achieve at all. With a
    *  working-precision residual, refinement drives the BACKWARD error to O(eps)
    *  and the forward error stalls at about kappa(A) * eps -- more steps do not
    *  help, because r is already dominated by the rounding in forming it. With
    *  an extended-precision residual, the FORWARD error goes to O(eps) too, as
    *  long as kappa(A) * eps stays comfortably below 1 (Skeel 1980; the basis of
    *  LAPACK's xGESVXX). Past that point nothing recovers the answer, and
    *  estimatedForwardError() is what says which regime you are in.
    *
    *  The theory is about stationary refinement, and that is where the work
    *  happens: Refinement::IterativeRefinement switches to a forward-error
    *  stopping rule (watch the correction, not the residual) when this is on,
    *  because the usual rule stops before applying a single correction -- the
    *  residual is ALREADY at machine precision after a backward-stable direct
    *  solve. Under BiCGStab the Krylov recurrence updates its residual from
    *  previous ones, which extended precision cannot reach, so the same
    *  stationary loop runs afterwards as a polish. Either way the flag means one
    *  thing: a forward error at machine precision rather than a backward one.
    *
    *  Cost is O(nnz) per refinement step against the correction solve's O(fill).
    *  It buys nothing when refinement does not run at all, which by default is
    *  whenever no static pivot was perturbed (see setRefineOnlyIfPerturbed).
    *
    *  \warning Compiling with -ffast-math or /fp:fast silently disables the
    *  effect: those flags let the compiler cancel the compensation terms
    *  algebraically. See LeftRightLUExtendedResidual.h. */
  void setExtendedPrecisionResidual(bool on) { m_extendedResidual = on; }
  bool extendedPrecisionResidual() const { return m_extendedResidual; }

  /** Compute the error bounds inside every solve() and let info() act on them.
    *  OFF by default, deliberately: it adds a condition estimate (once per
    *  factorization) and an O(nnz) backward error (once per solve) to a path
    *  whose cost is otherwise exactly the factor solve.
    *
    *  With it on, solve() fills lastBackwardError()/lastForwardError() and
    *  downgrades info() to NumericalIssue when the forward-error estimate
    *  reaches 1 -- i.e. when NO digit of the answer is supported -- even though
    *  the residual check passed. That combination is the signature of an
    *  ill-conditioned system: a backward-stable answer that is still wrong. */
  void setErrorBounds(bool on) { m_errorBounds = on; }
  bool errorBounds() const { return m_errorBounds; }

  /** Backward/forward error of the last solve(). NaN unless errorBounds() was on. */
  RealScalar lastBackwardError() const { return m_lastBackwardError; }
  RealScalar lastForwardError() const { return m_lastForwardError; }

  /** Correct digits supported for the last solve(), -1 unless errorBounds() was on. */
  int lastCorrectDigits() const {
    return (numext::isfinite)(m_lastForwardError) ? digitsFromForwardError(m_lastForwardError) : -1;
  }

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
  void setMatching(bool on) {
    m_matchingMethod = on ? supernodal_lu::MatchingMethod::Transversal
                          : supernodal_lu::MatchingMethod::None;
  }
  bool matching() const { return m_matchingMethod != supernodal_lu::MatchingMethod::None; }

  /** Select the matching algorithm; see SupernodalLU::setMatchingMethod. `MC64`
   *  is the exact maximum-product assignment plus its dual scaling, and is
   *  opt-in because it costs O(n) shortest-path searches. */
  void setMatchingMethod(supernodal_lu::MatchingMethod method) { m_matchingMethod = method; }
  supernodal_lu::MatchingMethod matchingMethod() const { return m_matchingMethod; }
  /** True if the last matching produced a full zero-free diagonal. A false here
   *  means the matrix is STRUCTURALLY SINGULAR: some column's nonzero rows are
   *  all claimed by other columns, so no row permutation can fill the diagonal.
   *  The factorization still runs (the unmatched columns get leftover rows and
   *  static pivoting bumps the resulting zero pivots), but the solution is
   *  meaningless; solve() then reports it through info()/solveResidual(). Valid
   *  after analyzePattern(), and only meaningful when matching() is on. */
  bool matchingIsPerfect() const { return m_matchingIsPerfect; }

  /** Structural symmetry of the pattern the symbolic analysis ran on, in [0,1]:
   *  the fraction of off-diagonal nonzeros A(i,j) whose mirror A(j,i) is also
   *  structurally present. 1 = already symmetric, so symmetrizing added nothing;
   *  0 = no nonzero has a mirror, so the symbolic pattern carries twice the
   *  off-diagonal entries A does. Measured after the matching permutation, since
   *  that is the pattern fill is actually computed from, and BEFORE the block
   *  triangular form sets the off-diagonal blocks aside -- so it describes the
   *  input, not the reduced problem. How much BTF removed is btfBlockCount() /
   *  largestBtfBlock(). Valid after analyzePattern(). */
  RealScalar patternSymmetry() const { return m_patternSymmetry; }

  /** True when patternSymmetry() is exactly 1 -- symmetrizing was a no-op and
   *  the fill is the same as an equivalent symmetric-pattern matrix's. */
  bool structurallySymmetric() const { return m_structurallySymmetric; }

  /** Permute to block upper triangular form and factor only the diagonal blocks
   *  (see \ref lrlu_btf). On by default. Costs one O(n + nnz) sweep on a matrix
   *  that turns out to be irreducible, which is every structurally symmetric
   *  one; saves an unbounded amount on a reducible one. Requires matching() --
   *  BTF's block structure is only well defined on a zero-free diagonal -- and
   *  is silently skipped without it. Set before analyzePattern(). */
  void setBlockTriangularForm(bool on) { m_btfEnabled = on; }
  bool blockTriangularForm() const { return m_btfEnabled; }

  /** Number of diagonal blocks found by the block triangular form. 1 means the
   *  matrix is irreducible and BTF bought nothing; 0 before analyzePattern().
   *  Equal to n exactly when the matched matrix is fully triangular, in which
   *  case the solve is pure substitution. */
  Index btfBlockCount() const {
    return m_btfBlockPtr.empty() ? Index(0) : static_cast<Index>(m_btfBlockPtr.size()) - 1;
  }

  /** Size of the largest diagonal block -- the one that actually sets the cost,
   *  since fill is superlinear in it. Equals n() when irreducible. */
  Index largestBtfBlock() const {
    Index largest = 0;
    for (std::size_t k = 1; k < m_btfBlockPtr.size(); ++k)
      largest = numext::maxi(largest, static_cast<Index>(m_btfBlockPtr[k] - m_btfBlockPtr[k - 1]));
    return largest;
  }

  /** Block boundaries in the internal (post-ordering) numbering: block k spans
   *  internal indices [btfBlockPointers()[k], btfBlockPointers()[k+1]). Size
   *  btfBlockCount()+1. Exposed for diagnostics -- to locate WHICH block is
   *  singular, intersect it with rowsPermutation()/colsPermutation(). */
  const std::vector<StorageIndex>& btfBlockPointers() const { return m_btfBlockPtr; }

  /** Number of nonzeros in the off-diagonal BTF blocks: entries of A that are
   *  never eliminated, only applied once each during the block
   *  back-substitution. 0 when BTF is inactive. */
  Index btfOffDiagonalNonzeros() const { return static_cast<Index>(m_offDiagValue.size()); }

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

  /** Parallelize INSIDE a supernode once the assembly DAG has run out of
   *  independent supernodes (the root-separator tail). On by default; it only
   *  ever engages when the DAG cannot fill the lanes, so turning it off simply
   *  leaves the tail serial. See the tail sweep in factorize(). */
  void setIntraSupernodeParallelism(bool on) { m_intraParallel = on; }
  bool intraSupernodeParallelism() const { return m_intraParallel; }

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
    m_conditionValid = false;
    m_conditionEstimate = NumTraits<RealScalar>::quiet_NaN();
    m_conditionSolves = 0;
    m_errorBounds = false;
    m_extendedResidual = false;
    m_lastBackwardError = NumTraits<RealScalar>::quiet_NaN();
    m_lastForwardError = NumTraits<RealScalar>::quiet_NaN();
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
    m_btfEnabled = true;
    m_matchingMethod = supernodal_lu::MatchingMethod::Transversal;
    m_pivoting = left_right_lu::Pivoting::Complete;
    m_intraParallel = true;
    m_matchingIsPerfect = true;
    m_patternSymmetry = RealScalar(1);
    m_structurallySymmetric = true;
    m_matchSign = 1;
    m_factorizationSign = Scalar(1);
    m_scalingDeterminant = RealScalar(1);
    m_nnzL = 0;
    m_nnzU = 0;
  }

  // analysis helpers (shared design with SupernodalLU).
  //
  // The symmetrized pattern is what makes an unsymmetric input pattern work: it
  // unions the pattern with its transpose. It is built ONCE, in the matrix's own
  // numbering (buildSymmetrizedPattern), and then relabelled into whichever
  // internal numbering the pipeline currently holds (relabelAdjacency) -- the
  // numbering changes when the postorder is folded in, but the graph does not.
  //
  // `blockOf` restricts the pattern to the BTF diagonal blocks: an entry (i, j)
  // whose endpoints are in different blocks is dropped, because that entry is
  // never eliminated (see \ref lrlu_btf) and including it would connect the
  // blocks in the elimination graph and reintroduce exactly the fill BTF
  // exists to avoid. Empty means "one block": no filtering.
  void buildSymmetrizedPattern(const MatrixType& matrix, const std::vector<StorageIndex>& blockOf);
  void relabelAdjacency(std::vector<std::vector<StorageIndex>>& adjacency) const;

  // Read a fill-reducing ordering functor's result as the DIRECT map
  // `direct[i] = new position of i`, which is what the pipeline needs.
  //
  // AMDOrdering and MetisOrdering return that mapping as the INVERSE in
  // indices(): perm.indices()(k) is the original column placed at new position
  // k, so it must be inverted here, not copied. Getting this backwards
  // eliminates the top separators FIRST and inflates fill enormously on
  // strongly directional matrices (measured: 250-300x on 3D FEM), while being
  // nearly invisible on near-symmetric orderings and leaving the residual at
  // machine precision either way -- a mistake that hides from every check
  // except fill. COLAMDOrdering (and anything else whose OrderingConvention
  // says returnsInverse=false) already hands back the direct map; inverting
  // that would be the same mistake in the other direction. An empty
  // permutation is NaturalOrdering's identity.
  static void readOrdering(const PermutationType& perm, StorageIndex m,
                           std::vector<StorageIndex>& direct);

  // Fill m_toInternal by ordering each BTF diagonal block on its own. Block k
  // keeps the internal index range [blockPtr[k], blockPtr[k+1]) it was given by
  // the block triangular form, so the blocks stay contiguous and in topological
  // order; the ordering only permutes within them.
  void orderWithinBlocks(const MatrixType& B, const std::vector<StorageIndex>& btfPosition,
                         const std::vector<StorageIndex>& blockPtr);
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

  // Gather ALL of a target's updates with the panel work chunked across the pool
  // (the intra-supernode counterpart of the applyUpdate loop). Every element
  // still accumulates its updates in SOURCE order, but the result is not
  // bit-identical to the serial gather: a chunk is a differently-shaped GEMM, so
  // Eigen may pick a different kernel and sum each dot product in a different
  // order. See the note on parallelChunks below.
  void applyAllUpdatesChunked(StorageIndex target);

  // Chunk-extent policy for splitting a panel op across the pool. Identical to
  // SupernodalLU's: kIntraChunkSize is a CEILING, kMinIntraChunkSize a floor, and
  // the extent actually used aims at one chunk per lane so that tall root panels
  // can fill the machine rather than being capped at ceil(total/128) chunks.
  static constexpr Index kIntraChunkSize = 128;
  static constexpr Index kMinIntraChunkSize = 32;

  // How far from "wide enough to fill the lanes" a level may be and still be
  // carved into the tail: it is taken when size * kTailLevelSlack <= lanes.
  //
  // Carving is a TRADE. The level loses its inter-supernode parallelism, and --
  // because this scheduler is barrier-free -- it also loses the overlap it had
  // with neighbouring levels, since the tail sweep is a hard phase boundary. In
  // exchange each of its supernodes gets the whole pool applied inside it.
  //
  // Re-measured 2026-08-22 on 16 cores / 32 lanes (clang 22, -O3, default
  // x86-64 ISA), factorize seconds, best of 3 INTERLEAVED rounds -- see the
  // protocol note in SupernodalLU.h before retuning this:
  //
  //   slack   lap3d_30^3 t2/t4/t8/t16/t32        lap2d_300^2 t2/t4/t8/t16/t32
  //     1     0.377 0.260 0.206 0.194 0.208      0.083 0.060 0.053 0.058 0.073
  //     2     0.379 0.263 0.197 0.187 0.199      0.083 0.059 0.050 0.054 0.064
  //     4     0.379 0.267 0.209 0.176 0.189      0.083 0.059 0.050 0.050 0.057
  //     8     0.379 0.267 0.215 0.192 0.178      0.082 0.058 0.049 0.050 0.051
  //    16     0.379 0.267 0.215 0.197 0.193      0.082 0.059 0.049 0.048 0.050
  //    32     0.375 0.268 0.215 0.198 0.197      0.082 0.058 0.049 0.048 0.049
  //
  // 4 stays, but read the table before quoting it: at 32 lanes NO value wins
  // both trees. 4 is best on the 3D tail at 16 lanes (0.176, against 0.192 for
  // 8); 8 is best on it at 32 (0.178 against 0.189) and better on the 2D tree
  // everywhere above 8 lanes; 1 and 2 lose the 2D tree badly at 32 lanes (0.073
  // and 0.064 against 0.049), which is the cross-level overlap this carve gives
  // up. The spread across the whole column is 5-15%, so the choice is a
  // compromise inside a shallow optimum rather than a peak. The t2/t4 columns
  // are flat: below 8 lanes the slack test cannot fire and the size==1 rule in
  // factorize() carries the tail on its own.
  //
  // What does NOT depend on the table: 8 cannot fire at 4 lanes at all without
  // that size==1 rule, so a slack chosen for the 32-lane column must not be the
  // only thing standing between a 4-thread run and its tail parallelism.
  static constexpr Index kTailLevelSlack = 4;

  Index intraChunkRows(Index total) const {
    const Index lanes = static_cast<Index>(m_executor.concurrency());
    if (lanes <= 1 || total <= 0) return numext::maxi(Index(1), total);
    const Index perLane = (total + lanes - 1) / lanes;
    return numext::maxi(kMinIntraChunkSize, numext::mini(kIntraChunkSize, perLane));
  }

  // Split [0, total) into contiguous chunks and dispatch body(start, length) for
  // each. Chunks cover disjoint output regions and leave each element's
  // accumulation order unchanged, so the result matches body(0, total) UP TO
  // Eigen choosing a different GEMM kernel for a differently-shaped block.
  //
  // That last clause is not a formality. Measured on this project's corpus, the
  // tail sweep changes the computed solution on ill-conditioned members
  // (Bai/cryg10000 by 0.88 relative, at an unchanged residual of 5e-15): the
  // factorization is exactly as good, but it is a DIFFERENT one, and on a matrix
  // that ill-conditioned any perturbation moves x a long way. Fill is unaffected.
  // Callers needing run-to-run reproducibility should use
  // setIntraSupernodeParallelism(false), which restores the previous behaviour.
  // Runs inline when only one chunk remains.
  template <typename F>
  void parallelChunks(Index total, Index chunkRows, const F& body) const {
    const Index chunkCount = (total + chunkRows - 1) / chunkRows;
    if (chunkCount <= 1) {
      if (total > 0) body(Index(0), total);
      return;
    }
    m_executor.parallelFor(Index(0), chunkCount, [&](Index c) {
      const Index start = c * chunkRows;
      body(start, numext::mini(chunkRows, total - start));
    });
  }

  // factor one supernode: gather all its sources' updates (left-looking), do the
  // pivoted LU of its diagonal block, then solve its off-diagonal panels. Writes
  // only this supernode's own panels. With intraParallel the gather GEMMs and the
  // panel TRSMs are chunked across the pool instead -- used only by the TAIL
  // sweep, where the DAG has run out of independent supernodes and the executor's
  // fork-join pool is therefore free.
  void factorizeSupernode(StorageIndex s, const RealScalar& staticPivot, Index& replacedCount,
                          bool& singular, int& sign, bool intraParallel = false);

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
  // Only ever called when errorBounds() is on -- this is the whole reason the
  // default solve path is unchanged.
  void recordErrorBounds(const DenseMatrix& rhs, const DenseMatrix& solution) const;

  /** Decimal digits supported by a forward-error estimate, floored at 0 and
    *  capped at the scalar type's own precision: claiming 20 digits of a double
    *  because the estimate underflowed would be worse than claiming none. */
  static int digitsFromForwardError(const RealScalar& ferr) {
    if (!(numext::isfinite)(ferr)) return 0;
    const int cap = std::numeric_limits<RealScalar>::digits10;
    if (!(ferr > RealScalar(0))) return cap;
    using std::log10;
    const RealScalar d = -log10(ferr);
    if (!(d > RealScalar(0))) return 0;
    return (d >= RealScalar(cap)) ? cap : static_cast<int>(d);
  }
  // Each of these sweeps the supernode range [sBegin, sEnd), which is one BTF
  // diagonal block (the whole factor when there is only one). Supernodes never
  // straddle a block boundary: a block boundary is a mandatory structural break
  // in the elimination tree, because the restricted pattern has no edge across
  // it, and computeSupernodePartition breaks a supernode wherever
  // parent[j-1] != j.
  template <typename Dest>
  void applyInverseL(Dest& y, StorageIndex sBegin, StorageIndex sEnd) const;
  template <typename Dest>
  void applyInverseU(Dest& y, StorageIndex sBegin, StorageIndex sEnd) const;
  template <bool Conjugate, typename Dest>
  void applyInverseLTransposed(Dest& y, StorageIndex sBegin, StorageIndex sEnd) const;
  template <bool Conjugate, typename Dest>
  void applyInverseUTransposed(Dest& y, StorageIndex sBegin, StorageIndex sEnd) const;

  template <typename Dest>
  void applyInverseL(Dest& y) const {
    applyInverseL(y, 0, static_cast<StorageIndex>(m_supernodes.size()));
  }
  template <typename Dest>
  void applyInverseU(Dest& y) const {
    applyInverseU(y, 0, static_cast<StorageIndex>(m_supernodes.size()));
  }

  // BTF block back-substitution. With A permuted to block upper triangular,
  // A y = r is solved one block at a time from the last block backwards:
  // solve the diagonal block, then subtract that block's contribution from the
  // right-hand sides of all the earlier blocks. With a single block this is
  // exactly L^-1 then U^-1 over the whole factor.
  // True when the block triangular form actually split the matrix. Derived from
  // the boundaries rather than stored, so it cannot disagree with them: an
  // inactive BTF leaves the single block {0, n}.
  bool btfActive() const { return m_btfBlockPtr.size() > 2; }

  template <typename Dest>
  void applyInverseBlocks(Dest& y) const;
  // The transpose is block LOWER triangular, so the same sweep runs forwards,
  // gathering each block's already-solved predecessors before solving it.
  template <bool Conjugate, typename Dest>
  void applyInverseBlocksTransposed(Dest& y) const;
  template <bool Conjugate>
  void solveTriangularTransposed(const DenseMatrix& rhs, DenseMatrix& solution) const;
  // FormResidual computes r = b - Op(x) for whichever operator this solve is
  // running (A, A^T or A^H), in working or extended precision as configured.
  // Passed in rather than derived here because only the call site knows which
  // operator it is, and getting that wrong would be invisible in the residual.
  template <typename ApplyA, typename ApplyPrec, typename FormResidual>
  void refineSolution(const DenseMatrix& rhs, DenseMatrix& solution, ApplyA applyA,
                      ApplyPrec applyPrec, FormResidual formResidual) const;
  template <typename ApplyPrec, typename FormResidual>
  void refineExtended(const DenseMatrix& rhs, DenseMatrix& solution, ApplyPrec applyPrec,
                      FormResidual formResidual) const;
  template <typename ApplyA, typename ApplyPrec, typename FormResidual>
  Index bicgstabColumn(ApplyA applyA, ApplyPrec applyPrec, FormResidual formResidual,
                       const DenseMatrix& b, DenseMatrix& x) const;

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
  RealScalar m_patternSymmetry;
  bool m_structurallySymmetric;
  int m_matchSign;

  // The symmetrized pattern of the matched matrix, in that matrix's own
  // numbering: CSR, diagonal excluded, each list duplicate-free but unordered.
  // Built once by buildSymmetrizedPattern, relabelled per numbering.
  std::vector<std::size_t> m_patternStart;
  std::vector<StorageIndex> m_patternIndex;

  // Per-supernode in-block pivot permutations (internal local order). *Row*
  // permutation P_s from partial/complete pivoting; *column* permutation Q_s from
  // complete pivoting. m_rowPivot[s][k] / m_colPivot[s][k] = local block row /
  // column at position k after pivoting; empty => identity.
  std::vector<std::vector<StorageIndex>> m_rowPivot;
  std::vector<std::vector<StorageIndex>> m_colPivot;
  Scalar m_factorizationSign;  // sign(matching) * prod sign(in-block row & col pivots)

  // --- block triangular form -----------------------------------------------
  // m_btfBlockPtr holds nblocks+1 boundaries in the INTERNAL numbering, and is
  // always populated after analyzePattern -- {0, n} when BTF is off or the
  // matrix is irreducible -- so the solve can run one uniform block loop.
  // m_btfSupernodePtr is the same partition expressed in supernode ids.
  std::vector<StorageIndex> m_btfBlockPtr;
  std::vector<StorageIndex> m_btfSupernodePtr;
  std::vector<StorageIndex> m_btfBlockOfInternal;  // internal index -> block id
  // The off-diagonal blocks: CSC over INTERNAL columns, scaled exactly as the
  // factor is. These entries are never eliminated, so they carry no fill and
  // are applied once per solve. Empty unless BTF found more than one block.
  std::vector<StorageIndex> m_offDiagStart;  // n+1, filled by analyzePattern
  std::vector<StorageIndex> m_offDiagRow;    // filled by factorize
  std::vector<Scalar> m_offDiagValue;

  std::vector<Supernode> m_supernodes;
  std::vector<RowBlock> m_rowBlocks;
  std::vector<StorageIndex> m_supernodeOfColumn;
  std::vector<std::vector<UpdateSource>> m_updateSources;  // per target: its sources
  std::vector<std::vector<StorageIndex>> m_consumers;      // per source: its targets
  std::vector<std::vector<StorageIndex>> m_levelGroups;    // also drives the tail sweep

  bool m_intraParallel;  // parallelize inside supernodes once the DAG runs dry

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
  // Condition estimation. m_conditionValid rather than a NaN sentinel because
  // infinity is a legitimate cached answer (a singular factorization).
  mutable bool m_conditionValid;
  mutable RealScalar m_conditionEstimate;
  mutable Index m_conditionSolves;
  bool m_errorBounds;
  bool m_extendedResidual;
  mutable RealScalar m_lastBackwardError;
  mutable RealScalar m_lastForwardError;
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
  bool m_btfEnabled;
  supernodal_lu::MatchingMethod m_matchingMethod;
  // MC64's dual scaling (original numbering), empty unless MC64 ran.
  std::vector<RealScalar> m_mc64RowScale, m_mc64ColScale;
  left_right_lu::Pivoting m_pivoting;
  Index m_nnzL;
  Index m_nnzU;
};

// ===========================================================================
//  Analysis (symbolic) phase  -- shared design with SupernodalLU
// ===========================================================================

// Build the symmetrized (pattern(A + A^T), diagonal excluded) adjacency of
// `matrix` in ITS OWN numbering, as a flat CSR in m_patternStart/m_patternIndex.
// Row order within a node's list is not maintained -- relabelAdjacency does not
// need it -- but each list is duplicate-free, which it does need.
//
// `blockOf` (empty = no BTF) keeps only the entries inside a diagonal block,
// which is exactly the set of entries the numeric phase eliminates.
template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::buildSymmetrizedPattern(
    const MatrixType& matrix, const std::vector<StorageIndex>& blockOf) {
  const StorageIndex n = m_size;
  const bool restrict = !blockOf.empty();
  m_patternStart.assign(static_cast<std::size_t>(n) + 1, 0);
  if (n == 0) {
    m_patternIndex.clear();
    m_patternSymmetry = RealScalar(1);
    m_structurallySymmetric = true;
    return;
  }

  // Pass 1: degree of every node with duplicates still in, so the CSR can be
  // filled without any per-row reallocation.
  Index offDiagonal = 0;  // off-diagonal nonzeros of `matrix`
  for (StorageIndex j = 0; j < n; ++j) {
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex i = static_cast<StorageIndex>(it.index());
      if (i == j) continue;
      ++offDiagonal;
      // Both directions go in: this union with the transpose is what lets an
      // unsymmetric pattern through without the caller symmetrizing it.
      ++m_patternStart[static_cast<std::size_t>(j) + 1];
      ++m_patternStart[static_cast<std::size_t>(i) + 1];
    }
  }
  for (StorageIndex j = 0; j < n; ++j) m_patternStart[j + 1] += m_patternStart[j];

  // Pass 2: scatter.
  m_patternIndex.resize(m_patternStart[static_cast<std::size_t>(n)]);
  std::vector<std::size_t> cursor(m_patternStart.begin(), m_patternStart.end() - 1);
  for (StorageIndex j = 0; j < n; ++j) {
    for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
      const StorageIndex i = static_cast<StorageIndex>(it.index());
      if (i == j) continue;
      m_patternIndex[cursor[static_cast<std::size_t>(j)]++] = i;
      m_patternIndex[cursor[static_cast<std::size_t>(i)]++] = j;
    }
  }

  // Pass 3: drop duplicates, compacting in place. A stamped marker array does
  // this in O(nnz) -- sorting each list first would cost O(nnz log d) for an
  // ordering nothing downstream reads.
  std::vector<StorageIndex> seen(static_cast<std::size_t>(n), StorageIndex(-1));
  std::size_t write = 0;
  for (StorageIndex j = 0; j < n; ++j) {
    const std::size_t begin = m_patternStart[static_cast<std::size_t>(j)];
    const std::size_t end = m_patternStart[static_cast<std::size_t>(j) + 1];
    m_patternStart[static_cast<std::size_t>(j)] = write;
    for (std::size_t k = begin; k < end; ++k) {
      const StorageIndex v = m_patternIndex[k];
      if (seen[static_cast<std::size_t>(v)] == j) continue;
      seen[static_cast<std::size_t>(v)] = j;
      m_patternIndex[write++] = v;
    }
  }
  const Index distinct = static_cast<Index>(write);
  m_patternStart[static_cast<std::size_t>(n)] = write;
  m_patternIndex.resize(write);

  // Every off-diagonal entry contributed 2 slots, so `distinct` = 2u for u
  // unordered pairs in the symmetrized pattern; a pair present in both
  // directions was written twice and collapsed. With m off-diagonal entries and
  // p mirrored pairs, m = u + p, so the mirrored fraction 2p/m is
  // (2m - distinct)/m. distinct == m is exactly the already-symmetric case.
  if (offDiagonal > 0) {
    const RealScalar ratio =
        RealScalar(2 * offDiagonal - distinct) / RealScalar(offDiagonal);
    // Duplicate entries in an uncompressed input would inflate `offDiagonal`;
    // clamp rather than report a nonsense ratio.
    m_patternSymmetry = numext::mini(RealScalar(1), numext::maxi(RealScalar(0), ratio));
    m_structurallySymmetric = (distinct == offDiagonal);
  } else {
    m_patternSymmetry = RealScalar(1);  // diagonal-only: trivially symmetric
    m_structurallySymmetric = true;
  }

  // Pass 4 (BTF only): drop the edges that cross a diagonal block. They are
  // deliberately absent from the elimination graph -- keeping them would
  // reconnect the blocks and reintroduce exactly the fill BTF exists to avoid.
  //
  // This runs AFTER the statistics above rather than as a filter inside passes
  // 1-3, so patternSymmetry() keeps describing the INPUT's pattern. That is
  // what a caller wants to know (and what the fill-guard message explains fill
  // with); how much of it BTF then set aside is reported separately by
  // btfBlockCount() and largestBtfBlock().
  if (!restrict) return;
  std::size_t keep = 0;
  for (StorageIndex j = 0; j < n; ++j) {
    const std::size_t begin = m_patternStart[static_cast<std::size_t>(j)];
    const std::size_t end = m_patternStart[static_cast<std::size_t>(j) + 1];
    m_patternStart[static_cast<std::size_t>(j)] = keep;
    for (std::size_t k = begin; k < end; ++k) {
      const StorageIndex v = m_patternIndex[k];
      if (blockOf[static_cast<std::size_t>(v)] != blockOf[static_cast<std::size_t>(j)]) continue;
      m_patternIndex[keep++] = v;
    }
  }
  m_patternStart[static_cast<std::size_t>(n)] = keep;
  m_patternIndex.resize(keep);
}

// Relabel the once-built symmetrized pattern into the current m_toInternal
// numbering. Downstream (computeEliminationTree, computeSupernodePartition)
// requires each list ASCENDING, and it comes out that way with no sort: the
// outer loop walks source nodes in increasing INTERNAL order and appends each
// one to its neighbours' lists, so every list is filled in ascending order by
// construction, and every list's exact length is known in advance from the
// node's degree. That is why the pattern is kept separately from the adjacency
// -- analyzePattern needs the adjacency twice in two different numberings, and
// sorting every list twice is the single most expensive thing the symbolic
// phase would otherwise do (on setfos_2 it is the difference between a 58 ms
// and a 38 ms analyzePattern).
template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::relabelAdjacency(
    std::vector<std::vector<StorageIndex>>& adjacency) const {
  const StorageIndex n = m_size;
  adjacency.assign(n, std::vector<StorageIndex>());
  if (n == 0) return;

  // Degree is a property of the node, not of the numbering, so the exact size
  // of every list is known up front and no list ever reallocates.
  std::vector<StorageIndex> inverse(static_cast<std::size_t>(n));
  for (StorageIndex o = 0; o < n; ++o) {
    const StorageIndex u = m_toInternal[o];
    inverse[static_cast<std::size_t>(u)] = o;
    adjacency[u].reserve(m_patternStart[static_cast<std::size_t>(o) + 1] -
                         m_patternStart[static_cast<std::size_t>(o)]);
  }

  for (StorageIndex u = 0; u < n; ++u) {
    const StorageIndex o = inverse[static_cast<std::size_t>(u)];
    const std::size_t end = m_patternStart[static_cast<std::size_t>(o) + 1];
    for (std::size_t k = m_patternStart[static_cast<std::size_t>(o)]; k < end; ++k)
      adjacency[m_toInternal[m_patternIndex[k]]].push_back(u);
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
      // Write into a PRE-SIZED buffer rather than through back_inserter. The
      // union can never exceed the sum of its inputs, so sizing scratch2 to
      // that bound up front lets set_union store through a contiguous iterator
      // -- a plain pointer write per element -- instead of a push_back whose
      // capacity check on every element also defeats any bulk copy of the tail
      // that remains when one input runs out. Same result, and this is the
      // single hottest line in analyzePattern.
      scratch2.resize(scratch.size() + static_cast<std::size_t>(cs.end() - childBeyond));
      const auto unionEnd =
          std::set_union(scratch.begin(), scratch.end(), childBeyond, cs.end(), scratch2.begin());
      scratch2.resize(static_cast<std::size_t>(unionEnd - scratch2.begin()));
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
      } else if (children[j].size() == 1 && rowsBeyond(structJ, j) == rowsBeyond(structPrev, j)) {
        // Fundamental continuation: one child AND the same off-diagonal row set,
        // so the merge is free. The structure half matters on chain elimination
        // trees (banded matrices), where the etree conditions alone hold at every
        // column and would collapse the whole matrix into one dense supernode.
        // See the fuller note in SupernodalLU.h's computeSupernodePartition.
        start = false;
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
void LeftRightLU<MatrixType, OrderingType, Executor>::readOrdering(const PermutationType& perm,
                                                                   StorageIndex m,
                                                                   std::vector<StorageIndex>& direct) {
  direct.resize(static_cast<std::size_t>(m));
  if (perm.size() == 0) {
    for (StorageIndex i = 0; i < m; ++i) direct[static_cast<std::size_t>(i)] = i;  // NaturalOrdering
  } else if (left_right_lu::OrderingConvention<OrderingType>::returnsInverse) {
    for (StorageIndex i = 0; i < m; ++i) direct[static_cast<std::size_t>(perm.indices()(i))] = i;
  } else {
    for (StorageIndex i = 0; i < m; ++i) direct[static_cast<std::size_t>(i)] = perm.indices()(i);
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::orderWithinBlocks(
    const MatrixType& B, const std::vector<StorageIndex>& btfPosition,
    const std::vector<StorageIndex>& blockPtr) {
  const StorageIndex n = m_size;
  m_toInternal.assign(static_cast<std::size_t>(n), 0);

  // Walk each block in position order: columnAt inverts btfPosition.
  std::vector<StorageIndex> columnAt(static_cast<std::size_t>(n));
  for (StorageIndex j = 0; j < n; ++j) columnAt[static_cast<std::size_t>(btfPosition[j])] = j;

  const StorageIndex nblocks = static_cast<StorageIndex>(blockPtr.size()) - 1;
  std::vector<StorageIndex> localDirect;
  std::vector<Triplet<Scalar, StorageIndex>> trips;
  MatrixType blockMatrix;
  PermutationType blockPerm;

  for (StorageIndex k = 0; k < nblocks; ++k) {
    const StorageIndex first = blockPtr[static_cast<std::size_t>(k)];
    const StorageIndex last = blockPtr[static_cast<std::size_t>(k) + 1];
    const StorageIndex size = last - first;

    // A singleton needs no ordering, and on a strongly reducible matrix it is
    // the overwhelmingly common case (SNAP/as-caida: 31,198 blocks, largest 95).
    // Handing tens of thousands of 1x1 matrices to the ordering functor would
    // cost more than the rest of analyzePattern put together.
    if (size == 1) {
      m_toInternal[static_cast<std::size_t>(columnAt[static_cast<std::size_t>(first)])] = first;
      continue;
    }

    trips.clear();
    for (StorageIndex p = first; p < last; ++p) {
      const StorageIndex j = columnAt[static_cast<std::size_t>(p)];
      for (typename MatrixType::InnerIterator it(B, j); it; ++it) {
        const StorageIndex q = btfPosition[static_cast<std::size_t>(it.index())];
        if (q < first || q >= last) continue;  // off-diagonal block: not eliminated
        trips.emplace_back(q - first, p - first, it.value());
      }
    }
    blockMatrix.resize(size, size);
    blockMatrix.setFromTriplets(trips.begin(), trips.end());
    blockMatrix.makeCompressed();

    m_orderingFunctor(blockMatrix, blockPerm);
    readOrdering(blockPerm, size, localDirect);
    for (StorageIndex c = 0; c < size; ++c)
      m_toInternal[static_cast<std::size_t>(columnAt[static_cast<std::size_t>(first + c)])] =
          first + localDirect[static_cast<std::size_t>(c)];
  }
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
  m_mc64RowScale.clear();
  m_mc64ColScale.clear();
  if (m_matchingMethod == supernodal_lu::MatchingMethod::MC64 && n > 0) {
    m_matchingIsPerfect =
        supernodal_lu::mc64Matching(matrix, m_matchRow, m_mc64RowScale, m_mc64ColScale);
  } else if (m_matchingMethod == supernodal_lu::MatchingMethod::Transversal && n > 0) {
    m_matchingIsPerfect = supernodal_lu::maximumWeightMatching(matrix, m_matchRow);
  } else {
    for (StorageIndex i = 0; i < n; ++i) m_matchRow[i] = i;
  }
  for (StorageIndex j = 0; j < n; ++j) m_matchRowInv[m_matchRow[j]] = j;
  m_matchSign = supernodal_lu::permutationSign(m_matchRow);

  MatrixType matched;
  if (m_matchingMethod != supernodal_lu::MatchingMethod::None) {
    std::vector<Triplet<Scalar, StorageIndex>> trips;
    trips.reserve(static_cast<std::size_t>(matrix.nonZeros()));
    for (StorageIndex j = 0; j < n; ++j)
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it)
        trips.emplace_back(m_matchRowInv[static_cast<StorageIndex>(it.index())], j, it.value());
    matched.resize(n, n);
    matched.setFromTriplets(trips.begin(), trips.end());
    matched.makeCompressed();
  }
  const MatrixType& B =
      m_matchingMethod != supernodal_lu::MatchingMethod::None ? matched : matrix;

  // 0.5) BLOCK TRIANGULAR FORM. The matching above is phase 1 of the
  //    Dulmage-Mendelsohn decomposition; this is phase 2 (strongly connected
  //    components, see LeftRightLUBlockTriangular.h). It hands back a
  //    block-contiguous position for every column, in an order that makes B
  //    block UPPER triangular. Only the diagonal blocks are eliminated.
  //
  //    Skipped without matching: the block structure is only well defined on a
  //    zero-free diagonal. Irreducible matrices fall through to exactly the
  //    single-block path below, having paid one O(n + nnz) sweep.
  std::vector<StorageIndex> btfPosition;
  std::vector<StorageIndex> blockOfB;  // B's numbering -> block id; empty = one block
  StorageIndex nblocks = 1;
  const bool btfWanted =
      m_btfEnabled && n > 0 && m_matchingMethod != supernodal_lu::MatchingMethod::None;
  if (btfWanted) nblocks = left_right_lu::blockTriangularOrder(B, btfPosition, m_btfBlockPtr);
  const bool btf = btfWanted && nblocks > 1;
  if (!btf) {
    nblocks = 1;
    m_btfBlockPtr.assign(2, 0);
    m_btfBlockPtr[1] = n;
    btfPosition.clear();
  }

  // 1) fill-reducing ordering. AMD/METIS order the pattern of B + B^T, which is
  //    exactly the graph this factorization eliminates -- so they remain the
  //    right choice on an unsymmetric pattern too (see the class docs). Under
  //    BTF each diagonal block is ordered on its own: the blocks are separate
  //    elimination problems, and the ordering has strictly more freedom on the
  //    smaller graphs.
  if (btf) {
    orderWithinBlocks(B, btfPosition, m_btfBlockPtr);
    std::vector<StorageIndex> blockOfPosition(static_cast<std::size_t>(n), 0);
    for (StorageIndex k = 0; k < nblocks; ++k)
      for (StorageIndex p = m_btfBlockPtr[static_cast<std::size_t>(k)];
           p < m_btfBlockPtr[static_cast<std::size_t>(k) + 1]; ++p)
        blockOfPosition[static_cast<std::size_t>(p)] = k;
    blockOfB.resize(static_cast<std::size_t>(n));
    for (StorageIndex j = 0; j < n; ++j)
      blockOfB[static_cast<std::size_t>(j)] =
          blockOfPosition[static_cast<std::size_t>(btfPosition[static_cast<std::size_t>(j)])];
  } else {
    PermutationType orderingPerm;
    m_orderingFunctor(B, orderingPerm);
    readOrdering(orderingPerm, n, m_toInternal);
  }

  // 2) elimination tree of the ordered, symmetrized (B + B^T) pattern. The
  //    pattern itself is numbering-independent, so it is built once here and
  //    only relabelled below.
  buildSymmetrizedPattern(B, blockOfB);
  std::vector<std::vector<StorageIndex>> adjacency;
  relabelAdjacency(adjacency);
  std::vector<StorageIndex> parent;
  computeEliminationTree(adjacency, parent);

  // 3) postorder + fold into the internal numbering (contiguous supernodes).
  std::vector<StorageIndex> postorder;
  computePostorder(parent, postorder);
  std::vector<StorageIndex> relabel(n);
  for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
  for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = relabel[m_toInternal[i]];

  // 4) re-express adjacency + tree in the final numbering. Nothing reads the CSR
  //    after this, and it is O(nnz) of the input -- tens of MB on a large matrix
  //    -- so release it rather than carry it for the solver's lifetime.
  relabelAdjacency(adjacency);
  computeEliminationTree(adjacency, parent);
  std::vector<std::size_t>().swap(m_patternStart);
  std::vector<StorageIndex>().swap(m_patternIndex);

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

  // 8) BTF bookkeeping in the FINAL internal numbering.
  //
  //    The block boundaries computed in step 0.5 survive steps 1-4 untouched,
  //    which is what lets the solve address a block as a plain index range.
  //    Step 1 permutes only within a block by construction; step 3's postorder
  //    is a permutation of the elimination FOREST, and no tree spans two blocks
  //    (the restricted pattern has no edge across a boundary), so the postorder
  //    visits every root of block k -- all of which have smaller indices than
  //    any root of block k+1 -- before any root of block k+1, and each subtree
  //    it emits stays inside its own block.
  m_btfBlockOfInternal.assign(static_cast<std::size_t>(n), 0);
  for (StorageIndex k = 0; k < nblocks; ++k)
    for (StorageIndex p = m_btfBlockPtr[static_cast<std::size_t>(k)];
         p < m_btfBlockPtr[static_cast<std::size_t>(k) + 1]; ++p)
      m_btfBlockOfInternal[static_cast<std::size_t>(p)] = k;
#ifndef NDEBUG
  for (StorageIndex j = 0; j < n; ++j)
    eigen_assert(!btf || m_btfBlockOfInternal[m_toInternal[j]] == blockOfB[j]);
#endif

  // Supernodes never straddle a block boundary (a boundary is a mandatory
  // structural break in computeSupernodePartition), so the same partition
  // expressed in supernode ids is just a lookup at each boundary column.
  m_btfSupernodePtr.assign(static_cast<std::size_t>(nblocks) + 1, 0);
  for (StorageIndex k = 0; k < nblocks; ++k) {
    const StorageIndex firstColumn = m_btfBlockPtr[static_cast<std::size_t>(k)];
    m_btfSupernodePtr[static_cast<std::size_t>(k)] =
        (firstColumn < n) ? m_supernodeOfColumn[static_cast<std::size_t>(firstColumn)]
                          : static_cast<StorageIndex>(m_supernodes.size());
  }
  m_btfSupernodePtr[static_cast<std::size_t>(nblocks)] = static_cast<StorageIndex>(m_supernodes.size());

  // Off-diagonal blocks: count per internal column now (pure pattern), fill the
  // rows and values in factorize, where the scaling is known.
  m_offDiagStart.assign(static_cast<std::size_t>(n) + 1, 0);
  m_offDiagRow.clear();
  m_offDiagValue.clear();
  if (btf) {
    for (StorageIndex j = 0; j < n; ++j) {
      const StorageIndex jj = m_toInternal[j];
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
        const StorageIndex ii = m_rowToInternal[static_cast<StorageIndex>(it.index())];
        if (m_btfBlockOfInternal[static_cast<std::size_t>(ii)] !=
            m_btfBlockOfInternal[static_cast<std::size_t>(jj)])
          ++m_offDiagStart[static_cast<std::size_t>(jj) + 1];
      }
    }
    for (StorageIndex jj = 0; jj < n; ++jj)
      m_offDiagStart[static_cast<std::size_t>(jj) + 1] += m_offDiagStart[static_cast<std::size_t>(jj)];
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

  // ROW BLOCK OUTER, COLUMN BLOCK INNER -- see the matching note in
  // SupernodalLU::applyUpdate. Each (rb, cb) pair writes its own disjoint
  // destination sub-block, so the nesting order does not affect the result;
  // putting rb outside hoists rowPanelPosition()'s binary search out of the
  // cb loop, where it was otherwise repeated once per facing column block.
  for (StorageIndex rb = firstFacingBlock; rb < lastBlock; ++rb) {
    const RowBlock& rowBlock = m_rowBlocks[rb];
    const StorageIndex rowCount = rowBlock.height();
    const auto lower = srcLower.block(rowBlock.panelOffset, 0, rowCount, wSrc);
    const bool facingRows = (rb < lastFacing);
    const StorageIndex destRow = facingRows ? StorageIndex(rowBlock.firstRow - targetFirstColumn)
                                            : rowPanelPosition(target, rowBlock.firstRow);
    for (StorageIndex cb = firstFacingBlock; cb < lastFacing; ++cb) {
      const RowBlock& colBlock = m_rowBlocks[cb];
      const StorageIndex columnCount = colBlock.height();
      const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;
      const auto facingUpper = srcUpper.block(0, colBlock.panelOffset, wSrc, columnCount);
      if (facingRows)
        targetDiag.block(destRow, targetColStart, rowCount, columnCount).noalias() -= lower * facingUpper;
      else
        targetLower.block(destRow, targetColStart, rowCount, columnCount).noalias() -= lower * facingUpper;
    }
  }

  for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
    const RowBlock& rowBlock = m_rowBlocks[rb];
    const StorageIndex rowCount = rowBlock.height();
    const StorageIndex destCol = rowPanelPosition(target, rowBlock.firstRow);
    const auto upper = srcUpper.block(0, rowBlock.panelOffset, wSrc, rowCount);
    for (StorageIndex cb = firstFacingBlock; cb < lastFacing; ++cb) {
      const RowBlock& colBlock = m_rowBlocks[cb];
      const StorageIndex columnCount = colBlock.height();
      const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;
      const auto facingLower = srcLower.block(colBlock.panelOffset, 0, columnCount, wSrc);
      targetUpper.block(targetColStart, destCol, columnCount, rowCount).noalias() -= facingLower * upper;
    }
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyAllUpdatesChunked(StorageIndex target) {
  const std::vector<UpdateSource>& sources = m_updateSources[target];
  if (sources.empty()) return;
  const StorageIndex targetFirstColumn = m_supernodes[target].firstColumn;
  const Index offDiag = static_cast<Index>(m_supernodes[target].offDiagonalRowCount);
  StridedPanel targetDiag = diagBlock(target);
  StridedPanel targetLower = lowerPanel(target);
  ContiguousPanel targetUpper = upperPanel(target);

  // Pass 1 (serial): contributions to the target's diagonal block. These are
  // small (<= width x width per source) and touch elements disjoint from the
  // off-diagonal panels, so ordering between passes is free per element.
  for (const UpdateSource& u : sources) {
    const Supernode& src = m_supernodes[u.sourceSupernode];
    const StorageIndex wSrc = src.width();
    const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
    const StridedPanel srcLower = lowerPanel(u.sourceSupernode);
    const ContiguousPanel srcUpper = upperPanel(u.sourceSupernode);
    StorageIndex lastFacing = u.facingRowBlock;
    while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;
    for (StorageIndex rb = u.facingRowBlock; rb < lastFacing; ++rb) {
      const RowBlock& rowBlock = m_rowBlocks[rb];
      const StorageIndex destRow = rowBlock.firstRow - targetFirstColumn;
      for (StorageIndex cb = u.facingRowBlock; cb < lastFacing; ++cb) {
        const RowBlock& colBlock = m_rowBlocks[cb];
        targetDiag.block(destRow, colBlock.firstRow - targetFirstColumn, rowBlock.height(),
                         colBlock.height())
            .noalias() -= srcLower.block(rowBlock.panelOffset, 0, rowBlock.height(), wSrc) *
                          srcUpper.block(0, colBlock.panelOffset, wSrc, colBlock.height());
      }
    }
  }

  // Pass 2: L off-diagonal panel, chunked by disjoint target-panel ROW ranges.
  // Each chunk walks the sources in order and applies the slice of every
  // below-facing GEMM that lands in its rows, so each element still accumulates
  // its updates in source order exactly as the serial gather does.
  parallelChunks(offDiag, intraChunkRows(offDiag), [&](Index chunkStart, Index chunkLen) {
    const Index chunkEnd = chunkStart + chunkLen;
    for (const UpdateSource& u : sources) {
      const Supernode& src = m_supernodes[u.sourceSupernode];
      const StorageIndex wSrc = src.width();
      const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
      const StridedPanel srcLower = lowerPanel(u.sourceSupernode);
      const ContiguousPanel srcUpper = upperPanel(u.sourceSupernode);
      StorageIndex lastFacing = u.facingRowBlock;
      while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;
      for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
        const RowBlock& rowBlock = m_rowBlocks[rb];
        const Index destRow = static_cast<Index>(rowPanelPosition(target, rowBlock.firstRow));
        if (destRow >= chunkEnd) break;  // panel positions increase with rb
        const Index a = numext::maxi(destRow, chunkStart);
        const Index b = numext::mini(destRow + static_cast<Index>(rowBlock.height()), chunkEnd);
        if (a >= b) continue;
        for (StorageIndex cb = u.facingRowBlock; cb < lastFacing; ++cb) {
          const RowBlock& colBlock = m_rowBlocks[cb];
          targetLower.block(a, colBlock.firstRow - targetFirstColumn, b - a, colBlock.height())
              .noalias() -= srcLower.block(rowBlock.panelOffset + (a - destRow), 0, b - a, wSrc) *
                            srcUpper.block(0, colBlock.panelOffset, wSrc, colBlock.height());
        }
      }
    }
  });

  // Pass 3: U off-diagonal panel, chunked by disjoint target-panel COLUMN ranges
  // (same slicing, transposed roles).
  parallelChunks(offDiag, intraChunkRows(offDiag), [&](Index chunkStart, Index chunkLen) {
    const Index chunkEnd = chunkStart + chunkLen;
    for (const UpdateSource& u : sources) {
      const Supernode& src = m_supernodes[u.sourceSupernode];
      const StorageIndex wSrc = src.width();
      const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
      const StridedPanel srcLower = lowerPanel(u.sourceSupernode);
      const ContiguousPanel srcUpper = upperPanel(u.sourceSupernode);
      StorageIndex lastFacing = u.facingRowBlock;
      while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;
      for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
        const RowBlock& rowBlock = m_rowBlocks[rb];
        const Index destCol = static_cast<Index>(rowPanelPosition(target, rowBlock.firstRow));
        if (destCol >= chunkEnd) break;  // panel positions increase with rb
        const Index a = numext::maxi(destCol, chunkStart);
        const Index b = numext::mini(destCol + static_cast<Index>(rowBlock.height()), chunkEnd);
        if (a >= b) continue;
        for (StorageIndex cb = u.facingRowBlock; cb < lastFacing; ++cb) {
          const RowBlock& colBlock = m_rowBlocks[cb];
          targetUpper.block(colBlock.firstRow - targetFirstColumn, a, colBlock.height(), b - a)
              .noalias() -= srcLower.block(colBlock.panelOffset, 0, colBlock.height(), wSrc) *
                            srcUpper.block(0, rowBlock.panelOffset + (a - destCol), wSrc, b - a);
        }
      }
    }
  });
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::computeEquilibration(const MatrixType& matrix) {
  const StorageIndex n = m_size;
  m_rowScale.assign(n, RealScalar(1));
  m_colScale.assign(n, RealScalar(1));

  // Seed with MC64's duals when it ran: they already put every matched diagonal
  // entry at magnitude 1 with nothing above 1, and Ruiz below iterates on
  // |value| * rowScale * colScale, so the two scalings compose.
  if (!m_mc64RowScale.empty()) {
    for (StorageIndex i = 0; i < n; ++i) m_rowScale[i] = m_mc64RowScale[i];
    for (StorageIndex j = 0; j < n; ++j) m_colScale[j] = m_mc64ColScale[j];
  }
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
      //
      // TWO PASSES, deliberately. maxCoeff(&index) has to track which lane holds
      // the running maximum, costing a compare plus a select and a horizontal
      // "any lane won?" test per packet on top of the max itself; plain
      // maxCoeff() is a bare vectorized reduction. Only the WINNING column's row
      // index is ever used, so the indexed form is needed once per elimination
      // step rather than once per trailing column. Profiling put this search at
      // roughly 44% of factorize() on the benchmark corpus -- it is what made
      // LeftRightLU slower than SupernodalLU on pivot-heavy matrices.
      //
      // The pivot CHOICE is unchanged: `best` still starts at the diagonal and
      // still only yields to a strictly larger magnitude, so ties keep going to
      // the earliest column-major position exactly as before, and bestCol is the
      // same column the one-pass loop would have settled on.
      RealScalar best = numext::abs(diag(k, k));
      StorageIndex bestCol = 0;
      bool found = false;
      for (StorageIndex jj = k; jj < w; ++jj) {
        const RealScalar m = diag.col(jj).segment(k, w - k).cwiseAbs().maxCoeff();
        if (m > best) {
          best = m;
          bestCol = jj;
          found = true;
        }
      }
      if (found) {
        Index rel = 0;
        diag.col(bestCol).segment(k, w - k).cwiseAbs().maxCoeff(&rel);
        pr = k + static_cast<StorageIndex>(rel);
        pc = bestCol;
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
                                                                         bool& singular, int& sign,
                                                                         bool intraParallel) {
  // LEFT-LOOKING gather: pull contributions from all (already-finished) sources.
  // Both paths accumulate each element's updates in source order; the chunked
  // path spreads the panel GEMMs across the pool, which reshapes them and so is
  // equal only up to floating-point reassociation (see parallelChunks).
  if (intraParallel)
    applyAllUpdatesChunked(s);
  else
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
    // L21 := L21 U11^{-1}: every ROW of the panel is an independent right-solve.
    // U12 := L11^{-1} U12: every COLUMN is independent. So both split cleanly.
    const Index offDiag = static_cast<Index>(m_supernodes[s].offDiagonalRowCount);
    auto lowerSolve = [&](Index start, Index len) {
      auto chunk = lower.middleRows(start, len);
      diag.template triangularView<Upper>().template solveInPlace<OnTheRight>(chunk);
    };
    auto upperSolve = [&](Index start, Index len) {
      auto chunk = upper.middleCols(start, len);
      diag.template triangularView<UnitLower>().solveInPlace(chunk);
    };
    if (intraParallel) {
      parallelChunks(offDiag, intraChunkRows(offDiag), lowerSolve);
      parallelChunks(offDiag, intraChunkRows(offDiag), upperSolve);
    } else {
      lowerSolve(Index(0), offDiag);
      upperSolve(Index(0), offDiag);
    }
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analysisDone && "analyzePattern must be called before factorize");
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_replacedPivots = 0;
  m_factorized = false;
  m_info = Success;
  // New values mean a new operator: the cached condition estimate describes the
  // factorization, not the pattern, so it cannot survive a refactorization.
  m_conditionValid = false;
  m_conditionSolves = 0;

  // Fail-fast fill guard: abort BEFORE allocating the factor arenas if the
  // symbolic structure predicts a factor larger than the configured limit. This
  // turns an (unrecoverable) multi-hundred-GB bad_alloc / OOM kill into a clean
  // NumericalIssue with a diagnostic -- the symmetrized-pattern fill of some
  // matrices (e.g. FEM systems without good separators) is orders of magnitude
  // larger than a partial-pivoting solver would need. Off by default (limit 0).
  if (m_maxFactorNonzeros > 0) {
    const Index predicted = predictedFactorNonzeros();
    if (predicted > m_maxFactorNonzeros) {
      char buf[640];
      std::snprintf(buf, sizeof(buf),
                    "LeftRightLU: predicted factor size %lld scalars (~%.1f GB) exceeds the "
                    "configured limit of %lld (setMaxFactorNonzeros). The fill of the "
                    "symmetrized (A + A^T) pattern is very large at pattern symmetry %.2f -- "
                    "the matrix likely lacks good vertex separators%s. Try MetisOrdering, or "
                    "an iterative (e.g. BiCGSTAB+ILUT) or partial-pivoting (SparseLU/PARDISO) "
                    "solver. Raise or clear the limit to force the factorization.",
                    static_cast<long long>(predicted),
                    static_cast<double>(predicted) * static_cast<double>(sizeof(Scalar)) / 1e9,
                    static_cast<long long>(m_maxFactorNonzeros),
                    static_cast<double>(m_patternSymmetry),
                    m_structurallySymmetric
                        ? ""
                        : ", and symmetrizing its unsymmetric pattern added structure on top");
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

  // 2) scatter the (permuted, scaled) values of A into the panels. Entries that
  //    cross a BTF block boundary are NOT part of any panel -- they are never
  //    eliminated -- and are peeled off into the off-diagonal CSC instead, for
  //    the block back-substitution to apply once per solve.
  const bool btf = btfActive();
  if (btf) {
    m_offDiagRow.resize(static_cast<std::size_t>(m_offDiagStart[static_cast<std::size_t>(m_size)]));
    m_offDiagValue.resize(m_offDiagRow.size());
  }
  std::vector<StorageIndex> offCursor;
  if (btf) offCursor.assign(m_offDiagStart.begin(), m_offDiagStart.end() - 1);

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
      if (btf && m_btfBlockOfInternal[static_cast<std::size_t>(ii)] !=
                     m_btfBlockOfInternal[static_cast<std::size_t>(jj)]) {
        const std::size_t slot = static_cast<std::size_t>(offCursor[static_cast<std::size_t>(jj)]++);
        m_offDiagRow[slot] = ii;
        m_offDiagValue[slot] = value;
      } else if (m_supernodeOfColumn[ii] == columnSupernode) {
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
    // Parallel dynamic DAG scheduler with PER-WORKER DEQUES AND WORK STEALING.
    // remaining[s] = number of unfinished update sources of s.
    //
    // Each worker owns a deque and pushes the consumers IT readied onto its own
    // back, popping from its own back as well: depth-first, and the node it just
    // produced (whose data is hot in this core's cache) is the node it takes
    // next. Only when its own deque runs dry does it steal, and it steals from
    // the FRONT of a victim -- the oldest entry, furthest from the victim's hot
    // end, so stealing rarely collides with the owner and tends to move a whole
    // coarse subtree rather than one leaf.
    //
    // The simpler alternative -- one global mutex-guarded ready stack with a
    // notify_all() after every completed supernode -- serializes every task
    // acquisition and every push behind a single lock, and wakes all P workers
    // per supernode when at most a couple can make progress. Measured, it stops
    // scaling around 8 threads and then goes BACKWARDS (lap2d_300^2 56.3ms at 16
    // threads -> 72.6ms at 32; lap3d_30^3 542 -> 634): the classic shape of
    // ready-queue contention rather than of a parallelism shortage. It also
    // defeats the depth-first subtree affinity the LIFO is there to provide, as
    // a worker's freshly-readied children land on the shared stack where any
    // other worker takes them immediately, so a subtree's data never stays on
    // one core.
    // THE TAIL. Near the root the assembly DAG narrows to a chain of separator
    // supernodes -- few nodes, each enormous -- and a scheduler that only
    // parallelizes ACROSS supernodes has nothing left to hand out. VTune's
    // threading analysis on lap3d_30^3, re-run 2026-08-22 on 16 cores / 32
    // lanes, with the tail sweep DISABLED: effective CPU utilization 1.18 of the
    // 8 lanes given, and 1.32 of 32 -- adding 24 lanes bought 0.14 of a lane.
    // The idle time is STARVATION, not contention: the per-worker deque locks
    // took 0.04s of wait across ~8500 acquisitions in a 4.6s factorization, so
    // the queue is not what the workers are waiting on; there is simply no
    // second supernode to hand them. Those levels are therefore carved out of
    // the DAG phase and swept afterwards with the pool applied INSIDE each
    // supernode instead, which takes the same measurement to 2.72 of 8 lanes and
    // 6.39 of 32 (wall clock: 3.08x at 32 threads, against 1.18x without it).
    //
    // The split is necessary because the executor's parallelFor is fork-join and
    // NOT nestable: while the DAG workers are running, all lanes are already
    // inside one parallelFor, so no supernode can dispatch panel work of its
    // own. Ending the parallel region first is what frees the pool.
    //
    // Level 0 is the leaves and the last group is the root, so the tail is the
    // suffix of m_levelGroups that stays too narrow to fill the lanes. The
    // `size * kTailLevelSlack <= lanes` test mirrors SupernodalLU's, for the
    // same measured reason: wider levels keep the outer per-supernode
    // parallelism, which beats chunking for moderate panels -- and on the wide
    // 2D tree carving them costs real time at high lane counts (see the measured
    // table at kTailLevelSlack).
    std::vector<char> isTail(static_cast<std::size_t>(supernodeNbr), 0);
    StorageIndex tailCount = 0;
    if (m_intraParallel && !m_levelGroups.empty()) {
      for (std::size_t L = m_levelGroups.size(); L-- > 0;) {
        const std::vector<StorageIndex>& g = m_levelGroups[L];
        // A level of ONE supernode always qualifies, whatever the slack: the DAG
        // phase would run it on one lane with the rest idle, so there is no
        // inter-supernode parallelism to give up. Without this the slack test
        // needs size <= lanes/4 and so never fires at 2 lanes at all -- the same
        // gap SupernodalLU had below 8 lanes.
        if (g.size() != 1 &&
            static_cast<Index>(g.size()) * kTailLevelSlack > static_cast<Index>(lanes))
          break;
        for (StorageIndex s : g) {
          isTail[static_cast<std::size_t>(s)] = 1;
          ++tailCount;
        }
      }
    }
    const StorageIndex dagCount = supernodeNbr - tailCount;

    std::vector<std::atomic<int>> remaining(static_cast<std::size_t>(supernodeNbr));
    for (StorageIndex s = 0; s < supernodeNbr; ++s)
      remaining[s].store(static_cast<int>(m_updateSources[s].size()), std::memory_order_relaxed);

    // One deque per lane, each cache-line aligned so two workers' locks and
    // bookkeeping never share a line.
    struct alignas(64) WorkerDeque {
      std::mutex mutex;
      std::deque<StorageIndex> items;
    };
    std::vector<WorkerDeque> queues(static_cast<std::size_t>(lanes));

    // Seed the initially-ready supernodes round-robin, so the sweep starts with
    // every lane holding work instead of one lane holding all of it.
    {
      std::size_t lane = 0;
      for (StorageIndex s = 0; s < supernodeNbr; ++s)
        if (!isTail[s] && remaining[s].load(std::memory_order_relaxed) == 0) {
          queues[lane].items.push_back(s);
          lane = (lane + 1) % static_cast<std::size_t>(lanes);
        }
    }

    std::atomic<StorageIndex> completed{0};
    std::atomic<bool> failed{false};

    // Idle workers park here rather than spinning. `sleepers` lets a producer
    // skip the notify entirely in the common case where nobody is parked, which
    // is what removes the per-supernode notify_all cost. Waits are bounded, so a
    // notify that races with a worker about to park costs latency, never a hang.
    std::mutex idleMutex;
    std::condition_variable idleCv;
    std::atomic<int> sleepers{0};

    auto pushLocal = [&](std::size_t lane, StorageIndex t) {
      {
        std::lock_guard<std::mutex> lk(queues[lane].mutex);
        queues[lane].items.push_back(t);
      }
      if (sleepers.load(std::memory_order_acquire) > 0) idleCv.notify_one();
    };

    auto popLocal = [&](std::size_t lane) -> StorageIndex {
      std::lock_guard<std::mutex> lk(queues[lane].mutex);
      if (queues[lane].items.empty()) return StorageIndex(-1);
      const StorageIndex s = queues[lane].items.back();  // LIFO: hottest first
      queues[lane].items.pop_back();
      return s;
    };

    auto trySteal = [&](std::size_t lane) -> StorageIndex {
      for (std::size_t k = 1; k < static_cast<std::size_t>(lanes); ++k) {
        const std::size_t victim = (lane + k) % static_cast<std::size_t>(lanes);
        std::lock_guard<std::mutex> lk(queues[victim].mutex);
        if (queues[victim].items.empty()) continue;
        const StorageIndex s = queues[victim].items.front();  // oldest, coldest
        queues[victim].items.pop_front();
        return s;
      }
      return StorageIndex(-1);
    };

    auto worker = [&](Index laneIndex) {
      const std::size_t lane = static_cast<std::size_t>(laneIndex);
      std::vector<StorageIndex> newReady;
      for (;;) {
        if (failed.load(std::memory_order_acquire)) return;

        StorageIndex s = popLocal(lane);
        if (s < 0) s = trySteal(lane);

        if (s < 0) {
          // Nothing anywhere right now. The DAG may still produce more, so this
          // is not a termination condition -- only the completed count is.
          if (completed.load(std::memory_order_acquire) == dagCount) return;
          sleepers.fetch_add(1, std::memory_order_release);
          {
            std::unique_lock<std::mutex> lk(idleMutex);
            idleCv.wait_for(lk, std::chrono::microseconds(200), [&] {
              return completed.load(std::memory_order_acquire) == dagCount ||
                     failed.load(std::memory_order_acquire);
            });
          }
          sleepers.fetch_sub(1, std::memory_order_release);
          continue;
        }

        bool localSingular = false;
        factorizeSupernode(s, staticPivot, replacedPerSupernode[s], localSingular, signPerSupernode[s]);
        if (localSingular) {
          singularPerSupernode[s] = 1;
          failed.store(true, std::memory_order_release);
          idleCv.notify_all();
          return;
        }

        // RIGHT-LOOKING push: decrement each consumer's in-degree and keep the
        // ones that just became ready on THIS lane (subtree affinity).
        newReady.clear();
        for (StorageIndex t : m_consumers[s])
          if (remaining[t].fetch_sub(1, std::memory_order_acq_rel) == 1) newReady.push_back(t);
        const StorageIndex done = completed.fetch_add(1, std::memory_order_acq_rel) + 1;

        // Tail supernodes are deliberately NOT enqueued: the post-DAG sweep
        // below takes them, with the pool free to work inside each one.
        for (StorageIndex t : newReady)
          if (!isTail[t]) pushLocal(lane, t);
        if (done == dagCount) idleCv.notify_all();  // wake everyone to exit
      }
    };
    m_executor.parallelFor(Index(0), static_cast<Index>(lanes), worker);

    if (failed.load(std::memory_order_acquire)) singular = true;

    // TAIL SWEEP. The parallel region has ended, so the pool is idle and
    // available again. Supernode ids are in postorder, so ascending id order is
    // a topological order: every source of a tail supernode -- whether it was
    // factored in the DAG phase or is an earlier member of the tail -- is
    // already done by the time we reach it.
    if (!singular && tailCount > 0) {
      for (StorageIndex s = 0; s < supernodeNbr; ++s) {
        if (!isTail[s]) continue;
        // Chunking is decided PER SUPERNODE, not per level: the very last
        // supernode is the root and has NO off-diagonal rows at all, so there is
        // nothing there to split, while the separator supernodes just below it
        // carry the tall panels this whole path exists for. Testing per level
        // (and giving up at the first level that fails) would let the root veto
        // the entire tail.
        const bool chunk =
            static_cast<Index>(m_supernodes[s].offDiagonalRowCount) >= 2 * kMinIntraChunkSize;
        bool localSingular = false;
        factorizeSupernode(s, staticPivot, replacedPerSupernode[s], localSingular,
                           signPerSupernode[s], chunk);
        if (localSingular) {
          singular = true;
          break;
        }
      }
    }
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
  auto formResidual = [this](const DenseMatrix& b2, const DenseMatrix& x2, DenseMatrix& r) {
    if (m_extendedResidual)
      left_right_lu::residualExtended(m_originalMatrix, b2, x2, r);
    else
      r.noalias() = b2 - m_originalMatrix * x2;
  };
  // PARDISO IPARM(8)=0: refine only if the factorization perturbed a pivot.
  if (!(m_refineOnlyIfPerturbed && m_replacedPivots == 0))
    refineSolution(rhs, solution, applyA,
                   [this](const DenseMatrix& in, DenseMatrix& out) { solveTriangular(in, out); },
                   formResidual);
  else
    m_lastRefinementIterations = 0;

  recordSolveStatus(rhs, solution, applyA);
  if (m_errorBounds) {
    recordErrorBounds(rhs, solution);
  } else {
    // Stale bounds from an earlier solve() that DID compute them would be worse
    // than none: NaN is what lastCorrectDigits() reads as "not measured".
    m_lastBackwardError = NumTraits<RealScalar>::quiet_NaN();
    m_lastForwardError = NumTraits<RealScalar>::quiet_NaN();
  }
  x = solution;
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseL(Dest& y, StorageIndex sBegin,
                                                                    StorageIndex sEnd) const {
  for (StorageIndex s = sBegin; s < sEnd; ++s) {
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
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseU(Dest& y, StorageIndex sBegin,
                                                                    StorageIndex sEnd) const {
  for (StorageIndex s = sEnd; s > sBegin;) {
    --s;
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
  }
}

// BTF block back-substitution, in the internal numbering. The matrix is block
// UPPER triangular there, so blocks are solved from the last backwards, and
// each solved block is immediately substituted out of the earlier blocks'
// right-hand sides. The off-diagonal entries are used exactly once and are
// never modified, which is why they neither fill nor accumulate rounding.
//
// With one block this is exactly L^-1 followed by U^-1 over the whole factor,
// and the entry loop below runs zero times.
template <typename MatrixType, typename OrderingType, typename Executor>
template <typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseBlocks(Dest& y) const {
  const StorageIndex nblocks = static_cast<StorageIndex>(m_btfSupernodePtr.size()) - 1;
  for (StorageIndex k = nblocks; k > 0;) {
    --k;
    const StorageIndex sBegin = m_btfSupernodePtr[static_cast<std::size_t>(k)];
    const StorageIndex sEnd = m_btfSupernodePtr[static_cast<std::size_t>(k) + 1];
    applyInverseL(y, sBegin, sEnd);
    applyInverseU(y, sBegin, sEnd);
    if (m_offDiagValue.empty()) continue;
    for (StorageIndex jj = m_btfBlockPtr[static_cast<std::size_t>(k)];
         jj < m_btfBlockPtr[static_cast<std::size_t>(k) + 1]; ++jj) {
      for (StorageIndex p = m_offDiagStart[static_cast<std::size_t>(jj)];
           p < m_offDiagStart[static_cast<std::size_t>(jj) + 1]; ++p)
        y.row(m_offDiagRow[static_cast<std::size_t>(p)]) -=
            m_offDiagValue[static_cast<std::size_t>(p)] * y.row(jj);
    }
  }
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::solveTriangular(const DenseMatrix& rhs,
                                                                      DenseMatrix& x) const {
  const StorageIndex n = m_size;
  const Index nrhs = rhs.cols();
  DenseMatrix y(n, nrhs);
  for (StorageIndex i = 0; i < n; ++i) y.row(m_rowToInternal[i]) = m_rowScale[i] * rhs.row(i);
  applyInverseBlocks(y);
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
//  Condition estimation and error bounds
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
typename LeftRightLU<MatrixType, OrderingType, Executor>::RealScalar
LeftRightLU<MatrixType, OrderingType, Executor>::conditionEstimate() const {
  if (m_conditionValid) return m_conditionEstimate;
  m_conditionValid = true;
  m_conditionSolves = 0;

  // Order matters: a default-constructed solver also has m_size == 0, and
  // answering "0" for it would read as a perfectly conditioned matrix rather
  // than as "there is no factorization to ask about".
  if (!m_factorized) {
    m_conditionEstimate = NumTraits<RealScalar>::infinity();
    return m_conditionEstimate;
  }
  if (m_size == 0) {
    m_conditionEstimate = RealScalar(0);
    return m_conditionEstimate;
  }

  const RealScalar anorm = left_right_lu::oneNorm(m_originalMatrix);
  if (!(anorm > RealScalar(0))) {
    // A zero matrix has no finite condition number, and saying "0" here would
    // read as perfectly conditioned.
    m_conditionEstimate = NumTraits<RealScalar>::infinity();
    return m_conditionEstimate;
  }

  // The operator handed to Hager's algorithm is the solver's own A^{-1}: the
  // scaled, permuted, BTF-blocked triangular solve, WITHOUT refinement. That is
  // deliberate -- refinement would make the operator nonlinear, and the estimate
  // is meant to describe the factors the caller is about to apply.
  typedef Matrix<Scalar, Dynamic, 1> Vector;
  DenseMatrix in(m_size, 1), out(m_size, 1);
  const RealScalar invNorm = left_right_lu::oneNormEstimate<Scalar>(
      Index(m_size),
      [&](const Vector& v, Vector& r) {
        in.col(0) = v;
        solveTriangular(in, out);
        r = out.col(0);
      },
      [&](const Vector& v, Vector& r) {
        in.col(0) = v;
        // Conjugate transpose for complex scalars: Hager's sign vector is
        // v/|v|, which pairs with A^H, not A^T.
        this->template solveTriangularTransposed<NumTraits<Scalar>::IsComplex>(in, out);
        r = out.col(0);
      },
      &m_conditionSolves);

  m_conditionEstimate = (numext::isfinite)(invNorm)
                            ? anorm * invNorm
                            : NumTraits<RealScalar>::infinity();
  return m_conditionEstimate;
}

template <typename MatrixType, typename OrderingType, typename Executor>
void LeftRightLU<MatrixType, OrderingType, Executor>::recordErrorBounds(
    const DenseMatrix& rhs, const DenseMatrix& solution) const {
  m_lastBackwardError =
      left_right_lu::componentwiseBackwardError(m_originalMatrix, rhs, solution);
  m_lastForwardError =
      left_right_lu::estimateForwardError(conditionEstimate(), m_lastBackwardError);

  // The case this exists to catch: the residual check passed -- the answer IS
  // the exact solution of a nearby system -- and yet the conditioning leaves no
  // digit of it standing. A residual alone cannot tell those apart, which is
  // the whole point of computing a forward error.
  if (m_info == Success && !(m_lastForwardError < RealScalar(1))) {
    m_info = NumericalIssue;
    m_lastError =
        "LeftRightLU: the solve is backward stable (see lastBackwardError()) but the estimated "
        "condition number (see conditionEstimate()) leaves no correct digits in the answer -- "
        "the system is too ill-conditioned to be solved usefully in this precision.";
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
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseUTransposed(Dest& y, StorageIndex sBegin,
                                                                              StorageIndex sEnd) const {
  for (StorageIndex s = sBegin; s < sEnd; ++s) {
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
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseLTransposed(Dest& y, StorageIndex sBegin,
                                                                              StorageIndex sEnd) const {
  for (StorageIndex s = sEnd; s > sBegin;) {
    --s;
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
  }
}

// The transpose of a block upper triangular matrix is block LOWER triangular,
// so the block sweep runs FORWARDS here: block k gathers the contributions of
// the blocks before it -- all already solved -- and then solves its own
// diagonal block. Each off-diagonal entry is read transposed (conjugated for
// the adjoint) from the same column-major storage, so no transpose is formed.
template <typename MatrixType, typename OrderingType, typename Executor>
template <bool Conjugate, typename Dest>
void LeftRightLU<MatrixType, OrderingType, Executor>::applyInverseBlocksTransposed(Dest& y) const {
  const StorageIndex nblocks = static_cast<StorageIndex>(m_btfSupernodePtr.size()) - 1;
  for (StorageIndex k = 0; k < nblocks; ++k) {
    if (!m_offDiagValue.empty()) {
      for (StorageIndex jj = m_btfBlockPtr[static_cast<std::size_t>(k)];
           jj < m_btfBlockPtr[static_cast<std::size_t>(k) + 1]; ++jj) {
        for (StorageIndex p = m_offDiagStart[static_cast<std::size_t>(jj)];
             p < m_offDiagStart[static_cast<std::size_t>(jj) + 1]; ++p) {
          const Scalar a = m_offDiagValue[static_cast<std::size_t>(p)];
          y.row(jj) -= (Conjugate ? numext::conj(a) : a) * y.row(m_offDiagRow[static_cast<std::size_t>(p)]);
        }
      }
    }
    const StorageIndex sBegin = m_btfSupernodePtr[static_cast<std::size_t>(k)];
    const StorageIndex sEnd = m_btfSupernodePtr[static_cast<std::size_t>(k) + 1];
    applyInverseUTransposed<Conjugate>(y, sBegin, sEnd);
    applyInverseLTransposed<Conjugate>(y, sBegin, sEnd);
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
  applyInverseBlocksTransposed<Conjugate>(y);
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
  auto formResidual = [this](const DenseMatrix& b2, const DenseMatrix& x2, DenseMatrix& r) {
    if (m_extendedResidual) {
      left_right_lu::residualExtendedTransposed<Conjugate>(m_originalMatrix, b2, x2, r);
    } else if (Conjugate) {
      r.noalias() = b2 - m_originalMatrix.adjoint() * x2;
    } else {
      r.noalias() = b2 - m_originalMatrix.transpose() * x2;
    }
  };
  if (!(m_refineOnlyIfPerturbed && m_replacedPivots == 0))
    refineSolution(rhs, solution, applyA,
                   [this](const DenseMatrix& in, DenseMatrix& out) {
                     this->template solveTriangularTransposed<Conjugate>(in, out);
                   },
                   formResidual);
  else
    m_lastRefinementIterations = 0;

  recordSolveStatus(rhs, solution, applyA);
  // Error bounds are NOT computed for a transposed solve: kappa_1(A^T) is not
  // kappa_1(A), and reporting the wrong one would be worse than reporting none.
  // NaN is what lastCorrectDigits() reads as "not measured".
  m_lastBackwardError = NumTraits<RealScalar>::quiet_NaN();
  m_lastForwardError = NumTraits<RealScalar>::quiet_NaN();
  x = solution;
}

// ===========================================================================
//  Iterative / Krylov refinement
// ===========================================================================

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA, typename ApplyPrec, typename FormResidual>
void LeftRightLU<MatrixType, OrderingType, Executor>::refineSolution(const DenseMatrix& rhs,
                                                                     DenseMatrix& solution,
                                                                     ApplyA applyA,
                                                                     ApplyPrec applyPrec,
                                                                     FormResidual formResidual) const {
  m_lastRefinementIterations = 0;
  const RealScalar rhsNorm = rhs.norm();
  if (m_maxRefinementIterations <= 0 || rhsNorm == RealScalar(0) ||
      m_refinementMethod == left_right_lu::Refinement::None)
    return;

  const Index n = rhs.rows();
  const Index nrhs = rhs.cols();

  if (m_extendedResidual && m_refinementMethod == left_right_lu::Refinement::IterativeRefinement) {
    refineExtended(rhs, solution, applyPrec, formResidual);
    return;
  }

  if (m_refinementMethod == left_right_lu::Refinement::IterativeRefinement) {
    DenseMatrix best = solution;
    RealScalar bestNorm = NumTraits<RealScalar>::highest();
    RealScalar prevNorm = NumTraits<RealScalar>::highest();
    DenseMatrix residual(n, nrhs), correction(n, nrhs);
    for (Index it = 0;; ++it) {
      // The one place extended precision changes anything: everything below --
      // the correction solve, the update, the stopping tests -- stays in working
      // precision, exactly as before.
      formResidual(rhs, solution, residual);
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
    const Index iters = bicgstabColumn(applyA, applyPrec, formResidual, bcol, xcol);
    solution.col(c) = xcol.col(0);
    m_lastRefinementIterations = numext::maxi(m_lastRefinementIterations, iters);
  }

  // BiCGStab's own residual comes from its recurrence, not from b - Ax, so
  // extended precision cannot reach it there. Polishing afterwards is what makes
  // setExtendedPrecisionResidual mean the same thing whichever method is
  // selected: a forward error at machine precision, not merely a backward one.
  if (m_extendedResidual) refineExtended(rhs, solution, applyPrec, formResidual);
}

// The forward-error refinement loop, shared by the stationary path and by the
// BiCGStab polish above.
template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyPrec, typename FormResidual>
void LeftRightLU<MatrixType, OrderingType, Executor>::refineExtended(
    const DenseMatrix& rhs, DenseMatrix& solution, ApplyPrec applyPrec,
    FormResidual formResidual) const {

    // Extended residuals need a DIFFERENT stopping rule, and this is the whole
    // reason they are not simply a drop-in accuracy improvement.
    //
    // The working-precision loop below stops when ||r|| <= eps*||b||. After a
    // direct solve that is already true -- the factorization is backward stable
    // -- so it stops before applying a single correction. That is correct when
    // the goal is a small BACKWARD error, and useless when the goal is a small
    // FORWARD one: on an ill-conditioned matrix the answer can be wrong in every
    // digit while its residual sits at machine precision.
    //
    // So this loop watches the CORRECTION instead: keep going while ||dx||/||x||
    // is still shrinking meaningfully, stop once it reaches eps (the solution has
    // converged) or stops halving (no further progress is available). That is the
    // criterion LAPACK's xGERFSX uses for the same reason.
  const Index n = rhs.rows();
  const Index nrhs = rhs.cols();
  DenseMatrix residual(n, nrhs), correction(n, nrhs);
    DenseMatrix best = solution;
    RealScalar bestResidual = NumTraits<RealScalar>::highest();
    RealScalar previousRatio = NumTraits<RealScalar>::highest();
    for (Index it = 0; it < m_maxRefinementIterations; ++it) {
      formResidual(rhs, solution, residual);
      const RealScalar resNorm = residual.norm();
      if (resNorm < bestResidual) {
        bestResidual = resNorm;
        best = solution;
      }
      applyPrec(residual, correction);
      const RealScalar xNorm = solution.norm();
      const RealScalar ratio =
          (xNorm > RealScalar(0)) ? correction.norm() / xNorm : correction.norm();
      solution += correction;
      ++m_lastRefinementIterations;
      if (!(numext::isfinite)(ratio)) break;
      if (ratio <= NumTraits<RealScalar>::epsilon()) break;
      if (it > 0 && ratio > previousRatio / RealScalar(2)) break;
      previousRatio = ratio;
    }
    // Keep the refined solution: unlike the backward-error loop, the last
    // iterate here is the most accurate one even when its residual is not the
    // smallest -- that mismatch IS the forward/backward distinction.
    formResidual(rhs, solution, residual);
    if (!solution.allFinite()) solution = best;
}

template <typename MatrixType, typename OrderingType, typename Executor>
template <typename ApplyA, typename ApplyPrec, typename FormResidual>
Index LeftRightLU<MatrixType, OrderingType, Executor>::bicgstabColumn(ApplyA applyA, ApplyPrec applyPrec,
                                                                      FormResidual formResidual,
                                                                      const DenseMatrix& b,
                                                                      DenseMatrix& x) const {
  const Index n = b.rows();
  const RealScalar bnorm = b.col(0).norm();
  const RealScalar threshold = m_refinementTolerance * bnorm;

  // Only the INITIAL residual benefits from extended precision here: every
  // later one comes from the Krylov recurrence r = s - omega*t, which is an
  // accumulation over previous residuals and not a fresh b - Ax. refineSolution
  // therefore follows BiCGStab with a stationary extended pass when the flag is
  // on -- see refineExtended.
  DenseMatrix r(n, 1);
  formResidual(b, x, r);
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
