// SupernodalLDLT -- a supernodal LDL^T factorization for symmetric matrices,
// positive definite or indefinite.
//
// WHAT IT IS FOR
//
// SupernodalLU and LeftRightLU factor a general matrix as L U. Handed a
// SYMMETRIC one they still compute both factors, and U is the transpose of what
// L already holds. This solver keeps only one of them: A = L D L^T with L unit
// lower triangular and D block diagonal, so both the stored factor and the flops
// that build it are roughly half. The saving is structural rather than a tuning
// win -- it is the redundant half of an LU of a symmetric matrix, not overhead.
//
// It is built on the SAME symbolic analysis as its two siblings
// (SupernodalLUSymbolic.h: elimination tree, postorder, supernode partition with
// amalgamation, block structure, update lists, scheduling levels). Only the
// numeric core and the solve differ. That is also why the analysis is worth
// reading there rather than here.
//
// SCOPE -- READ THIS FIRST
//
// The matrix must be symmetric (Hermitian for complex scalars). It need NOT be
// positive definite: D carries 1x1 and 2x2 blocks, chosen by the bounded
// Bunch-Kaufman criterion, which is what makes an indefinite matrix tractable.
// Saddle-point and KKT systems are exactly this case.
//
// Only one triangle is read, selected by the UpLo template parameter; whatever
// is in the other one is ignored, so a caller holding half the matrix pays
// nothing to use it, and one holding all of it pays nothing either.
//
// THE INTERCHANGES STAY INSIDE A SUPERNODE, which is what keeps the symbolic
// structure static. A 2x2 pivot needs a symmetric interchange, and that
// interchange is confined to the dense diagonal block, so the Schur complement
// the supernode sends out is unchanged by it:
//
//     L_R D L_R^H = A_R P^T (L_kk D L_kk^H)^-1 P A_R^H
//                 = A_R P^T (P A_kk P^T)^-1 P A_R^H
//                 = A_R A_kk^-1 A_R^H          <- P cancels
//
// So P is invisible outside the block. What it does cost is the ONE case where
// the block runs out of room: a 2x2 pivot wanted at the last column of a
// supernode has no second column to pair with, and growing the structure to find
// one is what this design refuses to do. A perturbed 1x1 pivot is taken instead
// and counted in straddlingPivots(), with refinement to clean up after it.
//
// Positive definiteness does not have to be asserted. setPivoting(None) is the
// fast path that assumes it, and its pivot test IS the check -- it reports a
// non-positive pivot rather than stepping over one. The default handles either
// case, and inertia() then says which it was, exactly.
//
// WHEN THE DIAGONAL ITSELF IS THE PROBLEM
//
// Bunch-Kaufman can only pivot inside a supernode. If the ordering never puts a
// good 2x2 candidate there, the solver perturbs its way through and the answer is
// flagged but worthless. setMatching(true) fixes that upstream: a symmetric
// weighted matching (Duff-Pralet) pairs columns into 2x2 candidates and keeps
// each pair together through the ordering. On a matrix with an all-zero diagonal
// this is the difference between an answer and none -- see SupernodalLDLTMatching.h.
//
// It is OFF by default because it is a real trade, not a free win: measured here,
// a saddle-point system that already had a usable diagonal paid 2x the fill and
// 3x the factorization time for pairs the numeric phase then declined.
// replacedPivots() is the number that says whether you need it.
//
// NO UNSYMMETRIC MATCHING, AND THAT IS NOT A GAP. The siblings permute rows to
// put large entries on the diagonal (MC64/transversal). That is an UNSYMMETRIC
// row permutation: applying it to a symmetric matrix destroys the symmetry this
// solver exists to exploit -- which is why the matching above is read as a
// symmetric pairing rather than applied as the LU solvers apply theirs.
//
// PARALLELISM. Levels of the elimination tree are dispatched across the executor,
// and a level too narrow to fill it switches to splitting one supernode's panel
// instead (setIntraSupernodeParallelism). Expect this solver to scale somewhat
// worse than SupernodalLU even so: half the flops sit against the same fixed
// costs, and there is one panel per supernode to chunk rather than two.
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
#include "SupernodalLDLTMatching.h"

namespace Eigen {

namespace supernodal_ldlt {

/** How each supernode's dense diagonal block is factored.
 *
 *   None          1x1 pivots taken in order, with no search. Correct only for a
 *                 POSITIVE DEFINITE matrix, where the diagonal is guaranteed
 *                 usable; a non-positive pivot is reported rather than stepped
 *                 over. The cheapest path, and the one with no permutation to
 *                 carry through the solve.
 *   BunchKaufman  1x1 and 2x2 pivot blocks chosen by the bounded Bunch-Kaufman
 *                 criterion, with SYMMETRIC interchanges confined to the block.
 *                 This is what makes an INDEFINITE matrix tractable: a symmetric
 *                 matrix can have an arbitrarily small diagonal and still be
 *                 perfectly well conditioned, and a 2x2 block is the smallest
 *                 pivot that is guaranteed to exist there.
 */
enum class Pivoting { None, BunchKaufman };

/** Counts of positive, negative and zero eigenvalues (Sylvester's law of
 *  inertia). Free from the signs of D once the factorization exists. */
struct Inertia {
  Index positive = 0;
  Index negative = 0;
  Index zero = 0;
};

}  // namespace supernodal_ldlt

/** \class SupernodalLDLT
 * \brief Supernodal LDL^T factorization of a sparse symmetric matrix, definite
 *        or indefinite, sharing its symbolic analysis with
 *        SupernodalLU/LeftRightLU.
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

  /** Numeric factorization. Under setPivoting(None) this reports NumericalIssue
   *  if the matrix turns out not to be positive definite -- see
   *  notPositiveDefiniteColumn(). The default handles that case instead. */
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
   *  factorization did not fail that way. Only Pivoting::None fails this way --
   *  Bunch-Kaufman handles a non-positive pivot rather than reporting it.
   *  Internal numbering: pass it through permutation() to reach the caller's. */
  Index notPositiveDefiniteColumn() const { return m_failColumn; }

  /** How many 1x1 pivots were perturbed away from zero by the last factorize()
   *  (see setStaticPivotThreshold). Always 0 under Pivoting::None, which reports
   *  a bad pivot instead of stepping over it. A count near n means the diagonal
   *  is unusable for this ordering and the answer should not be trusted without
   *  checking solveResidual(). */
  Index replacedPivots() const { return m_replacedPivots; }

  /** Number of 2x2 pivot blocks the last factorization used. Zero on a positive
   *  definite matrix; a positive count is the direct evidence that the matrix
   *  was indefinite and needed them. */
  Index pivotBlocks2x2() const { return m_pivot2x2Count; }

  /** How many times a 2x2 pivot was wanted at the LAST column of a supernode's
   *  diagonal block, where there is no second column to pair it with, and a
   *  perturbed 1x1 pivot was taken instead.
   *
   *  This is the one place the static-structure bargain costs accuracy rather
   *  than just time: a solver willing to grow its structure would delay the
   *  column into the parent supernode. Here the count is reported so the trade
   *  is visible, and refinement cleans up after it. A large count relative to
   *  supernodeCount() suggests a smaller setMaxBlockSize() is making blocks end
   *  in awkward places. */
  Index straddlingPivots() const { return m_straddlingPivots; }

  /** Counts of positive, negative and zero eigenvalues of A (Sylvester's law of
   *  inertia), read off the signs of D. Free once the factorization exists.
   *
   *  Exact for a factorization that perturbed nothing. Where static pivoting
   *  fired the inertia is that of the perturbed matrix, so read it next to
   *  replacedPivots(). */
  supernodal_ldlt::Inertia inertia() const { return m_inertia; }

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

  /** How many supernodes the last factorize() ran with intra-supernode
   *  parallelism rather than as one task on one lane. Always 0 for a serial
   *  executor or when setIntraSupernodeParallelism(false).
   *
   *  Expect a small number -- the root-separator chain only -- carrying a large
   *  share of the time. A count of 0 on a multi-lane run means every level was
   *  wide enough, or every panel too short, to leave alone. */
  Index intraParallelSupernodes() const { return m_intraSupernodes; }

  /** The fill-reducing permutation: indices()(i) is the internal index of
   *  original column i. */
  const PermutationType& permutation() const { return m_permutation; }

  // --- the factors themselves ------------------------------------------------
  //
  // These MATERIALIZE the factor rather than expose the panel arena, which is
  // the only honest option: the arena is a set of dense panels whose rows are
  // indirected through row blocks, and no Eigen expression maps onto it. They
  // cost an allocation and a pass over the factor, so they are for inspecting,
  // exporting or reusing the factorization, not for solving -- solve() reads the
  // panels directly and is what you want inside a loop.
  //
  // The identity they satisfy, with S = scalingS() and P = factorPermutation():
  //
  //     L * D * L^H  ==  P (S A S) P^T
  //
  // P is NOT permutation(): it also carries the local symmetric interchanges
  // Bunch-Kaufman chose inside each diagonal block. The two coincide exactly
  // when nothing was interchanged, which is every Pivoting::None factorization
  // and most positive definite ones.

  /** L: unit lower triangular, with its unit diagonal stored explicitly.
   *
   *  Includes the explicit zeros amalgamation introduces -- they are part of
   *  what the factor occupies, and nnzL() counts them the same way. */
  SparseMatrix<Scalar, ColMajor, StorageIndex> matrixL() const;

  /** D: block diagonal, 1x1 and 2x2 blocks. Equal to vectorD().asDiagonal()
   *  exactly when pivotBlocks2x2() is 0. */
  SparseMatrix<Scalar, ColMajor, StorageIndex> matrixD() const;

  /** The diagonal of D. This is the WHOLE of D only when pivotBlocks2x2() is 0;
   *  otherwise each 2x2 block also carries an off-diagonal that lives only in
   *  matrixD(). */
  Matrix<Scalar, Dynamic, 1> vectorD() const;

  /** The symmetric equilibration S, in the CALLER's numbering: A~ = S A S was
   *  what got factored. All ones when setEquilibration(false). */
  Matrix<RealScalar, Dynamic, 1> scalingS() const {
    Matrix<RealScalar, Dynamic, 1> s(m_size);
    for (StorageIndex i = 0; i < m_size; ++i) s[i] = m_scale[i];
    return s;
  }

  /** The permutation the FACTOR is expressed in: the fill-reducing ordering
   *  followed by the local symmetric interchanges. indices()(i) is the factor
   *  index of original column i. Equal to permutation() when no interchange
   *  happened. */
  PermutationType factorPermutation() const;

  /** Relative residual ||b - Ax|| / ||b|| measured by the last solve(). */
  RealScalar solveResidual() const { return m_lastSolveRelativeResidual; }
  /** Refinement steps taken by the last solve(). */
  Index iterativeRefinements() const { return m_lastRefinementIterations; }

  /** det(A), with the equilibration scaling divided back out. Overflows on
   *  moderately sized systems -- prefer logAbsDeterminant() with
   *  determinantSign(). */
  Scalar determinant() const {
    return determinantSign() * Scalar(numext::exp(logAbsDeterminant()));
  }

  /** Sign of det(A): +1 or -1, or 0 if a pivot block was singular.
   *
   *  L is unit triangular and the symmetric interchanges are applied to both
   *  sides, so neither contributes a sign -- det(A) has exactly the sign of
   *  det(D), and the equilibration scaling is positive. Every 2x2 Bunch-Kaufman
   *  block has a negative determinant by construction, so it contributes -1. */
  Scalar determinantSign() const {
    eigen_assert(m_factorized && "determinantSign() before a successful factorize()");
    int sign = 1;
    forEachPivotBlock([&](StorageIndex s, StorageIndex k, bool is2x2) {
      const RealScalar d = pivotBlockDeterminant(s, k, is2x2);
      if (d == RealScalar(0))
        sign = 0;
      else if (d < RealScalar(0))
        sign = -sign;
    });
    return Scalar(sign);
  }

  /** log|det(A)| accumulated as a sum of logs, so it stays finite where
   *  determinant() would overflow. A~ = S A S with a symmetric, positive
   *  scaling, so log|det A| = log|det D| - 2 sum log S. */
  RealScalar logAbsDeterminant() const {
    eigen_assert(m_factorized && "logAbsDeterminant() before a successful factorize()");
    RealScalar acc(0);
    forEachPivotBlock([&](StorageIndex s, StorageIndex k, bool is2x2) {
      acc += numext::log(numext::abs(pivotBlockDeterminant(s, k, is2x2)));
    });
    for (StorageIndex i = 0; i < m_size; ++i) acc -= RealScalar(2) * numext::log(m_scale[i]);
    return acc;
  }

  // --- options --------------------------------------------------------------

  /** How each dense diagonal block is factored; see supernodal_ldlt::Pivoting.
   *  Default **BunchKaufman**, which handles indefinite matrices.
   *
   *  Pivoting::None is the positive-definite fast path: no pivot search, no
   *  permutation to carry through the solve, and a non-positive pivot reported
   *  rather than handled. Choose it when you already know the matrix is SPD and
   *  have measured that the search costs you something. */
  void setPivoting(supernodal_ldlt::Pivoting mode) { m_pivoting = mode; }
  supernodal_ldlt::Pivoting pivoting() const { return m_pivoting; }

  /** Magnitude below which a 1x1 pivot is replaced by a same-sign value of this
   *  magnitude, so the factorization can continue past an (effectively) zero
   *  pivot. By default chosen automatically each factorize() as
   *  sqrt(eps) * max|A~_ij| of the equilibrated matrix: small enough to leave a
   *  usable pivot alone, large enough to step over a zero. Pass 0 to disable,
   *  which makes a zero pivot a reported failure instead.
   *
   *  Only Bunch-Kaufman perturbs; Pivoting::None reports. Perturbation is what
   *  refinement then repairs -- see setRefineOnlyIfPerturbed. */
  void setStaticPivotThreshold(const RealScalar& threshold) {
    m_staticPivotThreshold = threshold;
    m_thresholdIsAuto = false;
  }
  RealScalar staticPivotThreshold() const { return m_staticPivotThreshold; }

  /** Symmetric weighted matching before the ordering (Duff-Pralet), OFF by
   *  default. See SupernodalLDLTMatching.h for what it does and why the LU
   *  solvers' matching is not it.
   *
   *  Turn it on for a symmetric matrix whose DIAGONAL is the problem: a
   *  saddle-point or KKT system with a zero block, a mixed formulation, anything
   *  where Bunch-Kaufman ends up perturbing pivots or reports straddlingPivots().
   *  It pairs columns into 2x2 pivot candidates and keeps each pair together
   *  through the ordering, so the pivot the numeric phase wants is inside the
   *  supernode where it can be taken.
   *
   *  Off by default because it is not free and a positive definite matrix gets
   *  nothing from it: the diagonal is already the best pivot available, every
   *  pair the matching proposes is declined, and all that is left is the
   *  matching's own cost plus a quotient graph the ordering handles slightly
   *  worse than the original. replacedPivots() is the number to watch -- if it
   *  is 0, this has nothing to offer. */
  void setMatching(bool on) { m_matching = on; }
  bool matching() const { return m_matching; }

  /** 2x2 pivot candidate pairs the matching formed, or 0 when it did not run.
   *  Compare against pivotBlocks2x2(): the gap is the pairs the numeric phase
   *  looked at and declined, which is the matching working as intended rather
   *  than failing. */
  Index matchedPairs() const { return m_matchedPairs; }

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

  /** Stationary iterative refinement steps, x += M^-1 (b - Ax). Default 5, but
   *  gated on perturbation -- see setRefineOnlyIfPerturbed.
   *
   *  Stationary refinement is the right tool here only for small corrections;
   *  the natural Krylov method for a symmetric operator is conjugate gradients,
   *  or MINRES when it is indefinite, not the BiCGStab the unsymmetric siblings
   *  default to. */
  void setMaxIterativeRefinements(Index iters) { m_maxRefinementIterations = iters; }
  Index maxIterativeRefinements() const { return m_maxRefinementIterations; }
  void setRefinementTolerance(const RealScalar& tol) { m_refinementTolerance = tol; }

  /** Run refinement only when the factorization actually perturbed a pivot
   *  (replacedPivots() > 0). Default **true**.
   *
   *  An unperturbed LDL^T is backward stable, so there is nothing for refinement
   *  to repair and the first solve is already as good as the arithmetic allows;
   *  a perturbed one factored a slightly different matrix, and refinement is
   *  what recovers the difference. Gating on that keeps a positive definite
   *  solve at exactly the cost of the factor solve, while making the indefinite
   *  path robust without being asked. Set false to always refine. */
  void setRefineOnlyIfPerturbed(bool on) { m_refineOnlyIfPerturbed = on; }
  bool refineOnlyIfPerturbed() const { return m_refineOnlyIfPerturbed; }

  /** Relative-residual ceiling above which solve() downgrades info() to
   *  NumericalIssue instead of returning a bad answer silently. Default 1e-6. */
  void setSolveFailureThreshold(const RealScalar& tol) { m_solveFailureThreshold = tol; }
  RealScalar solveFailureThreshold() const { return m_solveFailureThreshold; }

  /** Dispatch the triangular sweeps across the executor, over elimination-tree
   *  levels. Results are unchanged (each supernode writes only its own rows). */
  void setParallelSolve(bool on) { m_parallelSolve = on; }
  bool parallelSolve() const { return m_parallelSolve; }

  /** Parallelize INSIDE a supernode on levels too narrow to fill the executor,
   *  on by default.
   *
   *  Level dispatch alone leaves the root separators -- a handful of supernodes
   *  that carry most of the work -- running on one lane. On such a level the
   *  supernodes are run sequentially and their panel operations are split across
   *  the pool instead: the Schur updates by disjoint target-panel row ranges, the
   *  panel solve by independent rows. Chunks write disjoint outputs and leave
   *  each element's accumulation order unchanged, so the answer does not depend
   *  on the setting. */
  void setIntraSupernodeParallelism(bool on) { m_intraParallel = on; }
  bool intraSupernodeParallelism() const { return m_intraParallel; }

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
    m_maxRefinementIterations = 5;
    m_refineOnlyIfPerturbed = true;
    m_refinementTolerance = NumTraits<RealScalar>::epsilon();
    m_solveFailureThreshold = RealScalar(1e-6);
    m_parallelSolve = true;
    m_intraParallel = true;
    m_matching = false;
    m_matchedPairs = 0;
    m_pivoting = supernodal_ldlt::Pivoting::BunchKaufman;
    m_staticPivotThreshold = RealScalar(0);
    m_thresholdIsAuto = true;
    m_replacedPivots = 0;
    m_pivot2x2Count = 0;
    m_straddlingPivots = 0;
    m_intraSupernodes = 0;
    m_inertia = supernodal_ldlt::Inertia();
    m_lastSolveRelativeResidual = RealScalar(0);
    m_lastRefinementIterations = 0;
    m_nnzL = 0;
    m_isInitialized = false;
  }

  // --- D as a block diagonal of 1x1 and 2x2 pivots --------------------------
  //
  // A 1x1 pivot d sits at diag(k,k). A 2x2 pivot occupying local columns (k,k+1)
  // stores its two diagonal entries at diag(k,k) and diag(k+1,k+1) and its
  // off-diagonal at diag(k+1,k) -- the slot L's implicit unit diagonal leaves
  // free, which is the LAPACK packed convention. m_pivotKind says which:
  //   1 = a 1x1 pivot, 2 = the leading column of a 2x2, 0 = its trailing column.

  bool is2x2Leader(StorageIndex internalColumn) const {
    return m_pivotKind[static_cast<std::size_t>(internalColumn)] == 2;
  }

  /** Visit every pivot block once, as (supernode, local column, is2x2). */
  template <typename F>
  void forEachPivotBlock(F&& body) const {
    for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
      const Supernode& sn = m_supernodes[s];
      for (StorageIndex k = 0; k < sn.width();) {
        const bool two = is2x2Leader(sn.firstColumn + k);
        body(s, k, two);
        k += two ? 2 : 1;
      }
    }
  }

  /** det of one pivot block. For a 2x2 [[a, conj(c)],[c, b]] that is
   *  a*b - |c|^2, which Bunch-Kaufman guarantees is negative. */
  RealScalar pivotBlockDeterminant(StorageIndex s, StorageIndex k, bool is2x2) const {
    const ConstStridedPanel diag = diagBlock(s);
    if (!is2x2) return numext::real(diag(k, k));
    const RealScalar a = numext::real(diag(k, k)), b = numext::real(diag(k + 1, k + 1));
    const Scalar c = m_dOffDiag[static_cast<std::size_t>(m_supernodes[s].firstColumn + k)];
    return a * b - numext::abs2(c);
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

  /** Numbering from the symmetric matching: pairs contracted, the quotient graph
   *  ordered and postordered, then expanded so each pair lands on two ADJACENT
   *  columns. Fills m_toInternal and returns true; returns false (leaving
   *  m_toInternal untouched) when the matching found no pair worth keeping, in
   *  which case the ordinary path is both cheaper and no worse. */
  bool computeMatchedNumbering(const MatrixType& matrix);

  // Symmetric adjacency in the internal numbering. The input holds one triangle,
  // and every stored off-diagonal entry contributes BOTH directions, so this is
  // the full elimination graph whichever triangle UpLo selects.
  void buildSymmetricAdjacency(const MatrixType& matrix,
                               std::vector<std::vector<StorageIndex>>& adjacency) const;

  // Symmetric Ruiz equilibration: one scaling vector applied on both sides.
  void computeEquilibration(const MatrixType& matrix);

  /** Everything one update needs to borrow rather than allocate. Caller-owned
   *  rather than a member: these run concurrently across a level, so a shared
   *  buffer would race -- and allocating instead would SERIALIZE, which is the
   *  more expensive mistake of the two. An LU update kernel gets to work
   *  entirely in block views of its two arenas; an LDL^T one has to form
   *  D * L^H somewhere, and doing that through the allocator once per source
   *  block is what flattens the parallel speedup. */
  struct UpdateScratch {
    std::vector<StorageIndex> destRow;  // panel positions of the below-facing blocks
    std::vector<Scalar> scaled;         // D_src * L_cb^H for the current column block
  };

  void applyUpdate(StorageIndex source, StorageIndex target, StorageIndex firstFacingBlock,
                   UpdateScratch& scratch);

  // Intra-parallel variant for the big root supernodes: applies ALL sources with
  // ONE fork-join dispatch over the target's panel rows rather than running the
  // sources on a single lane. Each chunk walks the sources in order and applies
  // only the slice of each update landing in its own rows, so chunks write
  // disjoint elements and every element still accumulates in source order.
  // Only legal from a sequential caller -- the pool is fork-join, not nestable.
  void applyAllUpdatesChunked(StorageIndex target);

  // Largest rows of a chunk when splitting a panel operation across the pool:
  // big enough that dispatch overhead is negligible against the chunk's BLAS-3
  // work, small enough to balance the tall root panels. A CEILING, not the chunk
  // size -- see intraChunkRows().
  static constexpr Index kIntraChunkSize = 128;
  // Floor on chunk extent: below this a panel GEMM is a strip too thin to
  // amortize either the BLAS call or the per-chunk walk over the update sources.
  static constexpr Index kMinIntraChunkSize = 32;
  // How far from "wide enough to fill the lanes" a MULTI-supernode level may be
  // and still switch to intra-supernode parallelism. A level of ONE supernode
  // qualifies unconditionally: there is no inter-supernode parallelism to give
  // up. These are SupernodalLU's tuned values, and they carry over because the
  // quantity they trade off -- panel rows per lane against dispatch overhead --
  // is the same in both solvers; only the number of panels per supernode differs.
  static constexpr Index kIntraLevelSlack = 8;

  // Chunk extent for splitting `total` rows across the pool. Aiming at one chunk
  // per lane is what keeps this from being capped by the ceiling: a fixed extent
  // would give ceil(total/128) chunks no matter how many lanes exist.
  Index intraChunkRows(Index total) const {
    const Index lanes = static_cast<Index>(m_executor.concurrency());
    if (lanes <= 1 || total <= 0) return numext::maxi(Index(1), total);
    const Index perLane = (total + lanes - 1) / lanes;
    return numext::maxi(kMinIntraChunkSize, numext::mini(kIntraChunkSize, perLane));
  }

  // Split [0, total) into contiguous chunks of at most `chunkRows` and dispatch
  // body(start, length) for each. Runs inline when only one chunk remains.
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

  // D_src * L_cb^H for one facing column block of `source`: the factor every row
  // block of that column block multiplies against. Formed once per column block
  // so the block-diagonal scaling never enters a GEMM inner loop, and a 2x2 pivot
  // MIXES the two rows rather than scaling them. Writes into `scratch`, which the
  // returned map views, so a caller in a parallel region allocates at most once.
  typedef Map<DenseMatrix> PlainPanel;
  PlainPanel scaledSourceBlock(StorageIndex source, const RowBlock& colBlock,
                               std::vector<Scalar>& scratch) const;

  /** Per-supernode outcome of the numeric phase, kept in a disjoint slot per
   *  supernode so a level can be factored concurrently without shared writes. */
  struct SupernodeResult {
    bool notPositiveDefinite = false;  // Pivoting::None only
    bool singular = false;             // a zero pivot that perturbation was not allowed to fix
    StorageIndex failColumn = 0;       // block-local on the way out, internal after
    Index replaced = 0;
    Index blocks2x2 = 0;
    Index straddling = 0;
  };

  void factorizeSupernode(StorageIndex s, const RealScalar& staticPivot, SupernodeResult& result,
                          bool intraParallel);
  /** 1x1 pivots taken in order, no search. Positive definite input only. */
  void factorizeDiagonalBlockUnpivoted(StridedPanel diag, signed char* pivotKind,
                                       SupernodeResult& result) const;
  /** Bounded Bunch-Kaufman: 1x1 and 2x2 pivots with symmetric interchanges
   *  confined to the block, so the global symbolic structure is untouched. */
  void factorizeDiagonalBlockBunchKaufman(StridedPanel diag, const RealScalar& staticPivot,
                                          std::vector<StorageIndex>& perm, signed char* pivotKind,
                                          Scalar* dOffDiag, SupernodeResult& result) const;

  /** Symmetric interchange of local indices i < j within a diagonal block.
   *  Swaps rows AND columns, so the block stays symmetric and only its lower
   *  triangle is touched. */
  void symmetricSwap(StridedPanel diag, StorageIndex i, StorageIndex j) const;

  /** head := P_s head, the local symmetric interchange of supernode s, applied
   *  to that supernode's own rows of the right-hand side. Its inverse is
   *  unpermuteHead. Both are no-ops when nothing moved. */
  template <typename Dest>
  void permuteHead(StorageIndex s, Dest& head) const {
    const std::vector<StorageIndex>& perm = m_diagPivot[s];
    if (perm.empty()) return;
    DenseMatrix tmp = head;
    for (StorageIndex k = 0; k < static_cast<StorageIndex>(perm.size()); ++k)
      head.row(k) = tmp.row(perm[k]);
  }
  template <typename Dest>
  void unpermuteHead(StorageIndex s, Dest& head) const {
    const std::vector<StorageIndex>& perm = m_diagPivot[s];
    if (perm.empty()) return;
    DenseMatrix tmp = head;
    for (StorageIndex k = 0; k < static_cast<StorageIndex>(perm.size()); ++k)
      head.row(perm[k]) = tmp.row(k);
  }

  /** toFactor[c] is the row/column the factor gives internal column c, i.e. the
   *  internal numbering with each supernode's local interchange folded in.
   *  Identity wherever no interchange happened, which is usually everywhere. */
  void buildFactorIndexMap(std::vector<StorageIndex>& toFactor) const {
    toFactor.resize(static_cast<std::size_t>(m_size));
    for (StorageIndex c = 0; c < m_size; ++c) toFactor[c] = c;
    for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
      const std::vector<StorageIndex>& perm = m_diagPivot[s];
      if (perm.empty()) continue;
      const StorageIndex first = m_supernodes[s].firstColumn;
      // perm[k] is the local index now sitting at position k, so it is position
      // k that the factor gives it.
      for (StorageIndex k = 0; k < static_cast<StorageIndex>(perm.size()); ++k)
        toFactor[first + perm[k]] = first + k;
    }
  }

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

  // Per internal column: 1 = a 1x1 pivot, 2 = leading column of a 2x2, 0 = its
  // trailing column. Under Pivoting::None every entry is 1.
  std::vector<signed char> m_pivotKind;
  // The off-diagonal entry of each 2x2 pivot, indexed by its LEADING internal
  // column. It is held here rather than in the (k+1, k) slot of the diagonal
  // block -- the packed LAPACK convention -- because that slot is also what
  // triangularView<UnitLower>() reads as L(k+1, k), and for a 2x2 pivot L has a
  // 2x2 identity there. Keeping D out of the block lets every triangular solve
  // use the plain unit-lower view with nothing to mask.
  std::vector<Scalar> m_dOffDiag;
  // Per supernode, the local symmetric interchange chosen inside its diagonal
  // block: perm[k] is the block-local index now sitting at position k. Empty
  // when nothing moved, which is the common case and the cheap one.
  std::vector<std::vector<StorageIndex>> m_diagPivot;

  std::vector<RealScalar> m_scale;  // symmetric equilibration, original numbering
  MatrixType m_originalMatrix;

  bool m_equilibrate, m_parallelSolve, m_intraParallel, m_matching;
  Index m_matchedPairs;
  Index m_relaxedSize, m_maxAmalgamationZeroRows;
  double m_amalgamationFillFraction;
  Index m_maxBlockSize, m_maxFactorNonzeros;
  Index m_maxRefinementIterations;
  bool m_refineOnlyIfPerturbed;
  supernodal_ldlt::Pivoting m_pivoting;
  RealScalar m_staticPivotThreshold;
  bool m_thresholdIsAuto;
  Index m_replacedPivots, m_pivot2x2Count, m_straddlingPivots, m_intraSupernodes;
  supernodal_ldlt::Inertia m_inertia;
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

// The matching path, end to end: pair, contract, order the quotient, expand.
//
// Everything here happens on the QUOTIENT graph rather than on A, and that is the
// whole trick. Ordering A and hoping the pairs survive does not work: a pair is
// connected, so one of its columns is the other's parent in the elimination tree,
// and a postorder only puts them side by side if the child happens to be visited
// last. Contracting first makes "side by side" a property of the construction
// instead of a coincidence.
//
// The expanded numbering is still a valid postorder of the expanded elimination
// tree, which is what the supernode partition downstream wants. Each quotient
// node expands to a CHAIN of one or two columns -- the leader's parent is its
// partner, since they are adjacent -- and a quotient node's descendants are
// exactly the expansions of its quotient descendants, all of which the quotient
// postorder already placed before it. So no further postorder is needed, and the
// caller must not apply one: it would undo this.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
bool SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::computeMatchedNumbering(
    const MatrixType& matrix) {
  const StorageIndex n = m_size;
  namespace symbolic = supernodal_lu::symbolic;

  // The matching reads rows and columns alike, so it needs both triangles: a
  // single stored triangle is a different, unsymmetric problem and would be
  // matched as one.
  MatrixType full = matrix.template selfadjointView<UpLo>();
  std::vector<StorageIndex> partner;
  m_matchedPairs = supernodal_ldlt::symmetricMatchingPairs(full, partner);
  if (m_matchedPairs == 0) return false;

  // 1) contract each pair to one quotient vertex.
  std::vector<StorageIndex> quotientOf(static_cast<std::size_t>(n), StorageIndex(-1));
  std::vector<StorageIndex> leader, follower;  // members of each quotient vertex
  leader.reserve(static_cast<std::size_t>(n));
  follower.reserve(static_cast<std::size_t>(n));
  for (StorageIndex i = 0; i < n; ++i) {
    if (quotientOf[i] != StorageIndex(-1)) continue;
    const StorageIndex c = static_cast<StorageIndex>(leader.size());
    quotientOf[i] = c;
    leader.push_back(i);
    const StorageIndex p = partner[static_cast<std::size_t>(i)];
    if (p >= 0) {
      quotientOf[p] = c;
      follower.push_back(p);
    } else {
      follower.push_back(StorageIndex(-1));
    }
  }
  const StorageIndex nc = static_cast<StorageIndex>(leader.size());

  // 2) quotient adjacency, in the original quotient numbering.
  std::vector<std::vector<StorageIndex>> quotientAdj(static_cast<std::size_t>(nc));
  for (StorageIndex j = 0; j < n; ++j) {
    const StorageIndex cj = quotientOf[j];
    for (typename MatrixType::InnerIterator it(full, j); it; ++it) {
      const StorageIndex ci = quotientOf[static_cast<StorageIndex>(it.index())];
      if (ci == cj) continue;
      quotientAdj[cj].push_back(ci);
      quotientAdj[ci].push_back(cj);
    }
  }
  for (StorageIndex c = 0; c < nc; ++c) {
    std::vector<StorageIndex>& row = quotientAdj[c];
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }

  // 3) fill-reducing ordering OF THE QUOTIENT. The functor wants a matrix, and
  //    only its pattern is read, so the values are placeholders.
  MatrixType quotient(nc, nc);
  {
    std::vector<Triplet<Scalar, StorageIndex>> entries;
    std::size_t count = static_cast<std::size_t>(nc);
    for (StorageIndex c = 0; c < nc; ++c) count += quotientAdj[c].size();
    entries.reserve(count);
    for (StorageIndex c = 0; c < nc; ++c) {
      entries.emplace_back(c, c, Scalar(1));
      for (StorageIndex v : quotientAdj[c]) entries.emplace_back(v, c, Scalar(1));
    }
    quotient.setFromTriplets(entries.begin(), entries.end());
  }
  PermutationType orderingPerm;
  m_orderingFunctor(quotient, orderingPerm);
  std::vector<StorageIndex> toOrdered(static_cast<std::size_t>(nc));
  if (orderingPerm.size() == 0) {
    for (StorageIndex c = 0; c < nc; ++c) toOrdered[c] = c;
  } else {
    for (StorageIndex c = 0; c < nc; ++c) toOrdered[orderingPerm.indices()(c)] = c;
  }

  // 4) elimination tree and postorder of the quotient, in that numbering.
  std::vector<std::vector<StorageIndex>> orderedAdj(static_cast<std::size_t>(nc));
  for (StorageIndex c = 0; c < nc; ++c) {
    std::vector<StorageIndex>& row = orderedAdj[toOrdered[c]];
    row.reserve(quotientAdj[c].size());
    for (StorageIndex v : quotientAdj[c]) row.push_back(toOrdered[v]);
    std::sort(row.begin(), row.end());
  }
  std::vector<StorageIndex> parent, postorder;
  symbolic::computeEliminationTree(nc, orderedAdj, parent);
  symbolic::computePostorder(nc, parent, postorder);

  // 5) expand, leader immediately before its partner.
  std::vector<StorageIndex> fromOrdered(static_cast<std::size_t>(nc));
  for (StorageIndex c = 0; c < nc; ++c) fromOrdered[toOrdered[c]] = c;
  StorageIndex next = 0;
  for (StorageIndex t = 0; t < nc; ++t) {
    const StorageIndex c = fromOrdered[postorder[t]];
    m_toInternal[leader[c]] = next++;
    if (follower[c] >= 0) m_toInternal[follower[c]] = next++;
  }
  eigen_assert(next == n && "matched numbering did not cover every column");
  return true;
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

  namespace symbolic = supernodal_lu::symbolic;
  m_matchedPairs = 0;

  // 1) the numbering: ordering plus postorder, from one of two paths. The
  //    matching path does both on the quotient graph and arrives with the
  //    numbering finished -- applying a second postorder to it would separate
  //    the pairs it exists to keep together.
  std::vector<std::vector<StorageIndex>> adjacency;
  std::vector<StorageIndex> parent;
  if (!(m_matching && computeMatchedNumbering(matrix))) {
    // Eigen's AMD/METIS functors symmetrize whatever they are given, so passing a
    // single triangle orders the same graph the factorization will eliminate.
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

    // Postorder the elimination tree and fold it into the numbering, so that
    // supernodes come out contiguous.
    buildSymmetricAdjacency(matrix, adjacency);
    symbolic::computeEliminationTree(n, adjacency, parent);
    std::vector<StorageIndex> postorder;
    symbolic::computePostorder(n, parent, postorder);
    std::vector<StorageIndex> relabel(n);
    for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
    for (StorageIndex i = 0; i < n; ++i) m_toInternal[i] = relabel[m_toInternal[i]];
  }

  // 4) adjacency + tree in the final numbering.
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
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorizeDiagonalBlockUnpivoted(
    StridedPanel diag, signed char* pivotKind, SupernodeResult& result) const {
  const StorageIndex w = static_cast<StorageIndex>(diag.rows());

  for (StorageIndex j0 = 0; j0 < w; j0 += kDiagBlockSize) {
    const StorageIndex jb = std::min<StorageIndex>(kDiagBlockSize, w - j0);
    const StorageIndex panelEnd = j0 + jb;

    // (1) unblocked LDL^T over the panel's columns, updating all rows below so
    //     that L21 is complete when the trailing update needs it.
    for (StorageIndex k = j0; k < panelEnd; ++k) {
      const RealScalar d = numext::real(diag(k, k));
      if (!(d > RealScalar(0))) {
        result.notPositiveDefinite = true;
        result.failColumn = k;
        return;
      }
      diag(k, k) = Scalar(d);  // D is real even for a Hermitian input
      pivotKind[k] = 1;
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

// Symmetric interchange of local indices i < j. Only the LOWER triangle of the
// block is stored, so a swap of row i with row j and column i with column j
// touches four disjoint regions, and the piece between i and j is REFLECTED
// across the diagonal -- entry (m, i) and entry (j, m) are the same element of
// the symmetric matrix seen from the two sides, which is why that stretch swaps
// against the conjugate rather than straight across.
//
// Note the extent of (a): EVERY column left of i, not just the factored ones.
// Those columns hold L where they are already factored and active values where
// they are not, and both have to move -- a 2x2 pivot swaps at i = k+1 while
// column k is still active and is part of the pivot being formed, so stopping at
// the factored boundary would silently leave that column behind. (A 1x1 pivot
// swaps at i = k, where the two extents coincide, which is what makes this easy
// to get wrong and hard to notice.)
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::symmetricSwap(
    StridedPanel diag, StorageIndex i, StorageIndex j) const {
  if (i == j) return;
  eigen_assert(i < j);
  const StorageIndex w = static_cast<StorageIndex>(diag.rows());

  // (a) every column left of i: L where factored, active values otherwise.
  if (i > 0) diag.row(i).segment(0, i).swap(diag.row(j).segment(0, i));

  // (b) the two diagonal entries (both real for a Hermitian block).
  std::swap(diag(i, i), diag(j, j));

  // (c) the reflected stretch strictly between i and j.
  for (StorageIndex m = i + 1; m < j; ++m) {
    const Scalar t = diag(m, i);
    diag(m, i) = numext::conj(diag(j, m));
    diag(j, m) = numext::conj(t);
  }

  // (d) the two column tails below j.
  const StorageIndex below = w - j - 1;
  if (below > 0)
    diag.col(i).segment(j + 1, below).swap(diag.col(j).segment(j + 1, below));
}

// Bounded Bunch-Kaufman. At each step the choice is between a 1x1 pivot and a
// 2x2 one, and the criterion bounds the growth of L either way -- which is the
// whole point, because a symmetric matrix can have an arbitrarily small diagonal
// and still be perfectly well conditioned, so "divide by the diagonal" is not an
// option the way it is for a positive definite matrix.
//
// Deliberately UNBLOCKED. The blocked form (LAPACK's xSYTRF) exists because a
// dense factorization has no other source of BLAS-3 work; here the block is
// capped by setMaxBlockSize (128 by default) and the BLAS-3 work that matters is
// the off-diagonal panel, so an unblocked kernel on a small block costs a worse
// constant on a term that is not the bottleneck.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorizeDiagonalBlockBunchKaufman(
    StridedPanel diag, const RealScalar& staticPivot, std::vector<StorageIndex>& perm,
    signed char* pivotKind, Scalar* dOffDiag, SupernodeResult& result) const {
  const StorageIndex w = static_cast<StorageIndex>(diag.rows());
  // (1 + sqrt(17)) / 8: the constant that minimises the bound on element growth.
  const RealScalar alpha = (RealScalar(1) + numext::sqrt(RealScalar(17))) / RealScalar(8);

  std::vector<StorageIndex> local(static_cast<std::size_t>(w));
  for (StorageIndex k = 0; k < w; ++k) local[k] = k;
  bool anySwap = false;

  StorageIndex k = 0;
  while (k < w) {
    const StorageIndex below = w - k - 1;

    // omega_k: the largest subdiagonal magnitude in column k, and where it is.
    RealScalar omegaK(0);
    StorageIndex r = k;
    if (below > 0) {
      Index rel = 0;
      omegaK = diag.col(k).segment(k + 1, below).cwiseAbs().maxCoeff(&rel);
      r = k + 1 + static_cast<StorageIndex>(rel);
    }
    const RealScalar absAkk = numext::abs(diag(k, k));

    StorageIndex pivotSize = 1;
    StorageIndex swapWith = k;
    if (omegaK > RealScalar(0)) {
      if (absAkk >= alpha * omegaK) {
        pivotSize = 1;  // the diagonal is big enough on its own
      } else {
        // omega_r: the largest off-diagonal magnitude in row/column r, read from
        // both sides of the diagonal because only the lower triangle is stored.
        RealScalar omegaR(0);
        for (StorageIndex m = k; m < r; ++m) omegaR = numext::maxi(omegaR, numext::abs(diag(r, m)));
        if (r + 1 < w)
          omegaR = numext::maxi(omegaR, diag.col(r).segment(r + 1, w - r - 1).cwiseAbs().maxCoeff());

        if (absAkk * omegaR >= alpha * omegaK * omegaK)
          pivotSize = 1;  // still fine at k once the whole row is accounted for
        else if (numext::abs(diag(r, r)) >= alpha * omegaR)
          pivotSize = 1, swapWith = r;  // r's diagonal is usable instead
        else
          pivotSize = 2, swapWith = r;  // neither is: take the 2x2 spanning k and r
      }
    }

    // A 2x2 pivot needs a second column inside THIS block. At the last column
    // there is none, and growing the structure to find one is exactly what this
    // solver refuses to do -- so take a perturbed 1x1 instead and record it.
    if (pivotSize == 2 && k + 1 >= w) {
      pivotSize = 1;
      swapWith = k;
      ++result.straddling;
    }

    if (pivotSize == 1) {
      if (swapWith != k) {
        symmetricSwap(diag, k, swapWith);
        std::swap(local[k], local[swapWith]);
        anySwap = true;
      }
      Scalar d = diag(k, k);
      const RealScalar absd = numext::abs(d);
      if (staticPivot > RealScalar(0) && absd < staticPivot) {
        // Bump to the threshold, keeping the sign: the inertia of the perturbed
        // matrix should still reflect which side of zero the pivot was on.
        d = (absd == RealScalar(0)) ? Scalar(staticPivot) : d * (staticPivot / absd);
        diag(k, k) = d;
        ++result.replaced;
      }
      if (d == Scalar(0)) {
        result.singular = true;
        result.failColumn = k;
        return;
      }
      pivotKind[k] = 1;
      if (below > 0) {
        auto col = diag.col(k).segment(k + 1, below);
        col /= d;
        diag.block(k + 1, k + 1, below, below).template triangularView<Lower>() -=
            (col * d) * col.adjoint();
      }
      k += 1;
    } else {
      // Bring r alongside k so the 2x2 occupies consecutive columns.
      if (swapWith != k + 1) {
        symmetricSwap(diag, k + 1, swapWith);
        std::swap(local[k + 1], local[swapWith]);
        anySwap = true;
      }
      // D = [[a, conj(c)], [c, b]]. Move c out of the block: the (k+1, k) slot is
      // what the unit-lower view reads as L(k+1, k), and L is the identity across
      // a 2x2 pivot, so it has to be left at zero.
      const Scalar a = diag(k, k), c = diag(k + 1, k), b = diag(k + 1, k + 1);
      dOffDiag[k] = c;
      diag(k + 1, k) = Scalar(0);
      const RealScalar det = numext::real(a) * numext::real(b) - numext::abs2(c);
      if (det == RealScalar(0)) {
        result.singular = true;
        result.failColumn = k;
        return;
      }
      pivotKind[k] = 2;
      pivotKind[k + 1] = 0;
      ++result.blocks2x2;

      const StorageIndex below2 = w - k - 2;
      if (below2 > 0) {
        // L(k+2:, k:k+1) = A(k+2:, k:k+1) * D^-1, with
        //   D^-1 = (1/det) [[b, -conj(c)], [-c, a]].
        auto c0 = diag.col(k).segment(k + 2, below2);
        auto c1 = diag.col(k + 1).segment(k + 2, below2);
        DenseMatrix pair(below2, 2);
        pair.col(0) = c0;
        pair.col(1) = c1;
        c0 = (pair.col(0) * Scalar(numext::real(b)) - pair.col(1) * c) / Scalar(det);
        c1 = (pair.col(1) * Scalar(numext::real(a)) - pair.col(0) * numext::conj(c)) / Scalar(det);

        // Trailing update A22 -= L21 D L21^H. L21 is (below2 x 2) and D is the
        // 2x2 block, so forming D * L21^H once keeps this one small GEMM.
        DenseMatrix dl(2, below2);
        dl.row(0) = Scalar(numext::real(a)) * c0.adjoint() + numext::conj(c) * c1.adjoint();
        dl.row(1) = c * c0.adjoint() + Scalar(numext::real(b)) * c1.adjoint();
        DenseMatrix l21(below2, 2);
        l21.col(0) = c0;
        l21.col(1) = c1;
        diag.block(k + 2, k + 2, below2, below2).template triangularView<Lower>() -= l21 * dl;
      }
      k += 2;
    }
  }
  if (anySwap) perm = std::move(local);  // empty => identity
}

// Subtract one source supernode's Schur contribution from a target.
//
// The LU form of this walks BOTH an L-side and a U-side; here the source sends
// only -L_R D L_R^H, so there is one side, and within the target's diagonal block
// only the lower triangle is computed. That is the whole factor-of-two in the
// numeric phase.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
typename SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::PlainPanel
SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::scaledSourceBlock(
    StorageIndex source, const RowBlock& colBlock, std::vector<Scalar>& scratch) const {
  const Supernode& src = m_supernodes[source];
  const StorageIndex wSrc = src.width(), cc = colBlock.height();
  const ConstStridedPanel srcLower = lowerPanel(source);
  const ConstStridedPanel srcDiag = diagBlock(source);
  const signed char* srcPivotKind = m_pivotKind.data() + static_cast<std::size_t>(src.firstColumn);
  const Scalar* srcDOffDiag = m_dOffDiag.data() + static_cast<std::size_t>(src.firstColumn);

  const std::size_t need = static_cast<std::size_t>(wSrc) * static_cast<std::size_t>(cc);
  if (scratch.size() < need) scratch.resize(need);
  PlainPanel dl(scratch.data(), wSrc, cc);
  dl = srcLower.block(colBlock.panelOffset, 0, cc, wSrc).adjoint();
  for (StorageIndex q = 0; q < wSrc;) {
    if (srcPivotKind[q] != 2) {
      dl.row(q) *= srcDiag(q, q);
      q += 1;
    } else {
      // D = [[a, conj(c)], [c, b]] mixes the two rows rather than scaling them.
      // Done entry by entry: the two-row temporary the expression form needs
      // would be another allocation, in the hottest loop there is.
      const Scalar a = Scalar(numext::real(srcDiag(q, q))), c = srcDOffDiag[q],
                   b = Scalar(numext::real(srcDiag(q + 1, q + 1)));
      const Scalar cbar = numext::conj(c);
      for (StorageIndex col = 0; col < cc; ++col) {
        const Scalar x0 = dl(q, col), x1 = dl(q + 1, col);
        dl(q, col) = a * x0 + cbar * x1;
        dl(q + 1, col) = c * x0 + b * x1;
      }
      q += 2;
    }
  }
  return dl;  // wSrc x cc
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::applyUpdate(
    StorageIndex source, StorageIndex target, StorageIndex firstFacingBlock,
    UpdateScratch& scratch) {
  const Supernode& src = m_supernodes[source];
  const StorageIndex wSrc = src.width();
  const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
  const StorageIndex targetFirstColumn = m_supernodes[target].firstColumn;

  const StridedPanel srcLower = lowerPanel(source);
  StridedPanel targetDiag = diagBlock(target);
  StridedPanel targetLower = lowerPanel(target);

  // Blocks of the source facing the target are consecutive: [firstFacing, lastFacing).
  StorageIndex lastFacing = firstFacingBlock;
  while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;

  // Hoist the panel-position search for the below-facing blocks: it depends on
  // the row block alone, and the column-block loop below would otherwise repeat
  // it once per facing column block.
  scratch.destRow.clear();
  for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb)
    scratch.destRow.push_back(rowPanelPosition(target, m_rowBlocks[rb].firstRow));

  // COLUMN BLOCK OUTER: D_src * L_cb^H is formed once per column block and every
  // row block reuses it, so the diagonal scaling never enters a GEMM inner loop.
  for (StorageIndex cb = firstFacingBlock; cb < lastFacing; ++cb) {
    const RowBlock& colBlock = m_rowBlocks[cb];
    const StorageIndex cc = colBlock.height();
    const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;

    const PlainPanel dl = scaledSourceBlock(source, colBlock, scratch.scaled);  // wSrc x cc

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
      const StorageIndex destRow = scratch.destRow[static_cast<std::size_t>(rb - lastFacing)];
      targetLower.block(destRow, targetColStart, rc, cc).noalias() -=
          srcLower.block(rowBlock.panelOffset, 0, rc, wSrc) * dl;
    }
  }
}

// All of a target's updates, with the off-diagonal panel split across the pool.
//
// The split is by TARGET PANEL ROWS, which is the only axis that keeps the
// outputs disjoint: two different sources write the same target rows, so
// chunking by source would collide, while a row range of the target's panel is
// written by exactly one chunk. Within a chunk the sources are still walked in
// order, and for a fixed source exactly one (column block, row block) pair
// reaches any given element -- so each element sees the same subtractions in the
// same order as the serial sweep.
//
// The one thing this repeats rather than shares is D_src * L_cb^H, which every
// chunk recomputes for each source's column blocks. That is wSrc x cc of work
// against the chunk's rows x cc x wSrc of GEMM, and the chunk extent never falls
// below kMinIntraChunkSize, so the repeat is bounded by ~1/32 of the work it
// feeds -- cheaper than materializing every source's scaled block up front,
// which for a root supernode is the size of the panel again.
template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::applyAllUpdatesChunked(
    StorageIndex target) {
  const std::vector<UpdateSource>& sources = m_updateSources[target];
  if (sources.empty()) return;
  const Supernode& tn = m_supernodes[target];
  const StorageIndex targetFirstColumn = tn.firstColumn;
  const Index offDiag = static_cast<Index>(tn.offDiagonalRowCount);
  StridedPanel targetDiag = diagBlock(target);
  StridedPanel targetLower = lowerPanel(target);

  // Pass 1 (serial): the target's diagonal block. At most width x width per
  // source, and disjoint from the panel, so there is nothing to gain by
  // splitting it and no ordering constraint against pass 2.
  std::vector<Scalar> dlScratch;
  for (const UpdateSource& u : sources) {
    const Supernode& src = m_supernodes[u.sourceSupernode];
    const StorageIndex wSrc = src.width();
    const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
    const StridedPanel srcLower = lowerPanel(u.sourceSupernode);
    StorageIndex lastFacing = u.facingRowBlock;
    while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target) ++lastFacing;
    for (StorageIndex cb = u.facingRowBlock; cb < lastFacing; ++cb) {
      const RowBlock& colBlock = m_rowBlocks[cb];
      const StorageIndex cc = colBlock.height();
      const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;
      const PlainPanel dl = scaledSourceBlock(u.sourceSupernode, colBlock, dlScratch);
      for (StorageIndex rb = cb; rb < lastFacing; ++rb) {
        const RowBlock& rowBlock = m_rowBlocks[rb];
        const StorageIndex rc = rowBlock.height();
        const StorageIndex destRow = rowBlock.firstRow - targetFirstColumn;
        const auto lower = srcLower.block(rowBlock.panelOffset, 0, rc, wSrc);
        if (rb == cb)
          targetDiag.block(destRow, targetColStart, rc, cc).template triangularView<Lower>() -=
              lower * dl;
        else
          targetDiag.block(destRow, targetColStart, rc, cc).noalias() -= lower * dl;
      }
    }
  }

  // Pass 2: the off-diagonal panel, chunked by disjoint target row ranges.
  parallelChunks(offDiag, intraChunkRows(offDiag), [&](Index chunkStart, Index chunkLen) {
    const Index chunkEnd = chunkStart + chunkLen;
    std::vector<Scalar> chunkScratch;  // one per chunk, not one per source block
    for (const UpdateSource& u : sources) {
      const Supernode& src = m_supernodes[u.sourceSupernode];
      const StorageIndex wSrc = src.width();
      const StorageIndex lastBlock = src.firstRowBlock + src.rowBlockCount;
      const StridedPanel srcLower = lowerPanel(u.sourceSupernode);
      StorageIndex lastFacing = u.facingRowBlock;
      while (lastFacing < lastBlock && m_rowBlocks[lastFacing].facingSupernode == target)
        ++lastFacing;
      for (StorageIndex cb = u.facingRowBlock; cb < lastFacing; ++cb) {
        const RowBlock& colBlock = m_rowBlocks[cb];
        const StorageIndex cc = colBlock.height();
        const StorageIndex targetColStart = colBlock.firstRow - targetFirstColumn;
        const PlainPanel dl = scaledSourceBlock(u.sourceSupernode, colBlock, chunkScratch);
        for (StorageIndex rb = lastFacing; rb < lastBlock; ++rb) {
          const RowBlock& rowBlock = m_rowBlocks[rb];
          const Index destRow = static_cast<Index>(rowPanelPosition(target, rowBlock.firstRow));
          if (destRow >= chunkEnd) break;  // panel positions increase with rb
          const Index a = numext::maxi(destRow, chunkStart);
          const Index b = numext::mini(destRow + static_cast<Index>(rowBlock.height()), chunkEnd);
          if (a >= b) continue;
          targetLower.block(a, targetColStart, b - a, cc).noalias() -=
              srcLower.block(rowBlock.panelOffset + (a - destRow), 0, b - a, wSrc) * dl;
        }
      }
    }
  });
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorizeSupernode(
    StorageIndex s, const RealScalar& staticPivot, SupernodeResult& result, bool intraParallel) {
  // Both paths accumulate each element's updates in source order; the chunked
  // one spreads the panel GEMMs across the pool instead of the supernodes.
  if (intraParallel) {
    applyAllUpdatesChunked(s);
  } else {
    UpdateScratch scratch;  // per-call: this runs concurrently across the level
    for (const UpdateSource& u : m_updateSources[s])
      applyUpdate(u.sourceSupernode, s, u.facingRowBlock, scratch);
  }

  const Supernode& sn = m_supernodes[s];
  const StorageIndex w = sn.width();
  signed char* pivotKind = m_pivotKind.data() + static_cast<std::size_t>(sn.firstColumn);
  Scalar* dOffDiag = m_dOffDiag.data() + static_cast<std::size_t>(sn.firstColumn);
  StridedPanel diag = diagBlock(s);

  if (m_pivoting == supernodal_ldlt::Pivoting::None)
    factorizeDiagonalBlockUnpivoted(diag, pivotKind, result);
  else
    factorizeDiagonalBlockBunchKaufman(diag, staticPivot, m_diagPivot[s], pivotKind, dOffDiag,
                                       result);
  if (result.notPositiveDefinite || result.singular) {
    result.failColumn += sn.firstColumn;  // block-local -> internal index
    return;
  }

  if (sn.offDiagonalRowCount == 0) return;
  StridedPanel lower = lowerPanel(s);

  // The block's symmetric interchange permutes its COLUMNS, and the panel's
  // columns are the same set, so they move with it. Its ROWS are other equations
  // entirely and do not: that is exactly why the Schur complement this supernode
  // sends out is unaffected by the local permutation.
  const std::vector<StorageIndex>& perm = m_diagPivot[s];
  if (!perm.empty()) {
    DenseMatrix permuted(lower.rows(), w);
    for (StorageIndex k = 0; k < w; ++k) permuted.col(k) = lower.col(perm[k]);
    lower = permuted;
  }

  // A_Rk = L_Rk D_kk L_kk^H, so L_Rk = A_Rk L_kk^-H D_kk^-1: one right-solve
  // against the unit upper triangle, then a right-multiply by D^-1 that has to
  // treat a 2x2 pivot as a block rather than two scalars. Every ROW of the panel
  // goes through both independently, which is what lets chunks of rows run
  // concurrently on a level too narrow to fill the pool any other way.
  auto panelSolve = [&](Index start, Index len) {
    auto rows = lower.middleRows(start, len);
    diag.template triangularView<UnitLower>().adjoint().template solveInPlace<OnTheRight>(rows);
    for (StorageIndex q = 0; q < w;) {
      if (pivotKind[q] != 2) {
        rows.col(q) /= diag(q, q);
        q += 1;
      } else {
        // [x0 x1] * D^-1 with D^-1 = (1/det) [[b, -conj(c)], [-c, a]], taken row
        // by row so the two-column temporary never reaches the allocator.
        const Scalar c = dOffDiag[q], cbar = numext::conj(c);
        const RealScalar a = numext::real(diag(q, q)), b = numext::real(diag(q + 1, q + 1));
        const Scalar det = Scalar(a * b - numext::abs2(c));
        for (Index r = 0; r < rows.rows(); ++r) {
          const Scalar x0 = rows(r, q), x1 = rows(r, q + 1);
          rows(r, q) = (x0 * Scalar(b) - x1 * c) / det;
          rows(r, q + 1) = (x1 * Scalar(a) - x0 * cbar) / det;
        }
        q += 2;
      }
    }
  };
  const Index offDiag = static_cast<Index>(sn.offDiagonalRowCount);
  if (intraParallel)
    parallelChunks(offDiag, intraChunkRows(offDiag), panelSolve);
  else
    panelSolve(Index(0), offDiag);
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
void SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analysisDone && "analyzePattern must be called before factorize");
  eigen_assert(matrix.rows() == m_size && matrix.cols() == m_size &&
               "factorize(): matrix size differs from analyzePattern()");
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(m_supernodes.size());
  m_factorized = false;
  m_failColumn = -1;
  m_replacedPivots = 0;
  m_pivot2x2Count = 0;
  m_straddlingPivots = 0;
  m_intraSupernodes = 0;
  m_inertia = supernodal_ldlt::Inertia();
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

  // Resolve the static-pivot threshold against the SCALED matrix, since that is
  // what gets factored. Pivoting::None never perturbs -- it reports instead --
  // so the threshold is only meaningful under Bunch-Kaufman.
  RealScalar staticPivot = m_staticPivotThreshold;
  if (m_thresholdIsAuto) {
    RealScalar maxAbs(0);
    for (StorageIndex j = 0; j < m_size; ++j)
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
        const StorageIndex i = static_cast<StorageIndex>(it.index());
        if (!inputTriangleHolds(i, j)) continue;
        maxAbs = numext::maxi(maxAbs, numext::abs(it.value()) * m_scale[i] * m_scale[j]);
      }
    staticPivot = numext::sqrt(NumTraits<RealScalar>::epsilon()) * maxAbs;
  }

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
  m_pivotKind.assign(static_cast<std::size_t>(m_size), 1);
  m_dOffDiag.assign(static_cast<std::size_t>(m_size), Scalar(0));
  m_diagPivot.assign(supernodeNbr, std::vector<StorageIndex>());

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
  std::vector<SupernodeResult> results(static_cast<std::size_t>(supernodeNbr));
  bool bad = false;
  const Index lanes = static_cast<Index>(m_executor.concurrency());
  for (const std::vector<StorageIndex>& group : m_levelGroups) {
    const Index groupSize = static_cast<Index>(group.size());
    // A level too narrow to fill the lanes switches to INTRA-supernode
    // parallelism: its supernodes run sequentially here and their panel work is
    // chunked across the pool instead. That is the root-separator chain, where
    // level dispatch has nothing left to give -- one supernode on one lane, the
    // rest idle -- and where most of the factorization time is. The pool is
    // fork-join and not nestable, so the two modes are exclusive per level.
    //
    // A level of ONE supernode qualifies regardless of the slack: there is no
    // inter-supernode parallelism there to trade away, so the only question is
    // whether the panel is tall enough to chunk, which the next guard answers.
    bool innerMode = m_intraParallel && lanes > 1 &&
                     (groupSize == 1 || groupSize * kIntraLevelSlack <= lanes);
    if (innerMode) {
      StorageIndex maxOffDiag = 0;
      for (StorageIndex s : group) maxOffDiag = std::max(maxOffDiag, m_supernodes[s].offDiagonalRowCount);
      // Tall enough for at least two chunks at the finest extent intraChunkRows()
      // will pick. Tracks the FLOOR rather than the ceiling, because the extent
      // scales with the lane count.
      innerMode = static_cast<Index>(maxOffDiag) >= 2 * kMinIntraChunkSize;
    }
    if (innerMode) {
      m_intraSupernodes += groupSize;
      for (Index k = 0; k < groupSize; ++k) {
        const StorageIndex s = group[static_cast<std::size_t>(k)];
        factorizeSupernode(s, staticPivot, results[s], /*intraParallel=*/true);
      }
    } else {
      m_executor.parallelFor(Index(0), groupSize, [&](Index k) {
        const StorageIndex s = group[static_cast<std::size_t>(k)];
        factorizeSupernode(s, staticPivot, results[s], /*intraParallel=*/false);
      });
    }
    for (StorageIndex s : group)
      if (results[s].notPositiveDefinite || results[s].singular) bad = true;
    if (bad) break;  // the factor is unusable; stop launching further levels
  }

  if (bad) {
    const SupernodeResult* failure = nullptr;
    for (StorageIndex s = 0; s < supernodeNbr && !failure; ++s)
      if (results[s].notPositiveDefinite || results[s].singular) failure = &results[s];
    m_failColumn = Index(failure->failColumn);
    char buf[600];
    if (failure->notPositiveDefinite)
      std::snprintf(buf, sizeof(buf),
                    "SupernodalLDLT: the matrix is not positive definite -- the pivot at internal "
                    "column %lld is not positive, and setPivoting(Pivoting::None) asked for the "
                    "positive definite fast path. Use the default Pivoting::BunchKaufman, which "
                    "handles indefinite matrices with 2x2 pivot blocks.",
                    static_cast<long long>(failure->failColumn));
    else
      std::snprintf(buf, sizeof(buf),
                    "SupernodalLDLT: singular pivot block at internal column %lld, which "
                    "perturbation was not allowed to repair (setStaticPivotThreshold is 0). The "
                    "matrix is numerically singular in the scaled arithmetic actually used.",
                    static_cast<long long>(failure->failColumn));
    m_lastError = buf;
    m_info = NumericalIssue;
    return;
  }

  for (const SupernodeResult& r : results) {
    m_replacedPivots += r.replaced;
    m_pivot2x2Count += r.blocks2x2;
    m_straddlingPivots += r.straddling;
  }

  // Inertia, by Sylvester's law: congruence preserves the signs, so the signs of
  // D are the signs of A's eigenvalues. A Bunch-Kaufman 2x2 block always has a
  // negative determinant, hence exactly one eigenvalue of each sign -- which is
  // also an invariant worth asserting rather than assuming.
  forEachPivotBlock([&](StorageIndex s, StorageIndex k, bool is2x2) {
    const RealScalar det = pivotBlockDeterminant(s, k, is2x2);
    if (is2x2) {
      eigen_assert(det < RealScalar(0) && "a 2x2 Bunch-Kaufman block must have negative determinant");
      ++m_inertia.positive;
      ++m_inertia.negative;
    } else if (det > RealScalar(0)) {
      ++m_inertia.positive;
    } else if (det < RealScalar(0)) {
      ++m_inertia.negative;
    } else {
      ++m_inertia.zero;
    }
  });

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
//  Factor accessors
// ===========================================================================

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
typename SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::PermutationType
SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::factorPermutation() const {
  eigen_assert(m_factorized && "factorPermutation() before a successful factorize()");
  std::vector<StorageIndex> toFactor;
  buildFactorIndexMap(toFactor);
  PermutationType p(m_size);
  for (StorageIndex i = 0; i < m_size; ++i) p.indices()(i) = toFactor[m_toInternal[i]];
  return p;
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
SparseMatrix<typename MatrixType::Scalar, ColMajor, typename MatrixType::StorageIndex>
SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::matrixL() const {
  eigen_assert(m_factorized && "matrixL() before a successful factorize()");
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> SparseFactor;
  SparseFactor L(m_size, m_size);
  if (m_size == 0) return L;

  std::vector<StorageIndex> toFactor;
  buildFactorIndexMap(toFactor);

  std::vector<Triplet<Scalar, StorageIndex>> entries;
  entries.reserve(static_cast<std::size_t>(m_nnzL));
  for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
    const Supernode& sn = m_supernodes[s];
    const StorageIndex w = sn.width(), first = sn.firstColumn;
    const ConstStridedPanel diag = diagBlock(s);

    // The diagonal block is ALREADY in factor coordinates -- the interchange was
    // applied to it in place -- so its rows go out unmapped.
    for (StorageIndex j = 0; j < w; ++j) {
      entries.emplace_back(first + j, first + j, Scalar(1));  // L's implicit unit diagonal
      for (StorageIndex i = j + 1; i < w; ++i)
        entries.emplace_back(first + i, first + j, diag(i, j));
    }

    // The panel's rows are other supernodes' equations, and the solve defers
    // THEIR interchanges until those supernodes are reached -- which is exactly
    // why the Schur complement is invariant to a local permutation. Assembling a
    // standalone L is where that deferral has to be paid off, by mapping each
    // panel row through the interchange of the supernode that owns it.
    const ConstStridedPanel lower = lowerPanel(s);
    for (StorageIndex b = 0; b < sn.rowBlockCount; ++b) {
      const RowBlock& block = m_rowBlocks[sn.firstRowBlock + b];
      for (StorageIndex p = 0; p < block.height(); ++p) {
        const StorageIndex row = toFactor[block.firstRow + p];
        for (StorageIndex j = 0; j < w; ++j)
          entries.emplace_back(row, first + j, lower(block.panelOffset + p, j));
      }
    }
  }
  L.setFromTriplets(entries.begin(), entries.end());
  return L;
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
SparseMatrix<typename MatrixType::Scalar, ColMajor, typename MatrixType::StorageIndex>
SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::matrixD() const {
  eigen_assert(m_factorized && "matrixD() before a successful factorize()");
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> SparseFactor;
  SparseFactor D(m_size, m_size);
  if (m_size == 0) return D;

  std::vector<Triplet<Scalar, StorageIndex>> entries;
  entries.reserve(static_cast<std::size_t>(m_size) + 2 * static_cast<std::size_t>(m_pivot2x2Count));
  for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
    const Supernode& sn = m_supernodes[s];
    const ConstStridedPanel diag = diagBlock(s);
    for (StorageIndex k = 0; k < sn.width(); ++k) {
      const StorageIndex g = sn.firstColumn + k;
      entries.emplace_back(g, g, diag(k, k));
      if (!is2x2Leader(g)) continue;
      const Scalar c = m_dOffDiag[static_cast<std::size_t>(g)];
      entries.emplace_back(g + 1, g, c);
      entries.emplace_back(g, g + 1, numext::conj(c));
    }
  }
  D.setFromTriplets(entries.begin(), entries.end());
  return D;
}

template <typename MatrixType, int UpLo, typename OrderingType, typename Executor>
Matrix<typename MatrixType::Scalar, Dynamic, 1>
SupernodalLDLT<MatrixType, UpLo, OrderingType, Executor>::vectorD() const {
  eigen_assert(m_factorized && "vectorD() before a successful factorize()");
  Matrix<Scalar, Dynamic, 1> d(m_size);
  for (StorageIndex s = 0; s < static_cast<StorageIndex>(m_supernodes.size()); ++s) {
    const Supernode& sn = m_supernodes[s];
    const ConstStridedPanel diag = diagBlock(s);
    for (StorageIndex k = 0; k < sn.width(); ++k) d[sn.firstColumn + k] = diag(k, k);
  }
  return d;
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
  // The factor's diagonal block is P^T L_kk, so solving with it means permuting
  // first. Panel contributions above needed no permutation: they live in
  // un-permuted global rows, which is what keeps P_s local to this supernode.
  permuteHead(t, head);
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
  // Undo the local interchange, so this supernode's rows are back in global
  // coordinates before any lower-numbered supernode reads them through its panel.
  unpermuteHead(s, head);
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
    permuteHead(s, head);
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
    const signed char* kind = m_pivotKind.data() + static_cast<std::size_t>(sn.firstColumn);
    for (StorageIndex k = 0; k < sn.width();) {
      if (kind[k] != 2) {
        y.row(sn.firstColumn + k) /= diag(k, k);
        k += 1;
      } else {
        // Solve the 2x2 system rather than dividing twice: D^-1 = (1/det) [[b, -conj(c)], [-c, a]].
        const Scalar a = diag(k, k), c = m_dOffDiag[sn.firstColumn + k], b = diag(k + 1, k + 1);
        const RealScalar det = numext::real(a) * numext::real(b) - numext::abs2(c);
        const DenseMatrix pair = y.middleRows(sn.firstColumn + k, 2);
        y.row(sn.firstColumn + k) =
            (Scalar(numext::real(b)) * pair.row(0) - numext::conj(c) * pair.row(1)) / Scalar(det);
        y.row(sn.firstColumn + k + 1) =
            (Scalar(numext::real(a)) * pair.row(1) - c * pair.row(0)) / Scalar(det);
        k += 2;
      }
    }
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

  // Stationary refinement, gated on perturbation by default: an unperturbed
  // LDL^T is backward stable and there is nothing to repair, so a positive
  // definite solve costs exactly the factor solve and nothing more.
  m_lastRefinementIterations = 0;
  const bool refine =
      m_maxRefinementIterations > 0 && (!m_refineOnlyIfPerturbed || m_replacedPivots > 0);
  if (refine && m_size > 0) {
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
