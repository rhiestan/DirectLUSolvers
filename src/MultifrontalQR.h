// MultifrontalQR -- a rank-revealing multifrontal sparse QR factorization, for
// numerical rank determination, least-squares problems and ill-conditioned or
// rank-deficient systems.
//
// WHAT IT IS FOR
//
//   A P = Q R        with P a column permutation, Q orthogonal, R upper
//                    trapezoidal and its numerical rank r decided on the way.
//
// The LU solvers of this project answer "solve A x = b" and stop at the first
// sign that no LU exists. This one is for the questions that follow: what is the
// numerical rank of A, which columns are dependent, what is the least-squares
// solution of an inconsistent or rectangular system, and which of the infinitely
// many solutions of a rank-deficient one should be returned. Orthogonal
// transformations are backward stable without any pivoting growth, so the price
// of the answer is fill (R is the Cholesky factor of A^H A) rather than accuracy.
//
// WHY NOT Eigen::SparseQR
//
// Measured on this project's setfos corpus, two defects of Eigen's left-looking
// SparseQR decide the matter, and this design removes both:
//
//   * FILL. SparseQR never permutes rows: its column elimination tree assumes a
//     structural diagonal (firstRowElt(k) = k) and each column's reflector is
//     forced to include row k. After the column ordering that fictitious entry
//     couples unrelated columns and R becomes nearly dense -- 5.6M entries in R
//     for a 4737-column, 14k-nonzero matrix whose true R has 14k. Here rows are
//     never "diagonal": each row enters the front of its leftmost column, which
//     is what the multifrontal method does by construction.
//   * RANK ON BADLY SCALED MATRICES. SparseQR's threshold is 20 (m+n) eps times
//     the largest column norm of the raw matrix. With entries spanning 1e53 it
//     declared rank 1 or 2 on matrices every LU solver solved to 1e-13. Numerical
//     rank is a property of a SCALED matrix, so this solver equilibrates first
//     (powers of two, so the scaling is exact) and decides rank on the result.
//
// And it is faster where it matters: every front is a dense block factored with
// blocked (compact-WY) Householder transformations, and independent subtrees of
// the elimination tree run in parallel through the executor.
//
// HOW RANK IS DECIDED -- AND CHECKED
//
// Rank is decided in two stages, because the cheap stage alone is known to fail.
//
//   1. Heath's rule while factoring (as SuiteSparseQR): a pivotal column whose
//      remaining norm is at most tol = rankTolerance() * max column norm is
//      declared DEAD. It keeps its entries above the current row (they belong to
//      R12) and its remainder is dropped; droppedNorm() accumulates the Frobenius
//      norm of everything dropped. This catches exact and structural rank
//      deficiency, and near-deficiency that shows up in a single column.
//   2. VERIFICATION. Heath's rule cannot see a dependency spread over several
//      columns, none of which is small on its own -- then R11 (the live part) is
//      ill-conditioned while the rank looks full. After factoring, the smallest
//      singular values of R11 are computed (exactly by a dense SVD for small r,
//      by block inverse iteration with Rayleigh-Ritz otherwise). If any lies
//      below tol, the columns carrying the near-null vectors are chosen by a
//      pivoted QR of those vectors, moved to the END of the column order, and the
//      matrix is refactored. The deferred columns form one final front that is
//      factored with column pivoting (LAPACK's xGEQP3 strategy), so their
//      decision is made by a genuinely rank-revealing kernel with everything else
//      already eliminated. This repeats until the check passes.
//
// The outcome is reported, not just used: rank(), smallestSingularValues() (of
// R11, i.e. estimates of sigma_r, sigma_{r-1}, ... of the scaled matrix) and
// droppedNorm() (an upper bound on sigma_{r+1} up to rounding) together say how
// large the gap at the rank really is. rankIsVerified() is the one-bit summary.
//
// WHAT solve() RETURNS
//
//   full rank, m == n     the solution of A x = b, refined.
//   full rank, m > n      the least-squares solution, refined on the augmented
//                         system (Bjorck), which is what stays accurate when A is
//                         ill-conditioned and the residual is not small.
//   rank r < n            by default the MINIMUM-NORM least-squares solution
//                         (pinv(A) b up to the rank decision): the basic solution
//                         from R11 projected orthogonally onto the complement of
//                         the null space. Solution::Basic returns the basic one
//                         (dead columns zero), which is cheaper and sparse in the
//                         dead columns.
//
// Refinement residuals are computed in double-double (see
// LeftRightLUExtendedResidual.h). That is exact on the SCALED matrix only
// because the scaling is by powers of two -- a general scaling would make the
// scaled entries themselves rounded, and extended precision would be refining
// towards a slightly different matrix.
//
// ROW SCALING AND LEAST SQUARES -- READ THIS FOR m != n
//
// Scaling rows changes a least-squares problem into a WEIGHTED one: the solver
// minimizes ||Dr (A x - b)|| instead of ||A x - b||. For a consistent system the
// two coincide, which is why Scaling::Auto scales rows only for square matrices
// and columns alone otherwise. Column scaling never changes the solution set.
//
// Usage:
//   #include <MultifrontalQR>
//   Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr(A);
//   if (qr.info() != Eigen::Success) std::cerr << qr.lastErrorMessage();
//   Eigen::Index r = qr.rank();
//   Eigen::VectorXd x = qr.solve(b);
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef MULTIFRONTAL_QR_H
#define MULTIFRONTAL_QR_H

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "SupernodalLUSupport.h"
#include "SupernodalLUSymbolic.h"
#include "SupernodalLUExecutor.h"
#include "LeftRightLUConditionEstimate.h"
#include "LeftRightLUExtendedResidual.h"

namespace Eigen {

namespace multifrontal_qr {

/** Fill-reducing column ordering.
 *
 *   COLAMD   column approximate minimum degree on A itself. Never forms A^H A,
 *            so a dense row costs nothing extra to order.
 *   AMD      approximate minimum degree on the pattern of A^H A. Often less fill
 *            than COLAMD on FEM-like matrices, but forming the pattern costs
 *            sum_i (row count)^2 -- quadratic in any dense row.
 *   Natural  the columns as given.
 *   Auto     COLAMD and AMD both, keeping whichever predicts the smaller R. AMD
 *            is skipped when A^H A would be too large to form cheaply.
 */
enum class Ordering { Auto, COLAMD, AMD, Natural };

/** Which solution solve() returns when the matrix is rank deficient. */
enum class Solution { MinimumNorm, Basic };

/** Equilibration before factoring. All factors are powers of two.
 *
 *   Auto            RowsAndColumns for a square matrix, Columns otherwise (row
 *                   scaling turns least squares into WEIGHTED least squares).
 *   RowsAndColumns  row max-norm scaling, then column 2-norm scaling.
 *   Columns         column 2-norm scaling only.
 *   None            factor A as given; rank decided on the raw column norms.
 */
enum class Scaling { Auto, RowsAndColumns, Columns, None };

namespace detail {

// 2^e with e chosen so that value * 2^e lies in [0.5, 1): the power of two
// closest to 1/value from below. Zero and non-finite values scale by 1.
template <typename RealScalar>
RealScalar inversePowerOfTwo(RealScalar value) {
  if (!(value > RealScalar(0)) || !(numext::isfinite)(value)) return RealScalar(1);
  int e = 0;
  std::frexp(value, &e);  // value = f * 2^e, f in [0.5, 1)
  return std::ldexp(RealScalar(1), -e);
}

}  // namespace detail
}  // namespace multifrontal_qr

/** \class MultifrontalQR
 * \brief Rank-revealing multifrontal sparse QR with verified rank, least-squares
 *        and minimum-norm solutions.
 *
 * \tparam MatrixType_ A column-major Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>.
 *                     Real or complex; any shape (m > n, m == n or m < n).
 * \tparam Executor_   A parallel-execution backend (default SerialExecutor); see
 *                     SupernodalLUExecutor.h. Fronts in the same elimination-tree
 *                     level are factored concurrently.
 */
template <typename MatrixType_, typename Executor_ = supernodal_lu::SerialExecutor>
class MultifrontalQR : public SparseSolverBase<MultifrontalQR<MatrixType_, Executor_>> {
 protected:
  typedef SparseSolverBase<MultifrontalQR<MatrixType_, Executor_>> Base;
  using Base::m_isInitialized;

 public:
  typedef MatrixType_ MatrixType;
  typedef Executor_ Executor;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrix;
  typedef Matrix<Scalar, Dynamic, 1> Vector;
  typedef Matrix<RealScalar, Dynamic, 1> RealVector;
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

  enum { ColsAtCompileTime = Dynamic, MaxColsAtCompileTime = Dynamic };

  using Base::_solve_impl;

  MultifrontalQR() {}
  explicit MultifrontalQR(const MatrixType& matrix) { compute(matrix); }

  // --- main driver ----------------------------------------------------------

  /** Symbolic analysis: column ordering, column elimination tree, fronts. Reads
   *  the pattern only. Forgets any columns a previous factorize() deferred. */
  void analyzePattern(const MatrixType& matrix);

  /** Numeric factorization, rank decision and (unless disabled) rank
   *  verification. Verification may REPEAT the symbolic analysis with some
   *  columns deferred to the end; deferredColumns() lists them. */
  void factorize(const MatrixType& matrix);

  void compute(const MatrixType& matrix) {
    analyzePattern(matrix);
    if (m_info == Success) factorize(matrix);
  }

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const;

  // --- results --------------------------------------------------------------

  Index rows() const { return m_rows; }
  Index cols() const { return m_cols; }
  ComputationInfo info() const { return m_info; }
  const std::string& lastErrorMessage() const { return m_lastError; }
  bool isFactorized() const { return m_factorized; }

  /** Numerical rank: the number of live pivots, i.e. the order of R11. */
  Index rank() const { return m_rank; }

  /** The absolute rank threshold the last factorize() used, in the SCALED
   *  matrix: rankTolerance() times the largest scaled column norm. */
  RealScalar absoluteRankThreshold() const { return m_absTol; }

  /** Frobenius norm of everything Heath's rule dropped, in the scaled matrix.
   *  Up to rounding, the scaled matrix is within this distance of a matrix of
   *  rank rank(), so it bounds sigma_{r+1} from above. */
  RealScalar droppedNorm() const { return m_droppedNorm; }

  /** Smallest singular values of R11 in ascending order, from the verification
   *  step (empty if verification is off or r == 0). They estimate sigma_r,
   *  sigma_{r-1}, ... of the scaled matrix: exact for a small R11 (dense SVD),
   *  converged Ritz values otherwise. Compare the first against droppedNorm():
   *  the ratio is the size of the gap the rank decision rests on. */
  const RealVector& smallestSingularValues() const { return m_sigma; }

  /** True when verification ran and found no singular value of R11 at or below
   *  absoluteRankThreshold() -- i.e. no column dependency was missed. False when
   *  it was disabled or the repair loop gave up; the rank is then Heath's
   *  alone. */
  bool rankIsVerified() const { return m_rankVerified; }

  /** Original column indices the verification moved to the final front, in
   *  that front's column order (the order deferredRotation() acts on). Empty
   *  when Heath's rule was right the first time. */
  const std::vector<StorageIndex>& deferredColumns() const { return m_deferredOrder; }
  /** How many times verification refactored the matrix. */
  Index repairIterations() const { return m_repairs; }

  /** Original column indices Heath's rule judged dependent (dead), ascending.
   *  Deficiency inside the deferred block is not attributed to columns -- the
   *  SVD there finds dependent DIRECTIONS; see deferredNullity(). */
  std::vector<StorageIndex> deadColumns() const {
    std::vector<StorageIndex> out;
    for (const FrontFactor& ff : m_factors) {
      if (ff.rotated) continue;
      for (Index c = ff.rank; c < ff.rank + ff.dead; ++c)
        out.push_back(m_sym.internalToOrig[std::size_t(ff.cols[std::size_t(c)])]);
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  /** Dimension of the null space found inside the deferred block. rank() ==
   *  cols() - deadColumns().size() - deferredNullity(). */
  Index deferredNullity() const {
    return m_rotatedFront < 0 ? 0 : m_factors[std::size_t(m_rotatedFront)].dead;
  }

  /** The unitary V of the deferred block's SVD (k x k, k = deferredColumns()
   *  size): the factorization is As * P * diag(I, V) == Q * R. Empty when
   *  nothing was deferred. */
  const DenseMatrix& deferredRotation() const {
    static const DenseMatrix empty;
    return m_rotatedFront < 0 ? empty : m_factors[std::size_t(m_rotatedFront)].rotV;
  }

  /** Scalars of R (the upper trapezoid of every front, amalgamation zeros
   *  included) and of the Householder vectors. */
  Index nnzR() const { return m_nnzR; }
  Index nnzH() const { return m_nnzH; }
  /** nnz(R) the symbolic analysis predicts for a full-rank factorization. */
  Index predictedFactorNonzeros() const { return m_sym.predictedNnzR; }
  Index frontCount() const { return Index(m_sym.fronts.size()); }
  Index levelCount() const { return Index(m_sym.levels.size()); }
  /** Largest dense front the last factorize() formed, rows x columns. */
  Index largestFrontEntries() const { return m_largestFront; }
  /** The ordering the analysis ended up using (Auto resolves to one). */
  multifrontal_qr::Ordering orderingUsed() const { return m_sym.ordering; }

  /** Row and column equilibration factors (powers of two): the factored matrix
   *  is diag(rowScaling()) * A * diag(colScaling()). */
  const RealVector& rowScaling() const { return m_rowScale; }
  const RealVector& colScaling() const { return m_colScale; }

  /** The column permutation of the factor: indices()(j) is the ORIGINAL column
   *  at position j of R. Live columns come first in pivot order, then the dead
   *  ones, then the deferred block. With V = deferredRotation() acting on that
   *  trailing block, As * P * diag(I, V) == Q * R up to droppedNorm(). */
  PermutationType colsPermutation() const {
    PermutationType p(m_cols);
    Index pos = 0;
    for (Index piv = 0; piv < m_rank; ++piv)
      if (m_pivotCol[std::size_t(piv)] >= 0)
        p.indices()(pos++) = m_sym.internalToOrig[std::size_t(m_pivotCol[std::size_t(piv)])];
    for (StorageIndex c : deadColumns()) p.indices()(pos++) = c;
    for (StorageIndex c : m_deferredOrder) p.indices()(pos++) = c;
    return p;
  }

  /** R materialized as a sparse rank x n matrix in the column order of
   *  colsPermutation(), the deferred block already rotated by
   *  deferredRotation(). Rows are in pivot order, so R is upper trapezoidal
   *  except that the deferred block's rows sit below the dead columns.
   *  Includes amalgamation zeros only where they are numerically nonzero. For
   *  inspection and export; solve() never needs it. */
  SparseMatrix<Scalar, ColMajor, StorageIndex> matrixR() const;

  /** Orthonormal basis of the (numerical) null space of A, n x (n - rank), in
   *  ORIGINAL column coordinates and the ORIGINAL (unscaled) inner product.
   *  Computed on first use and cached; empty if the null space is larger than
   *  setMaxNullSpaceScalars() allows. */
  const DenseMatrix& nullSpace() const {
    ensureNullSpace();
    if (m_nullSpace.cols() != m_nullDim && !m_nullSpaceTooLarge)
      m_nullSpace = m_nullQR.householderQ() * DenseMatrix::Identity(m_cols, m_nullDim);
    return m_nullSpace;
  }

  /** Hager-Higham estimate of kappa_1(R11) of the scaled matrix -- the
   *  conditioning of the live least-squares problem. Computed on first call. */
  RealScalar conditionEstimate() const;

  /** ||b - A x|| / ||b|| of the last solve(), in the original matrix. For a
   *  least-squares problem this is the (necessarily nonzero) optimal residual,
   *  not an error. */
  RealScalar solveResidual() const { return m_lastResidual; }
  /** ||A^H r|| / (||A||_F ||r||) of the last solve(): 0 at an exact least-squares
   *  solution. Reported as 0 when r is at rounding level (100 eps (||A|| ||x||
   *  + ||b||)), where the ratio is noise -- the solve is consistent then. */
  RealScalar leastSquaresOptimality() const { return m_lastOptimality; }
  /** Why the last solve() returned something other than what setSolution()
   *  asked for (currently: a null space too large to form), or empty. */
  const std::string& lastSolveMessage() const { return m_solveNote; }
  /** Refinement steps the last solve() took (on its worst column). */
  Index iterativeRefinements() const { return m_lastRefinements; }
  /** True when the last factorization's rank is below n or m > n: solve() then
   *  returns a least-squares solution. */
  bool isLeastSquares() const { return m_rank < m_cols || m_rows > m_cols; }

  // --- options --------------------------------------------------------------

  void setOrdering(multifrontal_qr::Ordering o) { m_ordering = o; }
  multifrontal_qr::Ordering ordering() const { return m_ordering; }
  void setSolution(multifrontal_qr::Solution s) { m_solution = s; }
  multifrontal_qr::Solution solution() const { return m_solution; }
  void setScaling(multifrontal_qr::Scaling s) { m_scaling = s; }
  multifrontal_qr::Scaling scaling() const { return m_scaling; }

  /** Relative rank tolerance tau: a column is dead when its remaining norm is at
   *  most tau * max column norm (after scaling). A negative value (default)
   *  selects SuiteSparseQR's 20 (m + n) eps. */
  void setRankTolerance(RealScalar tau) { m_relTol = tau; }
  RealScalar rankTolerance() const {
    if (m_relTol >= RealScalar(0)) return m_relTol;
    return RealScalar(20) * RealScalar(m_rows + m_cols) * NumTraits<RealScalar>::epsilon();
  }

  /** Check the rank decision and repair it (default on). See the header. */
  void setRankVerification(bool on) { m_verifyRank = on; }
  bool rankVerification() const { return m_verifyRank; }
  /** Bound on verification refactorizations (default 8). */
  void setMaxRepairs(Index n) { m_maxRepairs = n; }
  /** R11 order up to which verification uses a dense SVD (default 24). */
  void setDenseVerificationLimit(Index r) { m_denseVerifyLimit = r; }

  /** Refinement steps per solve (default 3; 0 disables). Stops early once the
   *  correction is negligible or stops shrinking. */
  void setMaxRefinements(Index n) { m_maxRefinements = n; }
  Index maxRefinements() const { return m_maxRefinements; }
  /** Double-double refinement residuals (default on). */
  void setExtendedPrecisionResidual(bool on) { m_extendedResidual = on; }

  /** Largest null-space basis (n x (n - rank) scalars) minimum-norm solves will
   *  form. Default 5e7. Beyond it solve() falls back to the basic solution and
   *  says so in lastSolveMessage(). */
  void setMaxNullSpaceScalars(double n) { m_maxNullSpaceScalars = n; }

  /** Relaxed amalgamation of fronts (default on). */
  void setAmalgamation(bool on) { m_amalgamate = on; }
  /** Split the trailing updates of large fronts across the executor on levels
   *  with fewer fronts than lanes (default on). The answer does not depend on
   *  it: the column chunks are the same either way. */
  void setIntraFrontParallelism(bool on) { m_intraParallel = on; }
  /** Panel width of the blocked Householder kernel (default 32). */
  void setBlockSize(Index nb) { m_blockSize = nb < 1 ? 1 : nb; }

  /** Fail-fast guard: factorize() refuses before allocating when the predicted
   *  nnz(R) exceeds this. 0 (default) = off. */
  void setMaxFactorNonzeros(Index limit) { m_maxFactorNonzeros = limit; }

  /** Largest pattern of A^H A (sum of squared row counts) Ordering::Auto will
   *  form to try AMD. Default max(1e7, 50 nnz(A)). */
  void setMaxAtAPattern(double n) { m_maxAtA = n; }

  Executor& executor() { return m_executor; }
  const Executor& executor() const { return m_executor; }

 private:
  // --- symbolic -------------------------------------------------------------

  struct Front {
    StorageIndex firstCol = 0, lastCol = 0;  // internal pivotal range, inclusive
    StorageIndex parent = -1;
    Index offBegin = 0, offEnd = 0;          // into Symbolic::offCols
    Index rowBegin = 0, rowEnd = 0;          // into Symbolic::frontRows
    Index childBegin = 0, childEnd = 0;      // into Symbolic::children
    bool deferred = false;
    Index npiv() const { return Index(lastCol) - Index(firstCol) + 1; }
    Index noff() const { return offEnd - offBegin; }
  };

  struct Symbolic {
    multifrontal_qr::Ordering ordering = multifrontal_qr::Ordering::COLAMD;
    std::vector<StorageIndex> internalToOrig, origToInternal;
    std::vector<Front> fronts;
    std::vector<StorageIndex> offCols, frontRows, children;
    std::vector<std::vector<StorageIndex>> levels;
    // A in row form, columns relabelled to internal ids and sorted per row.
    std::vector<Index> rowPtr;
    std::vector<StorageIndex> rowCols;
    std::vector<Index> entryMap;  // CSC position -> row-form position
    Index predictedNnzR = 0;
  };

  // --- numeric --------------------------------------------------------------

  struct FrontFactor {
    Index rank = 0;                   // live pivots (= R rows of this front)
    Index dead = 0;                   // dead pivotal columns (or directions, if rotated)
    // The deferred front: R_D = U diag(sigma) V^H. Its pivots are the leading
    // columns of V (directions, x_D = V y), U is part of Q, and R holds
    // [diag(sigma_live) 0].
    bool rotated = false;
    DenseMatrix rotU, rotV;
    RealVector sigma;
    std::vector<StorageIndex> cols;   // internal ids: live (pivot order), dead, off
    DenseMatrix R;                    // rank x cols.size()
    std::vector<StorageIndex> slots;  // row slots of front rows [0, H.rows())
    DenseMatrix H;                    // essential Householder parts, strictly below row i in column i
    std::vector<Scalar> tau;
    std::vector<StorageIndex> reach;  // Householder i acts on front rows [i, reach[i])
  };

  struct Contribution {
    DenseMatrix C;                    // rows x noff, columns = the front's off columns
    std::vector<StorageIndex> slots;
  };

  void buildSymbolic(const MatrixType& A, multifrontal_qr::Ordering ordering, Symbolic& sym) const;
  void orderColumns(const MatrixType& A, const std::vector<char>& isDeferred, multifrontal_qr::Ordering ordering,
                    std::vector<StorageIndex>& ordered) const;
  double atAPatternSize(const MatrixType& A) const;
  void computeScaling(const MatrixType& A);
  void factorOnce();
  void processFront(Index f, bool intraParallel, RealScalar& dropped, bool& nonFinite, Index& frontEntries);
  void verifyRank(bool& needsRepair, std::vector<StorageIndex>& newDeferred);
  void finishPivotMaps();
  void releaseNumeric();

  void applyQAdjoint(Vector& y) const;
  void applyQ(Vector& y) const;
  void upperSolve(const Vector& c, Vector& z, const Vector* rotatedDead = nullptr) const;
  // Live coordinates: one entry per pivot. For an ordinary front that is the
  // value of the pivot's column; for the rotated front it is the direction
  // coefficient y (x_D = V [y; 0]).
  void liveToInternal(const Vector& v, Vector& z) const;
  void internalToLive(const Vector& z, Vector& v) const;
  void upperAdjointSolve(const Vector& g, Vector& h) const;
  void upperMultiply(const Vector& z, Vector& u) const;
  void ensureNullSpace() const;
  void toOriginal(const Vector& z, Vector& x) const {
    x.resize(m_cols);
    for (Index c = 0; c < m_cols; ++c) x[c] = z[m_sym.origToInternal[std::size_t(c)]];
  }
  void toInternal(const Vector& x, Vector& z) const {
    z.resize(m_cols);
    for (Index c = 0; c < m_cols; ++c) z[m_sym.origToInternal[std::size_t(c)]] = x[c];
  }

  // options
  multifrontal_qr::Ordering m_ordering = multifrontal_qr::Ordering::Auto;
  multifrontal_qr::Solution m_solution = multifrontal_qr::Solution::MinimumNorm;
  multifrontal_qr::Scaling m_scaling = multifrontal_qr::Scaling::Auto;
  RealScalar m_relTol = RealScalar(-1);
  bool m_verifyRank = true;
  Index m_maxRepairs = 8;
  Index m_denseVerifyLimit = 24;
  Index m_maxRefinements = 3;
  bool m_extendedResidual = true;
  double m_maxNullSpaceScalars = 5e7;
  bool m_amalgamate = true;
  Index m_blockSize = 32;
  bool m_intraParallel = true;
  static constexpr Index kTrailingChunk = 64;
  Index m_maxFactorNonzeros = 0;
  double m_maxAtA = -1.0;
  Executor m_executor;

  // state
  Index m_rows = 0, m_cols = 0;
  ComputationInfo m_info = InvalidInput;
  std::string m_lastError;
  bool m_factorized = false;
  Symbolic m_sym;
  std::vector<StorageIndex> m_deferredCols;
  MatrixType m_pattern;    // compressed copy of the analyzed pattern (for re-analysis)
  MatrixType m_scaled;     // diag(rowScale) * A * diag(colScale), original order
  std::vector<Scalar> m_rowVals;  // scaled values in Symbolic row form
  RealVector m_rowScale, m_colScale;
  RealScalar m_absTol = RealScalar(0);
  RealScalar m_scaledFrobenius = RealScalar(0);

  std::vector<FrontFactor> m_factors;
  std::vector<Contribution> m_contrib;
  std::vector<Index> m_liveOffset;              // first pivot index of each front
  std::vector<StorageIndex> m_pivotCol;         // pivot index -> internal column
  std::vector<StorageIndex> m_pivotSlot;        // pivot index -> row slot
  std::vector<StorageIndex> m_internalToPivot;  // internal column -> pivot index, -1 dead, -2 deferred
  std::vector<StorageIndex> m_deferredOrder;    // original ids, in the rotated front's column order
  Index m_rotatedFront = -1;
  Index m_rank = 0;
  RealScalar m_droppedNorm = RealScalar(0);
  RealVector m_sigma;
  bool m_rankVerified = false;
  Index m_repairs = 0;
  Index m_nnzR = 0, m_nnzH = 0, m_largestFront = 0;

  mutable bool m_nullSpaceReady = false;
  mutable bool m_nullSpaceTooLarge = false;
  mutable DenseMatrix m_nullSpace;        // explicit basis, formed only on request
  mutable HouseholderQR<DenseMatrix> m_nullQR;  // the basis in factored form
  mutable Index m_nullDim = 0;
  mutable RealScalar m_conditionEstimate = RealScalar(-1);
  mutable RealScalar m_lastResidual = RealScalar(0);
  mutable RealScalar m_lastOptimality = RealScalar(0);
  mutable Index m_lastRefinements = 0;
  mutable std::string m_solveNote;
};

// ============================================================================
// symbolic analysis
// ============================================================================

template <typename MatrixType, typename Executor>
double MultifrontalQR<MatrixType, Executor>::atAPatternSize(const MatrixType& A) const {
  std::vector<double> rowCount(std::size_t(A.rows()), 0.0);
  for (Index c = 0; c < A.outerSize(); ++c)
    for (typename MatrixType::InnerIterator it(A, c); it; ++it) rowCount[std::size_t(it.row())] += 1.0;
  double total = 0.0;
  for (double r : rowCount) total += r * r;
  return total;
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::orderColumns(const MatrixType& A, const std::vector<char>& isDeferred,
                                                        multifrontal_qr::Ordering ordering,
                                                        std::vector<StorageIndex>& ordered) const {
  const Index m = A.rows(), n = A.cols();
  std::vector<StorageIndex> keep;
  keep.reserve(std::size_t(n));
  for (Index c = 0; c < n; ++c)
    if (!isDeferred[std::size_t(c)]) keep.push_back(StorageIndex(c));
  const Index nk = Index(keep.size());
  ordered.clear();
  ordered.reserve(std::size_t(n));

  if (ordering == multifrontal_qr::Ordering::Natural || nk == 0) {
    ordered = keep;
  } else {
    // Pattern of the kept columns, 1-byte values (the orderings read the pattern only).
    typedef SparseMatrix<signed char, ColMajor, StorageIndex> Pattern;
    Pattern sub(m, nk);
    {
      Index nnz = 0;
      for (StorageIndex c : keep) nnz += A.outerIndexPtr()[c + 1] - A.outerIndexPtr()[c];
      sub.resizeNonZeros(nnz);
      Index pos = 0;
      for (Index j = 0; j < nk; ++j) {
        sub.outerIndexPtr()[j] = StorageIndex(pos);
        const StorageIndex c = keep[std::size_t(j)];
        for (Index p = A.outerIndexPtr()[c]; p < A.outerIndexPtr()[c + 1]; ++p) {
          sub.innerIndexPtr()[pos] = A.innerIndexPtr()[p];
          sub.valuePtr()[pos] = 1;
          ++pos;
        }
      }
      sub.outerIndexPtr()[nk] = StorageIndex(pos);
    }
    PermutationType perm;
    std::vector<StorageIndex> position(static_cast<std::size_t>(nk));  // new position of kept column j
    if (ordering == multifrontal_qr::Ordering::COLAMD) {
      // COLAMDOrdering returns the DIRECT map: indices()(j) is the new position
      // of column j (see left_right_lu::OrderingConvention).
      COLAMDOrdering<StorageIndex> colamd;
      colamd(sub, perm);
      for (Index j = 0; j < nk; ++j) position[std::size_t(j)] = perm.indices()(j);
    } else {
      // AMD on the pattern of A^H A. AMDOrdering returns the INVERSE map:
      // indices()(k) is the column placed at position k.
      Pattern rowForm = sub.transpose();  // row-major view as a column-major matrix: column i = row i
      std::vector<StorageIndex> mark(std::size_t(nk), StorageIndex(-1));
      std::vector<Index> colPtr(std::size_t(nk) + 1, 0);
      std::vector<StorageIndex> colIdx;
      for (Index j = 0; j < nk; ++j) {
        mark[std::size_t(j)] = StorageIndex(j);
        for (typename Pattern::InnerIterator it(sub, j); it; ++it)
          for (typename Pattern::InnerIterator jt(rowForm, it.row()); jt; ++jt) {
            const StorageIndex k = StorageIndex(jt.row());
            if (mark[std::size_t(k)] != StorageIndex(j)) {
              mark[std::size_t(k)] = StorageIndex(j);
              colIdx.push_back(k);
            }
          }
        colPtr[std::size_t(j) + 1] = Index(colIdx.size());
      }
      Pattern ata(nk, nk);
      ata.resizeNonZeros(Index(colIdx.size()));
      for (Index j = 0; j <= nk; ++j) ata.outerIndexPtr()[j] = StorageIndex(colPtr[std::size_t(j)]);
      for (std::size_t p = 0; p < colIdx.size(); ++p) {
        ata.innerIndexPtr()[p] = colIdx[p];
        ata.valuePtr()[p] = 1;
      }
      AMDOrdering<StorageIndex> amd;
      amd(ata, perm);
      for (Index k = 0; k < nk; ++k) position[std::size_t(perm.indices()(k))] = StorageIndex(k);
    }
    ordered.assign(std::size_t(nk), StorageIndex(0));
    for (Index j = 0; j < nk; ++j) ordered[std::size_t(position[std::size_t(j)])] = keep[std::size_t(j)];
  }
  for (Index c = 0; c < n; ++c)
    if (isDeferred[std::size_t(c)]) ordered.push_back(StorageIndex(c));
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::buildSymbolic(const MatrixType& A, multifrontal_qr::Ordering ordering,
                                                         Symbolic& sym) const {
  namespace symbolic = supernodal_lu::symbolic;
  const Index m = A.rows(), n = A.cols();
  sym = Symbolic();
  sym.ordering = ordering;

  std::vector<char> isDeferred(std::size_t(n), 0);
  for (StorageIndex c : m_deferredCols) isDeferred[std::size_t(c)] = 1;
  const Index d = Index(m_deferredCols.size());

  std::vector<StorageIndex> ordered;
  orderColumns(A, isDeferred, ordering, ordered);

  // Column elimination tree (the elimination tree of A^H A) straight from A,
  // Liu's algorithm with a per-row "previous column" (CSparse's cs_etree, ata=1).
  // Two or more deferred columns get a fictitious dense row joining them into a
  // chain, which is what keeps them contiguous and last through the postorder:
  // the chain's top is the largest root, and the postorder of a chain visits
  // every side subtree before the chain itself.
  const bool fakeRow = d >= 2;
  std::vector<StorageIndex> parent(std::size_t(n), StorageIndex(-1));
  {
    std::vector<StorageIndex> ancestor(std::size_t(n), StorageIndex(-1));
    std::vector<StorageIndex> prev(std::size_t(m) + 1, StorageIndex(-1));
    auto visit = [&](Index row, StorageIndex k) {
      StorageIndex j = prev[std::size_t(row)];
      while (j != StorageIndex(-1) && j < k) {
        const StorageIndex next = ancestor[std::size_t(j)];
        ancestor[std::size_t(j)] = k;
        if (next == StorageIndex(-1)) {
          parent[std::size_t(j)] = k;
          break;
        }
        j = next;
      }
      prev[std::size_t(row)] = k;
    };
    for (Index k = 0; k < n; ++k) {
      const StorageIndex c = ordered[std::size_t(k)];
      for (Index p = A.outerIndexPtr()[c]; p < A.outerIndexPtr()[c + 1]; ++p) visit(A.innerIndexPtr()[p], StorageIndex(k));
      if (fakeRow && k >= n - d) visit(m, StorageIndex(k));
    }
  }

  std::vector<StorageIndex> postorder;
  symbolic::computePostorder(StorageIndex(n), parent, postorder);
  std::vector<StorageIndex> relabel(static_cast<std::size_t>(n));
  for (Index t = 0; t < n; ++t) relabel[std::size_t(postorder[std::size_t(t)])] = StorageIndex(t);
  std::vector<StorageIndex> finalParent(static_cast<std::size_t>(n));
  sym.internalToOrig.resize(std::size_t(n));
  sym.origToInternal.resize(std::size_t(n));
  for (Index t = 0; t < n; ++t) {
    const StorageIndex k = postorder[std::size_t(t)];
    finalParent[std::size_t(t)] = parent[std::size_t(k)] < 0 ? StorageIndex(-1) : relabel[std::size_t(parent[std::size_t(k)])];
    const StorageIndex orig = ordered[std::size_t(k)];
    sym.internalToOrig[std::size_t(t)] = orig;
    sym.origToInternal[std::size_t(orig)] = StorageIndex(t);
  }
  for (Index t = n - d; t < n; ++t)
    eigen_assert(isDeferred[std::size_t(sym.internalToOrig[std::size_t(t)])] && "deferred columns must stay last");

  // Row form of A in internal column ids, each row sorted (columns visited in
  // increasing internal order), plus the CSC -> row-form position map.
  const Index nnz = A.outerIndexPtr()[n];
  sym.rowPtr.assign(std::size_t(m) + 1, 0);
  for (Index p = 0; p < nnz; ++p) ++sym.rowPtr[std::size_t(A.innerIndexPtr()[p]) + 1];
  for (Index i = 0; i < m; ++i) sym.rowPtr[std::size_t(i) + 1] += sym.rowPtr[std::size_t(i)];
  sym.rowCols.resize(std::size_t(nnz));
  sym.entryMap.resize(std::size_t(nnz));
  {
    std::vector<Index> fill(sym.rowPtr.begin(), sym.rowPtr.end() - 1);
    for (Index t = 0; t < n; ++t) {
      const StorageIndex c = sym.internalToOrig[std::size_t(t)];
      for (Index p = A.outerIndexPtr()[c]; p < A.outerIndexPtr()[c + 1]; ++p) {
        const Index pos = fill[std::size_t(A.innerIndexPtr()[p])]++;
        sym.rowCols[std::size_t(pos)] = StorageIndex(t);
        sym.entryMap[std::size_t(p)] = pos;
      }
    }
  }

  // Pseudo-adjacency: column k's own contribution to R's row structure is the
  // union of the rows whose leftmost column is k. Every other row reaches k
  // through the elimination tree, which is what the children-merge supplies.
  std::vector<std::vector<StorageIndex>> adjacency(static_cast<std::size_t>(n));
  std::vector<StorageIndex> leftmost(std::size_t(m), StorageIndex(-1));
  for (Index i = 0; i < m; ++i) {
    const Index b = sym.rowPtr[std::size_t(i)], e = sym.rowPtr[std::size_t(i) + 1];
    if (b == e) continue;
    const StorageIndex k = sym.rowCols[std::size_t(b)];
    leftmost[std::size_t(i)] = k;
    std::vector<StorageIndex>& adj = adjacency[std::size_t(k)];
    adj.insert(adj.end(), sym.rowCols.begin() + b, sym.rowCols.begin() + e);
  }
  if (fakeRow)
    for (Index t = n - d; t < n; ++t) adjacency[std::size_t(n - d)].push_back(StorageIndex(t));
  for (auto& adj : adjacency) {
    std::sort(adj.begin(), adj.end());
    adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
  }

  symbolic::PartitionOptions opts;
  opts.maxBlockSize = 0;  // a front is one dense block; splitting only adds copies
  opts.cumulativeZeroRule = true;
  if (!m_amalgamate) {
    opts.zeroRuleSmall = 0;
    opts.zeroFraction[0] = opts.zeroFraction[1] = opts.zeroFraction[2] = -1.0;
  }
  std::vector<supernodal_lu::Supernode<StorageIndex>> supernodes;
  std::vector<StorageIndex> supernodeOfColumn;
  std::vector<std::vector<StorageIndex>> offRows;
  if (n > 0) symbolic::computeSupernodePartition(StorageIndex(n), adjacency, finalParent, opts, supernodes, supernodeOfColumn, offRows);
  std::vector<std::vector<StorageIndex>>().swap(adjacency);

  // Deferred columns become exactly one front: merge every supernode inside
  // [n - d, n) and cut one that straddles the boundary (its lower part keeps the
  // straddling columns as off columns -- a superset of its true structure).
  if (d > 0) {
    const StorageIndex cut = StorageIndex(n - d);
    std::vector<supernodal_lu::Supernode<StorageIndex>> sn2;
    std::vector<std::vector<StorageIndex>> off2;
    for (std::size_t s = 0; s < supernodes.size(); ++s) {
      supernodal_lu::Supernode<StorageIndex> sn = supernodes[s];
      if (sn.lastColumn < cut) {
        sn2.push_back(sn);
        off2.push_back(std::move(offRows[s]));
      } else if (sn.firstColumn < cut) {
        std::vector<StorageIndex> off;
        for (StorageIndex c = cut; c <= sn.lastColumn; ++c) off.push_back(c);
        off.insert(off.end(), offRows[s].begin(), offRows[s].end());
        sn.lastColumn = cut - 1;
        sn2.push_back(sn);
        off2.push_back(std::move(off));
      }
    }
    supernodal_lu::Supernode<StorageIndex> def;
    def.firstColumn = cut;
    def.lastColumn = StorageIndex(n - 1);
    sn2.push_back(def);
    off2.emplace_back();
    supernodes.swap(sn2);
    offRows.swap(off2);
    for (std::size_t s = 0; s < supernodes.size(); ++s)
      for (StorageIndex c = supernodes[s].firstColumn; c <= supernodes[s].lastColumn; ++c)
        supernodeOfColumn[std::size_t(c)] = StorageIndex(s);
  }

  const Index nf = Index(supernodes.size());
  sym.fronts.resize(std::size_t(nf));
  for (Index f = 0; f < nf; ++f) {
    Front& fr = sym.fronts[std::size_t(f)];
    fr.firstCol = supernodes[std::size_t(f)].firstColumn;
    fr.lastCol = supernodes[std::size_t(f)].lastColumn;
    const StorageIndex p = finalParent[std::size_t(fr.lastCol)];
    fr.parent = p < 0 ? StorageIndex(-1) : supernodeOfColumn[std::size_t(p)];
    fr.deferred = d > 0 && f == nf - 1;
    fr.offBegin = Index(sym.offCols.size());
    sym.offCols.insert(sym.offCols.end(), offRows[std::size_t(f)].begin(), offRows[std::size_t(f)].end());
    fr.offEnd = Index(sym.offCols.size());
    const Index npiv = fr.npiv(), noff = fr.noff();
    sym.predictedNnzR += npiv * (npiv + 1) / 2 + npiv * noff;
  }
  // children
  {
    std::vector<Index> count(std::size_t(nf) + 1, 0);
    for (Index f = 0; f < nf; ++f)
      if (sym.fronts[std::size_t(f)].parent >= 0) ++count[std::size_t(sym.fronts[std::size_t(f)].parent) + 1];
    for (Index f = 0; f < nf; ++f) count[std::size_t(f) + 1] += count[std::size_t(f)];
    sym.children.resize(std::size_t(count[std::size_t(nf)]));
    std::vector<Index> fill(count.begin(), count.end() - 1);
    for (Index f = 0; f < nf; ++f) {
      const StorageIndex p = sym.fronts[std::size_t(f)].parent;
      eigen_assert(p < 0 || p > f);
      if (p >= 0) sym.children[std::size_t(fill[std::size_t(p)]++)] = StorageIndex(f);
    }
    for (Index f = 0; f < nf; ++f) {
      sym.fronts[std::size_t(f)].childBegin = count[std::size_t(f)];
      sym.fronts[std::size_t(f)].childEnd = count[std::size_t(f) + 1];
    }
  }
  // rows by front of their leftmost column
  {
    std::vector<Index> count(std::size_t(nf) + 1, 0);
    for (Index i = 0; i < m; ++i)
      if (leftmost[std::size_t(i)] >= 0) ++count[std::size_t(supernodeOfColumn[std::size_t(leftmost[std::size_t(i)])]) + 1];
    for (Index f = 0; f < nf; ++f) count[std::size_t(f) + 1] += count[std::size_t(f)];
    sym.frontRows.resize(std::size_t(count[std::size_t(nf)]));
    std::vector<Index> fill(count.begin(), count.end() - 1);
    for (Index i = 0; i < m; ++i)
      if (leftmost[std::size_t(i)] >= 0)
        sym.frontRows[std::size_t(fill[std::size_t(supernodeOfColumn[std::size_t(leftmost[std::size_t(i)])])]++)] = StorageIndex(i);
    for (Index f = 0; f < nf; ++f) {
      sym.fronts[std::size_t(f)].rowBegin = count[std::size_t(f)];
      sym.fronts[std::size_t(f)].rowEnd = count[std::size_t(f) + 1];
    }
  }
  // levels: a front after all of its children
  {
    std::vector<Index> level(std::size_t(nf), 0);
    Index maxLevel = -1;
    for (Index f = 0; f < nf; ++f) {
      const Front& fr = sym.fronts[std::size_t(f)];
      Index lv = 0;
      for (Index c = fr.childBegin; c < fr.childEnd; ++c)
        lv = std::max(lv, level[std::size_t(sym.children[std::size_t(c)])] + 1);
      level[std::size_t(f)] = lv;
      maxLevel = std::max(maxLevel, lv);
    }
    sym.levels.assign(std::size_t(maxLevel + 1), std::vector<StorageIndex>());
    for (Index f = 0; f < nf; ++f) sym.levels[std::size_t(level[std::size_t(f)])].push_back(StorageIndex(f));
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::analyzePattern(const MatrixType& matrix) {
  m_info = Success;
  m_lastError.clear();
  m_factorized = false;
  m_isInitialized = false;
  releaseNumeric();
  m_rows = matrix.rows();
  m_cols = matrix.cols();
  m_deferredCols.clear();
  m_pattern = matrix;
  m_pattern.makeCompressed();

  using multifrontal_qr::Ordering;
  if (m_ordering != Ordering::Auto) {
    buildSymbolic(m_pattern, m_ordering, m_sym);
  } else {
    buildSymbolic(m_pattern, Ordering::COLAMD, m_sym);
    const double limit = m_maxAtA > 0 ? m_maxAtA : std::max(1e7, 50.0 * double(m_pattern.nonZeros()));
    if (m_cols > 0 && atAPatternSize(m_pattern) <= limit) {
      Symbolic alt;
      buildSymbolic(m_pattern, Ordering::AMD, alt);
      if (alt.predictedNnzR < m_sym.predictedNnzR) m_sym = std::move(alt);
    }
  }
}

// ============================================================================
// numeric factorization
// ============================================================================

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::releaseNumeric() {
  std::vector<FrontFactor>().swap(m_factors);
  std::vector<Contribution>().swap(m_contrib);
  m_pivotCol.clear();
  m_pivotSlot.clear();
  m_internalToPivot.clear();
  m_liveOffset.clear();
  m_deferredOrder.clear();
  m_rotatedFront = -1;
  m_rank = 0;
  m_sigma.resize(0);
  m_rankVerified = false;
  m_nullSpaceReady = false;
  m_nullSpaceTooLarge = false;
  m_nullSpace.resize(0, 0);
  m_nullQR = HouseholderQR<DenseMatrix>();
  m_nullDim = 0;
  m_conditionEstimate = RealScalar(-1);
  m_nnzR = m_nnzH = m_largestFront = 0;
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::computeScaling(const MatrixType& A) {
  using multifrontal_qr::Scaling;
  using multifrontal_qr::detail::inversePowerOfTwo;
  const Index m = A.rows(), n = A.cols();
  Scaling mode = m_scaling;
  if (mode == Scaling::Auto) mode = (m == n) ? Scaling::RowsAndColumns : Scaling::Columns;
  m_rowScale.setOnes(m);
  m_colScale.setOnes(n);
  if (mode == Scaling::RowsAndColumns) {
    RealVector rowMax = RealVector::Zero(m);
    for (Index c = 0; c < n; ++c)
      for (typename MatrixType::InnerIterator it(A, c); it; ++it)
        rowMax[it.row()] = (std::max)(rowMax[it.row()], numext::abs(it.value()));
    for (Index i = 0; i < m; ++i) m_rowScale[i] = inversePowerOfTwo(rowMax[i]);
  }
  if (mode != Scaling::None) {
    for (Index c = 0; c < n; ++c) {
      // Largest entry first, so the 2-norm is taken of values <= 1 and cannot
      // overflow however badly the input is scaled.
      RealScalar big(0);
      for (typename MatrixType::InnerIterator it(A, c); it; ++it)
        big = (std::max)(big, numext::abs(it.value()) * m_rowScale[it.row()]);
      const RealScalar pre = inversePowerOfTwo(big);
      RealScalar sq(0);
      for (typename MatrixType::InnerIterator it(A, c); it; ++it)
        sq += numext::abs2(it.value() * (m_rowScale[it.row()] * pre));
      m_colScale[c] = pre * inversePowerOfTwo(numext::sqrt(sq));
    }
  }
  m_scaled = A;
  m_scaled.makeCompressed();
  for (Index c = 0; c < n; ++c)
    for (typename MatrixType::InnerIterator it(m_scaled, c); it; ++it)
      it.valueRef() *= (m_rowScale[it.row()] * m_colScale[c]);
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::factorize(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == m_rows && matrix.cols() == m_cols && "factorize() needs the analyzed pattern");
  m_info = Success;
  m_lastError.clear();
  m_factorized = false;
  m_isInitialized = false;
  m_repairs = 0;
  releaseNumeric();

  computeScaling(matrix);
  if (m_scaled.nonZeros() != Index(m_sym.rowCols.size())) {
    m_info = InvalidInput;
    m_lastError = "MultifrontalQR: factorize() was given a different sparsity pattern than analyzePattern()";
    return;
  }
  RealScalar maxColNorm(0);
  m_scaledFrobenius = RealScalar(0);
  for (Index c = 0; c < m_cols; ++c) {
    const RealScalar cn = m_scaled.col(c).norm();
    if (!(numext::isfinite)(cn)) {
      m_info = NumericalIssue;
      m_lastError = "MultifrontalQR: the matrix has non-finite entries";
      return;
    }
    maxColNorm = (std::max)(maxColNorm, cn);
    m_scaledFrobenius += cn * cn;
  }
  m_scaledFrobenius = numext::sqrt(m_scaledFrobenius);
  m_absTol = rankTolerance() * (maxColNorm > RealScalar(0) ? maxColNorm : RealScalar(1));

  for (;;) {
    if (m_maxFactorNonzeros > 0 && m_sym.predictedNnzR > m_maxFactorNonzeros) {
      m_info = NumericalIssue;
      m_lastError = "MultifrontalQR: predicted nnz(R) = " + std::to_string(m_sym.predictedNnzR) +
                    " exceeds setMaxFactorNonzeros(" + std::to_string(m_maxFactorNonzeros) + ")";
      return;
    }
    factorOnce();
    if (m_info != Success) return;
    if (!m_verifyRank) break;
    bool needsRepair = false;
    std::vector<StorageIndex> extra;
    verifyRank(needsRepair, extra);
    if (!needsRepair) break;
    if (extra.empty() || m_repairs >= m_maxRepairs) {
      m_rankVerified = false;
      m_lastError = extra.empty()
                        ? "MultifrontalQR: R11 has a singular value below the rank threshold that column deferral "
                          "cannot isolate; the rank is Heath's estimate, unverified"
                        : "MultifrontalQR: rank verification gave up after setMaxRepairs() refactorizations; the "
                          "rank is unverified";
      break;
    }
    ++m_repairs;
    m_deferredCols.insert(m_deferredCols.end(), extra.begin(), extra.end());
    std::sort(m_deferredCols.begin(), m_deferredCols.end());
    const multifrontal_qr::Ordering used = m_sym.ordering;
    buildSymbolic(m_pattern, used, m_sym);
    releaseNumeric();
  }
  m_factorized = true;
  m_isInitialized = true;
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::factorOnce() {
  const Index nf = Index(m_sym.fronts.size());
  // Scaled values in row form.
  m_rowVals.assign(m_sym.rowCols.size(), Scalar(0));
  {
    const Index nnz = m_scaled.nonZeros();
    for (Index p = 0; p < nnz; ++p) m_rowVals[std::size_t(m_sym.entryMap[std::size_t(p)])] = m_scaled.valuePtr()[p];
  }
  m_factors.assign(std::size_t(nf), FrontFactor());
  m_contrib.assign(std::size_t(nf), Contribution());
  std::vector<RealScalar> dropped(std::size_t(nf), RealScalar(0));
  std::vector<char> nonFinite(std::size_t(nf), 0);
  std::vector<Index> entries(std::size_t(nf), 0);
  // A level with at least as many fronts as lanes is dispatched front by
  // front. A narrower one -- near the root, where a few fronts carry most of
  // the work -- runs its fronts in turn and splits each one's trailing updates
  // across the executor instead.
  const Index lanes = Index(m_executor.concurrency());
  for (const std::vector<StorageIndex>& level : m_sym.levels) {
    const bool narrow = m_intraParallel && lanes > 1 && Index(level.size()) < lanes;
    auto run = [&](Index i) {
      const Index f = level[std::size_t(i)];
      bool bad = false;
      processFront(f, narrow, dropped[std::size_t(f)], bad, entries[std::size_t(f)]);
      nonFinite[std::size_t(f)] = bad ? 1 : 0;
    };
    if (narrow) {
      for (Index i = 0; i < Index(level.size()); ++i) run(i);
    } else {
      m_executor.parallelFor(0, Index(level.size()), run);
    }
  }
  std::vector<Contribution>().swap(m_contrib);
  RealScalar d2(0);
  for (Index f = 0; f < nf; ++f) {
    d2 += dropped[std::size_t(f)];
    m_largestFront = (std::max)(m_largestFront, entries[std::size_t(f)]);
    if (nonFinite[std::size_t(f)]) {
      m_info = NumericalIssue;
      m_lastError = "MultifrontalQR: non-finite values appeared during factorization";
    }
  }
  m_droppedNorm = numext::sqrt(d2);
  finishPivotMaps();
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::finishPivotMaps() {
  const Index nf = Index(m_factors.size());
  m_liveOffset.assign(std::size_t(nf) + 1, 0);
  m_rank = 0;
  m_nnzR = m_nnzH = 0;
  m_rotatedFront = -1;
  for (Index f = 0; f < nf; ++f) {
    m_liveOffset[std::size_t(f)] = m_rank;
    const FrontFactor& ff = m_factors[std::size_t(f)];
    if (ff.rotated) m_rotatedFront = f;
    m_rank += ff.rank;
    m_nnzR += ff.R.size() - ff.rank * (ff.rank - 1) / 2;  // the upper trapezoid
    for (std::size_t i = 0; i < ff.tau.size(); ++i) m_nnzH += Index(ff.reach[i]) - Index(i);
    m_nnzH += ff.rotU.size() + ff.rotV.size();
  }
  m_liveOffset[std::size_t(nf)] = m_rank;
  m_pivotCol.assign(std::size_t(m_rank), StorageIndex(-1));
  m_pivotSlot.assign(std::size_t(m_rank), 0);
  m_internalToPivot.assign(std::size_t(m_cols), StorageIndex(-1));
  m_deferredOrder.clear();
  for (Index f = 0; f < nf; ++f) {
    const FrontFactor& ff = m_factors[std::size_t(f)];
    for (Index t = 0; t < ff.rank; ++t) {
      const Index p = m_liveOffset[std::size_t(f)] + t;
      m_pivotSlot[std::size_t(p)] = ff.slots[std::size_t(t)];
      if (!ff.rotated) {
        m_pivotCol[std::size_t(p)] = ff.cols[std::size_t(t)];
        m_internalToPivot[std::size_t(ff.cols[std::size_t(t)])] = StorageIndex(p);
      }
    }
    if (ff.rotated)
      for (StorageIndex g : ff.cols) {
        m_internalToPivot[std::size_t(g)] = StorageIndex(-2);
        m_deferredOrder.push_back(m_sym.internalToOrig[std::size_t(g)]);
      }
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::processFront(Index f, bool intraParallel, RealScalar& dropped,
                                                        bool& nonFinite, Index& frontEntries) {
  const Front& fr = m_sym.fronts[std::size_t(f)];
  const Index npiv = fr.npiv(), noff = fr.noff(), fn = npiv + noff;
  const StorageIndex* off = m_sym.offCols.data() + fr.offBegin;
  const RealScalar tol = m_absTol;
  auto globalOf = [&](Index local) -> StorageIndex {
    return local < npiv ? StorageIndex(fr.firstCol + local) : off[local - npiv];
  };

  // ---- assembly: original rows and children's contribution blocks, sorted by
  //      leftmost column so the front has a staircase shape.
  struct RowRef {
    Index leftmost;
    Index source;         // -1: original row; else position among the front's children
    Index index;          // original row id, or row within the child's block
  };
  std::vector<RowRef> refs;
  refs.reserve(std::size_t(fr.rowEnd - fr.rowBegin));
  for (Index q = fr.rowBegin; q < fr.rowEnd; ++q) {
    const StorageIndex i = m_sym.frontRows[std::size_t(q)];
    refs.push_back({Index(m_sym.rowCols[std::size_t(m_sym.rowPtr[std::size_t(i)])]) - Index(fr.firstCol), -1, Index(i)});
  }
  std::vector<std::vector<Index>> childMap(static_cast<std::size_t>(fr.childEnd - fr.childBegin));
  for (Index ci = fr.childBegin; ci < fr.childEnd; ++ci) {
    const StorageIndex c = m_sym.children[std::size_t(ci)];
    const Front& cf = m_sym.fronts[std::size_t(c)];
    const Contribution& cb = m_contrib[std::size_t(c)];
    std::vector<Index>& map = childMap[std::size_t(ci - fr.childBegin)];
    map.resize(std::size_t(cf.noff()));
    Index pos = 0;
    for (Index t = 0; t < cf.noff(); ++t) {
      const StorageIndex g = m_sym.offCols[std::size_t(cf.offBegin + t)];
      if (g <= fr.lastCol) {
        map[std::size_t(t)] = Index(g) - Index(fr.firstCol);
      } else {
        while (off[pos] < g) ++pos;
        eigen_assert(off[pos] == g && "child structure must be inside the parent front");
        map[std::size_t(t)] = npiv + pos;
      }
    }
    for (Index r = 0; r < cb.C.rows(); ++r) {
      Index t = 0;
      while (t < cb.C.cols() && cb.C(r, t) == Scalar(0)) ++t;
      if (t == cb.C.cols()) continue;  // an all-zero row carries nothing
      refs.push_back({map[std::size_t(t)], ci - fr.childBegin, r});
    }
  }
  std::stable_sort(refs.begin(), refs.end(), [](const RowRef& a, const RowRef& b) { return a.leftmost < b.leftmost; });
  const Index mf = Index(refs.size());
  frontEntries = mf * fn;

  DenseMatrix F = DenseMatrix::Zero(mf, fn);
  std::vector<StorageIndex> slots(static_cast<std::size_t>(mf));
  std::vector<Index> stair(std::size_t(fn), 0);
  {
    Index offPos = 0;
    for (Index r = 0; r < mf; ++r) {
      const RowRef& ref = refs[std::size_t(r)];
      ++stair[std::size_t(ref.leftmost)];
      if (ref.source < 0) {
        const StorageIndex i = StorageIndex(ref.index);
        slots[std::size_t(r)] = i;
        offPos = 0;
        for (Index p = m_sym.rowPtr[std::size_t(i)]; p < m_sym.rowPtr[std::size_t(i) + 1]; ++p) {
          const StorageIndex g = m_sym.rowCols[std::size_t(p)];
          Index local;
          if (g <= fr.lastCol) {
            local = Index(g) - Index(fr.firstCol);
          } else {
            while (off[offPos] < g) ++offPos;
            local = npiv + offPos;
          }
          F(r, local) = m_rowVals[std::size_t(p)];
        }
      } else {
        const Contribution& cb = m_contrib[std::size_t(m_sym.children[std::size_t(fr.childBegin + ref.source)])];
        const std::vector<Index>& map = childMap[std::size_t(ref.source)];
        slots[std::size_t(r)] = cb.slots[std::size_t(ref.index)];
        for (Index t = 0; t < cb.C.cols(); ++t) {
          const Scalar v = cb.C(ref.index, t);
          if (v != Scalar(0)) F(r, map[std::size_t(t)]) = v;
        }
      }
    }
    for (Index j = 1; j < fn; ++j) stair[std::size_t(j)] += stair[std::size_t(j - 1)];
  }
  for (Index ci = fr.childBegin; ci < fr.childEnd; ++ci) {
    Contribution& cb = m_contrib[std::size_t(m_sym.children[std::size_t(ci)])];
    cb.C.resize(0, 0);
    std::vector<StorageIndex>().swap(cb.slots);
  }

  // ---- factorization
  Index k = 0, reach = 0;
  std::vector<Index> hCol, hReach;
  std::vector<Scalar> hTau;
  std::vector<Index> liveLocal, deadLocal;
  std::vector<Index> order(static_cast<std::size_t>(fn));
  std::iota(order.begin(), order.end(), Index(0));  // factored position -> assembled local column
  Vector work(fn + 1);
  RealScalar drop2(0);

  auto householder = [&](Index j, Index len, Index applyEnd) {
    Scalar tau;
    RealScalar beta;
    F.col(j).segment(k, len).makeHouseholderInPlace(tau, beta);
    F(k, j) = beta;
    if (j + 1 < applyEnd)
      F.block(k, j + 1, len, applyEnd - j - 1)
          .applyHouseholderOnTheLeft(F.col(j).segment(k + 1, len - 1), tau, work.data());
    hCol.push_back(j);
    hReach.push_back(reach);
    hTau.push_back(tau);
    ++k;
  };

  // Blocked phase over columns [jBegin, jEnd): unblocked within a panel, then a
  // compact-WY update of the trailing columns restricted to the panel's rows.
  // pivotPhase applies Heath's rule; otherwise zero columns are skipped.
  DenseMatrix V;
  auto blockedPhase = [&](Index jBegin, Index jEnd, bool pivotPhase) {
    Index j = jBegin;
    while (j < jEnd) {
      const Index panelEnd = (std::min)(jEnd, j + m_blockSize);
      const Index k0 = k;
      const std::size_t h0 = hTau.size();
      for (; j < panelEnd; ++j) {
        reach = (std::max)(reach, stair[std::size_t(j)]);
        const Index len = reach - k;
        const RealScalar nrm = len > 0 ? F.col(j).segment(k, len).norm() : RealScalar(0);
        if (!(numext::isfinite)(nrm)) {
          nonFinite = true;
          return;
        }
        if (pivotPhase) {
          if (!(nrm > tol)) {
            drop2 += nrm * nrm;
            if (len > 0) F.col(j).segment(k, len).setZero();
            deadLocal.push_back(j);
            continue;
          }
          liveLocal.push_back(j);
        } else if (nrm == RealScalar(0)) {
          continue;
        }
        householder(j, len, panelEnd);
      }
      const Index p = Index(hTau.size() - h0);
      if (p > 0 && panelEnd < fn) {
        const Index rowsP = reach - k0;
        V.setZero(rowsP, p);
        for (Index i = 0; i < p; ++i) {
          const Index hr = hReach[h0 + std::size_t(i)] - k0;
          V(i, i) = Scalar(1);
          if (hr > i + 1) V.col(i).segment(i + 1, hr - i - 1) = F.col(hCol[h0 + std::size_t(i)]).segment(k0 + i + 1, hr - i - 1);
        }
        // Apply H_p ... H_1 = (I - V T V^H)^H with T built from the conjugated
        // coefficients (Eigen's backward block reflector), to fixed-width column
        // chunks. The chunks are independent, and their width does not depend
        // on the executor, so a parallel factorization is bit-identical to a
        // serial one.
        Map<const Vector> taus(hTau.data() + h0, p);
        DenseMatrix T(p, p);
        internal::make_block_householder_triangular_factor(T, V, taus.conjugate());
        const Index trailingCols = fn - panelEnd;
        const Index chunk = kTrailingChunk;
        const Index chunks = (trailingCols + chunk - 1) / chunk;
        auto update = [&](Index c) {
          const Index c0 = panelEnd + c * chunk;
          const Index width = (std::min)(chunk, fn - c0);
          auto block = F.block(k0, c0, rowsP, width);
          DenseMatrix tmp = V.adjoint() * block;
          tmp = (T.template triangularView<Upper>().adjoint() * tmp).eval();
          block.noalias() -= V * tmp;
        };
        if (intraParallel && chunks > 1 && double(rowsP) * double(trailingCols) * double(p) > 2e6) {
          m_executor.parallelFor(0, chunks, update);
        } else {
          for (Index c = 0; c < chunks; ++c) update(c);
        }
      }
    }
  };

  if (!fr.deferred) {
    blockedPhase(0, npiv, true);
  } else {
    // The deferred block: plain Householder QR (no rank decision), then the
    // SVD of its triangular factor. Column-norm criteria -- Heath's rule, and
    // column pivoting too -- can only overestimate rank, since a column's
    // residual norm is never below the singular value it hides; the SVD decides
    // exactly, and what it discards is the smallest perturbation that achieves
    // the rank.
    eigen_assert(noff == 0 && "the deferred front is the last one");
    blockedPhase(0, npiv, false);
  }
  if (nonFinite) return;
  const Index rank = k;

  // Compress the contribution block when it has more rows than columns: an
  // orthogonal transformation among non-pivot rows is part of Q like any other,
  // and it caps what the parent must assemble at noff rows.
  const bool hasParent = fr.parent >= 0;
  bool compressed = false;
  if (hasParent && noff > 0 && mf - rank > noff) {
    blockedPhase(npiv, fn, false);
    compressed = true;
    if (nonFinite) return;
  }
  dropped = drop2;

  FrontFactor& ff = m_factors[std::size_t(f)];
  const Index nh = Index(hTau.size());
  if (fr.deferred) {
    const Index kq = k;
    DenseMatrix RD = F.topRows(kq);
    for (Index i = 0; i < nh; ++i) {
      const Index hr = (std::min)(Index(hReach[std::size_t(i)]), kq);
      if (hr > i + 1) RD.col(hCol[std::size_t(i)]).segment(i + 1, hr - i - 1).setZero();
    }
    ff.rotated = true;
    RealVector sv;
    if (kq > 0) {
      BDCSVD<DenseMatrix, ComputeFullU | ComputeFullV> svd(RD);
      ff.rotU = svd.matrixU();
      ff.rotV = svd.matrixV();
      sv = svd.singularValues();
    } else {
      ff.rotV = DenseMatrix::Identity(npiv, npiv);
    }
    if (!sv.allFinite()) {
      nonFinite = true;
      return;
    }
    Index live = 0;
    while (live < sv.size() && sv[live] > tol) ++live;
    for (Index i = live; i < sv.size(); ++i) drop2 += sv[i] * sv[i];
    ff.rank = live;
    ff.dead = npiv - live;
    ff.sigma = sv.head(live);
    ff.R.setZero(live, npiv);
    for (Index i = 0; i < live; ++i) ff.R(i, i) = Scalar(sv[i]);
    ff.cols.resize(std::size_t(npiv));
    for (Index j = 0; j < npiv; ++j) ff.cols[std::size_t(j)] = globalOf(j);
    dropped = drop2;
  } else {
    ff.rank = rank;
    ff.dead = Index(deadLocal.size());
    std::vector<Index> colOrder;
    colOrder.reserve(std::size_t(fn));
    colOrder.insert(colOrder.end(), liveLocal.begin(), liveLocal.end());
    colOrder.insert(colOrder.end(), deadLocal.begin(), deadLocal.end());
    for (Index j = npiv; j < fn; ++j) colOrder.push_back(j);
    ff.cols.resize(std::size_t(fn));
    for (Index c = 0; c < fn; ++c) ff.cols[std::size_t(c)] = globalOf(colOrder[std::size_t(c)]);
    ff.R.resize(rank, fn);
    for (Index c = 0; c < fn; ++c) {
      const Index j = colOrder[std::size_t(c)];
      ff.R.col(c) = F.col(j).head(rank);
      if (c < rank) ff.R.col(c).tail(rank - c - 1).setZero();  // below the diagonal: Householder storage
    }
  }
  Index hRows = 0;
  for (Index i = 0; i < nh; ++i) hRows = (std::max)(hRows, hReach[std::size_t(i)]);
  ff.H.setZero(hRows, nh);
  ff.tau = hTau;
  ff.reach.resize(std::size_t(nh));
  for (Index i = 0; i < nh; ++i) {
    const Index hr = hReach[std::size_t(i)];
    ff.reach[std::size_t(i)] = StorageIndex(hr);
    if (hr > i + 1) ff.H.col(i).segment(i + 1, hr - i - 1) = F.col(hCol[std::size_t(i)]).segment(i + 1, hr - i - 1);
  }
  ff.slots.assign(slots.begin(), slots.begin() + hRows);

  if (hasParent && noff > 0) {
    // The compression reflectors are stored below their diagonal, which is
    // inside the block handed to the parent: clear them (H has its own copy),
    // or the parent would assemble Householder vectors as matrix entries.
    for (Index i = rank; i < nh; ++i) {
      const Index hr = hReach[std::size_t(i)];
      if (hr > i + 1) F.col(hCol[std::size_t(i)]).segment(i + 1, hr - i - 1).setZero();
    }
    const Index cEnd = compressed ? k : mf;
    Contribution& cb = m_contrib[std::size_t(f)];
    cb.C = F.block(rank, npiv, cEnd - rank, noff);
    cb.slots.assign(slots.begin() + rank, slots.begin() + cEnd);
  }
}

// ============================================================================
// triangular kernels. Pivot index p runs over the pivots front by front; R11
// is upper triangular in that order. The rotated front, if any, is the last
// one, so it is solved first going up and last going down.
// ============================================================================

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::upperSolve(const Vector& c, Vector& z, const Vector* rotatedDead) const {
  // Solves R11 z_live = c - R12 z_dead; z's dead entries are inputs (for the
  // rotated front, its dead direction coefficients are *rotatedDead, or zero).
  const Index nf = Index(m_factors.size());
  Vector rhs;
  for (Index f = nf - 1; f >= 0; --f) {
    const FrontFactor& ff = m_factors[std::size_t(f)];
    const Index k = ff.rank;
    if (ff.rotated) {
      const Index w = Index(ff.cols.size());
      Vector y = Vector::Zero(w);
      y.head(k) = c.segment(m_liveOffset[std::size_t(f)], k).cwiseQuotient(ff.sigma.template cast<Scalar>());
      if (rotatedDead) y.tail(w - k) = *rotatedDead;
      const Vector x = ff.rotV * y;
      for (Index j = 0; j < w; ++j) z[ff.cols[std::size_t(j)]] = x[j];
      continue;
    }
    if (k == 0) continue;
    const Index w = Index(ff.cols.size());
    rhs = c.segment(m_liveOffset[std::size_t(f)], k);
    for (Index col = k; col < w; ++col) {
      const Scalar v = z[ff.cols[std::size_t(col)]];
      if (v != Scalar(0)) rhs.noalias() -= ff.R.col(col) * v;
    }
    ff.R.topLeftCorner(k, k).template triangularView<Upper>().solveInPlace(rhs);
    for (Index t = 0; t < k; ++t) z[ff.cols[std::size_t(t)]] = rhs[t];
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::upperAdjointSolve(const Vector& g, Vector& h) const {
  // Solves R11^H h = g_live; g is indexed by internal column (for the deferred
  // block, in x-coordinates: only V_live^H g_D is read), h by pivot.
  const Index nf = Index(m_factors.size());
  Vector w = g;
  h.setZero(m_rank);
  Vector rhs, gathered;
  for (Index f = 0; f < nf; ++f) {
    const FrontFactor& ff = m_factors[std::size_t(f)];
    const Index k = ff.rank;
    if (k == 0) continue;
    if (ff.rotated) {
      const Index wd = Index(ff.cols.size());
      Vector gd(wd);
      for (Index j = 0; j < wd; ++j) gd[j] = w[ff.cols[std::size_t(j)]];
      h.segment(m_liveOffset[std::size_t(f)], k) =
          (ff.rotV.leftCols(k).adjoint() * gd).cwiseQuotient(ff.sigma.template cast<Scalar>());
      continue;
    }
    rhs.resize(k);
    for (Index t = 0; t < k; ++t) rhs[t] = w[ff.cols[std::size_t(t)]];
    ff.R.topLeftCorner(k, k).template triangularView<Upper>().adjoint().solveInPlace(rhs);
    h.segment(m_liveOffset[std::size_t(f)], k) = rhs;
    const Index first = k + ff.dead, w0 = Index(ff.cols.size());
    if (w0 > first) {
      gathered.noalias() = ff.R.rightCols(w0 - first).adjoint() * rhs;
      for (Index col = first; col < w0; ++col) w[ff.cols[std::size_t(col)]] -= gathered[col - first];
    }
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::upperMultiply(const Vector& z, Vector& u) const {
  // u = R11 z_live. z must hold live-coordinate data as liveToInternal() lays
  // it out: dead columns are ignored, deferred columns are used as given.
  const Index nf = Index(m_factors.size());
  u.setZero(m_rank);
  Vector x;
  for (Index f = 0; f < nf; ++f) {
    const FrontFactor& ff = m_factors[std::size_t(f)];
    const Index k = ff.rank;
    if (k == 0) continue;
    const Index w = Index(ff.cols.size());
    x.resize(w);
    for (Index col = 0; col < w; ++col) x[col] = z[ff.cols[std::size_t(col)]];
    if (ff.rotated) {
      u.segment(m_liveOffset[std::size_t(f)], k) =
          (ff.rotV.leftCols(k).adjoint() * x).cwiseProduct(ff.sigma.template cast<Scalar>());
      continue;
    }
    for (Index col = 0; col < w; ++col) {
      const bool dead = col >= k && col < k + ff.dead;
      if (dead || m_internalToPivot[std::size_t(ff.cols[std::size_t(col)])] == StorageIndex(-1)) x[col] = Scalar(0);
    }
    u.segment(m_liveOffset[std::size_t(f)], k).noalias() = ff.R * x;
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::liveToInternal(const Vector& v, Vector& z) const {
  z.setZero(m_cols);
  for (Index p = 0; p < m_rank; ++p)
    if (m_pivotCol[std::size_t(p)] >= 0) z[m_pivotCol[std::size_t(p)]] = v[p];
  if (m_rotatedFront >= 0) {
    const FrontFactor& ff = m_factors[std::size_t(m_rotatedFront)];
    const Vector x = ff.rotV.leftCols(ff.rank) * v.segment(m_liveOffset[std::size_t(m_rotatedFront)], ff.rank);
    for (Index j = 0; j < x.size(); ++j) z[ff.cols[std::size_t(j)]] = x[j];
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::internalToLive(const Vector& z, Vector& v) const {
  v.resize(m_rank);
  for (Index p = 0; p < m_rank; ++p)
    if (m_pivotCol[std::size_t(p)] >= 0) v[p] = z[m_pivotCol[std::size_t(p)]];
  if (m_rotatedFront >= 0) {
    const FrontFactor& ff = m_factors[std::size_t(m_rotatedFront)];
    const Index w = Index(ff.cols.size());
    Vector x(w);
    for (Index j = 0; j < w; ++j) x[j] = z[ff.cols[std::size_t(j)]];
    v.segment(m_liveOffset[std::size_t(m_rotatedFront)], ff.rank) = ff.rotV.leftCols(ff.rank).adjoint() * x;
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::applyQAdjoint(Vector& y) const {
  Vector yf, work(1);
  for (const FrontFactor& ff : m_factors) {
    const Index hr = ff.H.rows(), nh = ff.H.cols();
    if (nh == 0) continue;
    yf.resize(hr);
    for (Index r = 0; r < hr; ++r) yf[r] = y[ff.slots[std::size_t(r)]];
    for (Index i = 0; i < nh; ++i) {
      const Index len = Index(ff.reach[std::size_t(i)]) - i;
      yf.segment(i, len).applyHouseholderOnTheLeft(ff.H.col(i).segment(i + 1, len - 1), ff.tau[std::size_t(i)], work.data());
    }
    if (ff.rotated && ff.rotU.rows() > 0) yf.head(ff.rotU.rows()) = ff.rotU.adjoint() * yf.head(ff.rotU.rows());
    for (Index r = 0; r < hr; ++r) y[ff.slots[std::size_t(r)]] = yf[r];
  }
}

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::applyQ(Vector& y) const {
  Vector yf, work(1);
  for (Index f = Index(m_factors.size()) - 1; f >= 0; --f) {
    const FrontFactor& ff = m_factors[std::size_t(f)];
    const Index hr = ff.H.rows(), nh = ff.H.cols();
    if (nh == 0) continue;
    yf.resize(hr);
    for (Index r = 0; r < hr; ++r) yf[r] = y[ff.slots[std::size_t(r)]];
    if (ff.rotated && ff.rotU.rows() > 0) yf.head(ff.rotU.rows()) = ff.rotU * yf.head(ff.rotU.rows());
    for (Index i = nh - 1; i >= 0; --i) {
      const Index len = Index(ff.reach[std::size_t(i)]) - i;
      yf.segment(i, len).applyHouseholderOnTheLeft(ff.H.col(i).segment(i + 1, len - 1),
                                                   numext::conj(ff.tau[std::size_t(i)]), work.data());
    }
    for (Index r = 0; r < hr; ++r) y[ff.slots[std::size_t(r)]] = yf[r];
  }
}

// ============================================================================
// rank verification
// ============================================================================

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::verifyRank(bool& needsRepair, std::vector<StorageIndex>& extra) {
  needsRepair = false;
  extra.clear();
  const Index r = m_rank;
  m_sigma.resize(0);
  m_rankVerified = true;
  if (r == 0) return;

  DenseMatrix X;  // right singular vectors of R11 in live coordinates, ascending sigma
  if (r <= m_denseVerifyLimit) {
    DenseMatrix R11 = DenseMatrix::Zero(r, r);
    Vector e = Vector::Zero(r), z, u;
    for (Index j = 0; j < r; ++j) {
      e[j] = Scalar(1);
      liveToInternal(e, z);
      upperMultiply(z, u);
      R11.col(j) = u;
      e[j] = Scalar(0);
    }
    JacobiSVD<DenseMatrix, ComputeFullV> svd(R11);
    const RealVector s = svd.singularValues();  // descending
    const Index keep = (std::min)(r, Index(8));
    m_sigma = s.tail(keep).reverse();
    X = svd.matrixV().rightCols(keep).rowwise().reverse();
  } else {
    // Block inverse iteration on (R11^H R11)^-1 with Rayleigh-Ritz on R11.
    // Three vectors are enough to see a cluster of small singular values, and
    // the decision needs sigma_min only to within a modest factor of the
    // threshold -- so the iteration stops once the estimate settles to 10%.
    const Index p = (std::min)(r, Index(3));
    std::mt19937 rng(12345u);
    std::normal_distribution<double> gauss(0.0, 1.0);
    X.resize(r, p);
    for (Index j = 0; j < p; ++j)
      for (Index i = 0; i < r; ++i) X(i, j) = Scalar(RealScalar(gauss(rng)));
    X = HouseholderQR<DenseMatrix>(X).householderQ() * DenseMatrix::Identity(r, p);
    Vector g, h, z, u, v;
    RealVector previous;
    for (int it = 0; it < 8; ++it) {
      for (Index j = 0; j < p; ++j) {
        liveToInternal(X.col(j), g);
        upperAdjointSolve(g, h);
        z.setZero(m_cols);
        upperSolve(h, z);
        internalToLive(z, v);
        X.col(j) = v;
      }
      if (!X.allFinite()) {
        // (R11^H R11)^-1 overflowed: R11 is singular to working precision.
        m_sigma = RealVector::Zero(1);
        break;
      }
      X = HouseholderQR<DenseMatrix>(X).householderQ() * DenseMatrix::Identity(r, p);
      DenseMatrix Y(r, p);
      for (Index j = 0; j < p; ++j) {
        liveToInternal(X.col(j), z);
        upperMultiply(z, u);
        Y.col(j) = u;
      }
      JacobiSVD<DenseMatrix, ComputeThinV> svd(Y);
      RealVector sv = svd.singularValues().reverse();
      X = X * svd.matrixV().rowwise().reverse();
      m_sigma = sv;
      if (previous.size() == sv.size() && numext::abs(sv[0] - previous[0]) <= RealScalar(0.1) * sv[0]) break;
      previous = sv;
    }
  }
  if (m_sigma.size() == 0 || m_sigma[0] > m_absTol) return;

  m_rankVerified = false;
  needsRepair = true;
  if (X.cols() == 0 || !X.allFinite()) return;
  Index q = 0;
  while (q < m_sigma.size() && q < X.cols() && m_sigma[q] <= m_absTol) ++q;
  // Defer the ordinary columns the near-null vectors lean on most: a pivoted QR
  // of those vectors, restricted to ordinary pivots, picks a well-conditioned
  // selection. The rotated block needs no candidates -- its own singular values
  // are above the threshold by construction, so every small singular vector of
  // R11 has weight on the ordinary columns.
  std::vector<Index> ordinary;
  for (Index p = 0; p < r; ++p)
    if (m_pivotCol[std::size_t(p)] >= 0) ordinary.push_back(p);
  if (ordinary.empty()) return;
  DenseMatrix Xo(Index(ordinary.size()), q);
  for (std::size_t i = 0; i < ordinary.size(); ++i) Xo.row(Index(i)) = X.row(ordinary[i]).head(q);
  ColPivHouseholderQR<DenseMatrix> sel(Xo.adjoint());
  // A null vector spread over many columns is not isolated by deferring q of
  // them, so the selection grows geometrically with each repair: the q pivoted
  // picks first, then the heaviest remaining columns by weight in the vectors.
  const Index want = (std::min)(Index(ordinary.size()), q << (std::min)(m_repairs, Index(20)));
  std::vector<char> chosen(ordinary.size(), 0);
  std::vector<Index> picks;
  for (Index i = 0; i < (std::min)(q, want); ++i) {
    const Index o = sel.colsPermutation().indices()(i);
    chosen[std::size_t(o)] = 1;
    picks.push_back(o);
  }
  if (Index(picks.size()) < want) {
    std::vector<Index> rest;
    for (std::size_t o = 0; o < ordinary.size(); ++o)
      if (!chosen[o]) rest.push_back(Index(o));
    const RealVector weight = Xo.rowwise().squaredNorm();
    const Index more = want - Index(picks.size());
    std::partial_sort(rest.begin(), rest.begin() + more, rest.end(),
                      [&](Index a, Index b) { return weight[a] > weight[b]; });
    picks.insert(picks.end(), rest.begin(), rest.begin() + more);
  }
  for (Index o : picks) {
    const Index p = ordinary[std::size_t(o)];
    extra.push_back(m_sym.internalToOrig[std::size_t(m_pivotCol[std::size_t(p)])]);
  }
}

// ============================================================================
// solve
// ============================================================================

template <typename MatrixType, typename Executor>
void MultifrontalQR<MatrixType, Executor>::ensureNullSpace() const {
  if (m_nullSpaceReady) return;
  m_nullSpaceReady = true;
  const Index k = m_cols - m_rank;
  m_nullSpace.resize(m_cols, 0);
  m_nullDim = 0;
  if (k <= 0) return;
  if (double(m_cols) * double(k) > m_maxNullSpaceScalars) {
    m_nullSpaceTooLarge = true;
    return;
  }
  // Null space of [R11 R12]: [-R11^-1 R12 e_d; e_d] for every dead column d,
  // mapped back through the column scaling (x = Dc y), then orthonormalized.
  // All k right-hand sides at once, front by front: the same flops as k
  // separate solves, but as dense level-3 operations instead of k passes over
  // the front list.
  DenseMatrix Z = DenseMatrix::Zero(m_cols, k);  // rows: internal columns
  Index col = 0;
  Index rotBegin = -1;
  for (const FrontFactor& ff : m_factors) {
    if (ff.rotated) {
      rotBegin = col;
      col += ff.dead;
      continue;
    }
    for (Index dc = ff.rank; dc < ff.rank + ff.dead; ++dc) Z(ff.cols[std::size_t(dc)], col++) = Scalar(1);
  }
  eigen_assert(col == k);
  DenseMatrix block, gathered;
  for (Index f = Index(m_factors.size()) - 1; f >= 0; --f) {
    const FrontFactor& ff = m_factors[std::size_t(f)];
    const Index w = Index(ff.cols.size());
    if (ff.rotated) {
      // live directions get zero (the right-hand side is zero); dead ones are
      // the unit coefficients of their own null vectors.
      DenseMatrix Y = DenseMatrix::Zero(w, k);
      for (Index t = 0; t < ff.dead; ++t) Y(ff.rank + t, rotBegin + t) = Scalar(1);
      const DenseMatrix X = ff.rotV * Y;
      for (Index j = 0; j < w; ++j) Z.row(ff.cols[std::size_t(j)]) = X.row(j);
      continue;
    }
    const Index kf = ff.rank;
    if (kf == 0) continue;
    gathered.resize(w - kf, k);
    for (Index c2 = kf; c2 < w; ++c2) gathered.row(c2 - kf) = Z.row(ff.cols[std::size_t(c2)]);
    block.noalias() = -(ff.R.rightCols(w - kf) * gathered);
    ff.R.topLeftCorner(kf, kf).template triangularView<Upper>().solveInPlace(block);
    for (Index t = 0; t < kf; ++t) Z.row(ff.cols[std::size_t(t)]) = block.row(t);
  }
  DenseMatrix N(m_cols, k);
  for (Index c2 = 0; c2 < m_cols; ++c2)
    N.row(c2) = Z.row(m_sym.origToInternal[std::size_t(c2)]) * Scalar(m_colScale[c2]);
  Z.resize(0, 0);
  for (Index j = 0; j < k; ++j) N.col(j) /= N.col(j).norm();
  eigen_assert(col == k);
  HouseholderQR<DenseMatrix> qr(N);
  m_nullQR = std::move(qr);
  m_nullDim = k;
}

template <typename MatrixType, typename Executor>
typename MultifrontalQR<MatrixType, Executor>::RealScalar MultifrontalQR<MatrixType, Executor>::conditionEstimate()
    const {
  eigen_assert(m_factorized && "conditionEstimate() before a successful factorize()");
  if (m_conditionEstimate >= RealScalar(0)) return m_conditionEstimate;
  if (m_rank == 0) return m_conditionEstimate = RealScalar(0);
  // ||R11||_1 exactly (column sums in live coordinates), ||R11^-1||_1 by
  // Hager-Higham.
  RealVector colSum = RealVector::Zero(m_rank);
  const FrontFactor* rot = m_rotatedFront < 0 ? nullptr : &m_factors[std::size_t(m_rotatedFront)];
  std::vector<Index> rotPos;
  if (rot) {
    rotPos.assign(std::size_t(m_cols), -1);
    for (std::size_t j = 0; j < rot->cols.size(); ++j) rotPos[std::size_t(rot->cols[j])] = Index(j);
  }
  for (std::size_t f = 0; f < m_factors.size(); ++f) {
    const FrontFactor& ff = m_factors[f];
    if (ff.rotated) {
      colSum.segment(m_liveOffset[f], ff.rank) += ff.sigma;
      continue;
    }
    DenseMatrix toRot;
    if (rot && rot->rank > 0) toRot = DenseMatrix::Zero(ff.rank, rot->rank);
    for (Index col = 0; col < Index(ff.cols.size()); ++col) {
      if (col >= ff.rank && col < ff.rank + ff.dead) continue;
      const StorageIndex g = ff.cols[std::size_t(col)];
      const StorageIndex p = m_internalToPivot[std::size_t(g)];
      if (p >= 0) colSum[p] += ff.R.col(col).cwiseAbs().sum();
      if (p == StorageIndex(-2) && toRot.size() > 0) toRot += ff.R.col(col) * rot->rotV.row(rotPos[std::size_t(g)]).head(rot->rank);
    }
    if (toRot.size() > 0) colSum.segment(m_liveOffset[std::size_t(m_rotatedFront)], rot->rank) += toRot.cwiseAbs().colwise().sum().transpose();
  }
  auto applyInv = [this](const Vector& in, Vector& out) {
    Vector z = Vector::Zero(m_cols);
    upperSolve(in, z);
    internalToLive(z, out);
  };
  auto applyInvAdj = [this](const Vector& in, Vector& out) {
    Vector g;
    liveToInternal(in, g);
    upperAdjointSolve(g, out);
  };
  const RealScalar inv = left_right_lu::oneNormEstimate<Scalar>(m_rank, applyInv, applyInvAdj);
  m_conditionEstimate = colSum.maxCoeff() * inv;
  return m_conditionEstimate;
}

template <typename MatrixType, typename Executor>
template <typename Rhs, typename Dest>
void MultifrontalQR<MatrixType, Executor>::_solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const {
  eigen_assert(m_factorized && "solve() before a successful factorize()");
  eigen_assert(b.rows() == m_rows && "right-hand side has the wrong number of rows");
  const Index m = m_rows, n = m_cols, r = m_rank, nrhs = b.cols();
  dest.derived().resize(n, nrhs);
  m_lastRefinements = 0;
  m_lastResidual = RealScalar(0);
  m_lastOptimality = RealScalar(0);
  const bool minNorm = m_solution == multifrontal_qr::Solution::MinimumNorm && r < n;
  if (minNorm) ensureNullSpace();
  const bool project = minNorm && !m_nullSpaceTooLarge && m_nullDim > 0;
  const bool augmented = !(m == n && r == n);
  const RealScalar eps = NumTraits<RealScalar>::epsilon();

  Vector bs(m), y(m), c(r), z(n), dz(n), zo(n), res(m), rr(m), g(n), gi(n), h, x;
  for (Index col = 0; col < nrhs; ++col) {
    bs = b.col(col).template cast<Scalar>();
    bs.array() *= m_rowScale.array().template cast<Scalar>();

    auto basicSolve = [&](const Vector& rhs, Vector& out) {
      y = rhs;
      applyQAdjoint(y);
      for (Index p = 0; p < r; ++p) c[p] = y[m_pivotSlot[std::size_t(p)]];
      out.setZero(n);
      upperSolve(c, out);
    };
    auto residual = [&](const Vector& zi, Vector& out) {  // out = bs - As z
      toOriginal(zi, zo);
      if (m_extendedResidual) {
        left_right_lu::residualExtended(m_scaled, bs, zo, out);
      } else {
        out = bs - m_scaled * zo;
      }
    };

    basicSolve(bs, z);
    Index steps = 0;
    if (m_maxRefinements > 0 && r > 0) {
      RealScalar prevNorm = NumTraits<RealScalar>::highest();
      if (!augmented) {
        for (; steps < m_maxRefinements; ++steps) {
          residual(z, res);
          basicSolve(res, dz);
          const RealScalar dn = dz.template lpNorm<Infinity>();
          if (!(numext::isfinite)(dn) || dn >= prevNorm) break;
          z += dz;
          prevNorm = RealScalar(0.5) * dn;
          if (dn <= eps * z.template lpNorm<Infinity>()) break;
        }
      } else {
        // Bjorck's refinement of the augmented system [I B; B^H 0][r; z] = [b; 0],
        // B the live columns: f = b - r - B z, g = -B^H r, and the correction
        // is dr = Q [h; d2], dz = R11^-1 (d1 - h) with R11^H h = g, d = Q^H f.
        toOriginal(z, zo);
        Vector rv = bs - m_scaled * zo;
        Vector e(m);
        for (; steps < m_maxRefinements; ++steps) {
          residual(z, rr);
          Vector fv = rr - rv;
          Vector gzero = Vector::Zero(n);
          if (m_extendedResidual) {
            left_right_lu::residualExtendedTransposed<true>(m_scaled, gzero, rv, g);
          } else {
            g = -(m_scaled.adjoint() * rv);
          }
          toInternal(g, gi);
          for (Index q = 0; q < n; ++q)
            if (m_internalToPivot[std::size_t(q)] == StorageIndex(-1)) gi[q] = Scalar(0);
          e = fv;
          applyQAdjoint(e);
          upperAdjointSolve(gi, h);
          for (Index p = 0; p < r; ++p) c[p] = e[m_pivotSlot[std::size_t(p)]] - h[p];
          dz.setZero(n);
          upperSolve(c, dz);
          for (Index p = 0; p < r; ++p) e[m_pivotSlot[std::size_t(p)]] = h[p];
          applyQ(e);
          const RealScalar dn = dz.template lpNorm<Infinity>();
          if (!(numext::isfinite)(dn) || dn >= prevNorm) break;
          z += dz;
          rv += e;
          prevNorm = RealScalar(0.5) * dn;
          if (dn <= eps * z.template lpNorm<Infinity>()) break;
        }
      }
    }
    m_lastRefinements = (std::max)(m_lastRefinements, steps);

    toOriginal(z, x);
    x.array() *= m_colScale.array().template cast<Scalar>();
    if (project) {
      // x - N N^H x = Q diag(0, I) Q^H x with N the first nullDim columns of Q,
      // applied in factored form. Q is exactly unitary, so one pass suffices.
      const auto Qseq = m_nullQR.householderQ();
      x.applyOnTheLeft(Qseq.adjoint());
      x.head(m_nullDim).setZero();
      x.applyOnTheLeft(Qseq);
    }
    dest.col(col) = x;
  }

  // Diagnostics in the ORIGINAL matrix: A = Dr^-1 As Dc^-1.
  m_solveNote.clear();
  if (minNorm && !project && m_nullSpaceTooLarge)
    m_solveNote = "MultifrontalQR: null space too large for setMaxNullSpaceScalars(); returned the basic solution";
  RealScalar worstRes(0), worstOpt(0);
  RealScalar anorm(0);
  for (Index cc = 0; cc < n; ++cc)
    for (typename MatrixType::InnerIterator it(m_scaled, cc); it; ++it)
      anorm += numext::abs2(it.value() / (m_rowScale[it.row()] * m_colScale[cc]));
  anorm = numext::sqrt(anorm);
  for (Index col = 0; col < nrhs; ++col) {
    Vector bc = b.col(col).template cast<Scalar>();
    Vector xc = dest.col(col);
    Vector xs = xc.cwiseQuotient(m_colScale.template cast<Scalar>());
    Vector rs = m_scaled * xs;                                   // Dr A x
    Vector rOrig = bc - rs.cwiseQuotient(m_rowScale.template cast<Scalar>());
    const RealScalar bn = bc.norm();
    const RealScalar rn = rOrig.norm();
    worstRes = (std::max)(worstRes, bn > RealScalar(0) ? rn / bn : rn);
    // The optimality ratio is 0/0 noise once r is at rounding level -- a
    // consistent system solved to 1e-16 would read as far from optimal -- so it
    // is only measured for a residual clearly above that level.
    if (rn > RealScalar(100) * eps * (anorm * xc.norm() + bn)) {
      // ||A^H r|| / (||A||_F ||r||), with A^H r = Dc^-1 As^H Dr^-1 r.
      Vector w = rOrig.cwiseQuotient(m_rowScale.template cast<Scalar>());
      Vector atr = (m_scaled.adjoint() * w).cwiseQuotient(m_colScale.template cast<Scalar>());
      worstOpt = (std::max)(worstOpt, atr.norm() / (anorm * rn));
    }
  }
  m_lastResidual = worstRes;
  m_lastOptimality = worstOpt;
}

template <typename MatrixType, typename Executor>
SparseMatrix<typename MatrixType::Scalar, ColMajor, typename MatrixType::StorageIndex>
MultifrontalQR<MatrixType, Executor>::matrixR() const {
  eigen_assert(m_factorized);
  // Column positions in colsPermutation() order: ordinary live pivots, dead
  // columns, then the deferred block (whose entries are rotated by V).
  std::vector<StorageIndex> position(std::size_t(m_cols), 0);
  Index pos = 0;
  for (Index p = 0; p < m_rank; ++p)
    if (m_pivotCol[std::size_t(p)] >= 0) position[std::size_t(m_pivotCol[std::size_t(p)])] = StorageIndex(pos++);
  for (StorageIndex c : deadColumns()) position[std::size_t(m_sym.origToInternal[std::size_t(c)])] = StorageIndex(pos++);
  const Index rotStart = pos;
  const FrontFactor* rot = m_rotatedFront < 0 ? nullptr : &m_factors[std::size_t(m_rotatedFront)];
  std::vector<Index> rotPos(std::size_t(m_cols), -1);
  if (rot)
    for (std::size_t j = 0; j < rot->cols.size(); ++j) rotPos[std::size_t(rot->cols[j])] = Index(j);
  std::vector<Triplet<Scalar, StorageIndex>> t;
  for (std::size_t f = 0; f < m_factors.size(); ++f) {
    const FrontFactor& ff = m_factors[f];
    const StorageIndex row0 = StorageIndex(m_liveOffset[f]);
    if (ff.rotated) {
      for (Index i = 0; i < ff.rank; ++i) t.emplace_back(StorageIndex(row0 + i), StorageIndex(rotStart + i), ff.R(i, i));
      continue;
    }
    DenseMatrix toRot;
    if (rot) toRot = DenseMatrix::Zero(ff.rank, rot->rotV.cols());
    for (Index col = 0; col < Index(ff.cols.size()); ++col) {
      const StorageIndex g = ff.cols[std::size_t(col)];
      if (rot && rotPos[std::size_t(g)] >= 0) {
        toRot += ff.R.col(col) * rot->rotV.row(rotPos[std::size_t(g)]);
        continue;
      }
      for (Index row = 0; row < ff.rank; ++row) {
        const Scalar v = ff.R(row, col);
        if (v != Scalar(0)) t.emplace_back(StorageIndex(row0 + row), position[std::size_t(g)], v);
      }
    }
    for (Index j = 0; j < toRot.cols(); ++j)
      for (Index row = 0; row < ff.rank; ++row)
        if (toRot(row, j) != Scalar(0)) t.emplace_back(StorageIndex(row0 + row), StorageIndex(rotStart + j), toRot(row, j));
  }
  SparseMatrix<Scalar, ColMajor, StorageIndex> R(m_rank, m_cols);
  R.setFromTriplets(t.begin(), t.end());
  return R;
}

}  // namespace Eigen

#endif  // MULTIFRONTAL_QR_H
