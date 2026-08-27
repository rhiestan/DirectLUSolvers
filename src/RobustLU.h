// RobustLU -- a fallback ladder, not a single strategy.
//
// Every other solver here commits to one approach and reports honestly when it
// fails. That is the right contract for a solver, and it leaves the caller
// holding a question they usually cannot answer: WHICH other setting would have
// worked? This class answers it by trying, and by saying afterwards what it did.
//
// WHAT THE LADDER IS FOR, measured rather than assumed
//
// Over this project's 33-matrix SuiteSparse quick tier:
//
//   * 25 of 33 succeed on the first rung. The ladder must -- and does -- cost
//     them exactly one factorization and nothing else.
//   * 4 are rescued by MC64 matching (Chebyshev3, CAG_mat1916, fd12, lhr10c).
//     By far the highest-value rung.
//   * 1 is rescued by true partial pivoting (shyy41), which LeftRightLU cannot
//     do by construction -- so that rung delegates to PointBlockLU.
//   * 3 cannot be rescued at all (rw5151, foldoc, SmaGri). Recognising them
//     early matters as much as fixing the others: an MC64 attempt on lhr10c
//     costs 11.9 seconds, and spending that to confirm a failure is a bad trade.
//
// DIAGNOSIS-DIRECTED, NOT CHEAPEST-FIRST
//
// The obvious design climbs from cheap rungs to expensive ones. The measurement
// above says that is wrong. Extended-precision residuals need no refactorization
// at all -- the cheapest possible rung -- and they rescue NOTHING here, because
// every failure above is a BACKWARD-error failure and extended precision only
// converts an already-small backward error into a small forward one.
//
// So this class does not climb. It measures, diagnoses, and jumps to the rung
// that addresses the failure it actually observed:
//
//   backward error ~ eps, forward error small  -> done
//   structurally singular                      -> stop: no LU can fix this
//   backward error ~ eps, forward error large,
//       and kappa * eps < 1                    -> extended-residual RE-SOLVE
//   backward error ~ eps, forward error large,
//       and kappa * eps >= 1                   -> stop: the matrix, not the solver
//   backward error >> eps                      -> MC64 (bad diagonal)
//   backward error >> eps after MC64           -> true partial pivoting
//
// The stopping rules are the reason this step came after condition estimation.
// "The solver did badly" and "the matrix is unsolvable in this precision" look
// identical from a residual, and only the first is worth escalating. A small
// backward error next to a huge condition number is a solver that did
// everything right on a problem that has no answer in double.
//
// WHAT IT COSTS WHEN IT WORKS FIRST TIME
//
// One factorization, one solve, one O(nnz) backward error, and one condition
// estimate (4-5 triangular solves). No refactorization, no extra ordering. The
// diagnosis is cheap; only the rungs are expensive, and they run only when the
// diagnosis calls for them.
//
// WHY A SEPARATE CLASS
//
// Escalation is policy; LeftRightLU is mechanism. Keeping them apart preserves
// that class's contract -- one strategy, no hidden cost, nothing running that
// the caller did not ask for -- and avoids coupling it to PointBlockLU. Callers
// who want a single strategy keep using LeftRightLU and pay none of this.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef ROBUST_LU_H
#define ROBUST_LU_H

#include <Eigen/SparseCore>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>

#include "LeftRightLU.h"
#include "PointBlockLU.h"

namespace Eigen {

namespace robust_lu {

/** The rungs, in the order they are numbered rather than the order they run --
  * which rung runs next is a function of the diagnosis, not of this list. */
enum class Strategy {
  Default,           ///< LeftRightLU as it ships: equilibration, transversal matching, BTF.
  ExtendedResidual,  ///< Same factorization, re-solved with a double-double residual.
  MC64,              ///< Re-analyze with the exact maximum-product matching.
  MC64Extended,      ///< MC64 plus the extended-precision residual.
  PartialPivoting,   ///< PointBlockLU: true threshold pivoting, no perturbation.
  RankRevealing      ///< SparseQR: least squares plus a rank, when no LU exists.
};

const char* strategyName(Strategy s);

/** Why the ladder stopped where it did. */
enum class Outcome {
  Solved,               ///< A rung produced a trustworthy answer to A x = b.
  RankDeficient,        ///< A verified LEAST-SQUARES answer, plus a rank. See below.
  StructurallySingular, ///< No LU exists and rank-revealing was unavailable or refused.
  IllConditioned,       ///< Backward-stable but unsolvable in this precision.
  Exhausted,            ///< Every permitted rung was tried and none succeeded.
  Infeasible            ///< Predicted fill exceeded the guard on every rung.
};

const char* outcomeName(Outcome o);

/** One rung's attempt, kept whether it succeeded or not: the log IS the report. */
struct Attempt {
  Strategy strategy = Strategy::Default;
  bool factored = false;
  bool accepted = false;
  double backwardError = 0;     ///< Oettli-Prager omega; the acceptance criterion.
  double conditionEstimate = 0;
  double growthFactor = 0;
  double probeResidual = 0;
  long long replacedPivots = 0;
  long long factorNonzeros = -1;      ///< nnzL + nnzU for an LU rung; nnz(R) for QR.
  long long denseRows = -1;           ///< Vertices above AMD's dense threshold; -1 for QR.
  long long rank = -1;                ///< Rank-revealing rung only; -1 elsewhere.
  double leastSquaresOptimality = 0;  ///< ||A^T r|| / (||A|| ||r||); see acceptable().
  double milliseconds = 0;
  std::string note;             ///< Why it was rejected, when it was.
};

}  // namespace robust_lu

/** \brief Sparse LU that escalates through fallback strategies rather than
  *        failing, and reports what it tried.
  *
  * \tparam MatrixType_ a column-major Eigen::SparseMatrix.
  *
  * Usage is the ordinary Eigen solver interface:
  * \code
  *   Eigen::RobustLU<Eigen::SparseMatrix<double>> solver;
  *   solver.compute(A);
  *   Eigen::VectorXd x = solver.solve(b);
  *   if (solver.info() != Eigen::Success) std::cerr << solver.report();
  * \endcode
  *
  * Unlike every other option in this project, escalation is ON by default here:
  * choosing this class IS the opt-in.
  */
template <typename MatrixType_>
class RobustLU : public SparseSolverBase<RobustLU<MatrixType_>> {
 protected:
  typedef SparseSolverBase<RobustLU<MatrixType_>> Base;
  using Base::m_isInitialized;

 public:
  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrix;
  typedef Matrix<Scalar, Dynamic, 1> DenseVector;
  typedef Eigen::LeftRightLU<MatrixType> DirectSolver;
  typedef Eigen::PointBlockLU<MatrixType> PivotingSolver;
  // COLAMD rather than AMD: QR's fill follows the pattern of A^T A, which is
  // what COLAMD is designed to order.
  typedef Eigen::SparseQR<MatrixType, COLAMDOrdering<StorageIndex>> RankRevealingSolver;

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  using Base::_solve_impl;

  RobustLU() { init(); }
  explicit RobustLU(const MatrixType& matrix) {
    init();
    compute(matrix);
  }

  void compute(const MatrixType& matrix);

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

  inline Index rows() const { return m_size; }
  inline Index cols() const { return m_size; }
  ComputationInfo info() const { return m_info; }
  const std::string& lastErrorMessage() const { return m_lastError; }

  /** The strategy that produced the factorization in use. */
  robust_lu::Strategy strategy() const { return m_strategy; }
  /** Why the ladder stopped. */
  robust_lu::Outcome outcome() const { return m_outcome; }
  /** Every rung tried, in order, accepted or not. */
  const std::vector<robust_lu::Attempt>& attempts() const { return m_attempts; }
  /** A human-readable account of what was tried and why. */
  std::string report() const;

  /** How far the ladder may climb. Default: all the way.
    *  Set to Strategy::Default to disable escalation entirely, which makes this
    *  class a thin wrapper over LeftRightLU plus the diagnosis. */
  void setMaxStrategy(robust_lu::Strategy s) { m_maxStrategy = s; }
  robust_lu::Strategy maxStrategy() const { return m_maxStrategy; }

  /** Backward error above which a rung is rejected. Default 1e-6.
    *
    *  Calibrated, not guessed. Over this project's SuiteSparse quick tier the
    *  omega of every matrix that solves acceptably tops out at 3.9e-08 (median
    *  7.8e-16), while the worst omega of any matrix that does NOT starts at
    *  1.2e-02 -- six clear orders of magnitude of gap. 1e-6 sits in the middle
    *  of it with four orders of margin on each side, and matches the scale
    *  LeftRightLU::setSolveFailureThreshold already uses.
    *
    *  A tighter threshold is expensive rather than safer: the first version of
    *  this class used 64*eps and escalated 8 matrices that were already fine,
    *  at up to 11.9 seconds each. */
  void setBackwardErrorTolerance(const RealScalar& tol) { m_backwardTolerance = tol; }
  RealScalar backwardErrorTolerance() const { return m_backwardTolerance; }

  /** Relative residual above which a rung is rejected. Default 1e-6, matching
    *  LeftRightLU::setSolveFailureThreshold -- a rung must satisfy BOTH this and
    *  the backward-error tolerance (see acceptable()). */
  void setResidualTolerance(const RealScalar& tol) { m_residualTolerance = tol; }
  RealScalar residualTolerance() const { return m_residualTolerance; }

  /** Fill guard applied to every rung (see LeftRightLU::setMaxFactorNonzeros). */
  void setMaxFactorNonzeros(Index limit) { m_maxFactorNonzeros = limit; }
  Index maxFactorNonzeros() const { return m_maxFactorNonzeros; }

  /** The right-hand side the ladder uses to judge a rung.
    *
    * The awkward part of the design, stated plainly: a backward error needs a
    * right-hand side, and compute() has none. The ladder therefore judges each
    * rung against a deterministic probe -- b = A*x with x_i = 1 + sin(i)/2 --
    * because a factorization broken enough to matter is broken for essentially
    * any b. Supply your own when you know it; the per-solve honesty check in
    * LeftRightLU still measures the REAL right-hand side afterwards either way. */
  void setProbeRightHandSide(const DenseVector& b) { m_probe = b; }

  // --- dense rows -----------------------------------------------------------

  /** Vertices above AMD's dense threshold in the eliminated graph, or -1 if no
    *  LU rung ran. Free -- measured during the symbolic analysis. */
  Index denseRowCount() const { return m_denseRowCount; }

  /** How much fill those vertices are costing, as a ratio against what a
    *  bordered factorization could achieve (see
    *  LeftRightLU::denseRowFillPenalty). NaN unless the ladder had to escalate.
    *
    *  Computed only when a rung is rejected, because that is the only time it is
    *  actionable, and it costs a symbolic analysis -- trivial next to the
    *  refactorization the escalation is about to do, but not worth imposing on
    *  the 19 of 33 corpus matrices that never leave the first rung.
    *
    *  **The usual response is a different ordering, not a different solver.**
    *  AMD already orders dense vertices last, and on the corpus matrices with a
    *  handful of very dense rows this measures exactly 1.00 -- meaning they cost
    *  nothing at all. Where it is larger the cause is many moderately dense
    *  rows, and METIS or COLAMD often recovers most of it. */
  double denseRowFillPenalty() const { return m_denseRowPenalty; }

  /** Numerical rank, from the rank-revealing rung. -1 when that rung never ran. */
  Index rank() const { return m_rank; }

  /** True when the answer is a LEAST-SQUARES solution rather than a solution of
    *  A x = b, i.e. when outcome() == RankDeficient.
    *
    *  Worth checking explicitly. The answer is a BASIC least-squares solution --
    *  free variables set to zero -- not the minimum-norm one, which would need a
    *  complete orthogonal decomposition that Eigen has no sparse version of. On
    *  a matrix with a large null space the two differ substantially: measured on
    *  Pajek/SmaGri (rank 511 of 1059) the basic solution has norm 288 against
    *  the reference solution's 35. Both satisfy A x = b to machine precision;
    *  they simply differ by a null-space vector. If your problem cares which
    *  solution it gets, this rung is not enough on its own. */
  bool isLeastSquares() const { return m_outcome == robust_lu::Outcome::RankDeficient; }

  /** Guards on what the rank-revealing rung will attempt.
    *
    * QR is not a cheaper LU, and the guard has to be on FILL rather than on
    * size -- measured, after a first version guarded on rows and would have let
    * Pajek/foldoc run for over 25 minutes at only 13356 rows:
    *
    *     matrix     LU fill    QR time   QR outcome
    *     SmaGri       348 k    0.14 s    rank 511/1059, residual 2.7e-16
    *     shyy41       203 k     7.8 s    rank 4718/4720
    *     rw5151       583 k      26 s    rank 5150/5151 -- nothing else could
    *     lhr10c      57.1 M     265 s    LU already had a better answer
    *     foldoc      51.6 M     515 s    rank 12919/13356, residual 5.8e-14
    *
    * The separation is four orders of magnitude wide in LU fill and clean, so
    * the LU fill an earlier rung already measured is the predictor: if LU could
    * not stay under it, QR will not either. Default 5e6 scalars, midway across
    * that gap. Set 0 to disable.
    *
    * Note what the default gives up, because it is a real trade rather than a
    * free win: Pajek/foldoc IS solvable this way -- 8.6 minutes for a correct
    * answer where every LU strategy returns 1.8e+33 -- and the default guard
    * declines it. A silent 8.6-minute stall is judged the worse failure, and the
    * rung is logged as declined with its reason, so a caller who wants that
    * trade can raise the guard and get it.
    *
    * The row cap is a backstop for the case where no LU rung produced a fill
    * figure at all. Default 50000; set 0 to disable. */
  void setMaxRankRevealingFill(long long scalars) { m_maxRankRevealingFill = scalars; }
  long long maxRankRevealingFill() const { return m_maxRankRevealingFill; }
  void setMaxRankRevealingSize(Index rows) { m_maxRankRevealingRows = rows; }
  Index maxRankRevealingSize() const { return m_maxRankRevealingRows; }

  /** Diagnostics from the accepted rung. */
  RealScalar backwardError() const { return m_backwardError; }
  RealScalar conditionEstimate() const { return m_conditionEstimate; }
  Index nnzL() const;
  Index nnzU() const;

 private:
  void init() {
    m_size = 0;
    m_info = InvalidInput;
    m_strategy = robust_lu::Strategy::Default;
    m_outcome = robust_lu::Outcome::Exhausted;
    m_maxStrategy = robust_lu::Strategy::RankRevealing;  // the full ladder
    m_backwardTolerance = RealScalar(1e-6);
    m_residualTolerance = RealScalar(1e-6);
    m_maxRankRevealingRows = 50000;
    m_maxRankRevealingFill = 5000000;
    m_leastSquaresTolerance = RealScalar(1e-8);
    m_rank = -1;
    m_denseRowCount = -1;
    m_denseRowPenalty = NumTraits<double>::quiet_NaN();
    m_maxFactorNonzeros = 0;
    m_backwardError = NumTraits<RealScalar>::quiet_NaN();
    m_conditionEstimate = NumTraits<RealScalar>::quiet_NaN();
    m_usePivoting = false;
    m_useRankRevealing = false;
    m_isInitialized = false;
  }

  /** Configure and run one LeftRightLU rung; returns its attempt record. */
  robust_lu::Attempt runDirect(const MatrixType& matrix, robust_lu::Strategy s);
  /** Re-solve the EXISTING factorization with a double-double residual. No
    *  refactorization, no re-analysis -- this is the one genuinely cheap rung. */
  robust_lu::Attempt resolveExtended(const MatrixType& matrix);
  robust_lu::Attempt runPivoting(const MatrixType& matrix);
  robust_lu::Attempt runRankRevealing(const MatrixType& matrix);
  /** The terminal rung, entered only from a diagnosis that says no LU exists. */
  bool tryRankRevealing(const MatrixType& matrix);
  DenseVector probeFor(const MatrixType& matrix) const;
  /** Whether a rung's result may be handed to the caller.
    *
    * BOTH conditions are needed, and the second was learned the hard way. The
    * componentwise backward error is relative to |A||x| + |b|, so a solution
    * whose magnitude has blown up makes omega look excellent while ||Ax-b||/||b||
    * sits at 0.17 -- measured on Bai/rw5151, which this class accepted and
    * returned as a success until the residual joined the test. An unflagged
    * wrong answer is the one outcome this class exists to prevent, so the
    * criterion is the conjunction. */
  static std::string rejection(const robust_lu::Attempt& a) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "rejected: omega=%.2e, relative residual=%.2e", a.backwardError,
                  a.probeResidual);
    return buf;
  }
  /** Acceptance for the rank-revealing rung, which needs a DIFFERENT test.
    *
    * Demanding a small residual would be wrong: on an inconsistent system a
    * nonzero residual IS the answer, and least squares is what was asked for.
    * The right test is that the residual is orthogonal to the range of A.
    *
    * But that test is only meaningful when the residual is genuinely nonzero.
    * When the system turns out consistent and is solved exactly, ||r|| ~ 1e-16
    * and ||A^T r|| / (||A|| ||r||) becomes a ratio of two tiny numbers -- pure
    * noise. Measured on Pajek/SmaGri, solved to a relative residual of 2.7e-16,
    * that ratio reads 9.9e-02, which would reject a perfect answer. So: use the
    * residual test when the residual is negligible, the optimality test when it
    * is not. Getting this the wrong way round rejects every good answer. */
  bool acceptableLeastSquares(const robust_lu::Attempt& a, bool finite) const {
    if (!finite) return false;
    const RealScalar resid = RealScalar(a.probeResidual);
    const RealScalar optimality = RealScalar(a.leastSquaresOptimality);
    if (!(numext::isfinite)(resid) || !(numext::isfinite)(optimality)) return false;
    if (resid <= m_residualTolerance) return true;   // consistent, and solved
    return optimality <= m_leastSquaresTolerance;    // inconsistent, but optimal
  }

  bool acceptable(const robust_lu::Attempt& a, bool finite) const {
    if (!finite) return false;
    const RealScalar omega = RealScalar(a.backwardError);
    const RealScalar resid = RealScalar(a.probeResidual);
    if (!(numext::isfinite)(omega) || !(numext::isfinite)(resid)) return false;
    return omega <= m_backwardTolerance && resid <= m_residualTolerance;
  }

  StorageIndex m_size;
  ComputationInfo m_info;
  std::string m_lastError;
  robust_lu::Strategy m_strategy;
  robust_lu::Outcome m_outcome;
  robust_lu::Strategy m_maxStrategy;
  RealScalar m_backwardTolerance;
  RealScalar m_residualTolerance;
  Index m_maxRankRevealingRows;
  long long m_maxRankRevealingFill;
  RealScalar m_leastSquaresTolerance;
  Index m_rank;
  Index m_denseRowCount;
  double m_denseRowPenalty;
  Index m_maxFactorNonzeros;
  RealScalar m_backwardError;
  RealScalar m_conditionEstimate;
  std::vector<robust_lu::Attempt> m_attempts;
  DenseVector m_probe;

  // Which solver holds the accepted factors. Only one is ever true.
  bool m_usePivoting;
  bool m_useRankRevealing;
  // Held by pointer, not by value: neither solver is copy-assignable (a thread
  // pool and a mutex among the members), and each rung needs a clean instance
  // rather than one carrying the previous rung's settings.
  std::unique_ptr<DirectSolver> m_direct;
  std::unique_ptr<PivotingSolver> m_pivoting;
  std::unique_ptr<RankRevealingSolver> m_rankRevealing;
};

// ===========================================================================
//  Implementation
// ===========================================================================

namespace robust_lu {

inline const char* strategyName(Strategy s) {
  switch (s) {
    case Strategy::Default: return "default (transversal matching + BTF)";
    case Strategy::ExtendedResidual: return "extended-precision residual re-solve";
    case Strategy::MC64: return "MC64 maximum-product matching";
    case Strategy::MC64Extended: return "MC64 + extended-precision residual";
    case Strategy::PartialPivoting: return "true partial pivoting (PointBlockLU)";
    default: return "rank-revealing QR (least squares)";
  }
}

inline const char* outcomeName(Outcome o) {
  switch (o) {
    case Outcome::Solved: return "solved";
    case Outcome::RankDeficient: return "rank deficient -- least-squares answer";
    case Outcome::StructurallySingular: return "structurally singular";
    case Outcome::IllConditioned: return "too ill-conditioned for this precision";
    case Outcome::Infeasible: return "predicted fill exceeds the guard";
    default: return "no strategy succeeded";
  }
}

}  // namespace robust_lu

template <typename MatrixType>
typename RobustLU<MatrixType>::DenseVector RobustLU<MatrixType>::probeFor(
    const MatrixType& matrix) const {
  if (m_probe.size() == matrix.rows()) return m_probe;
  DenseVector x(matrix.rows());
  for (Index i = 0; i < matrix.rows(); ++i)
    x[i] = Scalar(RealScalar(1) + RealScalar(0.5) * numext::sin(RealScalar(i)));
  return matrix * x;
}

template <typename MatrixType>
robust_lu::Attempt RobustLU<MatrixType>::runDirect(const MatrixType& matrix,
                                                   robust_lu::Strategy s) {
  using namespace robust_lu;
  Attempt a;
  a.strategy = s;
  const auto t0 = std::chrono::steady_clock::now();

  m_direct.reset(new DirectSolver());
  m_direct->setErrorBounds(true);
  if (m_maxFactorNonzeros > 0) m_direct->setMaxFactorNonzeros(m_maxFactorNonzeros);
  if (s == Strategy::MC64 || s == Strategy::MC64Extended)
    m_direct->setMatchingMethod(supernodal_lu::MatchingMethod::MC64);
  if (s == Strategy::ExtendedResidual || s == Strategy::MC64Extended) {
    m_direct->setExtendedPrecisionResidual(true);
    m_direct->setRefinementMethod(left_right_lu::Refinement::IterativeRefinement);
    m_direct->setRefineOnlyIfPerturbed(false);
  }

  m_direct->compute(matrix);
  const auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  if (m_direct->info() != Success) {
    a.milliseconds = elapsed();
    a.note = m_direct->lastErrorMessage();
    return a;
  }
  a.factored = true;

  const DenseVector probe = probeFor(matrix);
  const DenseVector x = m_direct->solve(probe);
  a.milliseconds = elapsed();
  a.backwardError = double(m_direct->lastBackwardError());
  a.conditionEstimate = double(m_direct->conditionEstimate());
  a.growthFactor = double(m_direct->growthFactor());
  a.replacedPivots = (long long)m_direct->replacedPivots();
  a.factorNonzeros = (long long)m_direct->nnzL() + m_direct->nnzU();
  a.denseRows = (long long)m_direct->denseRowCount();
  const RealScalar bn = probe.norm();
  a.probeResidual = double(bn > RealScalar(0) ? (matrix * x - probe).norm() / bn
                                              : (matrix * x - probe).norm());
  a.accepted = acceptable(a, x.allFinite());
  if (!a.accepted) a.note = rejection(a);
  return a;
}

template <typename MatrixType>
robust_lu::Attempt RobustLU<MatrixType>::resolveExtended(const MatrixType& matrix) {
  using namespace robust_lu;
  Attempt a;
  a.strategy = Strategy::ExtendedResidual;
  const auto t0 = std::chrono::steady_clock::now();

  // The factorization is reused untouched: only the refinement settings change,
  // and those are consumed at solve() time. That is what makes this rung cost
  // one solve rather than one factorization.
  m_direct->setExtendedPrecisionResidual(true);
  m_direct->setRefinementMethod(left_right_lu::Refinement::IterativeRefinement);
  m_direct->setRefineOnlyIfPerturbed(false);

  const DenseVector probe = probeFor(matrix);
  const DenseVector x = m_direct->solve(probe);
  a.milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  a.factored = true;
  a.backwardError = double(m_direct->lastBackwardError());
  a.conditionEstimate = double(m_direct->conditionEstimate());
  a.growthFactor = double(m_direct->growthFactor());
  a.replacedPivots = (long long)m_direct->replacedPivots();
  const RealScalar bn = probe.norm();
  a.probeResidual = double(bn > RealScalar(0) ? (matrix * x - probe).norm() / bn
                                              : (matrix * x - probe).norm());
  a.accepted = acceptable(a, x.allFinite());
  if (a.accepted)
    a.note =
        "recovers digits lost to conditioning; kept (never worse than the plain residual, and "
        "the factorization is unchanged)";
  return a;
}

template <typename MatrixType>
robust_lu::Attempt RobustLU<MatrixType>::runPivoting(const MatrixType& matrix) {
  using namespace robust_lu;
  Attempt a;
  a.strategy = Strategy::PartialPivoting;
  const auto t0 = std::chrono::steady_clock::now();

  m_pivoting.reset(new PivotingSolver());
  m_pivoting->compute(matrix);
  const auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  if (m_pivoting->info() != Success) {
    a.milliseconds = elapsed();
    a.note = m_pivoting->lastErrorMessage();
    return a;
  }
  a.factored = true;

  const DenseVector probe = probeFor(matrix);
  const DenseVector x = m_pivoting->solve(probe);
  a.milliseconds = elapsed();
  // PointBlockLU exposes no backward error of its own, so it is measured here
  // with the same Oettli-Prager routine the other rungs are judged by -- the
  // comparison would be meaningless otherwise.
  a.backwardError = double(left_right_lu::componentwiseBackwardError(matrix, probe, x));
  const RealScalar bn = probe.norm();
  a.probeResidual = double(bn > RealScalar(0) ? (matrix * x - probe).norm() / bn
                                              : (matrix * x - probe).norm());
  a.accepted = acceptable(a, x.allFinite());
  if (!a.accepted) a.note = rejection(a);
  return a;
}

template <typename MatrixType>
robust_lu::Attempt RobustLU<MatrixType>::runRankRevealing(const MatrixType& matrix) {
  using namespace robust_lu;
  Attempt a;
  a.strategy = Strategy::RankRevealing;
  const auto t0 = std::chrono::steady_clock::now();
  const auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };

  MatrixType compressed = matrix;
  compressed.makeCompressed();  // SparseQR requires it and does not check

  m_rankRevealing.reset(new RankRevealingSolver());
  m_rankRevealing->compute(compressed);
  if (m_rankRevealing->info() != Success) {
    a.milliseconds = elapsed();
    a.note = "rank-revealing QR failed to factorize";
    return a;
  }
  a.factored = true;
  a.rank = (long long)m_rankRevealing->rank();
  a.factorNonzeros = (long long)m_rankRevealing->matrixR().nonZeros();

  const DenseVector probe = probeFor(matrix);
  const DenseVector x = m_rankRevealing->solve(probe);
  a.milliseconds = elapsed();

  const DenseVector r = matrix * x - probe;
  const RealScalar rnorm = r.norm();
  const RealScalar bnorm = probe.norm();
  a.probeResidual = double(bnorm > RealScalar(0) ? rnorm / bnorm : rnorm);
  a.backwardError = double(left_right_lu::componentwiseBackwardError(matrix, probe, x));

  // Least-squares optimality: for the true minimiser of ||Ax - b||, the residual
  // is orthogonal to the range of A, so ||A^T r|| / (||A|| ||r||) is ~eps.
  const DenseVector atr = matrix.adjoint() * r;
  const RealScalar anorm = left_right_lu::oneNorm(matrix);
  a.leastSquaresOptimality =
      double((rnorm > RealScalar(0) && anorm > RealScalar(0)) ? atr.norm() / (anorm * rnorm)
                                                             : RealScalar(0));
  a.accepted = acceptableLeastSquares(a, x.allFinite());
  if (!a.accepted) {
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "rejected: rank=%lld, relative residual=%.2e, optimality=%.2e", a.rank,
                  a.probeResidual, a.leastSquaresOptimality);
    a.note = buf;
  }
  return a;
}

template <typename MatrixType>
bool RobustLU<MatrixType>::tryRankRevealing(const MatrixType& matrix) {
  using namespace robust_lu;
  if (int(m_maxStrategy) < int(Strategy::RankRevealing)) {
    Attempt a;
    a.strategy = Strategy::RankRevealing;
    a.note = "not attempted: setMaxStrategy caps the ladder below this rung";
    m_attempts.push_back(a);
    return false;
  }
  // The fill predictor: whatever the LU rungs managed. QR's R is comparable to
  // or larger than the LU factors on every matrix measured here, so an LU fill
  // that is already enormous is proof the rung cannot pay.
  long long luFill = -1;
  for (const Attempt& prior : m_attempts)
    if (prior.strategy != Strategy::RankRevealing && prior.factorNonzeros > luFill)
      luFill = prior.factorNonzeros;
  if (m_maxRankRevealingFill > 0 && luFill > m_maxRankRevealingFill) {
    Attempt a;
    a.strategy = Strategy::RankRevealing;
    a.factorNonzeros = luFill;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "not attempted: the LU factors already carry %lld scalars, past the "
                  "rank-revealing fill guard of %lld (see setMaxRankRevealingFill) -- QR's R is "
                  "no smaller, and on a matrix this dense it runs for tens of minutes",
                  luFill, m_maxRankRevealingFill);
    a.note = buf;
    m_attempts.push_back(a);
    return false;
  }
  if (m_maxRankRevealingRows > 0 && matrix.rows() > m_maxRankRevealingRows) {
    Attempt a;
    a.strategy = Strategy::RankRevealing;
    char buf[220];
    std::snprintf(buf, sizeof(buf),
                  "not attempted: %lld rows exceeds the rank-revealing size cap of %lld "
                  "(see setMaxRankRevealingSize) -- QR is 17-24x the fill of LU on this "
                  "project's corpus and is not a cheaper fallback",
                  (long long)matrix.rows(), (long long)m_maxRankRevealingRows);
    a.note = buf;
    m_attempts.push_back(a);
    return false;
  }

  Attempt a = runRankRevealing(matrix);
  m_attempts.push_back(a);
  if (!a.accepted) return false;

  m_useRankRevealing = true;
  m_usePivoting = false;
  m_strategy = Strategy::RankRevealing;
  m_rank = Index(a.rank);
  m_backwardError = RealScalar(a.backwardError);
  m_conditionEstimate = NumTraits<RealScalar>::quiet_NaN();

  if (m_rank < Index(matrix.cols())) {
    // A verified answer, but to a DIFFERENT question than the caller asked: it
    // minimises ||Ax - b|| rather than solving A x = b, and it is the basic
    // solution rather than the minimum-norm one. info() stays Success because
    // the answer is trustworthy; the outcome and the message are what say it is
    // a least-squares answer, and isLeastSquares() is the programmatic check.
    m_outcome = Outcome::RankDeficient;
    m_info = Success;
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "RobustLU: no LU factorization exists (numerical rank %lld of %lld). The "
                  "answer is a verified BASIC least-squares solution -- free variables set to "
                  "zero, not the minimum-norm solution -- so it may differ from another "
                  "method's by a null-space vector. See rank() and isLeastSquares().",
                  (long long)m_rank, (long long)matrix.cols());
    m_lastError = buf;
  } else {
    m_outcome = Outcome::Solved;
    m_info = Success;
  }
  return true;
}

template <typename MatrixType>
void RobustLU<MatrixType>::compute(const MatrixType& matrix) {
  using namespace robust_lu;
  eigen_assert(matrix.rows() == matrix.cols() && "RobustLU requires a square matrix");
  m_size = StorageIndex(matrix.rows());
  m_attempts.clear();
  m_lastError.clear();
  m_usePivoting = false;
  m_useRankRevealing = false;
  m_rank = -1;
  m_denseRowCount = -1;
  m_denseRowPenalty = NumTraits<double>::quiet_NaN();
  m_info = NumericalIssue;
  m_outcome = Outcome::Exhausted;
  m_isInitialized = true;

  const int ceiling = int(m_maxStrategy);
  auto accept = [&](const Attempt& a) {
    m_strategy = a.strategy;
    m_outcome = Outcome::Solved;
    m_info = Success;
    m_backwardError = RealScalar(a.backwardError);
    m_conditionEstimate = RealScalar(a.conditionEstimate);
  };

  // ---- rung 0: the default strategy ------------------------------------
  Attempt first = runDirect(matrix, Strategy::Default);
  m_attempts.push_back(first);
  if (first.factored) m_denseRowCount = Index(first.denseRows);

  // Structural singularity is checked BEFORE the acceptance test, not only when
  // a rung is rejected. A singular matrix can still produce a small backward
  // error on a consistent right-hand side -- static pivoting bumps the zero
  // pivot, the probe happens to lie in the range of A, and the answer looks
  // fine. It is not: the factors are of a perturbed matrix, and a different b
  // will expose that. Saying so is the whole point of the class.
  if (first.factored && !m_direct->matchingIsPerfect()) {
    m_strategy = Strategy::Default;
    m_backwardError = RealScalar(first.backwardError);
    m_conditionEstimate = RealScalar(first.conditionEstimate);
    // No zero-free diagonal means no LU exists, so every LU rung is waste. This
    // is exactly the diagnosis the rank-revealing rung is for, and the only one
    // that reaches it without an LU failure first.
    if (tryRankRevealing(matrix)) return;
    m_outcome = Outcome::StructurallySingular;
    m_info = NumericalIssue;
    m_lastError =
        "RobustLU: the matrix is structurally singular (no zero-free diagonal exists), so no LU "
        "strategy can succeed -- the factors are of a perturbed matrix and another right-hand "
        "side will expose it. The rank-revealing rung did not produce a verified answer either "
        "(see report()).";
    return;
  }

  if (first.accepted) {
    // Backward-stable. The only remaining question is whether conditioning has
    // eaten the answer anyway, and whether anything can be done about it.
    const RealScalar kappaEps =
        RealScalar(first.conditionEstimate) * NumTraits<RealScalar>::epsilon();
    if (kappaEps < RealScalar(1) || !(numext::isfinite)(kappaEps)) {
      accept(first);
      if (kappaEps >= RealScalar(0.001) && ceiling >= int(Strategy::ExtendedResidual)) {
        // Backward-stable but losing digits to conditioning, and still inside
        // the range where extra residual precision can recover them. This rung
        // re-solves the SAME factors -- one extra solve, no refactorization.
        // Its result is kept whether or not it is formally "accepted", because
        // an extended residual is never worse than a working-precision one.
        Attempt ext = resolveExtended(matrix);
        m_attempts.push_back(ext);
        if (ext.accepted) accept(ext);
      }
      return;
    }
    // kappa * eps >= 1: no digit of the answer is supported and no rung can
    // change that. Escalating here would burn full factorizations to rediscover
    // a property of the matrix.
    accept(first);
    m_outcome = Outcome::IllConditioned;
    m_info = NumericalIssue;
    m_lastError =
        "RobustLU: the factorization is backward stable and its residual is small, but the "
        "estimated condition number leaves no correct digits guaranteed. The answer may still be "
        "accurate -- a condition number is a worst-case bound, not a measurement -- and there is "
        "no way to tell from the residual, which is exactly why this is flagged. This is the "
        "matrix, not the solver: no fallback strategy can recover it in this precision.";
    return;
  }

  // ---- the diagnosis -----------------------------------------------------
  if (!first.factored && first.note.find("predicted") != std::string::npos) {
    m_outcome = Outcome::Infeasible;
    m_lastError = first.note;
    return;
  }

  // The first rung was rejected, so escalation is coming. That makes the
  // dense-row penalty actionable -- and worth its symbolic analysis, which is
  // trivial next to the refactorization about to happen.
  if (first.factored && m_denseRowCount > 0)
    m_denseRowPenalty = m_direct->denseRowFillPenalty(matrix);

  // ---- rung: MC64 --------------------------------------------------------
  // The measured workhorse: 4 of the 5 rescuable corpus failures. A large
  // backward error after static pivoting means the diagonal the ordering
  // produced was unusable, and MC64 is the one tool here that maximises it
  // globally rather than greedily.
  if (ceiling >= int(Strategy::MC64)) {
    Attempt mc = runDirect(matrix, Strategy::MC64);
    m_attempts.push_back(mc);
    if (mc.accepted) {
      accept(mc);
      return;
    }
  }

  // ---- rung: true partial pivoting --------------------------------------
  // LeftRightLU confines pivoting to a diagonal block by construction, so a
  // failure that survives MC64 needs a factorization that can take a pivot from
  // anywhere. PointBlockLU is that factorization and it never perturbs a pivot.
  if (ceiling >= int(Strategy::PartialPivoting)) {
    Attempt pp = runPivoting(matrix);
    m_attempts.push_back(pp);
    if (pp.accepted) {
      m_usePivoting = true;
      m_strategy = Strategy::PartialPivoting;
      m_outcome = Outcome::Solved;
      m_info = Success;
      m_backwardError = RealScalar(pp.backwardError);
      m_conditionEstimate = NumTraits<RealScalar>::quiet_NaN();
      return;
    }
  }

  // ---- terminal rung: rank-revealing QR ---------------------------------
  // Every LU strategy has now failed on a matrix that HAS a zero-free diagonal,
  // which means the obstacle is numerical rank rather than structure. Measured
  // on this project's corpus, this is what rescues Bai/rw5151, where no LU
  // configuration reaches a usable answer at all.
  if (tryRankRevealing(matrix)) return;

  // ---- exhausted ---------------------------------------------------------
  // Re-establish the best factors available so solve() still returns the least
  // bad answer rather than nothing -- flagged, as always.
  m_useRankRevealing = false;
  runDirect(matrix, Strategy::Default);
  m_strategy = Strategy::Default;
  m_outcome = Outcome::Exhausted;
  m_info = NumericalIssue;
  m_lastError = "RobustLU: no strategy reached an acceptable backward error. See report().";
}

template <typename MatrixType>
template <typename Rhs, typename Dest>
void RobustLU<MatrixType>::_solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const {
  if (m_useRankRevealing)
    x = m_rankRevealing->solve(b.derived());
  else if (m_usePivoting)
    x = m_pivoting->solve(b.derived());
  else
    x = m_direct->solve(b.derived());
}

template <typename MatrixType>
Index RobustLU<MatrixType>::nnzL() const {
  if (m_useRankRevealing) return Index(m_rankRevealing->matrixR().nonZeros());
  return m_usePivoting ? m_pivoting->nnzL() : m_direct->nnzL();
}
template <typename MatrixType>
Index RobustLU<MatrixType>::nnzU() const {
  if (m_useRankRevealing) return Index(m_rankRevealing->matrixR().nonZeros());
  return m_usePivoting ? m_pivoting->nnzU() : m_direct->nnzU();
}

template <typename MatrixType>
std::string RobustLU<MatrixType>::report() const {
  std::string out = "RobustLU: ";
  out += robust_lu::outcomeName(m_outcome);
  if (m_outcome == robust_lu::Outcome::Solved) {
    out += " by ";
    out += robust_lu::strategyName(m_strategy);
  }
  out += "\n";
  char buf[512];
  for (std::size_t k = 0; k < m_attempts.size(); ++k) {
    const robust_lu::Attempt& a = m_attempts[k];
    if (a.strategy == robust_lu::Strategy::RankRevealing)
      std::snprintf(buf, sizeof(buf),
                    "  %zu. %-38s %-8s rank=%lld resid=%.2e optimality=%.2e %.1f ms", k + 1,
                    robust_lu::strategyName(a.strategy),
                    a.accepted ? "ACCEPT" : (a.factored ? "reject" : "declined"), a.rank,
                    a.probeResidual, a.leastSquaresOptimality, a.milliseconds);
    else
      std::snprintf(buf, sizeof(buf),
                    "  %zu. %-38s %-8s omega=%.2e kappa=%.2e growth=%.2e pivots=%lld %.1f ms",
                    k + 1, robust_lu::strategyName(a.strategy),
                    a.accepted ? "ACCEPT" : (a.factored ? "reject" : "declined"), a.backwardError,
                    a.conditionEstimate, a.growthFactor, a.replacedPivots, a.milliseconds);
    out += buf;
    if (!a.note.empty()) {
      out += "\n       ";
      out += a.note;
    }
    out += "\n";
  }
  // Only worth saying when it is both true and actionable: a penalty of 1.00 is
  // AMD having already dealt with the dense rows, which is the common case.
  if ((numext::isfinite)(m_denseRowPenalty) && m_denseRowPenalty > 1.2) {
    std::snprintf(buf, sizeof(buf),
                  "  note: %lld dense rows are costing %.2fx the fill a bordered factorization "
                  "would need. A different ordering (METIS, COLAMD) usually recovers most of "
                  "that and costs nothing to try.\n",
                  (long long)m_denseRowCount, m_denseRowPenalty);
    out += buf;
  }
  if (!m_lastError.empty()) {
    out += "  ";
    out += m_lastError;
    out += "\n";
  }
  return out;
}

}  // namespace Eigen

#endif  // ROBUST_LU_H
