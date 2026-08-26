// Condition estimation and backward error, for turning "it solved" into "it
// solved, and here is how much of the answer you may believe".
//
// A direct solver that reports only a residual cannot answer the question a
// caller actually has. A tiny residual means the computed x is the exact
// solution of a NEARBY system -- it says nothing about how far x is from the
// solution of the system that was asked about. On a well-conditioned matrix
// those are the same statement; on an ill-conditioned one they differ by orders
// of magnitude, and the residual is the reassuring half.
//
// Two quantities close that gap, and both are cheap once L and U exist:
//
//   BACKWARD ERROR (Oettli-Prager, componentwiseBackwardError below)
//       omega = max_i |b - Ax|_i / (|A||x| + |b|)_i
//   is the smallest RELATIVE, COMPONENTWISE perturbation of A and b for which
//   the computed x is the exact answer. It costs one sparse matrix-vector
//   product over |A| and is exact -- not an estimate. omega ~ eps means the
//   solver did everything a backward-stable method can do.
//
//   CONDITION NUMBER (Hager-Higham, oneNormInverseEstimate below)
//       kappa_1(A) = ||A||_1 ||A^{-1}||_1
//   is how much the answer can move per unit of perturbation. ||A^{-1}||_1 is
//   never formed: Hager's algorithm needs only products A^{-1}v and A^{-H}v,
//   which is what a factorization already provides, so the whole estimate costs
//   a handful of triangular solves rather than n of them.
//
// Together they bound the forward error:
//
//       ||x - x_exact||         kappa(A) * omega
//       ---------------  <~     (first order, see estimateForwardError)
//          ||x_exact||
//
// which is what lets a caller distinguish "13 correct digits" from "2".
//
// WHAT THESE NUMBERS DO NOT PROMISE
//
// The condition estimate is a LOWER BOUND on ||A^{-1}||_1 -- Hager's algorithm
// maximizes over a subset of the unit ball, so it can only underestimate.
// In practice it is almost always within a factor of 3 and very often exact,
// but a matrix can be constructed to defeat it, and underestimating means
// reporting a matrix as better conditioned than it is. It is an estimate, and
// callers should read it as one.
//
// The estimate also describes the operator the SOLVER inverts, which under
// static pivoting is a perturbed A, not A itself. That is the honest thing to
// report -- it is the operator whose inverse the caller is about to apply --
// but it means kappa can look better than the true kappa(A) on a matrix whose
// pivots were bumped. replacedPivots() is what says whether that happened.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef LEFT_RIGHT_LU_CONDITION_ESTIMATE_H
#define LEFT_RIGHT_LU_CONDITION_ESTIMATE_H

#include <Eigen/SparseCore>

#include <limits>

namespace Eigen {
namespace left_right_lu {

namespace detail {

// sign(z) in the sense Hager's algorithm needs: a unit-modulus number with z's
// direction, and +1 where z is zero (any unit vector will do there, and +1
// keeps the real case to a plain +-1 as LAPACK's dlacon has it).
template <typename Scalar>
inline Scalar unitSign(const Scalar& z) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const RealScalar mag = numext::abs(z);
  if (mag == RealScalar(0)) return Scalar(1);
  return z / mag;
}

}  // namespace detail

/** \brief Hager-Higham estimate of \f$\|B\|_1\f$ for an operator known only
  *        through products.
  *
  * \param n          dimension of the (square) operator
  * \param applyB     v -> B v,   writing into its second argument
  * \param applyBAdj  v -> B^H v (plain transpose for real scalars)
  * \param solveCount if non-null, incremented once per operator application
  * \param maxIterations  Hager iterations; LAPACK uses 5 and so do we
  *
  * The algorithm (Hager 1984, as improved by Higham and shipped as LAPACK's
  * xLACON) walks the vertices of the unit 1-norm ball uphill: the 1-norm of a
  * linear operator is attained at a vertex e_j, so it repeatedly asks the
  * ADJOINT which coordinate the current direction is most sensitive to, and
  * moves there. Each step needs one product with B and one with B^H, and the
  * sequence is finite because the estimate strictly increases until it stops.
  *
  * The final "alternating signs" probe is not a flourish: the uphill walk can
  * park on a local maximum, and that particular vector -- 1, -1-1/(n-1),
  * +1+2/(n-1), ... -- is a cheap second opinion that catches the classic
  * counterexamples. LAPACK keeps it for the same reason.
  *
  * \returns an estimate of \f$\|B\|_1\f$, never larger than the true value.
  */
template <typename Scalar, typename ApplyB, typename ApplyBAdj>
typename NumTraits<Scalar>::Real oneNormEstimate(Index n, ApplyB applyB, ApplyBAdj applyBAdj,
                                                 Index* solveCount = nullptr,
                                                 int maxIterations = 5) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vector;

  if (n <= 0) return RealScalar(0);

  Vector v(n), x(n);
  auto callB = [&](const Vector& in, Vector& out) {
    applyB(in, out);
    if (solveCount) ++(*solveCount);
  };
  auto callBAdj = [&](const Vector& in, Vector& out) {
    applyBAdj(in, out);
    if (solveCount) ++(*solveCount);
  };

  auto argAbsMax = [&](const Vector& u) {
    Index best = 0;
    RealScalar bestAbs = RealScalar(-1);
    for (Index i = 0; i < n; ++i) {
      const RealScalar a = numext::abs(u[i]);
      if (a > bestAbs) {
        bestAbs = a;
        best = i;
      }
    }
    return best;
  };

  // First probe: B applied to the uniform vector, which is the average column.
  x.setConstant(Scalar(RealScalar(1) / RealScalar(n)));
  callB(x, v);
  RealScalar est = v.template lpNorm<1>();
  if (n == 1) return numext::abs(v[0]);

  // Ask the adjoint which coordinate this direction is most sensitive to; that
  // is the first vertex to try. The loop below then alternates B / B^H, which
  // is the order LAPACK's xLACN2 uses -- and the order matters: checking for
  // convergence before rather than after the B e_j product cuts the walk short
  // and costs a factor of two in the estimate on some matrices.
  Vector xi(n);
  for (Index i = 0; i < n; ++i) xi[i] = detail::unitSign(v[i]);
  callBAdj(xi, x);
  Index j = argAbsMax(x);

  for (int iter = 1; iter < maxIterations; ++iter) {
    x.setZero();
    x[j] = Scalar(1);
    callB(x, v);
    const RealScalar estOld = est;
    est = v.template lpNorm<1>();

    // Two convergence tests, both from xLACN2. An unchanged sign vector means
    // the walk has closed a cycle and can only repeat itself; a non-increasing
    // estimate means the vertex it just tried was not an improvement.
    bool signsUnchanged = true;
    for (Index i = 0; i < n; ++i) {
      const Scalar sgn = detail::unitSign(v[i]);
      if (!numext::equal_strict(sgn, xi[i])) signsUnchanged = false;
      xi[i] = sgn;
    }
    if (signsUnchanged) break;
    if (est <= estOld) {
      // Deliberate deviation from LAPACK, which keeps the smaller value here.
      // Both are valid lower bounds; keeping the larger one is strictly the
      // better estimate and never breaks the bound.
      est = estOld;
      break;
    }

    callBAdj(xi, x);
    const Index jLast = j;
    j = argAbsMax(x);
    if (numext::abs(x[jLast]) == numext::abs(x[j])) break;
  }

  // Second opinion: the alternating-sign probe, which is designed to be a bad
  // fit for whatever local maximum the walk above may have settled on.
  if (n > 1) {
    for (Index i = 0; i < n; ++i) {
      const RealScalar t = RealScalar(1) + RealScalar(i) / RealScalar(n - 1);
      x[i] = Scalar((i % 2 == 0) ? t : -t);
    }
    Vector w(n);
    callB(x, w);
    const RealScalar alt = RealScalar(2) * w.template lpNorm<1>() / (RealScalar(3) * RealScalar(n));
    if (alt > est) est = alt;
  }
  return est;
}

/** \brief \f$\|A\|_1\f$, the largest absolute column sum. Exact, O(nnz). */
template <typename MatrixT>
typename NumTraits<typename MatrixT::Scalar>::Real oneNorm(const MatrixT& A) {
  typedef typename NumTraits<typename MatrixT::Scalar>::Real RealScalar;
  RealScalar best(0);
  for (Index k = 0; k < A.outerSize(); ++k) {
    RealScalar sum(0);
    for (typename MatrixT::InnerIterator it(A, k); it; ++it) sum += numext::abs(it.value());
    if (sum > best) best = sum;
  }
  return best;
}

/** \brief Oettli-Prager componentwise relative backward error of a computed
  *        solution.
  *
  * \returns \f$\max_i |b - Ax|_i / (|A||x| + |b|)_i\f$, maximised over the
  *          columns of a multi-column right-hand side.
  *
  * This is EXACT, not an estimate: it is the smallest \f$\epsilon\f$ for which
  * \f$(A + \delta A)x = b + \delta b\f$ with \f$|\delta A| \le \epsilon |A|\f$
  * and \f$|\delta b| \le \epsilon |b|\f$. A value near machine epsilon says the
  * solver was backward stable on this system; there is no better answer any
  * method could have produced in this precision, and any remaining inaccuracy
  * in x is the matrix's conditioning rather than the solver's doing.
  *
  * The guarded denominator follows LAPACK's xGERFS. Where a component of
  * \f$|A||x| + |b|\f$ is at or below the underflow threshold the ratio carries
  * no information -- it is 0/0 in exact arithmetic -- so a safe-minimum floor is
  * added to numerator and denominator alike, which keeps the quantity finite
  * without inventing a small backward error where none was measured.
  */
template <typename MatrixT, typename RhsT, typename SolT>
typename NumTraits<typename MatrixT::Scalar>::Real componentwiseBackwardError(const MatrixT& A,
                                                                              const RhsT& b,
                                                                              const SolT& x) {
  typedef typename MatrixT::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVector;

  const Index n = A.rows();
  const Index nrhs = b.cols();
  if (n == 0 || nrhs == 0) return RealScalar(0);

  const RealScalar safeMin = (std::numeric_limits<RealScalar>::min)();
  const RealScalar safe1 = RealScalar(n + 1) * safeMin;
  const RealScalar safe2 = safe1 / NumTraits<RealScalar>::epsilon();

  RealScalar worst(0);
  for (Index c = 0; c < nrhs; ++c) {
    // denom = |A| |x| + |b|, accumulated column-wise so A is traversed once in
    // its natural (compressed, column-major) order.
    RealVector denom = b.col(c).cwiseAbs();
    for (Index k = 0; k < A.outerSize(); ++k) {
      const RealScalar xk = numext::abs(x(k, c));
      if (xk == RealScalar(0)) continue;
      for (typename MatrixT::InnerIterator it(A, k); it; ++it)
        denom[it.row()] += numext::abs(it.value()) * xk;
    }
    const Matrix<Scalar, Dynamic, 1> residual = b.col(c) - A * x.col(c);
    for (Index i = 0; i < n; ++i) {
      const RealScalar r = numext::abs(residual[i]);
      const RealScalar term =
          (denom[i] > safe2) ? r / denom[i] : (r + safe1) / (denom[i] + safe1);
      if (term > worst) worst = term;
    }
  }
  return worst;
}

/** \brief First-order forward error estimate: \f$\kappa \cdot \omega\f$,
  *        clamped to 1.
  *
  * The bound behind it is standard (Higham, *Accuracy and Stability of
  * Numerical Algorithms*, thm 7.2): a backward error \f$\omega\f$ moves the
  * solution by at most \f$\mathrm{cond}(A,x)\,\omega\f$ to first order, and
  * \f$\kappa_1(A)\f$ is the usual computable stand-in for
  * \f$\mathrm{cond}(A,x)\f$. Two caveats travel with it and neither is
  * cosmetic: it is FIRST ORDER, so it is meaningless once it approaches 1, and
  * \f$\kappa\f$ is itself an underestimate (see oneNormEstimate). Clamping at 1
  * is therefore the honest ceiling -- "no digits are guaranteed" -- rather than
  * a claim that the answer is off by exactly 100%.
  */
template <typename RealScalar>
RealScalar estimateForwardError(const RealScalar& conditionEstimate,
                                const RealScalar& backwardError) {
  if (!(numext::isfinite)(conditionEstimate) || !(numext::isfinite)(backwardError))
    return NumTraits<RealScalar>::highest();
  const RealScalar bound = conditionEstimate * backwardError;
  return (bound < RealScalar(1)) ? bound : RealScalar(1);
}

}  // namespace left_right_lu
}  // namespace Eigen

#endif  // LEFT_RIGHT_LU_CONDITION_ESTIMATE_H
