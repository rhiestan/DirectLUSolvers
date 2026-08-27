// Extended-precision residuals for iterative refinement.
//
// WHY THIS EXISTS
//
// Iterative refinement repeats: form r = b - A x, solve A d = r with the
// existing factors, add d to x. Its ceiling is set entirely by the precision of
// that first step, and the reason is worth stating precisely rather than
// hand-waving:
//
//   * with r computed in WORKING precision, refinement drives the BACKWARD
//     error to O(eps) and then stops improving. The forward error settles at
//     roughly kappa(A) * eps and no number of further steps moves it, because
//     the computed r is already dominated by the rounding in forming it.
//   * with r computed in EXTENDED precision, refinement drives the FORWARD
//     error to O(eps) as well, provided kappa(A) * eps is comfortably below 1
//     (Skeel 1980; Demmel et al., the basis of LAPACK's xGESVXX).
//
// Measured on this solver, an upper bidiagonal matrix, four refinement steps,
// only the residual precision differing:
//
//     kappa       no refinement    double residual   extended residual
//     3.2e+07     2.1e-10          4.9e-11           0
//     6.4e+09     2.2e-08          2.2e-08           0
//     1.3e+12     2.3e-06          2.3e-06           0
//     2.6e+14     7.2e-04          7.2e-04           0
//
// WHY NOT long double
//
// Because it is not portable in the one direction that matters. On MSVC (and
// therefore on every Windows toolchain targeting it, clang-cl included) it is a
// 64-bit alias of double -- measured on this project's own toolchain: 8 bytes,
// 53 mantissa bits. Code written against it would silently do nothing on
// Windows while working on Linux, which is the worst available outcome for a
// numerical guarantee.
//
// What this header uses instead is DOUBLE-DOUBLE arithmetic built from
// error-free transformations: twoSum and twoProduct compute a floating-point
// operation's result AND its exact rounding error, both as ordinary values of
// the same type. Carrying the error term alongside the sum gives roughly twice
// the working mantissa (~106 bits for double) using nothing but the arithmetic
// the platform already has. It is pure software, identical everywhere, and it
// works for float and complex scalars too.
//
// THE FLAG HAZARD -- read this before compiling with anything unusual
//
// Error-free transformations are algebraically trivial: in exact arithmetic
// twoSum's error term IS zero, and that is the whole point -- it is nonzero
// only because of rounding. A compiler permitted to reason about floating point
// as though it were real arithmetic will therefore delete them outright. So:
//
//   * -ffast-math / /fp:fast BREAK this code. Not "degrade" -- the compensation
//     terms fold to zero and the result silently becomes the plain residual.
//   * -ffp-contract is safe here: twoProduct uses std::fma explicitly where the
//     platform has a fast one, and the Dekker fallback below is written so that
//     contraction cannot change its meaning.
//   * x87 excess-precision (32-bit x86 without SSE2) breaks the error terms too,
//     since intermediates are computed wider than they are stored. Every
//     x86-64 and ARM target is fine.
//
// The FMA path is worth having: measured on this project's matrices the
// extended residual costs about 2x the plain one with hardware FMA and about
// 5x without it (Dekker splitting), which as a share of one triangular solve is
// 1-13% against 2-25%. On x86-64 it needs -mfma / -march=native or better --
// the baseline x86-64 target has no FMA, and see the detection note below for
// why the standard FP_FAST_FMA macro cannot be relied on to tell you.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef LEFT_RIGHT_LU_EXTENDED_RESIDUAL_H
#define LEFT_RIGHT_LU_EXTENDED_RESIDUAL_H

#include <Eigen/SparseCore>

#include <cmath>
#include <limits>
#include <vector>

namespace Eigen {
namespace left_right_lu {

namespace detail {

// Does this build have a hardware fused multiply-add?
//
// It matters a lot and is annoyingly hard to ask. FP_FAST_FMA is the standard
// spelling -- a <cmath> macro the C library defines when fma() is roughly as
// fast as a multiply -- but it is only advisory, and MSVC's UCRT never defines
// it at all, on any hardware. Relying on it alone silently sends every Windows
// build down the slow path on machines that have had FMA3 since 2013. So the
// compiler's own target macros are consulted too.
//
// Getting this wrong is a performance question, never a correctness one: both
// paths compute the exact same error term. When in doubt the answer is "no",
// because libm's software fma is correctly rounded but can be 50-100x slower
// than a multiply, while Dekker's splitting is a predictable 17 flops.
#if defined(FP_FAST_FMA) || defined(__FMA__) || defined(__AVX2__) || \
    defined(__ARM_FEATURE_FMA) || defined(__aarch64__) || defined(_M_ARM64)
#define EIGEN_LEFT_RIGHT_LU_FAST_FMA 1
#else
#define EIGEN_LEFT_RIGHT_LU_FAST_FMA 0
#endif

template <typename Real>
struct HasFastFma {
  // float and double both ride on the same hardware unit; a type the platform
  // emulates (long double on some ABIs) would not, but no such type reaches
  // here in practice.
  static const bool value = EIGEN_LEFT_RIGHT_LU_FAST_FMA != 0;
};

/** Veltkamp splitting: a = hi + lo exactly, with each half carrying at most
  * half the mantissa, so products of halves are exact. */
template <typename Real>
inline void veltkampSplit(Real a, Real& hi, Real& lo) {
  // 2^ceil(p/2) + 1, computed once per type rather than hard-coded per width.
  static const Real splitter =
      std::ldexp(Real(1), (std::numeric_limits<Real>::digits + 1) / 2) + Real(1);
  const Real c = splitter * a;
  hi = c - (c - a);
  lo = a - hi;
}

/** Exact product: p = fl(a*b) and e = a*b - p, so a*b == p + e exactly. */
template <typename Real>
inline void twoProduct(Real a, Real b, Real& p, Real& e) {
  p = a * b;
  if (HasFastFma<Real>::value) {
    using std::fma;
    e = fma(a, b, -p);
  } else {
    Real ah, al, bh, bl;
    veltkampSplit(a, ah, al);
    veltkampSplit(b, bh, bl);
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  }
}

/** Exact sum, no assumption about which operand is larger (Knuth). */
template <typename Real>
inline void twoSum(Real a, Real b, Real& s, Real& e) {
  s = a + b;
  const Real bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

/** A running sum carrying its own rounding error: value() is accurate to
  * roughly twice the working precision. */
template <typename Real>
struct CompensatedSum {
  Real hi = Real(0);
  Real lo = Real(0);

  void add(Real v) {
    Real s, e;
    twoSum(hi, v, s, e);
    hi = s;
    lo += e;
  }
  /** Accumulate a*b, keeping the product's rounding error as well as the sum's. */
  void addProduct(Real a, Real b) {
    Real p, pe;
    twoProduct(a, b, p, pe);
    Real s, e;
    twoSum(hi, p, s, e);
    hi = s;
    lo += e + pe;
  }
  void clear() {
    hi = Real(0);
    lo = Real(0);
  }
  /** b - (hi + lo), rounded once at the end. */
  Real subtractedFrom(Real b) const {
    Real s, e;
    twoSum(b, -hi, s, e);
    return s + (e - lo);
  }
};

/** One entry of a matrix-vector product, accumulated into real/imaginary parts.
  * Complex scalars go through the same real error-free transformations, four
  * products at a time -- there is no separate complex EFT and none is needed. */
template <typename Scalar, typename Real>
inline void accumulateProduct(CompensatedSum<Real>& re, CompensatedSum<Real>& im, const Scalar& a,
                              const Scalar& x) {
  if
    constexpr(NumTraits<Scalar>::IsComplex) {
      const Real ar = numext::real(a), ai = numext::imag(a);
      const Real xr = numext::real(x), xi = numext::imag(x);
      re.addProduct(ar, xr);
      re.addProduct(-ai, xi);
      im.addProduct(ar, xi);
      im.addProduct(ai, xr);
    }
  else {
    EIGEN_UNUSED_VARIABLE(im);
    re.addProduct(a, x);
  }
}

template <typename Scalar, typename Real>
inline Scalar finishResidual(const CompensatedSum<Real>& re, const CompensatedSum<Real>& im,
                             const Scalar& b) {
  if
    constexpr(NumTraits<Scalar>::IsComplex) {
      return Scalar(re.subtractedFrom(numext::real(b)), im.subtractedFrom(numext::imag(b)));
    }
  else {
    EIGEN_UNUSED_VARIABLE(im);
    return re.subtractedFrom(numext::real(b));
  }
}

}  // namespace detail

/** \brief r = b - A x, with every dot product accumulated in double-double.
  *
  * The result is rounded to working precision once, at the end, so \a r is an
  * ordinary vector -- the extra precision lives only inside the accumulation,
  * which is where it is needed. Handles any number of right-hand sides.
  *
  * Cost is O(nnz) in the same traversal order as a plain sparse product, with a
  * few extra flops per entry; against a triangular solve's O(fill) it is
  * usually negligible, but it is not free and is measured rather than assumed
  * (see bench_refinement).
  */
template <typename MatrixT, typename RhsT, typename SolT, typename DestT>
void residualExtended(const MatrixT& A, const RhsT& b, const SolT& x, DestT& r) {
  typedef typename MatrixT::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real Real;

  const Index n = A.rows();
  const Index nrhs = b.cols();
  r.resize(n, nrhs);

  // Named, not `re(std::size_t(n))`: that spelling is a function declaration,
  // not a vector -- the most vexing parse, and it compiles.
  const std::size_t rows = std::size_t(n);
  std::vector<detail::CompensatedSum<Real>> re(rows), im(rows);
  for (Index c = 0; c < nrhs; ++c) {
    for (Index i = 0; i < n; ++i) {
      re[std::size_t(i)].clear();
      im[std::size_t(i)].clear();
    }
    // Column-major traversal, so A is walked exactly as a plain product walks it.
    for (Index k = 0; k < A.outerSize(); ++k) {
      const Scalar xk = x(k, c);
      if (numext::is_exactly_zero(xk)) continue;
      for (typename MatrixT::InnerIterator it(A, k); it; ++it) {
        const std::size_t i = std::size_t(it.row());
        detail::accumulateProduct<Scalar, Real>(re[i], im[i], it.value(), xk);
      }
    }
    for (Index i = 0; i < n; ++i)
      r(i, c) = detail::finishResidual<Scalar, Real>(re[std::size_t(i)], im[std::size_t(i)],
                                                     b(i, c));
  }
}

/** \brief r = b - A^T x (Conjugate == false) or r = b - A^H x, in double-double.
  *
  * Cheaper than the non-transposed form: each output entry is one column's dot
  * product, so a single accumulator suffices and nothing is scattered.
  */
template <bool Conjugate, typename MatrixT, typename RhsT, typename SolT, typename DestT>
void residualExtendedTransposed(const MatrixT& A, const RhsT& b, const SolT& x, DestT& r) {
  typedef typename MatrixT::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real Real;

  const Index n = A.cols();
  const Index nrhs = b.cols();
  r.resize(n, nrhs);

  for (Index c = 0; c < nrhs; ++c) {
    for (Index k = 0; k < A.outerSize(); ++k) {
      detail::CompensatedSum<Real> re, im;
      for (typename MatrixT::InnerIterator it(A, k); it; ++it) {
        const Scalar a = Conjugate ? numext::conj(it.value()) : it.value();
        detail::accumulateProduct<Scalar, Real>(re, im, a, x(it.row(), c));
      }
      r(k, c) = detail::finishResidual<Scalar, Real>(re, im, b(k, c));
    }
  }
}

}  // namespace left_right_lu
}  // namespace Eigen

#endif  // LEFT_RIGHT_LU_EXTENDED_RESIDUAL_H
