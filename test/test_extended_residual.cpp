// Extended-precision (double-double) residuals in iterative refinement.
//
//   ctest --test-dir build -R test_extended_residual --output-on-failure
//
// WHY THIS EXISTS
//
// Refinement's ceiling is the precision of r = b - Ax, and the difference is
// not a matter of degree: with r in working precision refinement reaches a
// small BACKWARD error and stops; with r in extended precision it reaches a
// small FORWARD error. This suite pins both halves of that claim, plus the
// arithmetic underneath it.
//
// MEASURING THIS IS EASY TO GET WRONG, so the setup deserves a note. If b is
// formed as a floating-point product A*xTrue, then xTrue is NOT the exact
// solution of the system actually stored -- b carries its own rounding, worth
// about kappa(A)*eps of forward error, and no residual precision can see past
// it. An experiment built that way shows extended precision doing nothing, or
// looking worse, for reasons that have nothing to do with the feature. Every
// ill-conditioned case below therefore uses an INTEGER matrix and an INTEGER
// solution, so b = A*xTrue is exact in binary floating point and xTrue really
// is the answer refinement should converge to.

#include <Eigen/SparseCore>

#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "LeftRightLUExtendedResidual.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::note;

typedef SparseMatrix<double> SpMat;
typedef Eigen::LeftRightLU<SpMat> Solver;
namespace lrlu = Eigen::left_right_lu;

namespace {

// Upper bidiagonal, diagonal 10, superdiagonal -17. Integer entries, so with an
// integer xTrue the product A*xTrue is exact; kappa grows like 1.7^n, so n alone
// dials the conditioning. The factorization still rounds (it divides by 10),
// which is what leaves refinement something to do.
SpMat integerBidiagonal(int n) {
  std::vector<Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    t.emplace_back(j, j, 10.0);
    if (j > 0) t.emplace_back(j - 1, j, -17.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

VectorXd integerSolution(Eigen::Index n) {
  VectorXd x(n);
  for (Eigen::Index i = 0; i < n; ++i) x[i] = double(1 + (i * 7) % 9);
  return x;
}

// ---------------------------------------------------------------------------
//  1. The error-free transformations themselves
// ---------------------------------------------------------------------------

void testErrorFreeTransformations() {
  std::printf("  error-free transformations are exact\n");

  // twoSum: s + e must reproduce a + b exactly, including when the addition
  // loses everything. 1 + eps/2 rounds to 1, and the error term must recover it.
  {
    const double a = 1.0, b = std::numeric_limits<double>::epsilon() / 2.0;
    double s, e;
    lrlu::detail::twoSum(a, b, s, e);
    checkTrue(s == 1.0, "twoSum: the sum rounds as expected");
    checkTrue(e == b, "twoSum: the error term recovers what the sum dropped");
  }
  // And with the operands the other way round, since Knuth's form must not care.
  {
    const double a = std::numeric_limits<double>::epsilon() / 2.0, b = 1.0;
    double s, e;
    lrlu::detail::twoSum(a, b, s, e);
    checkTrue(s == 1.0 && e == a, "twoSum: order of operands does not matter");
  }

  // twoProduct: p + e == a*b exactly. Checked on values whose product needs
  // more than 53 bits, so a plain multiply provably cannot be exact.
  {
    const double a = 1.0 + std::numeric_limits<double>::epsilon();
    const double b = 1.0 - std::numeric_limits<double>::epsilon();
    double p, e;
    lrlu::detail::twoProduct(a, b, p, e);
    checkTrue(p == 1.0, "twoProduct: the product rounds to 1");
    checkTrue(e != 0.0, "twoProduct: and the error term is not zero");
    // The exact product is 1 - eps^2; recovering it is the whole contract.
    const double eps = std::numeric_limits<double>::epsilon();
    checkTrue(std::abs(e - (-eps * eps)) <= 1e-40,
              "twoProduct: p + e reproduces the exact product");
  }
  {
    // A case with no cancellation, where the error term is a genuine low word.
    const double a = 1.0 / 3.0, b = 7.0 / 11.0;
    double p, e;
    lrlu::detail::twoProduct(a, b, p, e);
    checkTrue(p == a * b, "twoProduct: high word is the plain product");
    checkTrue(e != 0.0, "twoProduct: low word is nonzero for an inexact product");
  }

  // CompensatedSum against a known-hard summation: 1 + eps/2 added n times.
  // In working precision every term after the first vanishes; compensated, they
  // accumulate.
  {
    const double tiny = std::numeric_limits<double>::epsilon() / 2.0;
    lrlu::detail::CompensatedSum<double> acc;
    double naive = 1.0;
    acc.add(1.0);
    for (int i = 0; i < 1000; ++i) {
      acc.add(tiny);
      naive += tiny;
    }
    checkTrue(naive == 1.0, "naive summation loses all 1000 small terms");
    const double compensated = acc.subtractedFrom(0.0);  // -(hi+lo)
    checkTrue(std::abs(-compensated - (1.0 + 1000.0 * tiny)) < 1e-30,
              "compensated summation keeps them");
  }
}

// ---------------------------------------------------------------------------
//  2. The residual routine
// ---------------------------------------------------------------------------

void testResidualIsMoreAccurate() {
  std::printf("  the extended residual beats the working-precision one\n");
  const int n = 60;
  const SpMat A = integerBidiagonal(n);
  const VectorXd xTrue = integerSolution(n);
  const VectorXd b = A * xTrue;

  // Exact solution, exact right-hand side: the true residual is exactly zero,
  // and the extended computation must say so.
  VectorXd rExt;
  lrlu::residualExtended(A, b, xTrue, rExt);
  checkTrue(rExt.cwiseAbs().maxCoeff() == 0.0,
            "extended residual of the exact solution is exactly zero");

  // A perturbed x: both residuals should agree to working precision, since
  // there is now a genuine residual to represent.
  VectorXd xPerturbed = xTrue;
  xPerturbed[n / 2] += 1e-8;
  const VectorXd rPlain = b - A * xPerturbed;
  lrlu::residualExtended(A, b, xPerturbed, rExt);
  const double diff = (rExt - rPlain).cwiseAbs().maxCoeff();
  // The right yardstick is the PLAIN residual's own error bound, which scales
  // with |A||x| + |b| -- not with |r|. Those differ by orders of magnitude here
  // precisely because r is small, and asserting against |r| would be asserting
  // that the working-precision computation was already exact.
  const double scale = (A.cwiseAbs() * xPerturbed.cwiseAbs() + b.cwiseAbs()).maxCoeff();
  checkTrue(diff <= 8.0 * n * std::numeric_limits<double>::epsilon() * scale,
            "the two residuals agree to within the plain one's own error bound");
  checkTrue(diff > 0.0, "and they are not bit-identical -- the plain one really does round");

  // Multi-column, and the transposed form.
  Eigen::MatrixXd X(n, 2);
  X.col(0) = xTrue;
  X.col(1) = xPerturbed;
  Eigen::MatrixXd B(n, 2);
  B.col(0) = b;
  B.col(1) = b;
  Eigen::MatrixXd R;
  lrlu::residualExtended(A, B, X, R);
  checkTrue(R.col(0).cwiseAbs().maxCoeff() == 0.0, "multi-rhs: exact column is exactly zero");
  checkTrue((R.col(1) - rPlain).cwiseAbs().maxCoeff() <= 1e-8,
            "multi-rhs: perturbed column matches");

  const VectorXd bt = A.transpose() * xTrue;
  VectorXd rt;
  lrlu::residualExtendedTransposed<false>(A, bt, xTrue, rt);
  checkTrue(rt.cwiseAbs().maxCoeff() == 0.0, "transposed residual of the exact solution is zero");
}

// ---------------------------------------------------------------------------
//  3. The claim: backward error vs forward error
// ---------------------------------------------------------------------------

void testForwardErrorCollapses() {
  std::printf("  refinement reaches the FORWARD error only with an extended residual\n");
  std::printf("        %-4s %-10s %-13s %-13s %-13s\n", "n", "kappa", "no refinement",
              "double resid", "extended resid");

  bool sawBigGain = false;
  for (int n : {30, 40, 50, 60}) {
    const SpMat A = integerBidiagonal(n);
    const VectorXd xTrue = integerSolution(n);
    const VectorXd b = A * xTrue;
    auto err = [&](const VectorXd& v) { return (v - xTrue).norm() / xTrue.norm(); };

    // No refinement at all, for the baseline.
    Solver raw;
    raw.setRefinementMethod(Eigen::left_right_lu::Refinement::None);
    raw.compute(A);
    const VectorXd x0 = raw.solve(b);

    // Stationary refinement, working-precision residual. This is the mode the
    // theory is about; BiCGStab is covered separately below.
    Solver plain;
    plain.setRefinementMethod(Eigen::left_right_lu::Refinement::IterativeRefinement);
    plain.setRefineOnlyIfPerturbed(false);
    plain.setMaxIterativeRefinements(6);
    plain.compute(A);
    const VectorXd xPlain = plain.solve(b);

    Solver ext;
    ext.setRefinementMethod(Eigen::left_right_lu::Refinement::IterativeRefinement);
    ext.setRefineOnlyIfPerturbed(false);
    ext.setMaxIterativeRefinements(6);
    ext.setExtendedPrecisionResidual(true);
    ext.compute(A);
    const VectorXd xExt = ext.solve(b);

    std::printf("        %-4d %-10.2e %-13.2e %-13.2e %-13.2e\n", n, raw.conditionEstimate(),
                err(x0), err(xPlain), err(xExt));

    checkTrue(err(xExt) <= err(xPlain) * 1.000001 + 1e-300,
              "n=" + std::to_string(n) + ": the extended residual is never worse");
    // Where conditioning bites, it must be decisively better -- an order of
    // magnitude at least, or the feature is not earning its keep.
    if (err(xPlain) > 1e-12) {
      checkTrue(err(xExt) < err(xPlain) / 10.0,
                "n=" + std::to_string(n) + ": and decisively better where it matters");
      sawBigGain = true;
    }
    // Both are backward stable regardless: that is precisely why the plain
    // residual's forward error is invisible to the residual check.
    check(plain.solveResidual() < 1e-12,
          "n=" + std::to_string(n) + ": working-precision refinement is backward stable",
          plain.solveResidual());
  }
  checkTrue(sawBigGain, "at least one case was ill-conditioned enough to show the gain");
}

void testDefaultIsUnchanged() {
  std::printf("  the default path is untouched\n");
  const SpMat A = integerBidiagonal(50);
  const VectorXd b = A * integerSolution(50);

  Solver s;
  checkTrue(!s.extendedPrecisionResidual(), "extended residuals are OFF by default");

  // The switch must be inert when refinement does not run -- which by default
  // is whenever no pivot was perturbed.
  Solver a, bb;
  a.compute(A);
  bb.setExtendedPrecisionResidual(true);
  bb.compute(A);
  const VectorXd xa = a.solve(b);
  const VectorXd xb = bb.solve(b);
  if (a.replacedPivots() == 0 && a.refineOnlyIfPerturbed()) {
    checkTrue((xa - xb).cwiseAbs().maxCoeff() == 0.0,
              "with refinement gated off, the flag changes nothing at all");
    checkTrue(a.iterativeRefinements() == 0 && bb.iterativeRefinements() == 0,
              "and neither solver refined");
  } else {
    note("this matrix perturbs pivots; the inertness check does not apply");
  }
}

void testWorksWithBiCGStab() {
  std::printf("  BiCGStab gets there too, via the stationary polish pass\n");
  // BiCGStab cannot reach a forward error on its own: its residual comes from a
  // recurrence, not from b - Ax. refineSolution therefore polishes afterwards
  // with the stationary extended loop, so the flag means the same thing under
  // either refinement method -- which is what this checks.
  const SpMat A = integerBidiagonal(50);
  const VectorXd xTrue = integerSolution(50);
  const VectorXd b = A * xTrue;
  auto err = [&](const VectorXd& v) { return (v - xTrue).norm() / xTrue.norm(); };

  Solver plain, ext;
  for (Solver* s : {&plain, &ext}) {
    s->setRefinementMethod(Eigen::left_right_lu::Refinement::BiCGStab);
    s->setRefineOnlyIfPerturbed(false);
    s->setMaxIterativeRefinements(6);
  }
  ext.setExtendedPrecisionResidual(true);
  plain.compute(A);
  ext.compute(A);
  const VectorXd xp = plain.solve(b), xe = ext.solve(b);

  checkTrue(xe.allFinite(), "BiCGStab with an extended residual returns a finite answer");
  checkTrue(ext.info() == Eigen::Success, "and reports success");
  checkTrue(err(xe) < err(xp) / 10.0 || err(xe) <= 1e-15,
            "BiCGStab + extended residual reaches the forward error too");
  std::printf("        BiCGStab: plain %.2e, extended %.2e (polished by the stationary pass)\n",
              err(xp), err(xe));
}

void testTransposedRefinement() {
  std::printf("  the transposed solve refines with the right operator\n");
  // A wrong operator here (A instead of A^T, or a missing conjugate) would make
  // refinement diverge rather than converge, so accuracy IS the check.
  const SpMat A = integerBidiagonal(40);
  const VectorXd xTrue = integerSolution(40);
  const VectorXd bt = A.transpose() * xTrue;

  Solver s;
  s.setRefinementMethod(Eigen::left_right_lu::Refinement::IterativeRefinement);
  s.setRefineOnlyIfPerturbed(false);
  s.setMaxIterativeRefinements(6);
  s.setExtendedPrecisionResidual(true);
  s.compute(A);
  const VectorXd x = s.transpose().solve(bt);
  check((x - xTrue).norm() / xTrue.norm() < 1e-12, "transposed solve refines to high accuracy",
        (x - xTrue).norm() / xTrue.norm());
}

void testComplexAndFloat() {
  std::printf("  complex and float scalars\n");
  {
    typedef std::complex<double> Cplx;
    typedef SparseMatrix<Cplx> SpMatC;
    const int n = 30;
    std::vector<Triplet<Cplx>> t;
    for (int j = 0; j < n; ++j) {
      t.emplace_back(j, j, Cplx(10.0, 2.0));
      if (j > 0) t.emplace_back(j - 1, j, Cplx(-17.0, 3.0));
    }
    SpMatC A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    Eigen::Matrix<Cplx, Eigen::Dynamic, 1> xTrue(n);
    for (int i = 0; i < n; ++i) xTrue[i] = Cplx(double(1 + i % 7), double(i % 5));
    const Eigen::Matrix<Cplx, Eigen::Dynamic, 1> b = A * xTrue;

    Eigen::Matrix<Cplx, Eigen::Dynamic, 1> r;
    lrlu::residualExtended(A, b, xTrue, r);
    checkTrue(r.cwiseAbs().maxCoeff() == 0.0, "complex: exact solution gives an exact zero");

    Eigen::LeftRightLU<SpMatC> s;
    s.setRefinementMethod(Eigen::left_right_lu::Refinement::IterativeRefinement);
    s.setRefineOnlyIfPerturbed(false);
    s.setExtendedPrecisionResidual(true);
    s.compute(A);
    const Eigen::Matrix<Cplx, Eigen::Dynamic, 1> x = s.solve(b);
    check((x - xTrue).norm() / xTrue.norm() < 1e-13, "complex: refines accurately",
          (x - xTrue).norm() / xTrue.norm());

    // The adjoint path exercises the conjugated accumulation, which is separate
    // code from the plain transpose.
    const Eigen::Matrix<Cplx, Eigen::Dynamic, 1> bh = A.adjoint() * xTrue;
    const Eigen::Matrix<Cplx, Eigen::Dynamic, 1> xh = s.adjoint().solve(bh);
    check((xh - xTrue).norm() / xTrue.norm() < 1e-13, "complex: adjoint refines accurately",
          (xh - xTrue).norm() / xTrue.norm());
  }
  {
    // float has a 24-bit mantissa, so the compensated sum should reach roughly
    // double's precision -- the same mechanism, a different type.
    typedef SparseMatrix<float> SpMatF;
    const int n = 20;
    std::vector<Triplet<float>> t;
    for (int j = 0; j < n; ++j) {
      t.emplace_back(j, j, 10.0f);
      if (j > 0) t.emplace_back(j - 1, j, -17.0f);
    }
    SpMatF A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    Eigen::VectorXf xTrue(n);
    for (int i = 0; i < n; ++i) xTrue[i] = float(1 + (i * 7) % 9);
    const Eigen::VectorXf b = A * xTrue;
    Eigen::VectorXf r;
    lrlu::residualExtended(A, b, xTrue, r);
    checkTrue(r.cwiseAbs().maxCoeff() == 0.0f, "float: exact solution gives an exact zero");

    Eigen::LeftRightLU<SpMatF> s;
    s.setRefinementMethod(Eigen::left_right_lu::Refinement::IterativeRefinement);
    s.setRefineOnlyIfPerturbed(false);
    s.setExtendedPrecisionResidual(true);
    s.compute(A);
    const Eigen::VectorXf x = s.solve(b);
    check((x - xTrue).norm() / xTrue.norm() < 1e-5f, "float: refines accurately",
          double((x - xTrue).norm() / xTrue.norm()));
  }
}

void testInteractionWithErrorBounds() {
  std::printf("  pairs with the step-2 error bounds\n");
  // The two features answer each other: the forward-error estimate says whether
  // extended precision can help, and after it runs the estimate should improve.
  const SpMat A = integerBidiagonal(55);
  const VectorXd xTrue = integerSolution(55);
  const VectorXd b = A * xTrue;

  Solver s;
  s.setErrorBounds(true);
  s.setRefinementMethod(Eigen::left_right_lu::Refinement::IterativeRefinement);
  s.setRefineOnlyIfPerturbed(false);
  s.setMaxIterativeRefinements(6);
  s.setExtendedPrecisionResidual(true);
  s.compute(A);
  const VectorXd x = s.solve(b);

  checkTrue(s.info() == Eigen::Success, "the refined solve is reported as successful");
  check(s.lastBackwardError() < 1e-14, "backward error is at machine precision",
        s.lastBackwardError());
  const double trueErr = (x - xTrue).norm() / xTrue.norm();
  checkTrue(trueErr <= std::max(s.lastForwardError() * 100.0, 1e-14),
            "and the forward-error bound still holds after refinement");
  std::printf("        kappa=%.2e omega=%.1e ferr=%.1e true err=%.1e\n", s.conditionEstimate(),
              s.lastBackwardError(), s.lastForwardError(), trueErr);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("LeftRightLU extended-precision residuals\n");
  std::printf("  fast fma for double: %s\n",
              lrlu::detail::HasFastFma<double>::value ? "yes (std::fma)" : "no (Dekker split)");

  testErrorFreeTransformations();
  testResidualIsMoreAccurate();
  testForwardErrorCollapses();
  testDefaultIsUnchanged();
  testWorksWithBiCGStab();
  testTransposedRefinement();
  testComplexAndFloat();
  testInteractionWithErrorBounds();

  return lu_testing::summarize("extended residuals");
}
