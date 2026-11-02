// The solvers with a software floating-point scalar (Boost.Multiprecision's
// cpp_bin_float_quad: 113-bit mantissa, eps ~1.9e-34), run side by side with
// double through the same templated checks. Every tolerance is a multiple of
// the scalar's own epsilon, so a check that passes for double and fails for
// quad points at something in the solver tied to double -- a hard-wired
// tolerance, a std:: call that bypasses the type's own math, an fma the type
// does not round once. References are dense Eigen decompositions in the same
// scalar.
//
// Needs Boost.Multiprecision (header-only); built only with
// -DDLU_WITH_BOOST_MULTIPRECISION=ON.

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/eigen.hpp>

#include <LeftRightLU>
#include <MultifrontalQR>
#include <PointBlockLU>
#include <RobustLU>
#include <SupernodalLDLT>
#include <SupernodalLU>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "testing/Check.h"

using lu_testing::check;
using lu_testing::checkTrue;
using Eigen::Index;
using quad = boost::multiprecision::cpp_bin_float_quad;

namespace {

template <typename T>
using Dense = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using SpMat = Eigen::SparseMatrix<T>;

template <typename T>
double toDouble(const T& v) {
  return static_cast<double>(v);
}

template <typename T>
double eps() {
  return toDouble(Eigen::NumTraits<T>::epsilon());
}

template <typename T>
const char* typeName() {
  return std::is_same<T, double>::value ? "double" : "quad";
}

// Random entries are generated in double and converted, so both types see the
// same matrix.
template <typename T>
SpMat<T> randomSparse(int m, int n, double density, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0), v(-1.0, 1.0);
  std::vector<Eigen::Triplet<T>> t;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      if (u(rng) < density) t.emplace_back(i, j, T(v(rng)));
  for (int i = 0; i < std::min(m, n); ++i) t.emplace_back(i, i, T(4));
  SpMat<T> A(m, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

template <typename T>
Vec<T> randomVector(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> v(-1.0, 1.0);
  Vec<T> x(n);
  for (int i = 0; i < n; ++i) x[i] = T(v(rng));
  return x;
}

template <typename T>
double relErr(const Vec<T>& a, const Vec<T>& b) {
  const T d = b.norm();
  return toDouble(T((a - b).norm() / (d > T(0) ? d : T(1))));
}

// ||b - A x|| / (||A||_F ||x|| + ||b||): rounding level for a backward-stable solve.
template <typename T>
double backwardError(const SpMat<T>& A, const Vec<T>& x, const Vec<T>& b) {
  return toDouble(T((b - A * x).norm() / (A.norm() * x.norm() + b.norm())));
}

// ||A^H r|| / (||A|| (||A|| ||x|| + ||b||)): 0 at a least-squares solution, and
// meaningful even when r itself is at rounding level.
template <typename T>
double optimality(const SpMat<T>& A, const Vec<T>& x, const Vec<T>& b) {
  const T an = A.norm();
  return toDouble(T((A.transpose() * (A * x - b)).norm() / (an * (an * x.norm() + b.norm()))));
}

template <typename T>
Vec<T> pinvSolve(const Dense<T>& A, const Vec<T>& b, const T& relTol) {
  Eigen::JacobiSVD<Dense<T>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);
  svd.setThreshold(relTol);
  return svd.solve(b);
}

std::string tagOf(const char* type, const std::string& what) { return std::string(type) + ": " + what; }

// ---------------------------------------------------------------------------
// MultifrontalQR
// ---------------------------------------------------------------------------

template <typename T>
void qrSquare() {
  const std::string t = tagOf(typeName<T>(), "QR square 150");
  const SpMat<T> A = randomSparse<T>(150, 150, 0.03, 11);
  const Vec<T> b = randomVector<T>(150, 12);
  Eigen::MultifrontalQR<SpMat<T>> qr;
  const auto t0 = std::chrono::steady_clock::now();
  qr.compute(A);
  const Vec<T> x = qr.solve(b);
  const double ms = lu_testing::ms(t0, std::chrono::steady_clock::now());
  lu_testing::note(t + ": factor + solve " + std::to_string(ms) + " ms");
  checkTrue(qr.info() == Eigen::Success && qr.rank() == 150, t + ": full rank");
  check(backwardError(A, x, b) < 10 * eps<T>(), t + ": backward error / eps", backwardError(A, x, b) / eps<T>());
  const Vec<T> ref = Dense<T>(A).partialPivLu().solve(b);
  const double kappa = toDouble(qr.conditionEstimate());
  check(relErr(x, ref) < 100 * kappa * eps<T>(), t + ": matches dense LU (error / (kappa eps))",
        relErr(x, ref) / (kappa * eps<T>()));
}

template <typename T>
void qrLeastSquares() {
  const std::string t = tagOf(typeName<T>(), "QR tall 140x60");
  const SpMat<T> A = randomSparse<T>(140, 60, 0.06, 21);
  const Vec<T> b = randomVector<T>(140, 22);
  Eigen::MultifrontalQR<SpMat<T>> qr(A);
  const Vec<T> x = qr.solve(b);
  check(optimality(A, x, b) < 10 * eps<T>(), t + ": optimality / eps", optimality(A, x, b) / eps<T>());
  const Vec<T> ref = Dense<T>(A).colPivHouseholderQr().solve(b);
  const double kappa = toDouble(qr.conditionEstimate());
  check(relErr(x, ref) < 100 * kappa * kappa * eps<T>(), t + ": matches dense pivoted QR (error / (kappa^2 eps))",
        relErr(x, ref) / (kappa * kappa * eps<T>()));
}

// Exact dependencies: the rank is structural, the answer is pinv(A) b.
template <typename T>
void qrRankDeficient() {
  const std::string t = tagOf(typeName<T>(), "QR wide 40x60, dependent columns");
  Dense<T> Ad(randomSparse<T>(40, 60, 0.15, 31));
  for (int j = 3; j < 60; j += 7) Ad.col(j) = T(3) * Ad.col(j - 1) - Ad.col(j - 3);
  for (int i = 5; i < 40; i += 9) Ad.row(i) = Ad.row(i - 1) + T(2) * Ad.row(i - 2);
  SpMat<T> A = Ad.sparseView();
  A.makeCompressed();
  const Vec<T> b = randomVector<T>(40, 32);  // inconsistent: rows are dependent too
  Eigen::JacobiSVD<Dense<T>> svd(Ad);
  const T tol = T(1e3) * Eigen::NumTraits<T>::epsilon();
  Index refRank = 0;
  for (Index i = 0; i < svd.singularValues().size(); ++i)
    if (svd.singularValues()[i] > tol * svd.singularValues()[0]) ++refRank;
  Eigen::MultifrontalQR<SpMat<T>> qr(A);
  check(qr.rank() == refRank && refRank < 40, t + ": rank matches dense SVD", double(qr.rank()));
  checkTrue(qr.rankIsVerified(), t + ": rank verified");
  const Vec<T> x = qr.solve(b);
  const Vec<T> ref = pinvSolve<T>(Ad, b, tol);
  check(relErr(x, ref) < 1e4 * eps<T>(), t + ": min-norm == pinv(A) b (error / eps)", relErr(x, ref) / eps<T>());
  check(optimality(A, x, b) < 10 * eps<T>(), t + ": least squares (optimality / eps)", optimality(A, x, b) / eps<T>());
}

// Kahan: Heath's rule sees only healthy pivots, only verification finds the
// tiny singular value. The tolerance sits in the gap the dense SVD shows, so
// the test adapts to the precision rather than assuming double's.
template <typename T>
void qrKahan() {
  const std::string t = tagOf(typeName<T>(), "QR Kahan 90");
  const int n = 90;
  const double theta = 1.2;
  const T c = T(std::cos(theta)), s = T(std::sin(theta));
  std::vector<Eigen::Triplet<T>> tr;
  T sk(1);
  for (int i = 0; i < n; ++i) {
    tr.emplace_back(i, i, sk * (T(1) + T(1e-10) * T(i)));
    for (int j = i + 1; j < n; ++j) tr.emplace_back(i, j, -c * sk);
    sk *= s;
  }
  SpMat<T> A(n, n);
  A.setFromTriplets(tr.begin(), tr.end());
  const Dense<T> Ad(A);
  Eigen::JacobiSVD<Dense<T>> svd(Ad);
  const auto& sv = svd.singularValues();
  using std::sqrt;
  const T relTau = sqrt(sv[n - 1] * sv[n - 2]) / sv[0];  // geometric middle of the gap
  lu_testing::note(t + ": sigma_n/sigma_1 " + std::to_string(toDouble(T(sv[n - 1] / sv[0]))) + ", gap sigma_{n-1}/sigma_n " +
                   std::to_string(toDouble(T(sv[n - 2] / sv[n - 1]))));
  check(toDouble(T(sv[n - 2] / sv[n - 1])) > 1e3, t + ": reference has a clear gap", toDouble(T(sv[n - 2] / sv[n - 1])));
  Eigen::MultifrontalQR<SpMat<T>> qr;
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.setOrdering(Eigen::multifrontal_qr::Ordering::Natural);
  qr.setRankTolerance(relTau * sv[0] / Ad.colwise().norm().maxCoeff());
  qr.setRankVerification(false);
  qr.compute(A);
  check(qr.rank() == n, t + ": Heath's rule alone misses it", double(qr.rank()));
  qr.setRankVerification(true);
  qr.compute(A);
  check(qr.rank() == n - 1 && qr.rankIsVerified(), t + ": verification finds rank n-1", double(qr.rank()));
  const Vec<T> b = Vec<T>::Ones(n);
  const Vec<T> x = qr.solve(b);
  const Vec<T> ref = pinvSolve<T>(Ad, b, relTau);
  // Truncating sigma_n perturbs A by sigma_n: the answer is that accurate.
  const double bound = 100 * toDouble(T(sv[n - 1] / sv[n - 2]));
  check(relErr(x, ref) < bound, t + ": equals truncated pinv(A) b (error / bound)", relErr(x, ref) / bound);
}

// Column-scaled minimum norm is invariant under column scaling: A D -> D^-1 x.
template <typename T>
void qrColumnScaledInvariance() {
  const std::string t = tagOf(typeName<T>(), "QR column-scaled min-norm");
  Dense<T> Ad(randomSparse<T>(30, 50, 0.15, 41));
  for (int j = 3; j < 50; j += 7) Ad.col(j) = T(3) * Ad.col(j - 1) - Ad.col(j - 3);
  Vec<T> d(50);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> u(-6.0, 6.0);
  for (int j = 0; j < 50; ++j) d[j] = T(1.37 * std::pow(10.0, u(rng)));
  SpMat<T> A = Ad.sparseView(), AD = Dense<T>(Ad * d.asDiagonal()).sparseView();
  const Vec<T> b = randomVector<T>(30, 43);
  using Eigen::multifrontal_qr::Solution;
  Eigen::MultifrontalQR<SpMat<T>> qa, qd;
  qa.setSolution(Solution::ColumnScaledMinimumNorm);
  qd.setSolution(Solution::ColumnScaledMinimumNorm);
  qa.compute(A);
  qd.compute(AD);
  const Vec<T> xa = qa.solve(b), xd = qd.solve(b);
  const Vec<T> mapped = d.cwiseInverse().asDiagonal() * xa;
  check(relErr(xd, mapped) < 1e3 * eps<T>(), t + ": scale invariant (error / eps)", relErr(xd, mapped) / eps<T>());
}

// Refinement with double-T residuals reaches a forward error of a few eps on a
// system whose kappa * eps is far from small -- which needs twoProduct to be
// exact in T. Neumann Laplacian plus delta e1 e1^T: entries, solution and
// right-hand side are all exactly representable, kappa ~ 4 n / delta.
template <typename T>
void qrExtendedRefinement() {
  const std::string t = tagOf(typeName<T>(), "QR refinement, kappa ~ eps^-0.6");
  const int n = 200;
  using std::ldexp;
  using std::log2;
  const int k = int(0.6 * std::numeric_limits<T>::digits) - 10;  // delta = 2^-k
  const T delta = ldexp(T(1), -k);
  std::vector<Eigen::Triplet<T>> tr;
  for (int i = 0; i < n; ++i) {
    const int deg = (i > 0) + (i < n - 1);
    tr.emplace_back(i, i, T(deg) + (i == 0 ? delta : T(0)));
    if (i > 0) tr.emplace_back(i, i - 1, T(-1));
    if (i < n - 1) tr.emplace_back(i, i + 1, T(-1));
  }
  SpMat<T> A(n, n);
  A.setFromTriplets(tr.begin(), tr.end());
  Vec<T> xTrue(n);
  for (int i = 0; i < n; ++i) xTrue[i] = T((i * 7) % 11 + 1);
  const Vec<T> b = A * xTrue;  // exact: small integers plus delta * 1
  Eigen::MultifrontalQR<SpMat<T>> qr(A);
  const double kappa = 4.0 * n / toDouble(delta);
  qr.setExtendedPrecisionResidual(false);
  const Vec<T> xPlain = qr.solve(b);
  qr.setExtendedPrecisionResidual(true);
  const Vec<T> xExt = qr.solve(b);
  lu_testing::note(t + ": kappa ~" + std::to_string(kappa) + ", plain-residual error / eps " +
                   std::to_string(relErr(xPlain, xTrue) / eps<T>()));
  check(relErr(xExt, xTrue) < 10 * eps<T>(), t + ": extended-residual error / eps", relErr(xExt, xTrue) / eps<T>());
}

template <typename T>
void runQR() {
  std::printf("--- MultifrontalQR<%s> ---\n", typeName<T>());
  qrSquare<T>();
  qrLeastSquares<T>();
  qrRankDeficient<T>();
  qrKahan<T>();
  qrColumnScaledInvariance<T>();
  qrExtendedRefinement<T>();
}

// ---------------------------------------------------------------------------
// LeftRightLU
// ---------------------------------------------------------------------------

// Unsymmetric values and pattern, with pivoting.
template <typename T>
void luSquare() {
  const std::string t = tagOf(typeName<T>(), "LU unsymmetric 300");
  const SpMat<T> A = randomSparse<T>(300, 300, 0.015, 51);
  const Vec<T> b = randomVector<T>(300, 52);
  Eigen::LeftRightLU<SpMat<T>> lu;
  const auto t0 = std::chrono::steady_clock::now();
  lu.compute(A);
  const Vec<T> x = lu.solve(b);
  const double ms = lu_testing::ms(t0, std::chrono::steady_clock::now());
  lu_testing::note(t + ": factor + solve " + std::to_string(ms) + " ms");
  checkTrue(lu.info() == Eigen::Success, t + ": success");
  check(backwardError(A, x, b) < 10 * eps<T>(), t + ": backward error / eps", backwardError(A, x, b) / eps<T>());
  check(toDouble(lu.componentwiseBackwardError(b, x)) < 10 * eps<T>(), t + ": componentwise backward error / eps",
        toDouble(lu.componentwiseBackwardError(b, x)) / eps<T>());
  const Dense<T> Ad(A);
  const Vec<T> ref = Ad.partialPivLu().solve(b);
  const double kappa = toDouble(lu.conditionEstimate());
  check(relErr(x, ref) < 100 * kappa * eps<T>(), t + ": matches dense LU (error / (kappa eps))",
        relErr(x, ref) / (kappa * eps<T>()));
  // Hager-Higham: a lower bound on kappa_1, normally within a small factor.
  const Dense<T> Ainv = Ad.inverse();
  const double exact = toDouble(T(Ad.cwiseAbs().colwise().sum().maxCoeff() * Ainv.cwiseAbs().colwise().sum().maxCoeff()));
  check(kappa <= exact * (1 + 1e-6) && kappa >= exact / 10, t + ": condition estimate / exact kappa_1", kappa / exact);
  const double fwd = toDouble(lu.estimatedForwardError(b, x));
  check(relErr(x, ref) <= 10 * fwd + 100 * eps<T>(), t + ": forward error within the estimated bound (error / bound)",
        relErr(x, ref) / fwd);
  const T det = lu.determinant(), detRef = Ad.partialPivLu().determinant();
  check(toDouble(T(abs(det - detRef) / abs(detRef))) < 1e3 * eps<T>(), t + ": determinant (error / eps)",
        toDouble(T(abs(det - detRef) / abs(detRef))) / eps<T>());
}

// A reducible matrix: block upper triangular after a symmetric permutation,
// so BTF splits it into independent diagonal blocks.
template <typename T>
void luBlockTriangular() {
  const std::string t = tagOf(typeName<T>(), "LU reducible, BTF");
  const int nb = 6, bs = 25, n = nb * bs;
  std::vector<Eigen::Triplet<T>> tr;
  std::mt19937 rng(61);
  std::uniform_real_distribution<double> u(0.0, 1.0), v(-1.0, 1.0);
  for (int bi = 0; bi < nb; ++bi)
    for (int i = bi * bs; i < (bi + 1) * bs; ++i) {
      tr.emplace_back(i, i, T(5));
      for (int j = bi * bs; j < n; ++j)  // own block and everything to the right
        if (j != i && u(rng) < (j < (bi + 1) * bs ? 0.15 : 0.02)) tr.emplace_back(i, j, T(v(rng)));
    }
  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) perm[i] = (i * 37) % n;  // 37 is coprime to 150
  std::vector<Eigen::Triplet<T>> pt;
  for (const auto& e : tr) pt.emplace_back(perm[e.row()], perm[e.col()], e.value());
  SpMat<T> A(n, n);
  A.setFromTriplets(pt.begin(), pt.end());
  A.makeCompressed();
  const Vec<T> b = randomVector<T>(n, 62);
  Eigen::LeftRightLU<SpMat<T>> lu;
  lu.setBlockTriangularForm(true);
  lu.compute(A);
  const Vec<T> x = lu.solve(b);
  checkTrue(lu.info() == Eigen::Success, t + ": success");
  check(backwardError(A, x, b) < 10 * eps<T>(), t + ": backward error / eps", backwardError(A, x, b) / eps<T>());
}

// The system of qrExtendedRefinement, but with kappa ~ eps^-0.4: its smallest
// pivot (~delta / n) must stay above the static-pivot threshold sqrt(eps) max|a|,
// or the factorization is of a perturbed matrix and refinement can only contract
// by kappa sqrt(eps) -- by design, and not what this checks.
template <typename T>
void luExtendedRefinement() {
  const std::string t = tagOf(typeName<T>(), "LU refinement, kappa ~ eps^-0.4");
  const int n = 200;
  using std::ldexp;
  const int k = int(0.4 * std::numeric_limits<T>::digits);
  const T delta = ldexp(T(1), -k);
  std::vector<Eigen::Triplet<T>> tr;
  for (int i = 0; i < n; ++i) {
    const int deg = (i > 0) + (i < n - 1);
    tr.emplace_back(i, i, T(deg) + (i == 0 ? delta : T(0)));
    if (i > 0) tr.emplace_back(i, i - 1, T(-1));
    if (i < n - 1) tr.emplace_back(i, i + 1, T(-1));
  }
  SpMat<T> A(n, n);
  A.setFromTriplets(tr.begin(), tr.end());
  Vec<T> xTrue(n);
  for (int i = 0; i < n; ++i) xTrue[i] = T((i * 7) % 11 + 1);
  const Vec<T> b = A * xTrue;
  Eigen::LeftRightLU<SpMat<T>> lu;
  lu.setRefineOnlyIfPerturbed(false);
  lu.setMaxIterativeRefinements(5);
  lu.compute(A);
  check(lu.replacedPivots() == 0, t + ": no static pivot (precondition)", double(lu.replacedPivots()));
  lu.setExtendedPrecisionResidual(false);
  const Vec<T> xPlain = lu.solve(b);
  lu.setExtendedPrecisionResidual(true);
  const Vec<T> xExt = lu.solve(b);
  lu_testing::note(t + ": plain-residual error / eps " + std::to_string(relErr(xPlain, xTrue) / eps<T>()));
  check(relErr(xExt, xTrue) < 10 * eps<T>(), t + ": extended-residual error / eps", relErr(xExt, xTrue) / eps<T>());
}

// A zero row: the static pivot (sqrt(eps) * max|a|) replaces the zero pivot,
// the factorization succeeds, and solve() reports the singularity.
template <typename T>
void luStaticPivot() {
  const std::string t = tagOf(typeName<T>(), "LU singular row");
  const int g = 6, n = g * g;
  std::vector<Eigen::Triplet<T>> tr;
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j) {
      const int r = i * g + j;
      if (r == 10) continue;  // row 10 empty
      tr.emplace_back(r, r, T(4));
      if (i > 0) tr.emplace_back(r, r - g, T(-1));
      if (i < g - 1) tr.emplace_back(r, r + g, T(-1));
      if (j > 0) tr.emplace_back(r, r - 1, T(-1));
      if (j < g - 1) tr.emplace_back(r, r + 1, T(-1));
    }
  SpMat<T> A(n, n);
  A.setFromTriplets(tr.begin(), tr.end());
  Eigen::LeftRightLU<SpMat<T>> lu;
  lu.compute(A);
  checkTrue(lu.info() == Eigen::Success && lu.replacedPivots() > 0, t + ": factorizes with a replaced pivot");
  const Vec<T> x = lu.solve(Vec<T>::Ones(n));
  checkTrue(lu.info() == Eigen::NumericalIssue, t + ": solve() reports the singularity");
  Eigen::LeftRightLU<SpMat<T>> strict;
  strict.setStaticPivotThreshold(T(0));
  strict.compute(A);
  checkTrue(strict.info() == Eigen::NumericalIssue && !strict.isFactorized(), t + ": threshold 0 fails the factorization");
}

template <typename T>
void runLU() {
  std::printf("--- LeftRightLU<%s> ---\n", typeName<T>());
  luSquare<T>();
  luBlockTriangular<T>();
  luExtendedRefinement<T>();
  luStaticPivot<T>();
}

// ---------------------------------------------------------------------------
// SupernodalLU, PointBlockLU, SupernodalLDLT, RobustLU
// ---------------------------------------------------------------------------

// A determinant's relative error accumulates over every pivot (and the dense
// reference's does too): thousands of eps at n = 300, in both precisions.
constexpr double kDetTol = 1e5;

template <typename T>
double detErr(const T& det, const T& ref) {
  using std::abs;
  return toDouble(T(abs(det - ref) / abs(ref)));
}

template <typename T>
double exactKappa1(const Dense<T>& Ad) {
  const Dense<T> Ainv = Ad.inverse();
  return toDouble(T(Ad.cwiseAbs().colwise().sum().maxCoeff() * Ainv.cwiseAbs().colwise().sum().maxCoeff()));
}

template <typename T>
void supernodalSquare() {
  const std::string t = tagOf(typeName<T>(), "SupernodalLU 300");
  const SpMat<T> A = randomSparse<T>(300, 300, 0.015, 71);
  const Vec<T> b = randomVector<T>(300, 72);
  Eigen::SupernodalLU<SpMat<T>> lu(A);
  const Vec<T> x = lu.solve(b);
  checkTrue(lu.info() == Eigen::Success && lu.replacedPivots() == 0, t + ": success, no replaced pivots");
  check(backwardError(A, x, b) < 10 * eps<T>(), t + ": backward error / eps", backwardError(A, x, b) / eps<T>());
  const Dense<T> Ad(A);
  const double de = detErr(lu.determinant(), Ad.partialPivLu().determinant());
  check(de < kDetTol * eps<T>(), t + ": determinant (error / eps)", de / eps<T>());
}

template <typename T>
void pointBlockSquare() {
  const std::string t = tagOf(typeName<T>(), "PointBlockLU 300");
  const SpMat<T> A = randomSparse<T>(300, 300, 0.015, 81);
  const Vec<T> b = randomVector<T>(300, 82);
  Eigen::PointBlockLU<SpMat<T>> lu(A);
  const Vec<T> x = lu.solve(b);
  checkTrue(lu.info() == Eigen::Success, t + ": success");
  check(backwardError(A, x, b) < 10 * eps<T>(), t + ": backward error / eps", backwardError(A, x, b) / eps<T>());
  const double cbe = toDouble(lu.componentwiseBackwardError(b, x));
  check(cbe < 10 * eps<T>(), t + ": componentwise backward error / eps", cbe / eps<T>());
  const Dense<T> Ad(A);
  const double exact = exactKappa1(Ad), kappa = toDouble(lu.conditionEstimate());
  check(kappa <= exact * (1 + 1e-6) && kappa >= exact / 10, t + ": condition estimate / exact kappa_1", kappa / exact);
  const double de = detErr(lu.determinant(), Ad.partialPivLu().determinant());
  check(de < kDetTol * eps<T>(), t + ": determinant (error / eps)", de / eps<T>());
}

// Symmetric indefinite: Bunch-Kaufman 2x2 pivots, inertia against a dense
// eigensolver in the same scalar.
template <typename T>
void ldltIndefinite() {
  const std::string t = tagOf(typeName<T>(), "SupernodalLDLT indefinite 200");
  const int n = 200;
  const SpMat<T> R = randomSparse<T>(n, n, 0.01, 91);
  SpMat<T> A = SpMat<T>(R + SpMat<T>(R.transpose()));
  for (int i = 0; i < n; ++i) A.coeffRef(i, i) = T(i % 3 == 0 ? -3 : 5) + T(0.01) * T(i % 7);
  A.makeCompressed();
  const Vec<T> b = randomVector<T>(n, 92);
  Eigen::SupernodalLDLT<SpMat<T>> ldlt(A);
  const Vec<T> x = ldlt.solve(b);
  checkTrue(ldlt.info() == Eigen::Success && ldlt.replacedPivots() == 0, t + ": success, no replaced pivots");
  check(backwardError(A, x, b) < 10 * eps<T>(), t + ": backward error / eps", backwardError(A, x, b) / eps<T>());
  const Dense<T> Ad(A);
  // Inertia is discrete, so the reference eigenvalues are computed in double --
  // valid while none is near enough to zero for the cast to flip its sign. (A
  // dense quad eigensolver also overflows the stack under MSVC: Eigen's
  // Tridiagonalization gets a stack allocation inlined into a loop, C4750.)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Ad.template cast<double>(), Eigen::EigenvaluesOnly);
  Index neg = 0;
  for (Index i = 0; i < n; ++i) neg += eig.eigenvalues()[i] < 0.0;
  const double gap = eig.eigenvalues().cwiseAbs().minCoeff();
  check(gap > 1e-6, t + ": reference eigenvalues clear of zero (min |lambda|)", gap);
  const auto in = ldlt.inertia();
  check(in.negative == neg && in.positive == n - neg && in.zero == 0, t + ": inertia matches dense eigenvalues (negative)",
        double(in.negative));
  const double de = detErr(ldlt.determinant(), Ad.partialPivLu().determinant());
  check(de < kDetTol * eps<T>(), t + ": determinant (error / eps)", de / eps<T>());
  const double exact = exactKappa1(Ad), kappa = toDouble(ldlt.conditionEstimate());
  check(kappa <= exact * (1 + 1e-6) && kappa >= exact / 10, t + ": condition estimate / exact kappa_1", kappa / exact);
}

template <typename T>
void robustLadder() {
  using Eigen::robust_lu::Outcome;
  using Eigen::robust_lu::Strategy;
  const std::string t = tagOf(typeName<T>(), "RobustLU");
  {  // unsymmetric: first rung
    const SpMat<T> A = randomSparse<T>(200, 200, 0.02, 101);
    const Vec<T> b = randomVector<T>(200, 102);
    Eigen::RobustLU<SpMat<T>> r(A);
    const Vec<T> x = r.solve(b);
    checkTrue(r.outcome() == Outcome::Solved && r.strategy() == Strategy::Default, t + " unsymmetric: solved on the first rung");
    check(backwardError(A, x, b) < 10 * eps<T>(), t + " unsymmetric: backward error / eps", backwardError(A, x, b) / eps<T>());
  }
  {  // symmetric: LDL^T rung
    const SpMat<T> R = randomSparse<T>(200, 200, 0.01, 103);
    SpMat<T> A = SpMat<T>(R + SpMat<T>(R.transpose()));
    const Vec<T> b = randomVector<T>(200, 104);
    Eigen::RobustLU<SpMat<T>> r(A);
    const Vec<T> x = r.solve(b);
    checkTrue(r.matrixIsSymmetric() && r.outcome() == Outcome::Solved && r.strategy() == Strategy::SymmetricLDLT,
              t + " symmetric: solved by the LDL^T rung");
    check(backwardError(A, x, b) < 10 * eps<T>(), t + " symmetric: backward error / eps", backwardError(A, x, b) / eps<T>());
  }
  {  // structurally singular (empty columns: no LU exists), inconsistent:
     // the rank-revealing rung and its minimum-norm answer. A merely numerical
     // singularity would not get there -- an LU with a replaced pivot solves the
     // ladder's consistent probe, and is accepted.
    const int n = 60;
    Dense<T> Ad(randomSparse<T>(n, n, 0.08, 105));
    for (int j = 5; j < n; j += 11) Ad.col(j).setZero();
    SpMat<T> A = Ad.sparseView();
    A.makeCompressed();
    const Vec<T> b = randomVector<T>(n, 106);
    Eigen::RobustLU<SpMat<T>> r(A);
    const Vec<T> x = r.solve(b);
    const T tol = T(1e3) * Eigen::NumTraits<T>::epsilon();
    const Vec<T> ref = pinvSolve<T>(Ad, b, tol);
    checkTrue(r.outcome() == Outcome::RankDeficient && r.isLeastSquares(), t + " singular: rank-deficient outcome");
    check(r.rank() == n - 5, t + " singular: rank", double(r.rank()));
    check(relErr(x, ref) < 1e4 * eps<T>(), t + " singular: min-norm == pinv(A) b (error / eps)", relErr(x, ref) / eps<T>());
  }
}

// kappa ~ 3e16: kappa * eps > 1 in double, ~6e-18 in quad. Entries, solution
// and right-hand side are exact in both (1 + 2^-45 needs 46 bits).
template <typename T>
void robustBeyondDouble() {
  using Eigen::robust_lu::Outcome;
  const std::string t = tagOf(typeName<T>(), "RobustLU kappa ~3e16");
  const int n = 200;
  using std::ldexp;
  const T delta = ldexp(T(1), -45);
  std::vector<Eigen::Triplet<T>> tr;
  for (int i = 0; i < n; ++i) {
    const int deg = (i > 0) + (i < n - 1);
    tr.emplace_back(i, i, T(deg) + (i == 0 ? delta : T(0)));
    if (i > 0) tr.emplace_back(i, i - 1, T(-1));
    if (i < n - 1) tr.emplace_back(i, i + 1, T(-1));
  }
  SpMat<T> A(n, n);
  A.setFromTriplets(tr.begin(), tr.end());
  A.makeCompressed();
  Vec<T> xTrue(n);
  for (int i = 0; i < n; ++i) xTrue[i] = T((i * 7) % 11 + 1);
  const Vec<T> b = A * xTrue;
  Eigen::RobustLU<SpMat<T>> r;
  r.setSymmetryTolerance(T(-1));  // the unsymmetric ladder, where IllConditioned is decided
  r.compute(A);
  const Vec<T> x = r.solve(b);
  lu_testing::note(t + ": outcome '" + std::string(Eigen::robust_lu::outcomeName(r.outcome())) + "', kappa estimate " +
                   std::to_string(toDouble(r.conditionEstimate())) + ", forward error " + std::to_string(relErr(x, xTrue)));
  if (std::is_same<T, double>::value) {
    // kappa * eps ~ 6: no digit is guaranteed in double, and RobustLU says so --
    // even though the first rung replaced a pivot and its factors alone suggest
    // kappa ~ 1e10.
    checkTrue(r.outcome() == Outcome::IllConditioned, t + ": double flags it as ill-conditioned");
  } else {
    const double kappa = toDouble(r.conditionEstimate());
    checkTrue(r.outcome() == Outcome::Solved, t + ": quad solves it");
    check(kappa > 1e16 && kappa < 1e17, t + ": condition estimate", kappa);
    // kappa * eps ~ 5e-18 is below RobustLU's 1e-3 trigger for the extended
    // residual, so the answer carries the plain kappa * eps.
    check(relErr(x, xTrue) < 10 * kappa * eps<T>(), t + ": forward error / (kappa eps)",
          relErr(x, xTrue) / (kappa * eps<T>()));
  }
}

template <typename T>
void runOthers() {
  std::printf("--- SupernodalLU / PointBlockLU / SupernodalLDLT / RobustLU <%s> ---\n", typeName<T>());
  supernodalSquare<T>();
  pointBlockSquare<T>();
  ldltIndefinite<T>();
  robustLadder<T>();
  robustBeyondDouble<T>();
}

}  // namespace

int main() {
  runQR<double>();
  runQR<quad>();
  runLU<double>();
  runLU<quad>();
  runOthers<double>();
  runOthers<quad>();
  return lu_testing::summarize("multiprecision");
}
