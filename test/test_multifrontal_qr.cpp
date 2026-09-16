// MultifrontalQR correctness suite. Dependency-free: every matrix is generated,
// and every reference is a DENSE Eigen decomposition (SVD for rank and the
// pseudoinverse, pivoted QR for least squares), so the checks measure the sparse
// solver against an answer computed a different way.

#include <MultifrontalQR>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <complex>
#include <random>
#include <string>
#include <vector>

#include "testing/Check.h"
#include "testing/TestMatrices.h"

using lu_testing::check;
using lu_testing::checkTrue;
using Eigen::Index;

namespace {

template <typename Scalar>
Scalar randomScalar(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  return Scalar(u(rng));
}
template <>
std::complex<double> randomScalar<std::complex<double>>(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  return {u(rng), u(rng)};
}

template <typename Scalar>
Eigen::SparseMatrix<Scalar> randomSparse(int m, int n, double density, unsigned seed, bool addDiagonal = true) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<Eigen::Triplet<Scalar>> t;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      if (u(rng) < density) t.emplace_back(i, j, randomScalar<Scalar>(rng));
  if (addDiagonal)
    for (int i = 0; i < std::min(m, n); ++i) t.emplace_back(i, i, Scalar(4.0));
  Eigen::SparseMatrix<Scalar> A(m, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

template <typename Scalar>
using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <typename Scalar>
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
Index numericalRank(const Dense<Scalar>& A, double relTol) {
  Eigen::JacobiSVD<Dense<Scalar>> svd(A);
  const auto s = svd.singularValues();
  Index r = 0;
  for (Index i = 0; i < s.size(); ++i)
    if (s[i] > relTol * s[0]) ++r;
  return r;
}

template <typename Scalar>
Vec<Scalar> pinvSolve(const Dense<Scalar>& A, const Vec<Scalar>& b, double relTol) {
  Eigen::JacobiSVD<Dense<Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);
  svd.setThreshold(relTol);
  return svd.solve(b);
}

template <typename Scalar>
double relErr(const Vec<Scalar>& a, const Vec<Scalar>& b) {
  const double d = b.norm();
  return (a - b).norm() / (d > 0 ? d : 1.0);
}

// With D = diag(I, deferredRotation()), R^H R must equal (As P D)^H (As P D)
// on the live columns -- exactly, since nothing is dropped from them.
template <typename Scalar>
double factorIdentity(const Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>>& qr, const Eigen::SparseMatrix<Scalar>& A) {
  Dense<Scalar> As = Dense<Scalar>(A);
  As = qr.rowScaling().template cast<Scalar>().asDiagonal() * As * qr.colScaling().template cast<Scalar>().asDiagonal();
  Dense<Scalar> APD = As * qr.colsPermutation();
  const Index n = A.cols(), k = qr.deferredRotation().rows();
  if (k > 0) APD.rightCols(k) = APD.rightCols(k) * qr.deferredRotation();
  Dense<Scalar> R = Dense<Scalar>(qr.matrixR());
  // live columns: the ordinary pivots, then the rotated block's live directions
  const Index dead = Index(qr.deadColumns().size());
  const Index ordinaryLive = n - dead - k;
  std::vector<Index> live;
  for (Index j = 0; j < ordinaryLive; ++j) live.push_back(j);
  for (Index j = 0; j < qr.rank() - ordinaryLive; ++j) live.push_back(n - k + j);
  Dense<Scalar> Rl(R.rows(), Index(live.size())), Al(APD.rows(), Index(live.size()));
  for (std::size_t j = 0; j < live.size(); ++j) {
    Rl.col(Index(j)) = R.col(live[j]);
    Al.col(Index(j)) = APD.col(live[j]);
  }
  Dense<Scalar> lhs = Rl.adjoint() * Rl, rhs = Al.adjoint() * Al;
  return (lhs - rhs).norm() / std::max(1.0, rhs.norm());
}

template <typename Scalar>
void testSquare(const char* tag, const Eigen::SparseMatrix<Scalar>& A, double tolFactor = 1e-10) {
  using Solver = Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>>;
  const Index n = A.cols();
  Vec<Scalar> xTrue(n);
  for (Index i = 0; i < n; ++i) xTrue[i] = Scalar(1.0 + 0.5 * std::sin(double(i)));
  Vec<Scalar> b = A * xTrue;
  Solver qr(A);
  checkTrue(qr.info() == Eigen::Success, std::string(tag) + ": factorize");
  check(qr.rank() == n, std::string(tag) + ": full rank", double(qr.rank()));
  checkTrue(qr.rankIsVerified(), std::string(tag) + ": rank verified");
  Vec<Scalar> x = qr.solve(b);
  check(relErr(x, xTrue) < tolFactor, std::string(tag) + ": forward error", relErr(x, xTrue));
  check(qr.solveResidual() < 1e-13, std::string(tag) + ": residual", qr.solveResidual());
  if (n <= 400) check(factorIdentity(qr, A) < 1e-12, std::string(tag) + ": R^H R == (AP)^H AP", factorIdentity(qr, A));
}

void testOrderingsAgree() {
  const auto A = lu_testing::upwind2d(20, 20);
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), -1.0, 2.0);
  Eigen::VectorXd ref;
  using O = Eigen::multifrontal_qr::Ordering;
  for (O o : {O::COLAMD, O::AMD, O::Natural, O::Auto}) {
    Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
    qr.setOrdering(o);
    qr.compute(A);
    Eigen::VectorXd x = qr.solve(b);
    if (ref.size() == 0) ref = x;
    check(relErr<double>(x, ref) < 1e-12, "orderings agree (" + std::to_string(int(o)) + ")", relErr<double>(x, ref));
  }
}

void testTridiagonalFill() {
  const int n = 3000;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 2.0 + 0.001 * i);
    if (i > 0) t.emplace_back(i, i - 1, -1.0);
    if (i + 1 < n) t.emplace_back(i, i + 1, -0.7);
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr(A);
  // R of a tridiagonal matrix has 3 entries per row; amalgamation may add a few.
  check(qr.nnzR() < 6 * n, "tridiagonal: nnz(R) stays O(n)", double(qr.nnzR()) / n);
  Eigen::VectorXd x = qr.solve(Eigen::VectorXd::Ones(n));
  check((A * x - Eigen::VectorXd::Ones(n)).norm() < 1e-12 * std::sqrt(double(n)), "tridiagonal: residual",
        (A * x - Eigen::VectorXd::Ones(n)).norm());
}

template <typename Scalar>
void testLeastSquares(const char* tag, int m, int n, unsigned seed) {
  const auto A = randomSparse<Scalar>(m, n, 0.05, seed);
  std::mt19937 rng(seed + 1);
  Vec<Scalar> b(m);
  for (int i = 0; i < m; ++i) b[i] = randomScalar<Scalar>(rng);
  Dense<Scalar> Ad = Dense<Scalar>(A);
  Vec<Scalar> ref = Ad.colPivHouseholderQr().solve(b);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>> qr(A);
  check(qr.rank() == n, std::string(tag) + ": full column rank", double(qr.rank()));
  Vec<Scalar> x = qr.solve(b);
  check(relErr(x, ref) < 1e-11, std::string(tag) + ": matches dense least squares", relErr(x, ref));
  check(qr.leastSquaresOptimality() < 1e-14, std::string(tag) + ": optimality ||A^H r||", qr.leastSquaresOptimality());
  check(factorIdentity(qr, A) < 1e-12, std::string(tag) + ": R^H R == (AP)^H AP", factorIdentity(qr, A));
}

template <typename Scalar>
void testUnderdetermined(const char* tag, int m, int n, unsigned seed) {
  const auto A = randomSparse<Scalar>(m, n, 0.08, seed);
  std::mt19937 rng(seed + 7);
  Vec<Scalar> xAny(n);
  for (int i = 0; i < n; ++i) xAny[i] = randomScalar<Scalar>(rng);
  Vec<Scalar> b = A * xAny;
  Dense<Scalar> Ad = Dense<Scalar>(A);
  Vec<Scalar> ref = pinvSolve(Ad, b, 1e-12);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>> qr(A);
  check(qr.rank() == m, std::string(tag) + ": rank = m", double(qr.rank()));
  Vec<Scalar> x = qr.solve(b);
  check(relErr(x, ref) < 1e-11, std::string(tag) + ": minimum-norm solution", relErr(x, ref));
  check(qr.solveResidual() < 1e-13, std::string(tag) + ": consistent residual", qr.solveResidual());
  qr.setSolution(Eigen::multifrontal_qr::Solution::Basic);
  Vec<Scalar> xb = qr.solve(b);
  check((A * xb - b).norm() / b.norm() < 1e-13, std::string(tag) + ": basic solution solves", (A * xb - b).norm() / b.norm());
  check(xb.norm() >= x.norm() * (1 - 1e-12), std::string(tag) + ": basic is not shorter than min-norm",
        xb.norm() / x.norm());
}

// Rank deficiency Heath's rule sees: duplicated and empty columns.
void testExactRankDeficiency() {
  const int m = 120, n = 90;
  auto A = randomSparse<double>(m, n, 0.06, 11);
  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < n; ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
      if (j % 9 == 4) continue;                     // column empty
      const int src = (j % 9 == 7) ? j - 1 : j;     // column duplicates its neighbour
      (void)src;
      t.emplace_back(int(it.row()), j, it.value());
    }
  Eigen::SparseMatrix<double> B(m, n);
  B.setFromTriplets(t.begin(), t.end());
  // make columns j%9==7 exact copies of column j-1
  Eigen::MatrixXd Bd = Eigen::MatrixXd(B);
  for (int j = 0; j < n; ++j)
    if (j % 9 == 7) Bd.col(j) = 3.0 * Bd.col(j - 1);
  B = Bd.sparseView();
  const Index refRank = numericalRank<double>(Bd, 1e-12);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr(B);
  check(qr.rank() == refRank, "exact deficiency: rank matches SVD", double(qr.rank()));
  checkTrue(qr.rankIsVerified(), "exact deficiency: rank verified");
  std::mt19937 rng(3);
  Eigen::VectorXd b(m);
  for (int i = 0; i < m; ++i) b[i] = randomScalar<double>(rng);
  Eigen::VectorXd x = qr.solve(b);
  Eigen::VectorXd ref = pinvSolve<double>(Bd, b, 1e-12);
  check(relErr<double>(x, ref) < 1e-10, "exact deficiency: equals pinv(A) b", relErr<double>(x, ref));
  const Eigen::MatrixXd& N = qr.nullSpace();
  check(N.cols() == n - refRank, "exact deficiency: null space dimension", double(N.cols()));
  check((Bd * N).norm() < 1e-12, "exact deficiency: A N == 0", (Bd * N).norm());
  check((N.transpose() * N - Eigen::MatrixXd::Identity(N.cols(), N.cols())).norm() < 1e-12,
        "exact deficiency: N orthonormal", (N.transpose() * N - Eigen::MatrixXd::Identity(N.cols(), N.cols())).norm());
}

// Entries spanning 1e+-40: the scaled rank decision must see full rank.
void testBadScaling() {
  const int n = 150;
  auto A = randomSparse<double>(n, n, 0.04, 5);
  std::mt19937 rng(9);
  std::uniform_int_distribution<int> e(-40, 40);
  Eigen::VectorXd r(n), c(n);
  for (int i = 0; i < n; ++i) {
    r[i] = std::pow(10.0, e(rng));
    c[i] = std::pow(10.0, e(rng) / 4);
  }
  Eigen::SparseMatrix<double> B = r.asDiagonal() * A * c.asDiagonal();
  Eigen::VectorXd xTrue = Eigen::VectorXd::Ones(n).cwiseQuotient(c);
  Eigen::VectorXd b = B * xTrue;
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr(B);
  check(qr.rank() == n, "bad scaling: full rank", double(qr.rank()));
  Eigen::VectorXd x = qr.solve(b);
  check(relErr<double>(x, xTrue) < 1e-10, "bad scaling: forward error", relErr<double>(x, xTrue));
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.compute(B);
  lu_testing::note("unscaled, the rank would be " + std::to_string(qr.rank()));
}

// kappa ~1e9, with a right-hand side that is EXACT in binary (integer x,
// power-of-two coefficients): the solution of the stored system is then exactly
// xTrue, so the forward error is measurable below kappa * eps. A rounded
// b = A * xTrue would make xTrue the wrong reference, and extended-precision
// refinement would correctly move AWAY from it.
void testIllConditioned() {
  const int n = 50;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 1.0);
    if (i + 1 < n) t.emplace_back(i, i + 1, -2.0);
    if (i > 0) t.emplace_back(i, i - 1, 0.25);
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::VectorXd xTrue(n);
  for (int i = 0; i < n; ++i) xTrue[i] = double((i * 7) % 11 - 5);
  Eigen::VectorXd b = A * xTrue;
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.setRankTolerance(0);  // this test is about accuracy, not rank
  qr.compute(A);
  const double kappa = qr.conditionEstimate();
  check(kappa > 1e8, "ill-conditioned: kappa_1 estimate", kappa);
  qr.setMaxRefinements(0);
  const double e0 = relErr<double>(qr.solve(b), xTrue);
  qr.setMaxRefinements(4);
  const double e1 = relErr<double>(qr.solve(b), xTrue);
  check(e0 > 1e-12, "ill-conditioned: unrefined error ~ kappa eps", e0);
  check(e1 < 1e-15, "ill-conditioned: extended refinement is exact", e1);
}

// The Kahan matrix: upper triangular with moderate diagonal but a tiny smallest
// singular value. Its QR is itself, so Heath's rule sees only healthy pivots and
// reports full rank; only the verification can find the dependency.
void testHiddenRankDeficiency() {
  const int n = 100;
  const double theta = 1.2, c = std::cos(theta), s = std::sin(theta);
  std::vector<Eigen::Triplet<double>> t;
  double sk = 1.0;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, sk * (1.0 + 1e-10 * i));  // the classic perturbation keeps pivoting from reordering
    for (int j = i + 1; j < n; ++j) t.emplace_back(i, j, -c * sk);
    sk *= s;
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::MatrixXd Ad(A);
  const double tau = 1e-10;
  const Index refRank = numericalRank<double>(Ad, tau);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ad);
  lu_testing::note("Kahan: smallest diagonal " + std::to_string(Ad.diagonal().cwiseAbs().minCoeff()) +
                   ", sigma_min/sigma_max " + std::to_string(svd.singularValues()[n - 1] / svd.singularValues()[0]));
  check(refRank < n, "Kahan: reference rank is deficient", double(refRank));
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.setOrdering(Eigen::multifrontal_qr::Ordering::Natural);
  qr.setRankTolerance(tau / svd.singularValues()[0] * Ad.colwise().norm().maxCoeff());
  qr.setRankVerification(false);
  qr.compute(A);
  check(qr.rank() == n, "Kahan: Heath's rule alone misses it", double(qr.rank()));
  qr.setRankVerification(true);
  qr.compute(A);
  check(qr.rank() == refRank, "Kahan: verified rank matches SVD", double(qr.rank()));
  checkTrue(qr.rankIsVerified(), "Kahan: rank verified");
  check(qr.repairIterations() >= 1, "Kahan: repair refactorizations", double(qr.repairIterations()));
  check(factorIdentity(qr, A) < 1e-12, "Kahan: R^H R == (APD)^H APD on live columns", factorIdentity(qr, A));
  check(qr.deferredNullity() == n - refRank, "Kahan: deficiency found in the deferred block", double(qr.deferredNullity()));
  Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd x = qr.solve(b);
  Eigen::VectorXd ref = pinvSolve<double>(Ad, b, tau);
  check(relErr<double>(x, ref) < 1e-6, "Kahan: equals truncated pinv(A) b", relErr<double>(x, ref));
}

// 0/1 matrices with hub rows and empty columns: many exact dependencies, many
// dead columns, and fronts with more rows than columns (so the contribution
// blocks get compressed). The system is consistent, so the basic solution must
// solve it whatever the rank.
void testGraphMatrices() {
  double worstRes = 0, worstIdentity = 0;
  int rankMismatch = 0, runs = 0;
  for (unsigned seed = 1; seed <= 150; ++seed) {
    std::mt19937 rng(seed);
    const int n = 30 + int(seed % 60);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<Eigen::Triplet<double>> t;
    const int edges = n + int(seed % 3) * n / 2;
    for (int e = 0; e < edges; ++e) {
      const int i = pick(rng) % (1 + pick(rng));
      t.emplace_back(i, pick(rng), 1.0);
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    for (Index k = 0; k < A.nonZeros(); ++k) A.valuePtr()[k] = 1.0;
    A.makeCompressed();
    Eigen::MatrixXd Ad(A);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ad);
    Eigen::VectorXd b = A * Eigen::VectorXd::LinSpaced(n, 1.0, 2.0);
    using O = Eigen::multifrontal_qr::Ordering;
    for (O o : {O::COLAMD, O::AMD, O::Natural}) {
      Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
      qr.setOrdering(o);
      qr.setSolution(Eigen::multifrontal_qr::Solution::Basic);
      qr.setMaxRefinements(0);
      qr.compute(A);
      ++runs;
      const Eigen::VectorXd x = qr.solve(b);
      worstRes = std::max(worstRes, (A * x - b).norm() / b.norm());
      worstIdentity = std::max(worstIdentity, factorIdentity(qr, A));
      // Reference rank at the solver's own threshold, in the solver's scaling.
      Eigen::MatrixXd As = qr.rowScaling().asDiagonal() * Ad * qr.colScaling().asDiagonal();
      Eigen::JacobiSVD<Eigen::MatrixXd> ss(As);
      Index ref = 0;
      for (Index i = 0; i < n; ++i)
        if (ss.singularValues()[i] > qr.absoluteRankThreshold()) ++ref;
      if (ref != qr.rank()) ++rankMismatch;
    }
  }
  check(worstRes < 1e-13, "graph matrices: basic solution residual (worst)", worstRes);
  check(worstIdentity < 1e-12, "graph matrices: factor identity (worst)", worstIdentity);
  check(rankMismatch == 0, "graph matrices: rank == SVD rank (" + std::to_string(runs) + " runs)", double(rankMismatch));
}

void testParallelIdentical() {
  const auto A = lu_testing::laplacian2d(40, 40);
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 0.0, 1.0);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> s(A);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>, Eigen::supernodal_lu::PooledExecutor> p;
  p.executor() = Eigen::supernodal_lu::PooledExecutor(4);
  p.compute(A);
  Eigen::VectorXd xs = s.solve(b), xp = p.solve(b);
  check((xs - xp).norm() == 0.0, "parallel: bit-identical to serial", (xs - xp).norm());
}

void testEdgeCases() {
  // empty rows and columns, a zero matrix, a single column
  Eigen::SparseMatrix<double> Z(5, 4);
  Z.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qz(Z);
  check(qz.rank() == 0 && qz.info() == Eigen::Success, "zero matrix: rank 0", double(qz.rank()));
  Eigen::VectorXd xz = qz.solve(Eigen::VectorXd::Ones(5));
  check(xz.norm() == 0.0, "zero matrix: min-norm solution is 0", xz.norm());

  Eigen::SparseMatrix<double> C(4, 1);
  C.insert(1, 0) = 2.0;
  C.insert(3, 0) = 2.0;
  C.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qc(C);
  Eigen::VectorXd bc(4);
  bc << 1, 1, 1, 3;  // x = (2*1 + 2*3) / (2^2 + 2^2) = 1
  check(std::abs(qc.solve(bc)[0] - 1.0) < 1e-15, "single column: least squares", qc.solve(bc)[0]);
}

}  // namespace

int main() {
  std::printf("--- square ---\n");
  testSquare<double>("random 300", randomSparse<double>(300, 300, 0.01, 1));
  testSquare<double>("laplacian 30x30", lu_testing::laplacian2d(30, 30));
  testSquare<double>("upwind 25x25", lu_testing::upwind2d(25, 25));
  testSquare<std::complex<double>>("complex random 200", randomSparse<std::complex<double>>(200, 200, 0.02, 2));
  testSquare<double>("laplacian3d 12^3 (large R11)", lu_testing::laplacian3d(12, 12, 12));
  testOrderingsAgree();
  testTridiagonalFill();
  std::printf("--- least squares ---\n");
  testLeastSquares<double>("tall 300x120", 300, 120, 3);
  testLeastSquares<std::complex<double>>("complex tall 200x90", 200, 90, 4);
  testUnderdetermined<double>("wide 80x150", 80, 150, 5);
  testUnderdetermined<std::complex<double>>("complex wide 60x100", 60, 100, 6);
  std::printf("--- rank ---\n");
  testExactRankDeficiency();
  testHiddenRankDeficiency();
  testBadScaling();
  testGraphMatrices();
  std::printf("--- accuracy / parallel / edge ---\n");
  testIllConditioned();
  testParallelIdentical();
  testEdgeCases();
  return lu_testing::summarize("MultifrontalQR");
}
