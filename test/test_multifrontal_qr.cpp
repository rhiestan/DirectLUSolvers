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

// Every test runs once per engine; main() sets this.
Eigen::multifrontal_qr::Engine g_engine = Eigen::multifrontal_qr::Engine::Multifrontal;

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
  Solver qr;
  qr.setEngine(g_engine);
  qr.compute(A);
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
    qr.setEngine(g_engine);
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
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(A);
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
  Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>> qr;
  qr.setEngine(g_engine);
  qr.compute(A);
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
  Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>> qr;
  qr.setEngine(g_engine);
  qr.compute(A);
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
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(B);
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
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(B);
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
  qr.setEngine(g_engine);
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
  qr.setEngine(g_engine);
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
      qr.setEngine(g_engine);
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

// Square, rank deficient, and INCONSISTENT: exactly where row scaling would turn
// least squares into weighted least squares. Scaling::Auto must refactor without
// it and return pinv(A) b.
void testSquareInconsistentLeastSquares() {
  const int n = 200;
  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    if (j == n / 2) continue;  // an empty column
    t.emplace_back(j, j, 3.0);
    if (j > 0) t.emplace_back(j - 1, j, 1.0);
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  const Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
  Eigen::MatrixXd Ad(A);
  const Eigen::VectorXd ref = pinvSolve<double>(Ad, b, 1e-12);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(A);
  const Eigen::VectorXd x = qr.solve(b);
  check(qr.rank() == n - 1, "square inconsistent: rank n-1", double(qr.rank()));
  checkTrue(!qr.isWeightedLeastSquares(), "square inconsistent: least squares is unweighted");
  check((A.transpose() * (A * x - b)).norm() < 1e-12, "square inconsistent: A^T r == 0",
        (A.transpose() * (A * x - b)).norm());
  check(relErr<double>(x, ref) < 1e-12, "square inconsistent: equals pinv(A) b", relErr<double>(x, ref));

  // Rows spanning 1e+-40 and one empty column: column scaling alone decides a
  // different rank, so the row-scaled factorization is kept -- and said to be
  // weighted.
  auto B = randomSparse<double>(n, n, 0.03, 17);
  std::vector<Eigen::Triplet<double>> u;
  std::mt19937 rng(23);
  std::uniform_int_distribution<int> e(-40, 40);
  std::vector<double> rs(n);
  for (int i = 0; i < n; ++i) rs[i] = std::pow(10.0, e(rng));
  for (int j = 0; j < n; ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(B, j); it; ++it)
      if (j != 7) u.emplace_back(int(it.row()), j, it.value() * rs[it.row()]);
  Eigen::SparseMatrix<double> C(n, n);
  C.setFromTriplets(u.begin(), u.end());
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qc;
  qc.setEngine(g_engine);
  qc.compute(C);
  check(qc.rank() == n - 1, "badly row-scaled, singular: rank n-1", double(qc.rank()));
  checkTrue(qc.isWeightedLeastSquares(), "badly row-scaled, singular: reported as weighted");
}

void testParallelIdentical() {
  const auto A = lu_testing::laplacian2d(40, 40);
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 0.0, 1.0);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> s;
  s.setEngine(g_engine);
  s.compute(A);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>, Eigen::supernodal_lu::PooledExecutor> p;
  p.setEngine(g_engine);
  p.executor() = Eigen::supernodal_lu::PooledExecutor(4);
  p.compute(A);
  Eigen::VectorXd xs = s.solve(b), xp = p.solve(b);
  check((xs - xp).norm() == 0.0, "parallel: bit-identical to serial", (xs - xp).norm());
}

void testEdgeCases() {
  // empty rows and columns, a zero matrix, a single column
  Eigen::SparseMatrix<double> Z(5, 4);
  Z.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qz;
  qz.setEngine(g_engine);
  qz.compute(Z);
  check(qz.rank() == 0 && qz.info() == Eigen::Success, "zero matrix: rank 0", double(qz.rank()));
  Eigen::VectorXd xz = qz.solve(Eigen::VectorXd::Ones(5));
  check(xz.norm() == 0.0, "zero matrix: min-norm solution is 0", xz.norm());

  Eigen::SparseMatrix<double> C(4, 1);
  C.insert(1, 0) = 2.0;
  C.insert(3, 0) = 2.0;
  C.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qc;
  qc.setEngine(g_engine);
  qc.compute(C);
  Eigen::VectorXd bc(4);
  bc << 1, 1, 1, 3;  // x = (2*1 + 2*3) / (2^2 + 2^2) = 1
  check(std::abs(qc.solve(bc)[0] - 1.0) < 1e-15, "single column: least squares", qc.solve(bc)[0]);
}

// ---------------------------------------------------------------------------
// Rank-deficient matrices of every shape and scalar type, checked the same way:
// rank against a dense SVD of the SCALED matrix at the solver's own threshold,
// the factor identity, the minimum-norm solution against pinv(A) b, the
// least-squares optimality condition, and the null-space basis.
// ---------------------------------------------------------------------------

template <typename Scalar>
bool isPermutation(const Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>>& qr) {
  const auto p = qr.colsPermutation();
  std::vector<char> seen(std::size_t(p.size()), 0);
  for (Index i = 0; i < p.size(); ++i) {
    const Index v = p.indices()(i);
    if (v < 0 || v >= p.size() || seen[std::size_t(v)]) return false;
    seen[std::size_t(v)] = 1;
  }
  return true;
}

template <typename Scalar>
void checkRankDeficient(const std::string& tag, const Eigen::SparseMatrix<Scalar>& A, const Vec<Scalar>& b,
                        Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>>& qr, double solveTol = 1e-9) {
  const Index n = A.cols();
  qr.compute(A);
  checkTrue(qr.info() == Eigen::Success, tag + ": factorize");
  if (qr.info() != Eigen::Success) {
    lu_testing::note(qr.lastErrorMessage());
    return;
  }
  Dense<Scalar> Ad(A);
  Dense<Scalar> As = qr.rowScaling().template cast<Scalar>().asDiagonal() * Ad * qr.colScaling().template cast<Scalar>().asDiagonal();
  Eigen::JacobiSVD<Dense<Scalar>> svd(As);
  Index ref = 0;
  for (Index i = 0; i < svd.singularValues().size(); ++i)
    if (svd.singularValues()[i] > qr.absoluteRankThreshold()) ++ref;
  check(qr.rank() == ref, tag + ": rank == SVD rank (" + std::to_string(ref) + ")", double(qr.rank()));
  checkTrue(qr.rankIsVerified(), tag + ": rank verified");
  checkTrue(isPermutation(qr), tag + ": colsPermutation is a permutation");
  check(Index(qr.deadColumns().size()) + qr.deferredNullity() + qr.rank() == n, tag + ": dead + nullity + rank == n",
        double(qr.deadColumns().size()));
  if (n <= 400) check(factorIdentity(qr, A) < 1e-11, tag + ": R^H R == (APD)^H APD", factorIdentity(qr, A));
  const Vec<Scalar> x = qr.solve(b);
  const Vec<Scalar> ref_x = pinvSolve<Scalar>(Ad, b, 1e-11);
  check(relErr(x, ref_x) < solveTol, tag + ": min-norm == pinv(A) b", relErr(x, ref_x));
  const double opt = (Ad.adjoint() * (Ad * x - b)).norm() / std::max(1.0, b.norm());
  check(opt < 1e-10, tag + ": A^H r == 0", opt);
  if (qr.rank() < n) {
    const Dense<Scalar>& N = qr.nullSpace();
    check(N.cols() == n - qr.rank(), tag + ": null space dimension", double(N.cols()));
    if (N.cols() > 0) {
      check((Ad * N).norm() < 1e-10 * std::max(1.0, Ad.norm()), tag + ": A N == 0", (Ad * N).norm());
      const double orth = (N.adjoint() * N - Dense<Scalar>::Identity(N.cols(), N.cols())).norm();
      check(orth < 1e-11, tag + ": N orthonormal", orth);
    }
  }
}

// A sparse matrix with duplicated (complex-scaled) and empty columns, and
// duplicated and empty rows, in every shape.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> dependentColumns(int m, int n, unsigned seed, int period) {
  Dense<Scalar> Ad = Dense<Scalar>(randomSparse<Scalar>(m, n, 0.06, seed));
  for (int j = 0; j < n; ++j) {
    if (j % period == 2) Ad.col(j).setZero();
    if (j % period == period - 1) Ad.col(j) = Scalar(0.5) * Ad.col(j - 1) - Scalar(1.5) * Ad.col(j - 2);
  }
  if (m > 8) {
    Ad.row(7) = Scalar(3) * Ad.row(3);
    Ad.row(m / 2).setZero();
  }
  Eigen::SparseMatrix<Scalar> A = Ad.sparseView();
  A.makeCompressed();
  return A;
}

template <typename Scalar>
Vec<Scalar> randomVector(int m, unsigned seed) {
  std::mt19937 rng(seed);
  Vec<Scalar> b(m);
  for (int i = 0; i < m; ++i) b[i] = randomScalar<Scalar>(rng);
  return b;
}

template <typename Scalar>
void testDeficientShapes(const char* tag) {
  using Solver = Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>>;
  {
    Solver qr;
    qr.setEngine(g_engine);
    checkRankDeficient<Scalar>(std::string(tag) + " tall 200x90", dependentColumns<Scalar>(200, 90, 91, 8), randomVector<Scalar>(200, 15), qr);
    check(qr.leastSquaresOptimality() < 1e-13, std::string(tag) + " tall 200x90: optimality", qr.leastSquaresOptimality());
    qr.setSolution(Eigen::multifrontal_qr::Solution::Basic);
    const Vec<Scalar> b = randomVector<Scalar>(200, 15);
    const Vec<Scalar> xb = qr.solve(b);
    Dense<Scalar> Ad(dependentColumns<Scalar>(200, 90, 91, 8));
    check((Ad.adjoint() * (Ad * xb - b)).norm() < 1e-10, std::string(tag) + " tall 200x90: basic solution is least squares",
          (Ad.adjoint() * (Ad * xb - b)).norm());
  }
  {
    Solver qr;
    qr.setEngine(g_engine);
    checkRankDeficient<Scalar>(std::string(tag) + " wide 50x120", dependentColumns<Scalar>(50, 120, 31, 11), randomVector<Scalar>(50, 5), qr);
  }
  {
    Solver qr;
    qr.setEngine(g_engine);
    checkRankDeficient<Scalar>(std::string(tag) + " square 90x90", dependentColumns<Scalar>(90, 90, 21, 7), randomVector<Scalar>(90, 4), qr);
  }
}

// The Kahan matrix with unitary diagonal phase factors on both sides: the same
// singular values and the same hidden dependency, but every Householder
// reflector, the verification and the deferred block's SVD run in complex
// arithmetic.
void testComplexHiddenRankDeficiency() {
  typedef std::complex<double> cd;
  const int n = 100;
  const double theta = 1.2, c = std::cos(theta), s = std::sin(theta);
  std::vector<Eigen::Triplet<cd>> t;
  double sk = 1.0;
  for (int i = 0; i < n; ++i) {
    const cd rowPhase = std::polar(1.0, 0.7 * i);
    t.emplace_back(i, i, rowPhase * sk * (1.0 + 1e-10 * i) * std::polar(1.0, -0.3 * i));
    for (int j = i + 1; j < n; ++j) t.emplace_back(i, j, rowPhase * (-c * sk) * std::polar(1.0, -0.3 * j));
    sk *= s;
  }
  Eigen::SparseMatrix<cd> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Dense<cd> Ad(A);
  Eigen::JacobiSVD<Dense<cd>> svd(Ad);
  const double tau = 1e-10;
  const Index refRank = numericalRank<cd>(Ad, tau);
  check(refRank < n, "complex Kahan: reference rank is deficient", double(refRank));
  Eigen::MultifrontalQR<Eigen::SparseMatrix<cd>> qr;
  qr.setEngine(g_engine);
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.setOrdering(Eigen::multifrontal_qr::Ordering::Natural);
  qr.setRankTolerance(tau / svd.singularValues()[0] * Ad.colwise().norm().maxCoeff());
  qr.compute(A);
  check(qr.rank() == refRank, "complex Kahan: verified rank matches SVD", double(qr.rank()));
  checkTrue(qr.rankIsVerified(), "complex Kahan: rank verified");
  check(factorIdentity(qr, A) < 1e-11, "complex Kahan: R^H R == (APD)^H APD", factorIdentity(qr, A));
  const Vec<cd> b = Vec<cd>::Ones(n);
  const Vec<cd> x = qr.solve(b), ref = pinvSolve<cd>(Ad, b, tau);
  check(relErr(x, ref) < 1e-5, "complex Kahan: equals truncated pinv(A) b", relErr(x, ref));
}

// A Kahan block coupled (weakly) into a Laplacian: the deferred column is one
// of many, under every ordering, and the deferred front sits inside a real
// elimination tree instead of being the whole matrix.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> embeddedKahan(int k, int grid, unsigned seed) {
  const double theta = 1.2, c = std::cos(theta), s = std::sin(theta);
  std::vector<Eigen::Triplet<Scalar>> t;
  double sk = 1.0;
  for (int i = 0; i < k; ++i) {
    t.emplace_back(i, i, Scalar(sk * (1.0 + 1e-10 * i)));
    for (int j = i + 1; j < k; ++j) t.emplace_back(i, j, Scalar(-c * sk));
    sk *= s;
  }
  const auto L = lu_testing::laplacian2dAs<Scalar>(grid, grid);
  const int n = k + int(L.cols());
  for (int j = 0; j < L.outerSize(); ++j)
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(L, j); it; ++it) t.emplace_back(k + int(it.row()), k + j, it.value());
  std::mt19937 rng(seed);
  for (int e = 0; e < 60; ++e) {
    const int i = int(rng() % unsigned(k)), j = k + int(rng() % unsigned(L.cols()));
    t.emplace_back(i, j, Scalar(1e-13) * randomScalar<Scalar>(rng));  // far below the Kahan block's smallest singular value
    t.emplace_back(j, i, Scalar(1e-13) * randomScalar<Scalar>(rng));
  }
  Eigen::SparseMatrix<Scalar> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

template <typename Scalar>
void testEmbeddedHiddenDeficiency(const char* tag) {
  const auto A = embeddedKahan<Scalar>(100, 12, 13);
  const Vec<Scalar> b = Vec<Scalar>::Ones(A.rows());
  using O = Eigen::multifrontal_qr::Ordering;
  for (O o : {O::COLAMD, O::AMD, O::Natural}) {
    Eigen::MultifrontalQR<Eigen::SparseMatrix<Scalar>> qr;
    qr.setEngine(g_engine);
    qr.setOrdering(o);
    qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
    qr.setRankTolerance(1e-10);
    qr.setThoroughVerification(true);
    checkRankDeficient<Scalar>(std::string(tag) + " (ordering " + std::to_string(int(o)) + ")", A, b, qr, 1e-4);
    check(qr.deferredNullity() == 1, std::string(tag) + ": deficiency found in the deferred block", double(qr.deferredNullity()));
  }
}

// Serial and parallel factorizations must agree bit for bit also when the
// verification defers columns and refactors.
void testParallelIdenticalWithDeferral() {
  const auto A = embeddedKahan<double>(100, 30, 14);
  const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> s;
  s.setEngine(g_engine);
  s.setScaling(Eigen::multifrontal_qr::Scaling::None);
  s.setRankTolerance(1e-10);
  s.compute(A);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>, Eigen::supernodal_lu::PooledExecutor> p;
  p.setEngine(g_engine);
  p.setScaling(Eigen::multifrontal_qr::Scaling::None);
  p.setRankTolerance(1e-10);
  p.executor() = Eigen::supernodal_lu::PooledExecutor(4);
  p.compute(A);
  check(s.rank() == A.cols() - 1 && p.rank() == s.rank(), "parallel with deferral: rank n-1 on both", double(p.rank()));
  const Eigen::VectorXd xs = s.solve(b), xp = p.solve(b);
  check((xs - xp).norm() == 0.0, "parallel with deferral: bit-identical to serial", (xs - xp).norm());
}

// A front wide enough for the chunked trailing update to run across lanes.
void testLargeFrontParallel() {
  if (g_engine != Eigen::multifrontal_qr::Engine::Multifrontal) return;
  const auto A = lu_testing::laplacian3d(16, 16, 16);
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), -1.0, 1.0);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> s;
  s.compute(A);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>, Eigen::supernodal_lu::PooledExecutor> p;
  p.executor() = Eigen::supernodal_lu::PooledExecutor(8);
  p.compute(A);
  const Eigen::VectorXd xs = s.solve(b), xp = p.solve(b);
  check((xs - xp).norm() == 0.0, "laplacian3d 16^3, 8 lanes: bit-identical to serial", (xs - xp).norm());
  check((A * xs - b).norm() / b.norm() < 1e-12, "laplacian3d 16^3: residual", (A * xs - b).norm() / b.norm());
}

// Several right-hand sides at once must equal the columns solved one by one.
void testMultipleRightHandSides() {
  const auto A = dependentColumns<double>(80, 60, 41, 9);
  const int m = int(A.rows());
  Eigen::MatrixXd B(m, 3);
  for (int j = 0; j < 3; ++j) B.col(j) = randomVector<double>(m, 6 + unsigned(j));
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(A);
  const Eigen::MatrixXd X = qr.solve(B);
  double worst = 0.0;
  for (int j = 0; j < 3; ++j) {
    const Eigen::VectorXd xj = qr.solve(Eigen::VectorXd(B.col(j)));
    worst = std::max(worst, (X.col(j) - xj).norm() / xj.norm());
  }
  check(worst < 1e-13, "multiple rhs: equals column-by-column solves", worst);
}

// factorize() again on the same pattern, after a factorization that deferred
// columns: the deferred structure stays, the new values decide the rank.
void testRefactorizeAfterDeferral() {
  const int n = 100;
  const double theta = 1.2, c = std::cos(theta), s = std::sin(theta);
  std::vector<Eigen::Triplet<double>> t;
  double sk = 1.0;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, sk * (1.0 + 1e-10 * i));
    for (int j = i + 1; j < n; ++j) t.emplace_back(i, j, -c * sk);
    sk *= s;
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::MatrixXd Ad(A);
  const double tau = 1e-10;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ad);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.setOrdering(Eigen::multifrontal_qr::Ordering::Natural);
  qr.setRankTolerance(tau / svd.singularValues()[0] * Ad.colwise().norm().maxCoeff());
  qr.compute(A);
  const Index r1 = qr.rank();
  check(r1 < n && !qr.deferredColumns().empty(), "refactorize: first factorization deferred columns", double(qr.deferredColumns().size()));
  // same pattern, well-conditioned values
  Eigen::SparseMatrix<double> B = A;
  for (int j = 0; j < n; ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(B, j); it; ++it) it.valueRef() = it.row() == j ? 1.0 : 0.01;
  qr.factorize(B);
  checkTrue(qr.info() == Eigen::Success, "refactorize: factorize(B) succeeds");
  check(qr.rank() == n && qr.rankIsVerified(), "refactorize: B is full rank and verified", double(qr.rank()));
  const Eigen::VectorXd xt = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0), bb = B * xt;
  check(relErr<double>(qr.solve(bb), xt) < 1e-10, "refactorize: B forward error", relErr<double>(qr.solve(bb), xt));
  check(factorIdentity(qr, B) < 1e-11, "refactorize: B factor identity", factorIdentity(qr, B));
  qr.factorize(A);
  check(qr.rank() == r1, "refactorize: A again, same rank", double(qr.rank()));
  const Eigen::VectorXd b1 = Eigen::VectorXd::Ones(n);
  check(relErr<double>(qr.solve(b1), pinvSolve<double>(Ad, b1, tau)) < 1e-6, "refactorize: A again, equals truncated pinv(A) b",
        relErr<double>(qr.solve(b1), pinvSolve<double>(Ad, b1, tau)));
}

// Explicit zeros and an uncompressed input; analyzePattern on one, factorize on the other.
void testExplicitZerosAndUncompressed() {
  const int n = 80;
  const auto A = randomSparse<double>(n, n, 0.04, 61);
  Eigen::SparseMatrix<double> B(n, n);
  B.reserve(Eigen::VectorXi::Constant(n, 8));
  for (int j = 0; j < n; ++j) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) B.insert(it.row(), j) = it.value();
    B.coeffRef((j * 7 + 3) % n, j) += 0.0;  // an explicit zero (or an existing entry)
  }
  checkTrue(!B.isCompressed(), "explicit zeros: input is uncompressed");
  const Eigen::VectorXd xt = Eigen::VectorXd::LinSpaced(n, 1.0, 2.0), b = B * xt;
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(B);
  check(qr.info() == Eigen::Success && qr.rank() == n, "explicit zeros: full rank", double(qr.rank()));
  check(relErr<double>(qr.solve(b), xt) < 1e-11, "explicit zeros: forward error", relErr<double>(qr.solve(b), xt));
  Eigen::SparseMatrix<double> C = B;
  C.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> q2;
  q2.setEngine(g_engine);
  q2.analyzePattern(C);
  q2.factorize(B);
  checkTrue(q2.info() == Eigen::Success, "explicit zeros: analyzePattern(compressed) + factorize(uncompressed)");
  check(relErr<double>(q2.solve(b), xt) < 1e-11, "explicit zeros: mixed forward error", relErr<double>(q2.solve(b), xt));
}

// Dead columns at every position relative to the Householder panels.
void testBlockSizes() {
  const auto A = dependentColumns<double>(150, 120, 71, 5);
  const Eigen::VectorXd b = randomVector<double>(150, 9);
  for (int bs : {1, 2, 3, 5}) {
    Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
    qr.setEngine(g_engine);
    qr.setBlockSize(bs);
    qr.setAmalgamation(bs % 2 == 0);
    checkRankDeficient<double>("block size " + std::to_string(bs) + (bs % 2 ? ", no amalgamation" : ""), A, b, qr);
  }
}

// The Neumann Laplacian: one exact null vector spread over every column.
void testNeumannLaplacian() {
  static constexpr int g = 15;
  const int n = g * g;
  std::vector<Eigen::Triplet<double>> t;
  auto id = [](int x, int y) { return y * g + x; };
  for (int y = 0; y < g; ++y)
    for (int x = 0; x < g; ++x) {
      const int i = id(x, y);
      int deg = 0;
      if (x > 0) { t.emplace_back(i, id(x - 1, y), -1.0); ++deg; }
      if (x + 1 < g) { t.emplace_back(i, id(x + 1, y), -1.0); ++deg; }
      if (y > 0) { t.emplace_back(i, id(x, y - 1), -1.0); ++deg; }
      if (y + 1 < g) { t.emplace_back(i, id(x, y + 1), -1.0); ++deg; }
      t.emplace_back(i, i, double(deg));
    }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  const Eigen::VectorXd b = A * Eigen::VectorXd::LinSpaced(n, 0.0, 1.0);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  checkRankDeficient<double>("Neumann Laplacian", A, b, qr);
  const Eigen::MatrixXd& N = qr.nullSpace();
  if (N.cols() == 1) check(N.col(0).cwiseAbs().maxCoeff() - N.col(0).cwiseAbs().minCoeff() < 1e-10, "Neumann Laplacian: null vector is constant",
                           N.col(0).cwiseAbs().maxCoeff() - N.col(0).cwiseAbs().minCoeff());
}

// A rank tolerance large enough to kill or defer many columns: the factor
// identity on the live columns and a finite, LS-optimal solution still hold.
void testLargeRankTolerance() {
  const int n = 150;
  Eigen::MatrixXd Ad(randomSparse<double>(n, n, 0.03, 101, false));
  for (int i = 0; i < n; ++i) Ad(i, i) += 1e-6 * (i % 3);
  Eigen::SparseMatrix<double> A = Ad.sparseView();
  A.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.setRankTolerance(1e-2);
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.compute(A);
  checkTrue(qr.info() == Eigen::Success && qr.rank() < n, "large tolerance: factorizes, rank deficient");
  checkTrue(isPermutation(qr), "large tolerance: colsPermutation is a permutation");
  check(factorIdentity(qr, A) < 1e-11, "large tolerance: R^H R == (APD)^H APD on live columns", factorIdentity(qr, A));
  checkTrue(qr.solve(Eigen::VectorXd::Ones(n)).allFinite(), "large tolerance: finite solution");
}

// Non-finite entries are refused, not factored.
void testNonFiniteInput() {
  auto A = randomSparse<double>(30, 30, 0.1, 81);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  A.valuePtr()[5] = std::numeric_limits<double>::infinity();
  qr.compute(A);
  checkTrue(qr.info() != Eigen::Success, "inf entry: not a success");
  A.valuePtr()[5] = std::numeric_limits<double>::quiet_NaN();
  qr.compute(A);
  checkTrue(qr.info() != Eigen::Success, "nan entry: not a success");
  // Every column norm finite, but their squares overflow when summed: not a
  // non-finite matrix.
  Eigen::SparseMatrix<double> B(30, 30);
  for (int i = 0; i < 30; ++i) B.insert(i, i) = 1e154;
  B.makeCompressed();
  qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
  qr.compute(B);
  checkTrue(qr.lastErrorMessage().find("non-finite") == std::string::npos,
            "huge finite entries: not reported as non-finite");
}

// An uncompressed input factors exactly as its compressed copy does.
void testUncompressedInput() {
  auto A = randomSparse<double>(80, 60, 0.06, 83);
  Eigen::SparseMatrix<double> U = A;
  U.uncompress();
  U.coeffRef(0, 0) += 0.0;  // keeps the pattern, leaves the storage uncompressed
  checkTrue(!U.isCompressed(), "uncompressed input: storage really uncompressed");
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qa, qu;
  qa.setEngine(g_engine);
  qu.setEngine(g_engine);
  qa.compute(A);
  qu.compute(U);
  const Eigen::VectorXd b = A * Eigen::VectorXd::LinSpaced(60, 1.0, 2.0);
  checkTrue(qu.info() == Eigen::Success && qu.rank() == qa.rank() && qu.solve(b) == qa.solve(b),
            "uncompressed input: same rank and bit-identical solution");
}

// One solver object reused across shapes.
void testReuseAcrossShapes() {
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  const auto A = randomSparse<double>(100, 100, 0.03, 201);
  qr.compute(A);
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(100);
  check(relErr<double>(qr.solve(A * ones), ones) < 1e-10, "reuse: 100x100", relErr<double>(qr.solve(A * ones), ones));
  const auto B = randomSparse<double>(30, 50, 0.1, 207);
  qr.compute(B);
  const Eigen::VectorXd bb = B * Eigen::VectorXd::LinSpaced(50, 1.0, 2.0);
  check(qr.rank() == 30 && (B * qr.solve(bb) - bb).norm() / bb.norm() < 1e-12, "reuse: 30x50 after 100x100",
        (B * qr.solve(bb) - bb).norm() / bb.norm());
  const auto C = randomSparse<double>(60, 40, 0.1, 203);
  qr.compute(C);
  const Eigen::VectorXd xc = Eigen::VectorXd::LinSpaced(40, 1.0, 2.0);
  check(relErr<double>(qr.solve(C * xc), xc) < 1e-10, "reuse: 60x40 after 30x50", relErr<double>(qr.solve(C * xc), xc));
}

// Matrices with no rows or no columns -- also with assertions enabled, where a
// sparse reduction over zero rows would assert.
void testEmptyShapes() {
  auto run = [&](int m, int n, const std::string& tag) {
    Eigen::SparseMatrix<double> A(m, n);
    A.makeCompressed();
    Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
    qr.setEngine(g_engine);
    qr.compute(A);
    checkTrue(qr.info() == Eigen::Success && qr.rank() == 0, tag + ": rank 0");
    const Eigen::VectorXd x = qr.solve(Eigen::VectorXd::Ones(m));
    checkTrue(x.size() == n && x.norm() == 0.0, tag + ": solution is 0");
    checkTrue(qr.nullSpace().cols() == n, tag + ": null space is everything");
  };
  run(0, 0, "0x0");
  run(0, 5, "0x5");
  run(5, 0, "5x0");
  Eigen::SparseMatrix<double> R(1, 5);
  for (int j = 0; j < 5; ++j) R.insert(0, j) = j + 1.0;
  R.makeCompressed();
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(R);
  Eigen::VectorXd b1(1);
  b1 << 55.0;
  check(qr.rank() == 1 && relErr<double>(qr.solve(b1), pinvSolve<double>(Eigen::MatrixXd(R), b1, 1e-12)) < 1e-13, "1x5: min-norm solution",
        relErr<double>(qr.solve(b1), pinvSolve<double>(Eigen::MatrixXd(R), b1, 1e-12)));
}

// Options changed AFTER compute() must apply to later solves -- also on the
// cached column-scaled fallback factorization an inconsistent square solve
// builds, and to the "null space too large" decision when the cap is raised.
void testOptionsChangedAfterCompute() {
  const int n = 60;
  Eigen::MatrixXd Ad(randomSparse<double>(n, n, 0.05, 51));
  for (int j = 0; j < n; ++j)
    if (j % 10 == 5) Ad.col(j) = 2.0 * Ad.col(j - 1);
  Eigen::SparseMatrix<double> A = Ad.sparseView();
  A.makeCompressed();
  const Eigen::VectorXd b = Eigen::VectorXd::Ones(n);  // inconsistent: A is square and singular
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setEngine(g_engine);
  qr.compute(A);
  const Eigen::VectorXd xmin = qr.solve(b);
  checkTrue(!qr.isWeightedLeastSquares(), "options after compute: unweighted fallback in use");
  qr.setSolution(Eigen::multifrontal_qr::Solution::Basic);
  const Eigen::VectorXd xb = qr.solve(b);
  double deadNorm = 0.0;
  for (auto c : qr.deadColumns()) deadNorm += std::abs(xb[c]);
  check(deadNorm == 0.0, "options after compute: Basic after MinimumNorm zeroes the dead columns", deadNorm);
  check((A.transpose() * (A * xb - b)).norm() < 1e-10, "options after compute: basic solution is least squares",
        (A.transpose() * (A * xb - b)).norm());
  qr.setSolution(Eigen::multifrontal_qr::Solution::MinimumNorm);
  qr.setMaxRefinements(0);
  const Eigen::VectorXd x0 = qr.solve(b);  // solve() is lazy: it must be evaluated to run
  check(qr.iterativeRefinements() == 0, "options after compute: setMaxRefinements(0) honoured", double(qr.iterativeRefinements()));
  check(x0.allFinite(), "options after compute: unrefined solution is finite", x0.norm());

  const auto B = randomSparse<double>(30, 50, 0.1, 207);
  const Eigen::VectorXd bb = B * Eigen::VectorXd::LinSpaced(50, 1.0, 2.0);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qb;
  qb.setEngine(g_engine);
  qb.compute(B);
  qb.setMaxNullSpaceScalars(10);
  const Eigen::VectorXd xbasic = qb.solve(bb);
  checkTrue(!qb.lastSolveMessage().empty() && qb.nullSpace().cols() == 0, "null-space cap: basic solution, message set");
  qb.setMaxNullSpaceScalars(5e7);
  const Eigen::VectorXd xm = qb.solve(bb);
  checkTrue(qb.lastSolveMessage().empty(), "null-space cap lifted: min-norm solution again");
  check(xm.norm() < xbasic.norm() * (1 - 1e-6), "null-space cap lifted: shorter than the basic solution", xm.norm() / xbasic.norm());
}

// The minimum-norm solution is the basic solution projected off the null
// space, and the basic solution can be arbitrarily larger than the answer:
// here 1e12 times (two columns 1e-12 apart), and 1e6 times on ordinary random
// wide systems whose pivot columns happen to be nearly dependent. The
// projection alone would leave eps |x_basic|; the accuracy must be
// eps kappa(A) |x_minnorm| regardless.
void testMinimumNormAccuracy() {
  for (double delta : {1e-9, 1e-12}) {
    Eigen::MatrixXd Ad(2, 3);
    Ad << 1, 1, 0, 1, 1 + delta, 1;  // rank 2; columns 1,2 nearly parallel, column 3 completes the range
    Eigen::SparseMatrix<double> A = Ad.sparseView();
    A.makeCompressed();
    Eigen::VectorXd b(2);
    b << 1, 2;
    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(Ad);
    const Eigen::VectorXd ref = svd.solve(b);
    Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
    qr.setEngine(g_engine);
    qr.setOrdering(Eigen::multifrontal_qr::Ordering::Natural);
    qr.setScaling(Eigen::multifrontal_qr::Scaling::None);
    qr.compute(A);
    const Eigen::VectorXd x = qr.solve(b);
    const std::string tag = "min-norm accuracy 2x3, delta " + std::to_string(delta);
    check(qr.rank() == 2 && qr.rankIsVerified(), tag + ": full row rank", double(qr.rank()));
    check(relErr<double>(x, ref) < 1e-12, tag + ": equals pinv(A) b", relErr<double>(x, ref));
    check((Ad * x - b).norm() < 1e-13, tag + ": consistent residual", (Ad * x - b).norm());
  }
  // Ordinary random wide systems: the error follows kappa(R11), not kappa(A).
  double worstErr = 0.0, worstKappaA = 0.0;
  for (unsigned seed = 202; seed < 212; ++seed) {
    const auto B = randomSparse<double>(30, 50, 0.1, seed);
    const Eigen::MatrixXd Bd(B);
    const Eigen::VectorXd bb = B * Eigen::VectorXd::LinSpaced(50, 1.0, 2.0);
    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(Bd);
    worstKappaA = std::max(worstKappaA, svd.singularValues()[0] / svd.singularValues()[29]);
    Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
    qr.setEngine(g_engine);
    qr.compute(B);
    worstErr = std::max(worstErr, relErr<double>(qr.solve(bb), svd.solve(bb)));
  }
  lu_testing::note("wide 30x50: worst kappa(A) " + std::to_string(worstKappaA));
  check(worstErr < 1e-13 * worstKappaA, "min-norm accuracy, wide 30x50 (10 seeds): error ~ eps kappa(A)", worstErr);
}

}  // namespace

void testAutoEngine() {
  using Eigen::multifrontal_qr::Engine;
  const auto A = lu_testing::laplacian2d(30, 30);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setScalarEngineThreshold(1000000);
  qr.compute(A);
  checkTrue(qr.engineUsed() == Engine::Scalar, "auto engine: R below the threshold -> scalar");
  qr.setScalarEngineThreshold(1000);
  qr.setScalarEngineDensity(0.0, 0);
  qr.compute(A);
  checkTrue(qr.engineUsed() == Engine::Multifrontal, "auto engine: R above the threshold -> multifrontal");
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), -1.0, 1.0);
  Eigen::VectorXd xm = qr.solve(b);
  qr.setEngine(Engine::Scalar);
  qr.compute(A);
  Eigen::VectorXd xs = qr.solve(b);
  check(relErr<double>(xs, xm) < 1e-13, "engines agree", relErr<double>(xs, xm));
}

// Engine::Auto starts on the scalar engine and, past setScalarDeferralLimit()
// dependent columns, hands the matrix to the multifrontal one mid-factorize.
void testAutoEngineDeferralSwitch() {
  using Eigen::multifrontal_qr::Engine;
  const int n = 400;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 2.0 + 0.01 * i);
    if (i > 0) t.emplace_back(i - 1, i, -1.0);
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::MatrixXd Ad(A);
  for (int i = 2; i < n; i += 3) Ad.col(i) = 2.0 * Ad.col(i - 1);  // 133 dependent columns
  A = Ad.sparseView();
  A.makeCompressed();
  const Eigen::VectorXd b = A * Eigen::VectorXd::LinSpaced(n, 1.0, 2.0);
  const Eigen::VectorXd ref = pinvSolve<double>(Ad, b, 1e-12);
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr;
  qr.setScalarEngineThreshold(1000000);  // the scalar engine would be chosen on size alone
  qr.compute(A);
  checkTrue(qr.engineUsed() == Engine::Multifrontal, "auto engine: >64 deferrals hand over to multifrontal");
  check(qr.rank() == numericalRank<double>(Ad, 1e-12) && qr.rankIsVerified(), "auto engine switch: verified rank", double(qr.rank()));
  check(relErr<double>(qr.solve(b), ref) < 1e-9, "auto engine switch: equals pinv(A) b", relErr<double>(qr.solve(b), ref));
  check(factorIdentity(qr, A) < 1e-11, "auto engine switch: R^H R == (APD)^H APD", factorIdentity(qr, A));
  Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qs;
  qs.setEngine(Engine::Scalar);  // forced: every dependent column goes through the deferred block
  qs.compute(A);
  check(qs.rank() == qr.rank() && qs.deferredNullity() == n - qs.rank(), "forced scalar: same rank, all in the deferred block", double(qs.rank()));
  check(relErr<double>(qs.solve(b), ref) < 1e-9, "forced scalar: equals pinv(A) b", relErr<double>(qs.solve(b), ref));
}

int main() {
  using Eigen::multifrontal_qr::Engine;
  for (Engine e : {Engine::Multifrontal, Engine::Scalar}) {
    g_engine = e;
    const char* tag = e == Engine::Multifrontal ? "multifrontal" : "scalar";
    std::printf("=== engine: %s ===\n", tag);
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
    testSquareInconsistentLeastSquares();
    testGraphMatrices();
    std::printf("--- accuracy / parallel / edge ---\n");
    testIllConditioned();
    testParallelIdentical();
    testEdgeCases();
    std::printf("--- rank deficiency, every shape ---\n");
    testDeficientShapes<double>("real");
    testDeficientShapes<std::complex<double>>("complex");
    testComplexHiddenRankDeficiency();
    testEmbeddedHiddenDeficiency<double>("embedded Kahan");
    testEmbeddedHiddenDeficiency<std::complex<double>>("complex embedded Kahan");
    testNeumannLaplacian();
    testBlockSizes();
    testLargeRankTolerance();
    std::printf("--- solves, reuse, options ---\n");
    testMultipleRightHandSides();
    testRefactorizeAfterDeferral();
    testExplicitZerosAndUncompressed();
    testNonFiniteInput();
    testUncompressedInput();
    testReuseAcrossShapes();
    testEmptyShapes();
    testParallelIdenticalWithDeferral();
    testLargeFrontParallel();
    testOptionsChangedAfterCompute();
    testMinimumNormAccuracy();
  }
  std::printf("=== engine selection ===\n");
  testAutoEngine();
  testAutoEngineDeferralSwitch();
  return lu_testing::summarize("MultifrontalQR");
}
