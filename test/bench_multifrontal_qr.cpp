// MultifrontalQR against Eigen::SparseQR, one matrix and one solver per run so a
// driver script can put a wall-clock limit on each (Eigen::SparseQR takes minutes
// on some of these matrices).
//
//   bench_multifrontal_qr <matrix.mtx> <mfqr|mfqr-noverify|eigen> [threads]
//
// Prints one CSV line:
//   matrix,solver,m,n,nnz,analyze_ms,factor_ms,solve_ms,nnzR,nnzH,rank,verified,
//   repairs,sigma_min,dropped,err,resid,optimality,ordering
//
// The right-hand side is b = A * xTrue. err is ||x - xTrue|| / ||xTrue||, which
// is only an error when the matrix has full column rank; for a rank-deficient
// matrix read resid (the system is consistent, so it should be small) and the
// rank itself.

#include <MultifrontalQR>

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <complex>
#include <cstdio>
#include <fstream>
#include <string>

#include "testing/Check.h"
#include "testing/MatrixMarket.h"

namespace {

using lu_testing::Clock;
using lu_testing::ms;

std::string baseName(const std::string& path) {
  std::string s = path.substr(path.find_last_of("/\\") + 1);
  if (s.size() > 4 && s.compare(s.size() - 4, 4, ".mtx") == 0) s.resize(s.size() - 4);
  return s;
}

template <typename Scalar>
int run(const std::string& path, const std::string& solver, int threads) {
  using SpMat = Eigen::SparseMatrix<Scalar>;
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  SpMat A = lu_testing::loadMatrixMarketAs<Scalar>(path);
  A.makeCompressed();
  const Eigen::Index m = A.rows(), n = A.cols();
  Vec xTrue(n);
  for (Eigen::Index i = 0; i < n; ++i) xTrue[i] = Scalar(1.0 + 0.5 * std::sin(double(i)));
  const Vec b = A * xTrue;

  double tA = 0, tF = 0, tS = 0, sigma = -1, dropped = -1, opt = -1;
  long long nnzR = 0, nnzH = -1, rank = 0, repairs = 0;
  int verified = -1, ordering = -1;
  Vec x;
  if (solver.rfind("mfqr", 0) == 0) {
    Eigen::MultifrontalQR<SpMat, Eigen::supernodal_lu::PooledExecutor> qr;
    qr.executor() = Eigen::supernodal_lu::PooledExecutor(threads);
    if (solver == "mfqr-noverify") qr.setRankVerification(false);
    auto t0 = Clock::now();
    qr.analyzePattern(A);
    auto t1 = Clock::now();
    qr.factorize(A);
    auto t2 = Clock::now();
    if (qr.info() != Eigen::Success) {
      std::printf("%s,%s,FAILED,%s\n", baseName(path).c_str(), solver.c_str(), qr.lastErrorMessage().c_str());
      return 1;
    }
    x = qr.solve(b);
    auto t3 = Clock::now();
    tA = ms(t0, t1), tF = ms(t1, t2), tS = ms(t2, t3);
    nnzR = qr.nnzR(), nnzH = qr.nnzH(), rank = qr.rank(), repairs = qr.repairIterations();
    verified = qr.rankIsVerified() ? 1 : 0;
    sigma = qr.smallestSingularValues().size() ? double(qr.smallestSingularValues()[0]) : -1.0;
    dropped = double(qr.droppedNorm());
    opt = double(qr.leastSquaresOptimality());
    ordering = int(qr.orderingUsed());
  } else {
    Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> qr;
    auto t0 = Clock::now();
    qr.analyzePattern(A);
    auto t1 = Clock::now();
    qr.factorize(A);
    auto t2 = Clock::now();
    if (qr.info() != Eigen::Success) {
      std::printf("%s,%s,FAILED\n", baseName(path).c_str(), solver.c_str());
      return 1;
    }
    x = qr.solve(b);
    auto t3 = Clock::now();
    tA = ms(t0, t1), tF = ms(t1, t2), tS = ms(t2, t3);
    nnzR = qr.matrixR().nonZeros(), rank = qr.rank();
  }
  const double err = double((x - xTrue).norm() / xTrue.norm());
  const double res = double((A * x - b).norm() / b.norm());
  std::printf("%s,%s,%lld,%lld,%lld,%.2f,%.2f,%.2f,%lld,%lld,%lld,%d,%lld,%.3e,%.3e,%.3e,%.3e,%.3e,%d\n",
              baseName(path).c_str(), solver.c_str(), (long long)m, (long long)n, (long long)A.nonZeros(), tA, tF, tS,
              nnzR, nnzH, rank, verified, repairs, sigma, dropped, err, res, opt, ordering);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <matrix.mtx> <mfqr|mfqr-noverify|eigen> [threads]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1], solver = argv[2];
  const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
  std::ifstream in(path);
  std::string banner;
  std::getline(in, banner);
  if (banner.find("complex") != std::string::npos) return run<std::complex<double>>(path, solver, threads);
  return run<double>(path, solver, threads);
}
