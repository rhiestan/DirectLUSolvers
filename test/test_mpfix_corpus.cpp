// Real matrices with mpfix scalars: a few small SuiteSparse matrices from the
// checked-in corpus, solved by every solver in binary128, binary256 and
// binary512. test_mpfix covers the solvers' features on generated matrices;
// this one checks that they solve matrices nobody constructed for them, at the
// accuracy of the type and not of double.
//
// The criterion is the normwise backward error
//     omega = ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf),
// which a backward-stable solve brings to a modest multiple of the type's
// epsilon whatever the matrix's conditioning. An answer that went through
// double anywhere would stop near 1e-16, far above every bound here.
//
// Needs mpfix (header-only, C++20) and the unpacked corpus; built only with
// -DDLU_WITH_MPFIX=ON.

#include <MpfixSupport>

#include <LeftRightLU>
#include <MultifrontalQR>
#include <PointBlockLU>
#include <RobustLU>
#include <SupernodalLDLT>
#include <SupernodalLU>

#include <cstdio>
#include <string>
#include <vector>

#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::note;

namespace {

template <typename T>
const char* typeName();
template <>
const char* typeName<mpfix::binary128>() { return "binary128"; }
template <>
const char* typeName<mpfix::binary256>() { return "binary256"; }
template <>
const char* typeName<mpfix::binary512>() { return "binary512"; }

template <typename T>
double backwardError(const Eigen::SparseMatrix<T>& A, const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  // ||A||_inf: the largest absolute row sum.
  Eigen::Matrix<T, Eigen::Dynamic, 1> rowSums = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(A.rows());
  for (Eigen::Index j = 0; j < A.outerSize(); ++j)
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, j); it; ++it) rowSums[it.row()] += abs(it.value());
  const T normA = rowSums.maxCoeff();
  const T r = (b - A * x).template lpNorm<Eigen::Infinity>();
  const T denom = normA * x.template lpNorm<Eigen::Infinity>() + b.template lpNorm<Eigen::Infinity>();
  return double(r / denom);
}

template <typename T, typename Solver>
void solveWith(const char* solverName, const std::string& label, const Eigen::SparseMatrix<T>& A,
               const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  const std::string tag = std::string(typeName<T>()) + " " + label + " " + solverName;
  Solver s;
  s.compute(A);
  if (!checkTrue(s.info() == Eigen::Success, tag + ": factorized")) return;
  const Eigen::Matrix<T, Eigen::Dynamic, 1> x = s.solve(b);
  const double omega = backwardError(A, x, b);
  // 1e3 eps leaves room for the growth in a pivoted elimination and the size
  // of these matrices (n up to 2000); a solve that dropped to double anywhere
  // misses it by 15 orders of magnitude or more.
  const double eps = double(Eigen::NumTraits<T>::epsilon());
  check(x.allFinite() && omega <= 1e3 * eps, tag + ": backward error / eps", omega / eps);
}

template <typename T>
void solveMatrix(const lu_testing::SuiteSparseMatrix& m, bool symmetric) {
  using SpMat = Eigen::SparseMatrix<T>;
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  const SpMat A = lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(m.path)).template cast<T>();
  // A right-hand side with an exactly known, non-dyadic solution.
  Vec xTrue(A.cols());
  for (Eigen::Index i = 0; i < A.cols(); ++i) xTrue[i] = T(int(i % 7) + 1) / T(7);
  const Vec b = A * xTrue;

  solveWith<T, Eigen::LeftRightLU<SpMat>>("LeftRightLU", m.label(), A, b);
  solveWith<T, Eigen::SupernodalLU<SpMat>>("SupernodalLU", m.label(), A, b);
  solveWith<T, Eigen::PointBlockLU<SpMat>>("PointBlockLU", m.label(), A, b);
  solveWith<T, Eigen::MultifrontalQR<SpMat>>("MultifrontalQR", m.label(), A, b);
  solveWith<T, Eigen::RobustLU<SpMat>>("RobustLU", m.label(), A, b);
  if (symmetric) solveWith<T, Eigen::SupernodalLDLT<SpMat>>("SupernodalLDLT", m.label(), A, b);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("mpfix on SuiteSparse matrices\n");
  // Small enough for binary512, and different in kind: an unsymmetric
  // chemical-process matrix, an unsymmetric-pattern CFD matrix (symmetrized
  // here, as the supernodal solvers need), and a symmetric structural one.
  struct Pick {
    const char* label;
    bool symmetric;
  };
  const Pick picks[] = {{"Bai/bwm2000", false}, {"Bai/tols1090", false}, {"Boeing/nasa1824", true}};

  const std::vector<lu_testing::SuiteSparseMatrix> all = lu_testing::suitesparseMatrices();
  int found = 0;
  for (const Pick& p : picks) {
    const lu_testing::SuiteSparseMatrix* m = nullptr;
    for (const auto& c : all)
      if (c.label() == p.label && c.available) m = &c;
    if (!m) {
      note(std::string(p.label) + " not unpacked -- skipped (reconfigure with DLU_UNPACK_SUITESPARSE=ON)");
      continue;
    }
    ++found;
    std::printf("  %s (n = %lld)\n", p.label, m->n);
    solveMatrix<mpfix::binary128>(*m, p.symmetric);
    solveMatrix<mpfix::binary256>(*m, p.symmetric);
    solveMatrix<mpfix::binary512>(*m, p.symmetric);
  }
  checkTrue(found > 0, "at least one corpus matrix was available");
  return lu_testing::summarize("mpfix corpus");
}
