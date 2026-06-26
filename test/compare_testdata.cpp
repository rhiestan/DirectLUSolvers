// Compare SupernodalLU against Eigen::SparseLU on the real-world MatrixMarket
// matrices in testdata/.
//
// Build (from repo root):
//   clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src \
//       DirectLUSolvers/test/compare_testdata.cpp -o build/compare_testdata
//
// Run (from repo root, so relative testdata/ paths resolve):
//   ./build/compare_testdata
//
// For each matrix we build b = A*xTrue with a known xTrue, then solve A x = b
// with both solvers and report relative error, residual, factor sizes and time.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SupernodalLU.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::Triplet;
using Clock = std::chrono::steady_clock;

namespace {

// Minimal MatrixMarket coordinate reader for real (or pattern) matrices.
// Mirrors the lower triangle when the banner says "symmetric".
SparseMatrix<double> loadMatrixMarket(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty file " + path);

  // Banner: %%MatrixMarket matrix coordinate <field> <symmetry>
  bool symmetric = false;
  bool patternOnly = false;
  {
    std::string lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("symmetric") != std::string::npos) symmetric = true;
    if (lower.find("pattern") != std::string::npos) patternOnly = true;
  }

  // Skip comment lines, then read the size line.
  do {
    if (!std::getline(in, line)) throw std::runtime_error("unexpected EOF in " + path);
  } while (!line.empty() && line[0] == '%');

  int rows = 0, cols = 0;
  long long nnz = 0;
  {
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;
  }

  std::vector<Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(symmetric ? 2 * nnz : nnz));

  for (long long k = 0; k < nnz; ++k) {
    if (!std::getline(in, line)) throw std::runtime_error("unexpected EOF reading entries in " + path);
    if (line.empty()) { --k; continue; }
    std::istringstream iss(line);
    int i = 0, j = 0;
    double v = 1.0;
    iss >> i >> j;
    if (!patternOnly) iss >> v;
    --i; --j;  // MatrixMarket is 1-based
    triplets.emplace_back(i, j, v);
    if (symmetric && i != j) triplets.emplace_back(j, i, v);
  }

  SparseMatrix<double> A(rows, cols);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

// Return a copy of A whose sparsity pattern is symmetric, by inserting explicit
// zeros at (j,i) wherever (i,j) is present but (j,i) is not. This does not change
// the linear system; it only gives SupernodalLU the symmetric pattern it needs.
SparseMatrix<double> symmetrizePattern(const SparseMatrix<double>& A) {
  std::vector<Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(A.nonZeros()) * 2);
  for (int col = 0; col < A.outerSize(); ++col)
    for (SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
      triplets.emplace_back(it.col(), it.row(), 0.0);  // structural mirror
    }
  SparseMatrix<double> S(A.rows(), A.cols());
  S.setFromTriplets(triplets.begin(), triplets.end());  // sums dup -> keeps real value
  S.makeCompressed();
  return S;
}

bool patternIsSymmetric(const SparseMatrix<double>& A) {
  SparseMatrix<double> AT = A.transpose();
  AT.makeCompressed();
  if (AT.nonZeros() != A.nonZeros()) return false;
  // Compare inner-index sets column by column.
  for (int col = 0; col < A.outerSize(); ++col) {
    if (A.outerIndexPtr()[col] != AT.outerIndexPtr()[col]) return false;
  }
  for (int k = 0; k < A.nonZeros(); ++k)
    if (A.innerIndexPtr()[k] != AT.innerIndexPtr()[k]) return false;
  return true;
}

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Result {
  bool ok = false;
  double err = 0, resid = 0, factorMs = 0, solveMs = 0;
  long long nnzL = 0, nnzU = 0;
  long long replaced = 0;
  int refineIters = 0;
  std::string note;
};

// threshold <= 0 disables static pivoting. refineSteps > 0 enables iterative
// refinement (re-using the stored factorization), stopping early once the
// relative residual is at machine-precision level.
Result runSupernodal(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue,
                     double threshold, int refineSteps) {
  Result r;
  Eigen::SupernodalLU<SparseMatrix<double>> solver;
  try {
    if (threshold > 0.0) solver.setStaticPivotThreshold(threshold);
    auto t0 = Clock::now();
    solver.compute(A);
    auto t1 = Clock::now();
    if (solver.info() != Eigen::Success) {
      r.note = "factorize: " + solver.lastErrorMessage();
      return r;
    }
    VectorXd x = solver.solve(b);
    const double bn = b.norm();
    for (int it = 0; it < refineSteps; ++it) {
      VectorXd resid = b - A * x;
      if (resid.norm() / bn < 1e-14) break;
      x += solver.solve(resid);
      ++r.refineIters;
    }
    auto t2 = Clock::now();
    r.factorMs = ms(t0, t1);
    r.solveMs = ms(t1, t2);
    r.err = (x - xTrue).norm() / xTrue.norm();
    r.resid = (A * x - b).norm() / bn;
    r.nnzL = solver.nnzL();
    r.nnzU = solver.nnzU();
    r.replaced = solver.replacedPivots();
    r.ok = true;
  } catch (const std::exception& e) {
    r.note = std::string("threw: ") + e.what();
  }
  return r;
}

Result runSparseLU(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  Result r;
  Eigen::SparseLU<SparseMatrix<double>> solver;
  auto t0 = Clock::now();
  solver.compute(A);
  auto t1 = Clock::now();
  if (solver.info() != Eigen::Success) {
    r.note = "factorize failed";
    return r;
  }
  VectorXd x = solver.solve(b);
  auto t2 = Clock::now();
  r.factorMs = ms(t0, t1);
  r.solveMs = ms(t1, t2);
  r.err = (x - xTrue).norm() / xTrue.norm();
  r.resid = (A * x - b).norm() / b.norm();
  r.nnzL = solver.nnzL();
  r.nnzU = solver.nnzU();
  r.ok = true;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  const std::vector<std::string> matrices = {
      "testdata/dendrimer/dendrimer.mtx",
      "testdata/tomography/tomography.mtx",
      "testdata/rdb2048_noL/rdb2048_noL.mtx",
      "testdata/YaleB_10NN/YaleB_10NN.mtx",
      "testdata/bayer05/bayer05.mtx",
  };
  // Allow overriding the list on the command line.
  std::vector<std::string> files(matrices);
  if (argc > 1) files.assign(argv + 1, argv + argc);

  std::printf("Comparing SupernodalLU vs Eigen::SparseLU on testdata matrices\n");
  std::printf("(b = A*xTrue with random xTrue; pattern symmetrized for SupernodalLU)\n\n");

  int failures = 0;
  for (const std::string& path : files) {
    std::printf("==================================================================\n");
    std::printf("%s\n", path.c_str());
    SparseMatrix<double> A;
    try {
      A = loadMatrixMarket(path);
    } catch (const std::exception& e) {
      std::printf("  load error: %s\n", e.what());
      ++failures;
      continue;
    }

    const bool sym = patternIsSymmetric(A);
    std::printf("  n = %ld, nnz = %ld, pattern %s\n", (long)A.rows(),
                (long)A.nonZeros(), sym ? "symmetric" : "UNsymmetric (will symmetrize)");

    SparseMatrix<double> Asym = sym ? A : symmetrizePattern(A);

    VectorXd xTrue = VectorXd::Random(A.rows());
    VectorXd b = Asym * xTrue;

    // Static-pivot threshold ~ sqrt(eps) * max|A_ij| (SuperLU_DIST style): big
    // enough to step over zero/tiny pivots, small enough to barely perturb A.
    const double scale = Asym.coeffs().cwiseAbs().maxCoeff();
    const double threshold = std::sqrt(std::numeric_limits<double>::epsilon()) * scale;

    Result plain = runSupernodal(Asym, b, xTrue, /*threshold=*/0.0, /*refine=*/0);
    Result piv = runSupernodal(Asym, b, xTrue, threshold, /*refine=*/30);
    Result ref = runSparseLU(Asym, b, xTrue);

    auto report = [](const char* name, const Result& r) {
      if (r.ok) {
        std::printf("  %-22s  err=%.3e  resid=%.3e  factor=%8.2f ms  solve=%7.2f ms  nnzL=%lld nnzU=%lld",
                    name, r.err, r.resid, r.factorMs, r.solveMs, r.nnzL, r.nnzU);
        if (r.replaced > 0 || r.refineIters > 0)
          std::printf("  [bumped %lld pivots, %d refine its]", r.replaced, r.refineIters);
        std::printf("\n");
      } else {
        std::printf("  %-22s  DID NOT SOLVE: %s\n", name, r.note.c_str());
      }
    };
    report("SupernodalLU (plain)", plain);
    report("SupernodalLU (pivot+refine)", piv);
    report("Eigen SparseLU", ref);

    const bool pivAccurate = piv.ok && piv.resid < 1e-6;
    if (!plain.ok && pivAccurate)
      std::printf("  -> static pivoting + refinement RECOVERED a solution unpivoted LU could not.\n");
    else if (piv.ok && !pivAccurate)
      std::printf("  -> static pivoting produced a solution but residual stayed large"
                  " (%lld/%ld pivots bumped) -> needs true row pivoting.\n",
                  piv.replaced, (long)A.rows());
    if (!pivAccurate) ++failures;
    std::printf("\n");
  }

  std::printf("Done. %d matrix(es) where SupernodalLU did not produce a solution.\n", failures);
  return failures == 0 ? 0 : 1;
}
