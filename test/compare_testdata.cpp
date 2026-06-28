// Compare SupernodalLU against Eigen::SparseLU (and, optionally, Intel MKL
// PARDISO) on the real-world MatrixMarket matrices in testdata/.
//
// The METIS (HAVE_METIS) and PARDISO (HAVE_PARDISO) comparisons are each behind
// an independent compile guard, so any subset can be built.
//
// Plain build (SupernodalLU vs Eigen::SparseLU only, from repo root):
//   clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src \
//       DirectLUSolvers/test/compare_testdata.cpp -o build/compare_testdata
//
// PARDISO only (Eigen/PardisoSupport, statically linked MKL sequential LP64 -
//   no MKL runtime DLLs needed; MKL_INT==int matches StorageIndex):
//   clang++ -std=c++17 -O2 -DHAVE_PARDISO -I eigen -I DirectLUSolvers/src \
//       -I mkl/include DirectLUSolvers/test/compare_testdata.cpp \
//       -L mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
//       -o build/compare_testdata
//
// With METIS, and the full combination, use clang-cl: the installed GKlib/metis
// were built against the dynamic CRT (/MD), so everything must agree on /MD,
// which means MKL must be linked dynamically too (mkl_rt) when combined. clang++
// in its GNU driver does not inject the dynamic UCRT import libs cleanly, so
// clang-cl is the reliable Windows linker here.
//
//   clang-cl /std:c++17 /O2 /EHsc /MD /DHAVE_METIS /DHAVE_PARDISO \
//       /I eigen /I DirectLUSolvers/src /I mkl/include /I install/include \
//       DirectLUSolvers/test/compare_testdata.cpp \
//       /Fe:build/compare_testdata.exe \
//       /link /LIBPATH:mkl/lib mkl_rt.lib /LIBPATH:install/lib metis.lib GKlib.lib
//
// (Drop /DHAVE_PARDISO and the mkl bits for a self-contained METIS-only build.)
//
// Run (from repo root, so relative testdata/ paths resolve). When PARDISO was
// linked via mkl_rt, put mkl/bin on PATH and pick a threading layer that does
// not need libiomp5md.dll:
//   set MKL_THREADING_LAYER=SEQUENTIAL && set PATH=mkl\bin;%PATH%
//   ./build/compare_testdata
//
// For each matrix we build b = A*xTrue with a known xTrue, then solve A x = b
// with each solver and report relative error, residual, factor sizes and time.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#ifdef HAVE_PARDISO
#include <Eigen/PardisoSupport>
#endif

#ifdef HAVE_METIS
#include "SupernodalLUMetis.h"  // SupernodalLUMetis = SupernodalLU + MetisOrdering
#endif

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
  long long snodes = 0;
  int refineIters = 0;
  std::string note;
};

// Runs a SupernodalLU instance (any ordering) with the robust defaults (auto
// static pivot + iterative refinement). When amalgamate==false we force the
// pure fundamental-supernode partition (setAmalgamation(1,0)) so the effect of
// amalgamation is isolated. Templated on the solver type so the same body
// serves both the default AMD ordering and the METIS ordering.
template <typename Solver>
Result runSupernodalWith(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue,
                         bool amalgamate) {
  Result r;
  Solver solver;
  try {
    if (!amalgamate) solver.setAmalgamation(1, 0);
    auto t0 = Clock::now();
    solver.compute(A);
    auto t1 = Clock::now();
    if (solver.info() != Eigen::Success) {
      r.note = "factorize: " + solver.lastErrorMessage();
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
    r.replaced = solver.replacedPivots();
    r.refineIters = static_cast<int>(solver.iterativeRefinements());
    r.snodes = static_cast<long long>(solver.supernodeCount());
    r.ok = true;
  } catch (const std::exception& e) {
    r.note = std::string("threw: ") + e.what();
  }
  return r;
}

Result runSupernodal(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue,
                     bool amalgamate) {
  return runSupernodalWith<Eigen::SupernodalLU<SparseMatrix<double>>>(A, b, xTrue, amalgamate);
}

#ifdef HAVE_METIS
// Same solver, but with the METIS nested-dissection ordering wired in. Uses the
// default (amalgamated) supernode settings, like the AMD "amalgamated" run, so
// the two are directly comparable and the difference is purely the ordering.
Result runSupernodalMetis(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  return runSupernodalWith<Eigen::SupernodalLUMetis<SparseMatrix<double>>>(A, b, xTrue,
                                                                           /*amalgamate=*/true);
}
#endif

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

#ifdef HAVE_PARDISO
// Intel MKL PARDISO via Eigen's drop-in interface (Eigen/PardisoSupport).
// PardisoLU factorizes a general real unsymmetric matrix with PARDISO's own
// scaling + partial pivoting (mtype 11). The wrapper does not expose factor
// nnz, so nnzL/nnzU are left at 0.
Result runPardiso(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  Result r;
  Eigen::PardisoLU<SparseMatrix<double>> solver;
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
  r.ok = true;
  return r;
}
#endif

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

  std::printf("Comparing SupernodalLU vs Eigen::SparseLU"
#ifdef HAVE_METIS
              " (AMD & METIS orderings)"
#endif
#ifdef HAVE_PARDISO
              " vs MKL PARDISO"
#endif
              " on testdata matrices\n");
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

    Result plain = runSupernodal(Asym, b, xTrue, /*amalgamate=*/false);
    Result piv = runSupernodal(Asym, b, xTrue, /*amalgamate=*/true);
#ifdef HAVE_METIS
    Result metis = runSupernodalMetis(Asym, b, xTrue);
#endif
    Result ref = runSparseLU(Asym, b, xTrue);

    const long long nn = A.rows();
    auto report = [nn](const char* name, const Result& r) {
      if (r.ok) {
        std::printf("  %-22s  err=%.3e  resid=%.3e  factor=%8.2f ms  solve=%7.2f ms  nnzL=%lld nnzU=%lld",
                    name, r.err, r.resid, r.factorMs, r.solveMs, r.nnzL, r.nnzU);
        if (r.snodes > 0)
          std::printf("  snodes=%lld avgW=%.1f", r.snodes, double(nn) / double(r.snodes));
        if (r.replaced > 0 || r.refineIters > 0)
          std::printf("  [bumped %lld pivots, %d refine its]", r.replaced, r.refineIters);
        std::printf("\n");
      } else {
        std::printf("  %-22s  DID NOT SOLVE: %s\n", name, r.note.c_str());
      }
    };
    report("SNLU (fundamental)", plain);
    report("SNLU AMD (amalg)", piv);
#ifdef HAVE_METIS
    report("SNLU METIS (amalg)", metis);
#endif
    report("Eigen SparseLU", ref);
#ifdef HAVE_PARDISO
    Result pard = runPardiso(Asym, b, xTrue);
    report("MKL PARDISO", pard);
#endif

    const bool pivAccurate = piv.ok && piv.resid < 1e-6;
    if (plain.ok && piv.ok) {
      const double fillRatio = double(piv.nnzL + piv.nnzU) / double(plain.nnzL + plain.nnzU);
      const double speedup = piv.factorMs > 0 ? plain.factorMs / piv.factorMs : 0.0;
      std::printf("  -> amalgamation: %.2fx factor speed, %lld->%lld supernodes, %+.0f%% stored nz\n",
                  speedup, plain.snodes, piv.snodes, (fillRatio - 1.0) * 100.0);
    }
#ifdef HAVE_METIS
    if (piv.ok && metis.ok) {
      const double fillRatio = double(metis.nnzL + metis.nnzU) / double(piv.nnzL + piv.nnzU);
      const double speedup = metis.factorMs > 0 ? piv.factorMs / metis.factorMs : 0.0;
      std::printf("  -> METIS vs AMD ordering: %+.0f%% stored nz, %.2fx factor speed\n",
                  (fillRatio - 1.0) * 100.0, speedup);
    }
#endif
    if (piv.ok && !pivAccurate)
      std::printf("  -> residual stayed large (%lld/%ld pivots bumped) -> needs true row pivoting.\n",
                  piv.replaced, (long)A.rows());
    if (!pivAccurate) ++failures;
    std::printf("\n");
  }

  std::printf("Done. %d matrix(es) where SupernodalLU did not produce a solution.\n", failures);
  return failures == 0 ? 0 : 1;
}
