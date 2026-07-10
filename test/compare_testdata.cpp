// Compare SupernodalLU against Eigen::SparseLU (and, optionally, Intel MKL
// PARDISO) on the real-world MatrixMarket matrices in testdata/.
//
// The METIS (HAVE_METIS) and PARDISO (HAVE_PARDISO) comparisons are each behind
// an independent compile guard, so any subset can be built. HAVE_METIS also
// enables the SupernodalLU+Auto column (SupernodalLUAutoOrdering.h): AMD vs.
// several METIS restarts, keeping whichever predicts the least fill.
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
//   ./build/compare_testdata                 # runs the built-in matrix list
//   ./build/compare_testdata path/to/A.mtx   # or an explicit list
//
// For each matrix we build b = A*xTrue with a known xTrue, then solve A x = b
// with each solver and print one row of a precision/time table: relative error
// ||x-xTrue||/||xTrue||, relative residual ||Ax-b||/||b||, and factor+solve time.
// Rows stream out as they finish. SparseLU/PARDISO get the original A; only
// SupernodalLU gets the pattern-symmetrized Asym (numerically the same system).
// SupernodalLU is skipped (no row pivoting -> ruinous fill) above SNLU_MAX_N
// rows (default 100000; env override, 0 = never skip), e.g. on the 659k pre2.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#ifdef HAVE_PARDISO
#include <Eigen/PardisoSupport>
#endif

#ifdef HAVE_METIS
#include "SupernodalLUMetis.h"        // SupernodalLUMetis = SupernodalLU + MetisOrdering
#include "SupernodalLUAutoOrdering.h"  // SupernodalLUAuto = SupernodalLU + AutoOrdering (AMD vs METIS restarts)
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SupernodalLU.h"
#include "LeftRightLU.h"  // PARDISO-style sibling solver (left-right-looking + complete pivoting)

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
  bool skipped = false;
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

// LeftRightLU (PARDISO-style): reuses the same result-collection body -- it
// exposes the same nnzL/nnzU/replacedPivots/supernodeCount/iterativeRefinements
// surface -- and, like SupernodalLU, factors a symmetric pattern (so it also
// gets Asym). Default settings: complete pivoting + dynamic scheduler (serial
// executor here, matching the single-threaded SupernodalLU column).
Result runLeftRight(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue,
                    bool amalgamate) {
  return runSupernodalWith<Eigen::LeftRightLU<SparseMatrix<double>>>(A, b, xTrue, amalgamate);
}

#ifdef HAVE_METIS
// Same solver, but with the METIS nested-dissection ordering wired in. Uses the
// default (amalgamated) supernode settings, like the AMD "amalgamated" run, so
// the two are directly comparable and the difference is purely the ordering.
Result runSupernodalMetis(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  return runSupernodalWith<Eigen::SupernodalLUMetis<SparseMatrix<double>>>(A, b, xTrue,
                                                                           /*amalgamate=*/true);
}

// Same solver again, but with AutoOrdering: tries AMD plus several METIS
// restarts and keeps whichever predicts the least fill (see
// SupernodalLUAutoOrdering.h). Should never be worse than the plain-AMD
// column by more than noise, and can beat both AMD and a single METIS call
// when nested dissection helps but a lucky/unlucky seed would have mattered.
Result runSupernodalAuto(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  return runSupernodalWith<Eigen::SupernodalLUAuto<SparseMatrix<double>>>(A, b, xTrue,
                                                                          /*amalgamate=*/true);
}
#endif

Result runSparseLU(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  Result r;
  try {
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
  } catch (const std::exception& e) {
    r.note = std::string("threw: ") + e.what();
  }
  return r;
}

#ifdef HAVE_PARDISO
// Intel MKL PARDISO via Eigen's drop-in interface (Eigen/PardisoSupport).
// PardisoLU factorizes a general real unsymmetric matrix with PARDISO's own
// scaling + partial pivoting (mtype 11). The wrapper does not expose factor
// nnz, so nnzL/nnzU are left at 0.
Result runPardiso(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& xTrue) {
  Result r;
  try {
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
  } catch (const std::exception& e) {
    r.note = std::string("threw: ") + e.what();
  }
  return r;
}
#endif

// Short label for a matrix: its parent directory name (e.g. "sherman1" from
// "testdata/sherman1/sherman1.mtx", which also handles dirs whose .mtx file is
// named differently, like setfos/spmatrix.mtx).
std::string matrixLabel(const std::string& path) {
  size_t slash = path.find_last_of("/\\");
  std::string dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash);
  size_t slash2 = dir.find_last_of("/\\");
  std::string parent = (slash2 == std::string::npos) ? dir : dir.substr(slash2 + 1);
  if (!parent.empty()) return parent;
  std::string file = (slash == std::string::npos) ? path : path.substr(slash + 1);
  size_t dot = file.find_last_of('.');
  return dot == std::string::npos ? file : file.substr(0, dot);
}

// One solver's cell in the table: relative error, relative residual, and
// total (factor + solve) time in ms. 29 chars wide, matching the header.
void printCell(const Result& r) {
  if (r.skipped)
    std::printf(" %28s", "skipped (n too large)");
  else if (r.ok && std::isfinite(r.err) && std::isfinite(r.resid))
    std::printf(" %9.2e %9.2e %8.1f", r.err, r.resid, r.factorMs + r.solveMs);
  else
    std::printf(" %28s", r.ok ? "non-finite" : "FAILED");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  const std::vector<std::string> matrices = {
      "testdata/bcsstm13/bcsstm13.mtx",
      "testdata/dendrimer/dendrimer.mtx",
      "testdata/gemat11/gemat11.mtx",
      "testdata/rdb2048_noL/rdb2048_noL.mtx",
      "testdata/setfos/spmatrix.mtx",
      "testdata/sherman1/sherman1.mtx",
      "testdata/tomography/tomography.mtx",
      "testdata/YaleB_10NN/YaleB_10NN.mtx",
      "testdata/bayer05/bayer05.mtx",
      // Large, hard circuit matrix (n=659k). Kept last: SupernodalLU's
      // symmetric-pattern static-pivot factorization may run very long or
      // exhaust memory here; the other solvers are still listed.
      "testdata/pre2/pre2.mtx",
  };
  // Allow overriding the list on the command line.
  std::vector<std::string> files(matrices);
  if (argc > 1) files.assign(argv + 1, argv + argc);

  std::printf("Precision / time comparison: SupernodalLU vs LeftRightLU vs Eigen::SparseLU"
#ifdef HAVE_METIS
              " (+SupernodalLU METIS/Auto)"
#endif
#ifdef HAVE_PARDISO
              " vs MKL PARDISO"
#endif
              "\n");
  std::printf("b = A*xTrue with random xTrue; pattern symmetrized for SupernodalLU.\n");
  std::printf("Per solver: err = ||x-xTrue||/||xTrue||, resid = ||Ax-b||/||b||, "
              "ms = factor+solve time.\n\n");

  // Build the header. Each solver block is 29 chars: " err       resid     ms".
  auto solverHead = [](const char* name) { std::printf(" |%-28s", name); };
  auto solverSub = []() { std::printf(" | %9s %9s %8s", "err", "resid", "ms"); };

  std::printf("%-12s %8s %10s", "matrix", "n", "nnz");
  solverHead(" SupernodalLU");
#ifdef HAVE_METIS
  solverHead(" SupernodalLU+METIS");
  solverHead(" SupernodalLU+Auto");
#endif
  solverHead(" LeftRightLU");
  solverHead(" Eigen SparseLU");
#ifdef HAVE_PARDISO
  solverHead(" MKL PARDISO");
#endif
  std::printf("\n");
  std::printf("%-12s %8s %10s", "", "", "");
  solverSub();
#ifdef HAVE_METIS
  solverSub();
  solverSub();
#endif
  solverSub();
  solverSub();
#ifdef HAVE_PARDISO
  solverSub();
#endif
  std::printf("\n");

  // SupernodalLU factors a STATIC symmetric pattern with no row pivoting; on a
  // very large, ill-structured matrix (e.g. the 659k circuit matrix pre2) the
  // resulting fill is enormous and the factorization exhausts memory. Skip it
  // past this size so the table still reports the other solvers. Override with
  // the SNLU_MAX_N environment variable (0 = never skip).
  long long snluMaxN = 100000;
  if (const char* env = std::getenv("SNLU_MAX_N")) snluMaxN = std::atoll(env);

  int failures = 0;
  for (const std::string& path : files) {
    SparseMatrix<double> A;
    try {
      A = loadMatrixMarket(path);
    } catch (const std::exception& e) {
      std::printf("%-12s  load error: %s\n", matrixLabel(path).c_str(), e.what());
      ++failures;
      continue;
    }

    // Adding explicit structural zeros (symmetrizePattern) does not change the
    // operator, so b = A*xTrue works for every solver. SparseLU and PARDISO get
    // the original (lighter, possibly unsymmetric) A; only SupernodalLU needs a
    // symmetric pattern, and we build that Asym lazily -- not at all when SNLU
    // is skipped, which avoids a large needless allocation on e.g. pre2.
    VectorXd xTrue = VectorXd::Random(A.rows());
    VectorXd b = A * xTrue;

    // Print the row label first and flush, so progress is visible even while a
    // slow solver (e.g. on the large pre2 matrix) is still running.
    std::printf("%-12s %8lld %10lld", matrixLabel(path).c_str(),
                (long long)A.rows(), (long long)A.nonZeros());

    const bool runSnlu = !(snluMaxN > 0 && (long long)A.rows() > snluMaxN);
    SparseMatrix<double> Asym;  // built only if a SupernodalLU variant runs
    if (runSnlu) Asym = patternIsSymmetric(A) ? A : symmetrizePattern(A);

    Result snlu;
    if (!runSnlu) snlu.skipped = true;
    else snlu = runSupernodal(Asym, b, xTrue, /*amalgamate=*/true);
    std::printf(" |");
    printCell(snlu);
#ifdef HAVE_METIS
    Result metis;
    if (!runSnlu) metis.skipped = true;
    else metis = runSupernodalMetis(Asym, b, xTrue);
    std::printf(" |");
    printCell(metis);
    Result autoOrd;
    if (!runSnlu) autoOrd.skipped = true;
    else autoOrd = runSupernodalAuto(Asym, b, xTrue);
    std::printf(" |");
    printCell(autoOrd);
#endif
    Result lrlu;
    if (!runSnlu) lrlu.skipped = true;
    else lrlu = runLeftRight(Asym, b, xTrue, /*amalgamate=*/true);
    std::printf(" |");
    printCell(lrlu);
    Result ref = runSparseLU(A, b, xTrue);
    std::printf(" |");
    printCell(ref);
#ifdef HAVE_PARDISO
    Result pard = runPardiso(A, b, xTrue);
    std::printf(" |");
    printCell(pard);
#endif
    std::printf("\n");

    auto reachedTolerance = [](const Result& r) {
      return r.skipped || (r.ok && std::isfinite(r.resid) && r.resid < 1e-6);
    };
    if (!reachedTolerance(snlu) || !reachedTolerance(lrlu)) ++failures;
  }

  std::printf("\n%d matrix(es) where SupernodalLU or LeftRightLU did not reach resid < 1e-6.\n",
              failures);
  return failures == 0 ? 0 : 1;
}
