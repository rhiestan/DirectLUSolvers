// Benchmark: every solver in this library, plus Eigen's own direct and
// iterative solvers, against the ~5-categories x ~5-examples sample corpus in
// setfosmatrices_samples/ (see setfos_matrix_pipeline.py at the repo root).
//
// Each category is a real physics matrix family (drift-diffusion / optics FEM
// Jacobians) picked apart by pattern: real-symmetric-tridiagonal,
// real-unsymmetric-tridiagonal, real-unsymmetric-banded,
// real-unsymmetric-general-sparse, complex-unsymmetric-general-sparse.
//
// The provided _rhs.mtx files are NOT used here: to get a ground-truth error
// norm (not just a residual), this follows the same convention as
// bench_solvers.cpp / compare_testdata.cpp -- synthesize xTrue and b = A*xTrue.
//
// Build + run (from the DirectLUSolvers directory):
//   cmake --build build --target bench_setfos_samples
//   ./build/bench_setfos_samples
//
// USAGE
//   bench_setfos_samples                       # setfosmatrices_samples/ (default location)
//   bench_setfos_samples /path/to/samples       # explicit root
//   bench_setfos_samples --reps 5               # best-of-N timed repetitions (default 3)
//   bench_setfos_samples --csv path/to/out.csv  # where to write the raw rows (default analysis/benchmark_results.csv)

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "MultifrontalQR.h"
#include "PointBlockLU.h"
#include "RobustLU.h"
#include "SupernodalLDLT.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestMatrices.h"

namespace fs = std::filesystem;
using lu_testing::ms;
using Clock = lu_testing::Clock;

#ifndef DLU_SETFOS_SAMPLES_DIR
#define DLU_SETFOS_SAMPLES_DIR "../setfosmatrices_samples"
#endif

namespace {

std::string samplesDir(const std::string& argOverride) {
  if (!argOverride.empty()) return argOverride;
  if (const char* env = std::getenv("DLU_SETFOS_SAMPLES_DIR"))
    if (env[0] != '\0') return env;
  return DLU_SETFOS_SAMPLES_DIR;
}

struct Row {
  std::string category, matrix, solver, note;
  long long n = 0, nnz = 0;
  double analyzeMs = 0, factorMs = 0, solveMs = 0;
  double err = 0, resid = 0;
  long long fill = -1;
  bool ok = false;

  double total() const { return analyzeMs + factorMs + solveMs; }
};

std::vector<Row> g_rows;

// Run `body` reps+1 times (first is a discarded warm-up) and keep the MINIMUM
// of each timed phase -- see bench_solvers.cpp for why minimum, not mean.
template <typename Body>
Row measure(const std::string& cat, const std::string& mat, const std::string& solver, long long n,
            long long nnz, int reps, Body body) {
  Row r;
  r.category = cat;
  r.matrix = mat;
  r.solver = solver;
  r.n = n;
  r.nnz = nnz;
  r.analyzeMs = r.factorMs = r.solveMs = 1e300;
  for (int rep = 0; rep <= reps; ++rep) {
    Row one;
    if (!body(one)) {
      r.note = one.note;
      return r;
    }
    if (rep == 0) continue;  // warm-up
    r.analyzeMs = std::min(r.analyzeMs, one.analyzeMs);
    r.factorMs = std::min(r.factorMs, one.factorMs);
    r.solveMs = std::min(r.solveMs, one.solveMs);
    r.err = one.err;
    r.resid = one.resid;
    r.fill = one.fill;
    if (!one.note.empty()) r.note = one.note;
  }
  r.ok = true;
  return r;
}

// LeftRightLU / SupernodalLU: shared analyzePattern/factorize/solve/nnzL/nnzU
// interface. `input` may differ from `A` (SupernodalLU needs a symmetrized
// copy) -- the answer is always scored against the real `A`.
template <typename Solver, typename SpMat, typename Vec>
Row runOurs(const std::string& name, int reps, const std::string& cat, const std::string& mat,
            const SpMat& input, const SpMat& A, const Vec& b, const Vec& xTrue) {
  return measure(cat, mat, name, A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Solver s;
    const auto t0 = Clock::now();
    s.analyzePattern(input);
    const auto t1 = Clock::now();
    s.factorize(input);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed: " + s.lastErrorMessage();
      return false;
    }
    const Vec x = s.solve(b);
    const auto t3 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "solve failed";
      return false;
    }
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    return true;
  });
}

// SupernodalLDLT: same shape, but only nnzL() -- there is no separate U.
template <typename Solver, typename SpMat, typename Vec>
Row runLDLT(const std::string& name, int reps, const std::string& cat, const std::string& mat,
            const SpMat& A, const Vec& b, const Vec& xTrue) {
  return measure(cat, mat, name, A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Solver s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed: " + s.lastErrorMessage();
      return false;
    }
    const Vec x = s.solve(b);
    const auto t3 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "solve failed";
      return false;
    }
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.nnzL());
    return true;
  });
}

// PointBlockLU: the headline number is the REPLAY (factorize on an already-
// analyzed pattern), not the first factorization -- see bench_solvers.cpp.
template <typename Solver, typename SpMat, typename Vec>
Row runPointBlock(const std::string& name, int reps, const std::string& cat, const std::string& mat,
                  const SpMat& A, const Vec& b, const Vec& xTrue) {
  Row r;
  r.category = cat;
  r.matrix = mat;
  r.solver = name;
  r.n = A.rows();
  r.nnz = A.nonZeros();
  Solver s;
  const auto t0 = Clock::now();
  s.analyzePattern(A);
  const auto t1 = Clock::now();
  s.factorize(A);
  if (s.info() != Eigen::Success) {
    r.note = "factorize failed: " + s.lastErrorMessage();
    return r;
  }
  r.analyzeMs = ms(t0, t1);
  r.factorMs = r.solveMs = 1e300;
  Vec x;
  for (int rep = 0; rep <= reps; ++rep) {
    const auto a0 = Clock::now();
    s.factorize(A);
    const auto a1 = Clock::now();
    x = s.solve(b);
    const auto a2 = Clock::now();
    if (rep == 0) continue;
    r.factorMs = std::min(r.factorMs, ms(a0, a1));
    r.solveMs = std::min(r.solveMs, ms(a1, a2));
  }
  r.err = (x - xTrue).norm() / xTrue.norm();
  r.resid = (A * x - b).norm() / b.norm();
  r.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
  r.ok = true;
  return r;
}

// RobustLU: compute() is one-shot (diagnosis + whichever rung it lands on), no
// separate analyzePattern/factorize split -- timed as a single phase.
template <typename Solver, typename SpMat, typename Vec>
Row runRobust(const std::string& name, int reps, const std::string& cat, const std::string& mat,
              const SpMat& A, const Vec& b, const Vec& xTrue) {
  Row r;
  r.category = cat;
  r.matrix = mat;
  r.solver = name;
  r.n = A.rows();
  r.nnz = A.nonZeros();
  r.analyzeMs = 0;
  r.factorMs = r.solveMs = 1e300;
  std::string strategy;
  for (int rep = 0; rep <= reps; ++rep) {
    Solver s;
    const auto t0 = Clock::now();
    s.compute(A);
    const auto t1 = Clock::now();
    if (s.info() != Eigen::Success) {
      r.note = "compute failed: " + s.lastErrorMessage();
      return r;
    }
    const Vec x = s.solve(b);
    const auto t2 = Clock::now();
    if (rep == 0) {
      strategy = Eigen::robust_lu::strategyName(s.strategy());
      continue;
    }
    r.factorMs = std::min(r.factorMs, ms(t0, t1));
    r.solveMs = std::min(r.solveMs, ms(t1, t2));
    r.err = (x - xTrue).norm() / xTrue.norm();
    r.resid = (A * x - b).norm() / b.norm();
    r.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
  }
  r.note = "strategy=" + strategy;
  r.ok = true;
  return r;
}

template <typename SpMat, typename Vec>
Row runSparseLU(int reps, const std::string& cat, const std::string& mat, const SpMat& A, const Vec& b,
                const Vec& xTrue) {
  return measure(cat, mat, "Eigen::SparseLU", A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Eigen::SparseLU<SpMat> s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed";
      return false;
    }
    const Vec x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    return true;
  });
}

template <typename SpMat, typename Vec>
Row runSparseQR(int reps, const std::string& cat, const std::string& mat, const SpMat& A, const Vec& b,
                const Vec& xTrue) {
  typedef typename SpMat::Scalar Scalar;
  return measure(cat, mat, "Eigen::SparseQR", A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed";
      return false;
    }
    const Vec x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.matrixR().nonZeros());  // nnz(R) only, not L+U
    (void)Scalar();
    return true;
  });
}

// MultifrontalQR: rank-revealing QR with the default Auto engine (scalar or
// multifrontal, chosen from the symbolic nnz(R)). Fill is nnz(R).
template <typename SpMat, typename Vec>
Row runMultifrontalQR(int reps, const std::string& cat, const std::string& mat, const SpMat& A, const Vec& b,
                      const Vec& xTrue) {
  return measure(cat, mat, "MultifrontalQR", A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Eigen::MultifrontalQR<SpMat> s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed: " + s.lastErrorMessage();
      return false;
    }
    const Vec x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.nnzR());
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s rank %lld/%lld%s",
                  s.engineUsed() == Eigen::multifrontal_qr::Engine::Scalar ? "scalar" : "multifrontal",
                  (long long)s.rank(), (long long)A.cols(), s.rankIsVerified() ? " verified" : "");
    one.note = buf;
    return true;
  });
}

template <typename SpMat, typename Vec>
Row runSimplicialLDLT(int reps, const std::string& cat, const std::string& mat, const SpMat& A,
                      const Vec& b, const Vec& xTrue) {
  return measure(cat, mat, "Eigen::SimplicialLDLT", A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Eigen::SimplicialLDLT<SpMat> s;
    const auto t0 = Clock::now();
    s.analyzePattern(A);
    const auto t1 = Clock::now();
    s.factorize(A);
    const auto t2 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "factorize failed";
      return false;
    }
    const Vec x = s.solve(b);
    const auto t3 = Clock::now();
    one.analyzeMs = ms(t0, t1);
    one.factorMs = ms(t1, t2);
    one.solveMs = ms(t2, t3);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = static_cast<long long>(s.matrixL().nestedExpression().nonZeros());
    return true;
  });
}

template <typename Solver, typename SpMat, typename Vec>
Row runIterative(const std::string& name, int reps, const std::string& cat, const std::string& mat,
                 const SpMat& A, const Vec& b, const Vec& xTrue, int maxIter, double tol) {
  return measure(cat, mat, name, A.rows(), A.nonZeros(), reps, [&](Row& one) {
    Solver s;
    s.setMaxIterations(maxIter);
    s.setTolerance(tol);
    const auto t0 = Clock::now();
    s.compute(A);
    const auto t1 = Clock::now();
    if (s.info() != Eigen::Success) {
      one.note = "compute (preconditioner) failed";
      return false;
    }
    const Vec x = s.solve(b);
    const auto t2 = Clock::now();
    one.analyzeMs = 0;
    one.factorMs = ms(t0, t1);
    one.solveMs = ms(t1, t2);
    one.err = (x - xTrue).norm() / xTrue.norm();
    one.resid = (A * x - b).norm() / b.norm();
    one.fill = -1;
    if (s.info() != Eigen::Success) {
      char buf[128];
      std::snprintf(buf, sizeof buf, "no convergence: iters=%d est.err=%.2e",
                    static_cast<int>(s.iterations()), static_cast<double>(s.error()));
      one.note = buf;
      return false;
    }
    return true;
  });
}

void printHeader() {
  std::printf("  %-26s %9s %9s %8s %9s  %9s %9s %12s\n", "solver", "analyze", "factor", "solve",
              "total", "err", "resid", "fill");
}

void printRow(const Row& r) {
  if (!r.ok) {
    std::printf("  %-26s %s\n", r.solver.c_str(), r.note.c_str());
    return;
  }
  std::printf("  %-26s %9.3f %9.3f %8.3f %9.3f  %9.2e %9.2e", r.solver.c_str(), r.analyzeMs,
              r.factorMs, r.solveMs, r.total(), r.err, r.resid);
  if (r.fill >= 0)
    std::printf(" %12lld", r.fill);
  else
    std::printf(" %12s", "-");
  std::printf("  %s\n", r.note.c_str());
}

// True iff A(i,j) == A(j,i) (not just structurally, numerically) for every
// stored entry. Only meaningful -- and only called -- when patternIsSymmetric
// already holds, which is what makes the element-wise valuePtr comparison
// below valid (A and its transpose then share the identical compressed
// index arrays).
template <typename Scalar>
bool isValueSymmetric(const Eigen::SparseMatrix<Scalar>& A) {
  Eigen::SparseMatrix<Scalar> AT = A.transpose();
  AT.makeCompressed();
  Eigen::SparseMatrix<Scalar> Ac = A;
  Ac.makeCompressed();
  if (Ac.nonZeros() != AT.nonZeros()) return false;
  for (Eigen::Index k = 0; k < Ac.nonZeros(); ++k) {
    const auto d = Ac.valuePtr()[k] - AT.valuePtr()[k];
    const auto scale = (std::max)(Eigen::numext::abs(Ac.valuePtr()[k]), typename Eigen::NumTraits<Scalar>::Real(1));
    if (Eigen::numext::abs(d) > 1e-9 * scale) return false;
  }
  return true;
}

// Peek just the MatrixMarket field token ("real"/"complex"/...) without
// committing to a Scalar type -- loadMatrixMarketAs<Scalar> throws if a
// complex file is read as a real Scalar, so the field has to be known first.
bool isComplexFile(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line)) return false;
  std::istringstream iss(line);
  std::vector<std::string> tok;
  for (std::string t; iss >> t;) {
    for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    tok.push_back(t);
  }
  return tok.size() > 3 && tok[3] == "complex";
}

template <typename Scalar>
void runMatrix(const std::string& path, const std::string& category, int reps) {
  using SpMat = Eigen::SparseMatrix<Scalar>;
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  constexpr bool kComplex = Eigen::NumTraits<Scalar>::IsComplex;

  SpMat A;
  try {
    A = lu_testing::loadMatrixMarketAs<Scalar>(path);
  } catch (const std::exception& e) {
    std::printf("  load failed: %s: %s\n", path.c_str(), e.what());
    return;
  }
  if (A.rows() != A.cols() || A.rows() == 0) {
    std::printf("  skip (not square or empty): %s\n", path.c_str());
    return;
  }

  const std::string mat = fs::path(path).stem().string();
  const bool patSym = lu_testing::patternIsSymmetric(A);
  const bool valSym = patSym && isValueSymmetric(A);

  std::mt19937 rng(12345);  // same xTrue distribution seed for every matrix/solver
  (void)rng;
  std::srand(12345);
  const Vec xTrue = Vec::Random(A.rows());
  const Vec b = A * xTrue;

  std::printf("\n=== [%s] %s  (n=%lld nnz=%lld patSym=%s valSym=%s)\n", category.c_str(), mat.c_str(),
              (long long)A.rows(), (long long)A.nonZeros(), patSym ? "yes" : "no",
              valSym ? "yes" : "no");
  printHeader();

  const SpMat Asym = lu_testing::ensureSymmetricPattern(A);
  std::vector<Row> rows;

  rows.push_back(runOurs<Eigen::LeftRightLU<SpMat, Eigen::AMDOrdering<int>>>("LeftRightLU AMD", reps,
                                                                              category, mat, A, A, b, xTrue));
  rows.push_back(runOurs<Eigen::LeftRightLU<SpMat, Eigen::COLAMDOrdering<int>>>(
      "LeftRightLU COLAMD", reps, category, mat, A, A, b, xTrue));
  rows.push_back(runOurs<Eigen::SupernodalLU<SpMat, Eigen::AMDOrdering<int>>>(
      "SupernodalLU AMD", reps, category, mat, Asym, A, b, xTrue));
  rows.push_back(runPointBlock<Eigen::PointBlockLU<SpMat, Eigen::COLAMDOrdering<int>>>(
      "PointBlockLU COLAMD", reps, category, mat, A, b, xTrue));
  rows.push_back(runRobust<Eigen::RobustLU<SpMat>>("RobustLU", reps, category, mat, A, b, xTrue));

  rows.push_back(runSparseLU(reps, category, mat, A, b, xTrue));
  rows.push_back(runSparseQR(reps, category, mat, A, b, xTrue));
  rows.push_back(runMultifrontalQR(reps, category, mat, A, b, xTrue));

  if (valSym) {
    rows.push_back(runSimplicialLDLT(reps, category, mat, A, b, xTrue));
    if constexpr (!kComplex) {
      rows.push_back(runLDLT<Eigen::SupernodalLDLT<SpMat>>("SupernodalLDLT", reps, category, mat, A, b,
                                                            xTrue));
      const int maxIter = std::min<int>(2000, 20 * static_cast<int>(A.rows()) + 50);
      rows.push_back(
          runIterative<Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                                                Eigen::IncompleteCholesky<Scalar>>>(
              "Eigen::CG+IC", reps, category, mat, A, b, xTrue, maxIter, 1e-10));
    }
  }

  const int maxIter = std::min<int>(2000, 20 * static_cast<int>(A.rows()) + 50);
  rows.push_back(runIterative<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<Scalar>>>(
      "Eigen::BiCGSTAB+ILUT", reps, category, mat, A, b, xTrue, maxIter, 1e-10));
  rows.push_back(runIterative<Eigen::GMRES<SpMat, Eigen::IncompleteLUT<Scalar>>>(
      "Eigen::GMRES+ILUT", reps, category, mat, A, b, xTrue, maxIter, 1e-10));

  const Row* best = nullptr;
  for (const Row& r : rows) {
    printRow(r);
    if (r.ok && (!best || r.total() < best->total())) best = &r;
  }
  if (best)
    std::printf("  fastest: %s at %.3f ms\n", best->solver.c_str(), best->total());

  g_rows.insert(g_rows.end(), rows.begin(), rows.end());
}

void runOneFile(const std::string& path, const std::string& category, int reps) {
  if (isComplexFile(path))
    runMatrix<std::complex<double>>(path, category, reps);
  else
    runMatrix<double>(path, category, reps);
}

std::vector<std::string> listMatrixFiles(const fs::path& dir) {
  std::vector<std::string> files;
  if (!fs::exists(dir)) return files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    if (name.size() < 4 || name.substr(name.size() - 4) != ".mtx") continue;
    if (name.size() >= 8 && name.substr(name.size() - 8) == "_rhs.mtx") continue;
    files.push_back(entry.path().string());
  }
  std::sort(files.begin(), files.end());
  return files;
}

void writeCsv(const std::string& path) {
  fs::create_directories(fs::path(path).parent_path());
  std::ofstream out(path);
  out << "category,matrix,solver,n,nnz,analyze_ms,factor_ms,solve_ms,total_ms,err,resid,fill,ok,note\n";
  for (const Row& r : g_rows) {
    std::string note = r.note;
    std::replace(note.begin(), note.end(), ',', ';');
    out << r.category << ',' << r.matrix << ',' << r.solver << ',' << r.n << ',' << r.nnz << ','
        << r.analyzeMs << ',' << r.factorMs << ',' << r.solveMs << ',' << r.total() << ',' << r.err
        << ',' << r.resid << ',' << r.fill << ',' << (r.ok ? 1 : 0) << ',' << note << "\n";
  }
  std::printf("\nraw results written to %s (%zu rows)\n", path.c_str(), g_rows.size());
}

struct Agg {
  int nOk = 0, nTotal = 0;
  double sumTotalMs = 0, sumResid = 0;
};

void printCategorySummary() {
  std::printf("\n================ per-category solver ranking ================\n");
  std::vector<std::string> categories;
  for (const Row& r : g_rows)
    if (std::find(categories.begin(), categories.end(), r.category) == categories.end())
      categories.push_back(r.category);

  for (const std::string& cat : categories) {
    std::map<std::string, Agg> bySolver;
    for (const Row& r : g_rows) {
      if (r.category != cat) continue;
      Agg& a = bySolver[r.solver];
      ++a.nTotal;
      if (r.ok) {
        ++a.nOk;
        a.sumTotalMs += r.total();
        a.sumResid += r.resid;
      }
    }
    std::vector<std::pair<std::string, Agg>> ranked(bySolver.begin(), bySolver.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& x, const auto& y) {
      if (x.second.nOk != y.second.nOk) return x.second.nOk > y.second.nOk;
      const double mx = x.second.nOk ? x.second.sumTotalMs / x.second.nOk : 1e300;
      const double my = y.second.nOk ? y.second.sumTotalMs / y.second.nOk : 1e300;
      return mx < my;
    });
    std::printf("\n%s:\n", cat.c_str());
    std::printf("  %-26s %8s %12s %12s\n", "solver", "solved", "mean total", "mean resid");
    for (const auto& [solver, a] : ranked) {
      std::printf("  %-26s %4d/%-3d", solver.c_str(), a.nOk, a.nTotal);
      if (a.nOk > 0)
        std::printf(" %9.3f ms %12.2e\n", a.sumTotalMs / a.nOk, a.sumResid / a.nOk);
      else
        std::printf(" %12s %12s\n", "-", "-");
    }
    if (!ranked.empty() && ranked.front().second.nOk > 0)
      std::printf("  -> best: %s\n", ranked.front().first.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  int reps = 3;
  std::string root;
  std::string csv = "analysis/benchmark_results.csv";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (arg.rfind("--", 0) == 0) {
      std::printf("usage: %s [root_dir] [--reps N] [--csv path]\n", argv[0]);
      return 2;
    } else {
      root = arg;
    }
  }

  const std::string dir = samplesDir(root);
  std::printf("DirectLUSolvers setfos-samples benchmark\n");
  std::printf("  samples dir: %s\n", dir.c_str());
  std::printf("  reps (best-of, after a warm-up): %d\n", reps);
  std::printf("  xTrue/b are synthesized (b = A*xTrue); the paired _rhs.mtx files are not used.\n");
  std::printf("  fill = nnzL+nnzU as each solver reports it (SparseQR, MultifrontalQR: nnz(R) only).\n");

  if (!fs::exists(dir)) {
    std::printf("samples dir does not exist: %s\n", dir.c_str());
    return 2;
  }

  std::vector<std::string> categories;
  for (const auto& entry : fs::directory_iterator(dir))
    if (entry.is_directory()) categories.push_back(entry.path().filename().string());
  std::sort(categories.begin(), categories.end());

  for (const std::string& cat : categories) {
    for (const std::string& path : listMatrixFiles(fs::path(dir) / cat)) runOneFile(path, cat, reps);
  }

  printCategorySummary();
  writeCsv(csv);
  return 0;
}
