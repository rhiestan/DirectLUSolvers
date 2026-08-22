// Serial-vs-parallel consistency for BOTH solvers' intra-supernode parallelism.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_parallel_consistency --output-on-failure
//
// WHY THIS EXISTS
//
// Both solvers parallelize INSIDE a supernode once the elimination structure
// stops offering independent supernodes: SupernodalLU on its narrow levels,
// LeftRightLU in the tail sweep that follows its DAG phase. Both do it by
// splitting a panel operation into chunks dispatched across the pool. That is
// the part of each solver where a mistake is a data race -- silent, thread-count
// dependent, and invisible to a single-threaded test suite. Nothing else here
// exercised it against its serial counterpart.
//
// WHAT MUST HOLD, AND WHAT MUST NOT BE ASSERTED
//
// Chunking splits one big GEMM into several smaller ones. Eigen picks its kernel
// and blocking from the shape of the operands, so the chunked form sums each dot
// product in a different order, and the factorization it produces is a DIFFERENT
// (equally valid) one. Demanding a bit-identical solution would therefore encode
// a false claim, and on an ill-conditioned matrix it fails loudly for no good
// reason: Bai/cryg10000 solves to a 5e-15 residual both ways while the two
// solution vectors differ by 0.88 relative, because at that conditioning any
// perturbation moves x a long way. So this suite asserts the invariants that are
// genuinely thread-independent:
//
//   * FILL is identical. The symbolic structure is decided before any thread
//     starts, so nnzL+nnzU must match the serial run EXACTLY -- an exact
//     assertion, and the one a scheduling bug that corrupted the structure would
//     break first.
//   * The parallel solve is ACCURATE: its residual meets the same threshold the
//     serial run met.
//   * The two solutions AGREE, unless the parallel solve is NO LESS ACCURATE
//     than the serial one -- in which case the disagreement is the matrix's
//     conditioning talking, not a race. Comparing against the serial residual
//     rather than against a fixed "machine precision" bar is what makes this
//     robust: HB/nnc1374 solves to 1.7e-9 both ways, which is accurate for that
//     matrix and would fail an absolute 1e-11 threshold for no reason. A genuine
//     race degrades the answer; it does not politely produce another exact one.
//
// Divergences of that last kind are counted and reported, so a change in how
// many matrices diverge is visible in the log even though it is not a failure.
//
// The corpus is optional (see TestData.h); the synthetic Laplacians always run,
// and lap3d is the shape that actually drives the tail path on both solvers.

#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::supernodal_lu::PooledExecutor;
using lu_testing::check;
using lu_testing::checkTrue;

// PooledExecutor, not StdThreadExecutor: the thread count has to be set after
// construction so one solver type covers the whole sweep.
using Snlu = Eigen::SupernodalLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, PooledExecutor>;
using Lrlu = Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, PooledExecutor>;

namespace {

// Keep the deliberately-pathological corpus members from spending a minute each,
// several times over: past this they decline at factorization and are skipped.
constexpr long long kFillLimit = 20LL * 1000 * 1000;

// The residual a solve must reach to count as accurate.
constexpr double kResidTolerance = 1e-6;
// Below this, two solutions count as agreeing outright.
constexpr double kAgreeTolerance = 1e-8;
// How much worse than the serial residual the parallel one may be before a
// disagreement stops being attributable to conditioning. Generous on purpose:
// a race blows the residual up by orders of magnitude (often to inf/nan), so
// this does not need to be tight to catch one, and being tight would only make
// the suite flaky on matrices that sit near their accuracy limit either way.
constexpr double kResidSlack = 100.0;

int g_diverged = 0;  // reported, not failed -- see the header comment

VectorXd deterministicRhs(const SparseMatrix<double>& A) {
  VectorXd x(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) x(i) = 1.0 + 0.5 * std::sin(static_cast<double>(i));
  return A * x;
}

struct Outcome {
  VectorXd x;
  long long fill = 0;
  double resid = 0.0;
  bool ok = false;
};

template <typename Solver>
Outcome run(const SparseMatrix<double>& A, const VectorXd& b, int threads) {
  Outcome o;
  Solver s;
  s.setMaxFactorNonzeros(kFillLimit);
  if (threads > 1) s.executor() = PooledExecutor(threads);
  try {
    s.compute(A);
    if (s.info() != Eigen::Success) return o;
    o.x = s.solve(b);
    o.fill = static_cast<long long>(s.nnzL()) + s.nnzU();
    o.resid = (A * o.x - b).norm() / b.norm();
    o.ok = std::isfinite(o.resid);
  } catch (const std::exception& e) {
    lu_testing::fail(std::string("threw: ") + e.what());
  }
  return o;
}

// One solver, one matrix: compare every thread count against the serial run.
template <typename Solver>
void sweep(const char* who, const std::string& label, const SparseMatrix<double>& A) {
  const VectorXd b = deterministicRhs(A);
  const Outcome ref = run<Solver>(A, b, 1);
  // A matrix this solver declines or solves badly on its own is not this
  // suite's business -- test_suitesparse judges that. Skip rather than fail.
  if (!ref.ok || ref.resid > kResidTolerance) return;

  const std::string key = label + " [" + who + "]";
  for (int threads : {2, 4, 8}) {
    for (int rep = 0; rep < 2; ++rep) {  // a race need not show on the first run
      const Outcome p = run<Solver>(A, b, threads);
      const std::string tag =
          key + " t=" + std::to_string(threads) + " r=" + std::to_string(rep);
      if (!checkTrue(p.ok, tag + ": parallel run succeeded")) continue;
      if (!check(p.fill == ref.fill, tag + ": fill matches serial",
                 static_cast<double>(p.fill - ref.fill)))
        continue;
      if (!check(p.resid <= kResidTolerance, tag + ": parallel residual", p.resid)) continue;

      const double diff = (p.x - ref.x).norm() / std::max(1.0, ref.x.norm());
      if (diff <= kAgreeTolerance) continue;  // agrees outright; nothing to explain
      // Disagrees. Legitimate only if the parallel solve is no less accurate
      // than the serial one -- i.e. the matrix, not the schedule, decided x.
      // The evidence printed is that accuracy ratio, not the disagreement:
      // the ratio is what says whether anything is wrong.
      const bool noWorse = (p.resid <= kResidSlack * ref.resid);
      check(noWorse, tag + ": disagrees, but no less accurate than serial",
            p.resid / std::max(ref.resid, 1e-300));
      if (noWorse) ++g_diverged;
    }
  }
}

void checkMatrix(const std::string& label, const SparseMatrix<double>& A) {
  sweep<Snlu>("SupernodalLU", label, A);
  sweep<Lrlu>("LeftRightLU", label, A);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  // --quick keeps CTest to the synthetic matrices; bare runs add the corpus.
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--quick") quick = true;
    else { std::printf("usage: %s [--quick]\n", argv[0]); return 2; }
  }

  std::printf("Serial-vs-parallel consistency (threads 2/4/8, 2 reps each)\n\n");

  // lap3d is the load-bearing case: a big root separator is what forces both
  // solvers onto their intra-supernode path in the first place.
  std::printf("== synthetic\n");
  checkMatrix("lap3d_20x20x20", lu_testing::laplacian3d(20, 20, 20));
  checkMatrix("lap2d_120x120", lu_testing::laplacian2d(120, 120));

  if (!quick) {
    std::printf("== SuiteSparse corpus\n");
    for (const auto& m : lu_testing::suitesparseMatrices()) {
      if (!m.available) continue;
      try {
        checkMatrix(m.label(),
                    lu_testing::ensureSymmetricPattern(lu_testing::loadMatrixMarket(m.path)));
      } catch (const std::exception& e) {
        lu_testing::fail(m.label() + ": load failed: " + e.what());
      }
    }
  }

  std::printf(
      "\n%d solution(s) differed from serial while staying at least as accurate "
      "(conditioning, not a race -- see the header comment).\n",
      g_diverged);
  return lu_testing::summarize("Parallel consistency");
}
