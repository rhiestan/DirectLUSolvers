// One shared contract for every Executor backend.
//
//   cmake -S . -B build -G Ninja -DDLU_WITH_OPENMP=ON -DDLU_WITH_TBB=ON
//   ctest --test-dir build -R test_executors --output-on-failure
//
// WHY THIS EXISTS
//
// The README documents four backends -- SerialExecutor, StdThreadExecutor,
// OpenMPExecutor, TBBExecutor -- and claims they give "numerically consistent
// results for a fixed thread count". Only StdThreadExecutor was ever tested.
// The other two were verified once by hand, and the CMakeLists carried no switch
// for them at all, so nothing would notice if a change broke either.
//
// The three multithreaded backends are exercised in ONE binary against the SAME
// checks, deliberately: separate per-backend tests drift, and the interesting
// property is agreement BETWEEN them, not that each is individually plausible.
//
// OpenMP and TBB are optional; when their switches are off the backend is simply
// skipped and the suite still covers StdThreadExecutor. The checks are:
//
//   * concurrency() reports what was asked for;
//   * factor + solve agree with SerialExecutor to machine precision, and the
//     determinant with it;
//   * the parallel triangular solve stays BIT-IDENTICAL through each backend
//     (it is a different dispatch path per backend, and the bit-identity claim
//     is about the algorithm, so it must hold for all of them);
//   * LeftRightLU's barrier-free DAG scheduler runs correctly on each backend --
//     it is a single parallelFor whose body is a whole scheduler, which is a
//     much stranger thing to ask of an executor than a plain loop;
//   * repeated factorizations do not deadlock or drift.

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstdio>
#include <string>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

#ifdef DLU_HAVE_OPENMP
#include "SupernodalLUExecutorOpenMP.h"
#endif
#ifdef DLU_HAVE_TBB
#include "SupernodalLUExecutorTBB.h"
#endif

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;

namespace slu = Eigen::supernodal_lu;

namespace {

typedef SparseMatrix<double> SpMat;

// Big enough that the parallel triangular solve's work threshold
// (rows x nrhs >= 200000) is crossed, so the parallel dispatch path is really
// taken rather than silently falling back to serial.
constexpr int kGrid = 300;  // 300x300 = 90000 rows
constexpr int kNrhs = 4;

struct Reference {
  SpMat A;          // large: crosses the parallel-solve work threshold
  MatrixXd B;
  MatrixXd X;       // serial solution
  long long fill = 0;

  // The determinant needs its OWN, small system. det scales like the product of
  // the pivots, so the 90000-row Laplacian above has |det| ~ 4^90000 -- inf in
  // double, and inf - inf is NaN, which makes the comparison vacuous rather than
  // strict. (This exact trap already bit test_scalar_types; it is easy to walk
  // into twice.) A 30x20 grid keeps it representable.
  SpMat smallA;
  double smallDet = 0.0;
};

Reference buildReference() {
  Reference r;
  r.A = lu_testing::laplacian2d(kGrid, kGrid);
  const Eigen::Index n = r.A.rows();
  r.B.resize(n, kNrhs);
  for (Eigen::Index i = 0; i < n; ++i)
    for (int j = 0; j < kNrhs; ++j) r.B(i, j) = 1.0 + 0.1 * ((i + j) % 7);

  Eigen::SupernodalLU<SpMat> s;  // SerialExecutor
  s.setMaxIterativeRefinements(0);
  s.compute(r.A);
  r.X = s.solve(r.B);
  r.fill = static_cast<long long>(s.nnzL()) + s.nnzU();

  r.smallA = lu_testing::randomSymmetricPattern(120, 0.05, 31);
  Eigen::SupernodalLU<SpMat> t;
  t.compute(r.smallA);
  r.smallDet = t.determinant();
  return r;
}

// Run the shared contract for one executor type at a given thread count.
// `configure` installs a correctly-constructed executor, since the backends
// differ in how a thread count is requested.
template <typename Executor, typename Configure>
void exerciseBackend(const std::string& tag, const Reference& ref, Configure configure) {
  typedef Eigen::SupernodalLU<SpMat, Eigen::AMDOrdering<int>, Executor> Solver;
  typedef Eigen::LeftRightLU<SpMat, Eigen::AMDOrdering<int>, Executor> DagSolver;

  Solver s;
  configure(s);
  checkTrue(s.executor().concurrency() >= 1, tag + ": concurrency() is at least 1");

  s.setMaxIterativeRefinements(0);
  s.compute(ref.A);
  if (s.info() != Eigen::Success) {
    lu_testing::fail(tag + ": factorization failed: " + s.lastErrorMessage());
    return;
  }

  // The symbolic structure is executor-independent, so any difference here means
  // the backend perturbed something it should not have touched.
  checkTrue(static_cast<long long>(s.nnzL()) + s.nnzU() == ref.fill,
            tag + ": fill matches the serial factorization");

  const MatrixXd X = s.solve(ref.B);
  const double agree = (X - ref.X).norm() / ref.X.norm();
  check(agree < 1e-12, tag + ": solution agrees with SerialExecutor", agree);
  const double resid = (ref.A * X - ref.B).norm() / ref.B.norm();
  check(resid < 1e-8, tag + ": residual", resid);

  // Determinant on the small system, where it is representable. Worth checking
  // per backend because it folds in the sign of every supernode's in-block pivot
  // permutation, and those are produced concurrently.
  {
    Solver small;
    configure(small);
    small.compute(ref.smallA);
    const double detRel =
        std::abs(small.determinant() - ref.smallDet) / std::max(1.0, std::abs(ref.smallDet));
    check(detRel < 1e-10, tag + ": determinant agrees with SerialExecutor", detRel);
  }

  // The parallel triangular solve is a distinct dispatch path per backend, and
  // its bit-identity claim is about the algorithm, not about one pool.
  s.setParallelSolve(false);
  const MatrixXd serialSweep = s.solve(ref.B);
  s.setParallelSolve(true);
  const MatrixXd parallelSweep = s.solve(ref.B);
  const double sweepDiff = (serialSweep - parallelSweep).cwiseAbs().maxCoeff();
  check(sweepDiff == 0.0, tag + ": parallel triangular solve is bit-identical", sweepDiff);

  // LeftRightLU asks something much stranger of an executor: one parallelFor
  // whose body is an entire work-stealing scheduler that runs to completion.
  DagSolver d;
  configure(d);
  d.setMaxIterativeRefinements(0);
  d.compute(ref.A);
  if (d.info() != Eigen::Success) {
    lu_testing::fail(tag + ": LeftRightLU factorization failed: " + d.lastErrorMessage());
    return;
  }
  const MatrixXd Xd = d.solve(ref.B);
  const double dagResid = (ref.A * Xd - ref.B).norm() / ref.B.norm();
  check(dagResid < 1e-8, tag + ": LeftRightLU DAG scheduler on this backend", dagResid);

  // Repeat a few times: a backend that leaks pool state or races on teardown
  // tends to show it on the second or third go, not the first.
  for (int rep = 0; rep < 3; ++rep) {
    d.factorize(ref.A);
    if (d.info() != Eigen::Success) {
      lu_testing::fail(tag + ": repeated factorization failed");
      return;
    }
  }
  const double repeatResid = (ref.A * MatrixXd(d.solve(ref.B)) - ref.B).norm() / ref.B.norm();
  check(repeatResid < 1e-8, tag + ": stable across repeated factorizations", repeatResid);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("Executor backend contract\n");

  const Reference ref = buildReference();
  std::printf("  reference: n=%lld, fill=%lld (SerialExecutor)\n", (long long)ref.A.rows(),
              ref.fill);

  // StdThreadExecutor owns its pool, deletes its copy operations and declares a
  // destructor, so it has no move operations either: its thread count is fixed
  // at construction and it CANNOT be reassigned into a live solver. It therefore
  // gets exactly one run, at whatever the default pool size is -- labelling two
  // runs "2" and "8" would be a fiction, since both would use the default.
  // Thread-count variation for this backend goes through PooledExecutor, the
  // shared-pool wrapper that exists precisely to make it assignable.
  std::printf("StdThreadExecutor:\n");
  exerciseBackend<slu::StdThreadExecutor>("StdThread(default)", ref, [](auto&) {});
  for (int t : {2, 8})
    exerciseBackend<Eigen::supernodal_lu::PooledExecutor>(
        "Pooled/StdThread(" + std::to_string(t) + ")", ref,
        [t](auto& s) { s.executor() = Eigen::supernodal_lu::PooledExecutor(t); });

#ifdef DLU_HAVE_OPENMP
  std::printf("OpenMPExecutor:\n");
  for (int t : {2, 8}) {
    exerciseBackend<slu::OpenMPExecutor>("OpenMP(" + std::to_string(t) + ")", ref,
                                         [t](auto& s) { s.executor() = slu::OpenMPExecutor(t); });
    // The README states a positive thread count overrides the runtime default
    // for every dispatch issued through the instance.
    slu::OpenMPExecutor e(t);
    checkTrue(e.concurrency() == t, "OpenMP(" + std::to_string(t) + "): concurrency() honours the request");
  }
#else
  std::printf("OpenMPExecutor: not built (configure with -DDLU_WITH_OPENMP=ON)\n");
#endif

#ifdef DLU_HAVE_TBB
  std::printf("TBBExecutor:\n");
  for (int t : {2, 8}) {
    exerciseBackend<slu::TBBExecutor>("TBB(" + std::to_string(t) + ")", ref,
                                      [t](auto& s) { s.executor() = slu::TBBExecutor(t); });
  }
  // TBBExecutor caps concurrency with a global_control held by shared_ptr, so
  // reassignment must actually take effect -- and must do so repeatedly, since
  // the previous cap has to be released first. The README claims this works;
  // nothing checked it.
  {
    Eigen::SupernodalLU<SpMat, Eigen::AMDOrdering<int>, slu::TBBExecutor> s;
    s.executor() = slu::TBBExecutor(4);
    const int first = s.executor().concurrency();
    s.executor() = slu::TBBExecutor(2);
    const int second = s.executor().concurrency();
    checkTrue(first == 4 && second == 2,
              "TBB: reassigning executor() re-caps concurrency (4 then 2)");
    if (!(first == 4 && second == 2))
      lu_testing::note("observed " + std::to_string(first) + " then " + std::to_string(second));
  }
#else
  std::printf("TBBExecutor: not built (configure with -DDLU_WITH_TBB=ON)\n");
#endif

  return lu_testing::summarize("Executors");
}
