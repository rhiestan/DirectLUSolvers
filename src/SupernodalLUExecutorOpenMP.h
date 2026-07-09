// SupernodalLU - OpenMP parallel-execution backend.
//
// Implements the Executor concept from SupernodalLUExecutor.h
// (template<class F> void parallelFor(Index,Index,F&&) const; int concurrency()
// const) on top of OpenMP's `#pragma omp parallel for`, instead of the bundled
// std::thread pool (StdThreadExecutor).
//
// Kept SEPARATE from SupernodalLUExecutor.h (which stays free of any threading
// library dependency beyond <thread>): including this header requires the
// translation unit to be compiled with OpenMP enabled and linked against an
// OpenMP runtime.
//
//   clang++ -fopenmp ...          (clang, GNU driver)
//   clang-cl /openmp ...          (clang, MSVC driver)
//   cl /openmp ...                (MSVC)
//   g++ -fopenmp ...              (GCC)
//
// Unlike StdThreadExecutor, OpenMPExecutor does not own a persistent thread
// pool itself -- it drives whatever pool the OpenMP runtime maintains, which
// is normally created lazily on the first parallel region and then kept
// warm. That makes this backend a good fit when the surrounding application
// already uses OpenMP elsewhere (one shared runtime pool, no double
// subscription of CPU cores).
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_EXECUTOR_OPENMP_H
#define SUPERNODAL_LU_EXECUTOR_OPENMP_H

#include <Eigen/Core>

#include <omp.h>

namespace Eigen {
namespace supernodal_lu {

// Parallel-execution backend driving OpenMP's own thread pool. See file
// header for build requirements.
class OpenMPExecutor {
 public:
  // numThreads == 0 (default) uses the OpenMP runtime's own current default
  // (omp_get_max_threads(), i.e. whatever OMP_NUM_THREADS / a prior
  // omp_set_num_threads() call left in effect). A positive value overrides
  // the thread count for every parallelFor() issued through this instance
  // (via the `num_threads` clause) without touching the runtime's ambient
  // default, so it composes safely with unrelated OpenMP code in the same
  // process.
  explicit OpenMPExecutor(int numThreads = 0) : m_numThreads(numThreads) {}

  int concurrency() const { return m_numThreads > 0 ? m_numThreads : omp_get_max_threads(); }

  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    if (end <= begin) return;
    const int threads = concurrency();
    // A region of 1 or fewer usable threads is pure overhead; also lets this
    // class work correctly in a translation unit that links OpenMP but runs
    // with OMP_NUM_THREADS=1.
    if (threads <= 1 || end - begin == 1) {
      for (Index i = begin; i < end; ++i) f(i);
      return;
    }
    // schedule(dynamic): SupernodalLU's per-index costs are highly
    // non-uniform (a huge root supernode next to tiny leaves, or an uneven
    // chunk split), so a single index per grab -- like StdThreadExecutor's
    // fetch_add scheduling -- balances load far better than OpenMP's default
    // static, evenly-sized chunking.
#pragma omp parallel for num_threads(threads) schedule(dynamic) default(shared)
    for (Index i = begin; i < end; ++i) f(i);
  }

 private:
  int m_numThreads;
};

}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_EXECUTOR_OPENMP_H
