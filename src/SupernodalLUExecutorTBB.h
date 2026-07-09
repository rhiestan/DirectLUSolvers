// SupernodalLU - oneAPI Threading Building Blocks (oneTBB) parallel-execution
// backend.
//
// Implements the Executor concept from SupernodalLUExecutor.h
// (template<class F> void parallelFor(Index,Index,F&&) const; int concurrency()
// const) on top of oneapi::tbb::parallel_for, instead of the bundled
// std::thread pool (StdThreadExecutor).
//
// Kept SEPARATE from SupernodalLUExecutor.h (which stays free of any
// threading-library dependency beyond <thread>): including this header
// requires the oneTBB headers on the include path and linking against
// tbb12/tbb (e.g. -ltbb12, or /link tbb12.lib on MSVC/clang-cl).
//
// Like OpenMPExecutor, this backend does not own a private thread pool: TBB
// maintains one process-wide worker arena, shared by every TBBExecutor
// instance and any other TBB-based code linked into the same application --
// so this composes cleanly with other TBB-parallel libraries (e.g. oneMKL
// built with the TBB threading layer) without oversubscribing the machine.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_EXECUTOR_TBB_H
#define SUPERNODAL_LU_EXECUTOR_TBB_H

#include <Eigen/Core>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>

#include <memory>

namespace Eigen {
namespace supernodal_lu {

// Parallel-execution backend driving oneTBB's own worker arena. See file
// header for build requirements.
class TBBExecutor {
 public:
  // maxThreads == 0 (default) leaves TBB's ambient concurrency (normally
  // std::thread::hardware_concurrency(), or whatever an enclosing
  // tbb::global_control / task_arena already set) untouched. A positive value
  // installs a tbb::global_control that caps TBB's arena to that many threads
  // for the lifetime of this TBBExecutor -- oneTBB's documented way to bound
  // concurrency (TBB has no equivalent of a per-call thread-count argument).
  // Because the cap is process-wide for as long as this object (or a copy of
  // it) lives, avoid running two TBBExecutors with DIFFERENT positive
  // maxThreads concurrently; a single instance reused across solves is the
  // normal case. The control block is held by shared_ptr (global_control
  // itself has a user-declared destructor but no move/copy semantics of its
  // own) so TBBExecutor stays cheaply copyable/assignable -- needed because
  // SupernodalLU's only executor-reconfiguration hook is assigning through
  // the mutable executor() accessor, e.g. `solver.executor() = TBBExecutor(n);`.
  explicit TBBExecutor(int maxThreads = 0) {
    if (maxThreads > 0)
      m_control = std::make_shared<oneapi::tbb::global_control>(
          oneapi::tbb::global_control::max_allowed_parallelism, static_cast<std::size_t>(maxThreads));
  }

  // active_value(), NOT info::default_concurrency(): the latter is the
  // platform's static default and does not reflect a currently active
  // global_control cap (from this instance's own maxThreads, or one an
  // enclosing scope installed), which is what callers actually want to know.
  int concurrency() const {
    return static_cast<int>(
        oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism));
  }

  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    if (end <= begin) return;
    // grainsize 1 + simple_partitioner: always split down to one index per
    // task. SupernodalLU's per-index costs are highly non-uniform (a huge
    // root supernode next to tiny leaves), so forcing fine-grained tasks lets
    // TBB's work-stealing scheduler balance the load, the same reason
    // StdThreadExecutor/OpenMPExecutor use per-index dynamic scheduling
    // rather than static, evenly-sized chunks.
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<Index>(begin, end, 1),
        [&](const oneapi::tbb::blocked_range<Index>& range) {
          for (Index i = range.begin(); i != range.end(); ++i) f(i);
        },
        oneapi::tbb::simple_partitioner());
  }

 private:
  std::shared_ptr<oneapi::tbb::global_control> m_control;
};

}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_EXECUTOR_TBB_H
