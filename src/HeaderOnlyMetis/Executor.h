// The Executor concept used by the parallel nested-dissection driver, and a
// serial default so the port keeps its no-dependency property.
//
// This deliberately does NOT include SupernodalLUExecutor.h. The whole point of
// HeaderOnlyMetis is that it needs nothing linked and nothing else included, so
// the Executor stays a duck-typed template parameter: anything providing
//
//     template <class F> void parallelFor(I begin, I end, F&& f) const;  // barrier on return
//     int concurrency() const;                                          // >= 1
//
// works. Eigen::supernodal_lu::SerialExecutor / StdThreadExecutor / OpenMPExecutor
// / TBBExecutor from this project all satisfy it, so
//
//     #include <SupernodalLUExecutor.h>
//     Eigen::supernodal_lu::StdThreadExecutor exec(8);
//     HeaderOnlyMetisParallelOrdering<int, Eigen::supernodal_lu::StdThreadExecutor> ord(exec);
//
// works without this header knowing anything about them.
//
// ONE HARD REQUIREMENT, and it shapes the whole parallel design: parallelFor is
// NOT re-entrant in the backends this project ships. StdThreadExecutor::dispatch
// stores a single shared body pointer and generation counter, so a parallelFor
// issued from inside a worker would overwrite the in-flight one and deadlock.
// The driver therefore never nests them: it runs one flat dispatch per level of
// the dissection tree.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_EXECUTOR_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_EXECUTOR_H

#include <cstddef>

namespace header_only_metis {

// Runs the loop on the calling thread. Empty and trivially copyable, so the
// parallel driver instantiated with it compiles down to the plain loop.
struct SerialExecutor {
  template <typename F>
  void parallelFor(std::ptrdiff_t begin, std::ptrdiff_t end, F&& f) const {
    for (std::ptrdiff_t i = begin; i < end; ++i) f(i);
  }
  int concurrency() const { return 1; }
};

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_EXECUTOR_H
