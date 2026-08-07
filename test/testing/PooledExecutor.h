// A reconfigurable Executor for thread-count sweeps.
//
// SupernodalLU/LeftRightLU hold their Executor BY VALUE, and the only hook for
// changing it after construction is assigning through the executor() accessor.
// `StdThreadExecutor` owns its pool, deletes its copy operations and declares a
// destructor (so it has no move operations either) -- it therefore cannot be
// assigned, and its thread count is fixed at construction. That makes a sweep
// over thread counts impossible with it directly; the README says as much and
// suggests "a custom executor wrapping a shared pool", which is what this is.
//
// Holding the pool by shared_ptr makes the wrapper default-constructible,
// copyable and assignable while the pool itself stays unique and non-copied:
//
//   Eigen::SupernodalLU<Mat, Eigen::AMDOrdering<int>, lu_testing::PooledExecutor> s;
//   s.executor() = lu_testing::PooledExecutor(8);   // now actually reconfigurable
//
// The default constructor makes a one-thread pool (which spawns no threads at
// all and runs parallelFor inline), so default-constructing a solver costs
// nothing until a real pool is assigned in.

#ifndef DIRECTLUSOLVERS_TEST_TESTING_POOLEDEXECUTOR_H
#define DIRECTLUSOLVERS_TEST_TESTING_POOLEDEXECUTOR_H

#include <memory>
#include <utility>

#include "SupernodalLUExecutor.h"

namespace lu_testing {

class PooledExecutor {
 public:
  // The Executor concept's index type: inside namespace Eigen the solvers spell
  // this bare `Index`, which resolves to Eigen::Index by enclosing-namespace
  // lookup -- there is no supernodal_lu::Index to name.
  typedef Eigen::Index Index;

  // numThreads == 0 selects std::thread::hardware_concurrency(); the default of
  // 1 spawns nothing, so an unassigned solver behaves like SerialExecutor.
  explicit PooledExecutor(int numThreads = 1)
      : m_pool(std::make_shared<Eigen::supernodal_lu::StdThreadExecutor>(
            static_cast<unsigned>(numThreads < 0 ? 0 : numThreads))) {}

  int concurrency() const { return m_pool->concurrency(); }

  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    m_pool->parallelFor(begin, end, std::forward<F>(f));
  }

 private:
  std::shared_ptr<Eigen::supernodal_lu::StdThreadExecutor> m_pool;
};

}  // namespace lu_testing

#endif  // DIRECTLUSOLVERS_TEST_TESTING_POOLEDEXECUTOR_H
