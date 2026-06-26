// SupernodalLU - pluggable parallel execution backends.
//
// The numeric factorization parallelizes over the elimination tree using a
// level-set schedule: supernodes in the same level are independent and are
// dispatched through an Executor's parallelFor(begin, end, f), which must run
// f(i) for every i in [begin, end) and act as a barrier on return.
//
// The Executor is a compile-time policy (the 3rd template parameter of
// SupernodalLU). The concept it must satisfy is a single method:
//
//     template <class F> void parallelFor(Index begin, Index end, F&& f) const;
//     int concurrency() const;   // number of worker lanes (>= 1)
//
// SerialExecutor (the default) keeps the solver dependency-free. StdThreadExecutor
// is a ready-to-use std::thread pool. Backends such as oneTBB or OpenMP are a few
// lines: forward parallelFor to tbb::parallel_for / a `#pragma omp parallel for`.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_EXECUTOR_H
#define SUPERNODAL_LU_EXECUTOR_H

#include <Eigen/Core>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace Eigen {
namespace supernodal_lu {

// Default backend: runs the loop on the calling thread. Empty, trivially
// copyable, and pulls in no threading dependencies.
struct SerialExecutor {
  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    for (Index i = begin; i < end; ++i) f(i);
  }
  int concurrency() const { return 1; }
};

// Reference multithreaded backend: a persistent pool of std::thread workers with
// a fork-join, dynamically load-balanced parallelFor. Non-copyable (owns the
// pool); construct one per solver, or write an Executor that references a shared
// pool if you prefer.
class StdThreadExecutor {
 public:
  // numThreads == 0 selects std::thread::hardware_concurrency().
  explicit StdThreadExecutor(unsigned numThreads = 0) { start(numThreads); }
  ~StdThreadExecutor() { stop(); }

  StdThreadExecutor(const StdThreadExecutor&) = delete;
  StdThreadExecutor& operator=(const StdThreadExecutor&) = delete;

  int concurrency() const { return static_cast<int>(m_numThreads); }

  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    if (end <= begin) return;
    if (m_numThreads <= 1) {
      for (Index i = begin; i < end; ++i) f(i);
      return;
    }
    // Type-erase the body so the worker threads can share it for this call.
    std::function<void(Index)> body = [&f](Index i) { f(i); };
    dispatch(begin, end, body);
  }

 private:
  void start(unsigned numThreads) {
    if (numThreads == 0) numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;
    m_numThreads = numThreads;
    for (unsigned i = 1; i < numThreads; ++i)  // the calling thread is lane 0
      m_threads.emplace_back([this] { workerLoop(); });
  }

  void stop() {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_stop = true;
      ++m_generation;
    }
    m_workAvailable.notify_all();
    for (std::thread& t : m_threads) t.join();
    m_threads.clear();
  }

  // Grab indices one at a time (dynamic balancing); f(i) is heavy enough that
  // single-index granularity is fine.
  void processChunks(const std::function<void(Index)>& f) const {
    for (;;) {
      const Index i = m_next.fetch_add(1, std::memory_order_relaxed);
      if (i >= m_end) break;
      f(i);
    }
  }

  void dispatch(Index begin, Index end, const std::function<void(Index)>& body) const {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_body = &body;
      m_next.store(begin, std::memory_order_relaxed);
      m_end = end;
      m_activeWorkers = m_numThreads - 1;
      ++m_generation;
    }
    m_workAvailable.notify_all();
    processChunks(body);  // the calling thread participates
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_allDone.wait(lock, [this] { return m_activeWorkers == 0; });
      m_body = nullptr;
    }
  }

  void workerLoop() {
    std::uint64_t lastGeneration = 0;
    for (;;) {
      const std::function<void(Index)>* body = nullptr;
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_workAvailable.wait(lock, [this, lastGeneration] {
          return m_stop || m_generation != lastGeneration;
        });
        if (m_stop) return;
        lastGeneration = m_generation;
        body = m_body;
      }
      processChunks(*body);
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (--m_activeWorkers == 0) m_allDone.notify_one();
      }
    }
  }

  unsigned m_numThreads = 1;
  std::vector<std::thread> m_threads;

  mutable std::mutex m_mutex;
  mutable std::condition_variable m_workAvailable;
  mutable std::condition_variable m_allDone;
  mutable const std::function<void(Index)>* m_body = nullptr;
  mutable std::atomic<Index> m_next{0};
  mutable Index m_end = 0;
  mutable unsigned m_activeWorkers = 0;
  mutable std::uint64_t m_generation = 0;
  bool m_stop = false;
};

}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_EXECUTOR_H
