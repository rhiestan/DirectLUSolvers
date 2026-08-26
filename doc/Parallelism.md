# Parallelism

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [Testing](Testing.md)*

Both `SupernodalLU` and `LeftRightLU` thread their numeric factorization through the same
pluggable `Executor`, and both parallelize `solve()`. `PointBlockLU` is deliberately
single-threaded — see [Why PointBlockLU is not parallel](PointBlockLU.md#why-pointblocklu-is-not-parallel),
which is a measurement rather than an omission. This document covers the executor backends,
the two parallel mechanisms, and every scaling measurement in the project.

Numeric factorization parallelizes two ways, both driven by the same `Executor`:

1. **Elimination-tree level parallelism.** Independent supernodes within one elimination-tree
   level factor concurrently (`levelCount()` levels total; `widestLevel()` supernodes at the
   widest — an upper bound on how much this alone can use).
2. **Intra-supernode parallelism** (`setIntraSupernodeParallelism`, on by default). Chunks a
   single big supernode's GEMM/TRSM work across the executor when a level is too narrow to keep
   the machine busy on its own — this is what breaks the "serial tail" of the few huge
   root-separator supernodes and is responsible for most of the speedup on well-separated
   matrices (measured 3.21x on a 30³ 3D Laplacian at 32 threads, versus 1.15x
   from level parallelism alone; see [Parallel scaling](#parallel-scaling-measured)).

`LeftRightLU` has the same two knobs in a different shape: a barrier-free DAG in place of the
level barrier, and the same `setIntraSupernodeParallelism` switch, which there carves the narrow
top levels out of the DAG phase and sweeps them afterwards with the pool applied inside each
supernode. The split matters for the same reason and by the same order (1.18x → 3.08x on the
same matrix).

The `Executor` concept (`SupernodalLUExecutor.h`) is two methods:

```cpp
template <class F> void parallelFor(Index begin, Index end, F&& f) const;  // run f(i) for every i in [begin,end)
int concurrency() const;                                                    // worker lanes, >= 1
```

Four backends are provided:

| Executor | Header | Dependency | Notes |
|---|---|---|---|
| `supernodal_lu::SerialExecutor` | `SupernodalLU.h` (bundled) | none | Default. No threading. |
| `supernodal_lu::StdThreadExecutor` | `SupernodalLUExecutor.h` (bundled) | `<thread>` | Persistent `std::thread` pool, fork-join, dynamic work-stealing. Thread count fixed at construction (default `hardware_concurrency()`); the instance is non-copyable **and** non-movable, so it cannot be reconfigured via `solver.executor() = ...` — use `PooledExecutor` when you need that. |
| `supernodal_lu::PooledExecutor` | `SupernodalLUExecutor.h` (bundled) | `<thread>` | The same pool held through a `shared_ptr`, which makes it copyable and assignable: `solver.executor() = PooledExecutor(8)` works, and copies share one pool rather than spawning a second. Default-constructs to a single thread (spawns nothing). Use this for a runtime-chosen thread count. |
| `supernodal_lu::OpenMPExecutor` | `SupernodalLUExecutorOpenMP.h` | OpenMP runtime | See [below](#openmpexecutor). |
| `supernodal_lu::TBBExecutor` | `SupernodalLUExecutorTBB.h` | oneAPI TBB | See [below](#tbbexecutor). |

```cpp
#include <SupernodalLUExecutor.h>   // pulled in transitively by SupernodalLU.h too

Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>,
                    Eigen::supernodal_lu::StdThreadExecutor> solver;   // default-constructs an N-thread pool
solver.compute(A);
```

All four executors give **numerically consistent** results for a fixed thread count (whether or
not they're bit-identical to the serial factorization depends on `setIntraSupernodeParallelism`,
see above) — pick whichever backend fits how the rest of your application is threaded. This is
checked rather than asserted; see [Testing the executor backends](Testing.md#testing-the-executor-backends).

## `OpenMPExecutor`

```cpp
#include <SupernodalLUExecutorOpenMP.h>

Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>,
                    Eigen::supernodal_lu::OpenMPExecutor> solver;
solver.compute(A);
```

Full implementation (`src/SupernodalLUExecutorOpenMP.h`):

```cpp
class OpenMPExecutor {
 public:
  // numThreads == 0 (default) uses the OpenMP runtime's own current default
  // (omp_get_max_threads()). A positive value overrides the thread count for
  // every parallelFor() issued through this instance (via `num_threads`)
  // without touching the runtime's ambient default.
  explicit OpenMPExecutor(int numThreads = 0) : m_numThreads(numThreads) {}

  int concurrency() const { return m_numThreads > 0 ? m_numThreads : omp_get_max_threads(); }

  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    if (end <= begin) return;
    const int threads = concurrency();
    if (threads <= 1 || end - begin == 1) {
      for (Index i = begin; i < end; ++i) f(i);
      return;
    }
    // schedule(dynamic): per-index costs are highly non-uniform (a huge root
    // supernode next to tiny leaves), so single-index dynamic grabs balance
    // load far better than OpenMP's default static, evenly-sized chunking.
#pragma omp parallel for num_threads(threads) schedule(dynamic) default(shared)
    for (Index i = begin; i < end; ++i) f(i);
  }

 private:
  int m_numThreads;
};
```

It does not own a thread pool itself — it drives whichever pool the OpenMP runtime maintains
(created lazily on first use, then kept warm), so it composes with other OpenMP code in the same
process without oversubscription.

**Build** (needs OpenMP enabled and the runtime linked):

```sh
# clang++, GNU driver
clang++ -std=c++17 -O2 -fopenmp -I eigen -I DirectLUSolvers/src your_code.cpp -o your_binary
# clang-cl / MSVC (Windows)
clang-cl /std:c++17 /O2 /openmp /I eigen /I DirectLUSolvers/src your_code.cpp
cl /std:c++17 /O2 /openmp /I eigen /I DirectLUSolvers/src your_code.cpp
# GCC
g++ -std=c++17 -O2 -fopenmp -I eigen -I DirectLUSolvers/src your_code.cpp -o your_binary
```

On Windows with clang's `-fopenmp`, the runtime is linked dynamically — make sure `libomp.dll`
(ships alongside `clang++.exe` in the LLVM install's `bin/`) is on `PATH` at runtime.
Verified: `clang++ -fopenmp`, LLVM 22, gave a bit-exact-vs-serial solution (agreement `~1.7e-15`,
consistent with the intra-supernode-parallelism reassociation caveat above) and a 1.5x speedup
on a 14400-row 2D Laplacian with 32 threads.

## `TBBExecutor`

```cpp
#include <SupernodalLUExecutorTBB.h>

Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>,
                    Eigen::supernodal_lu::TBBExecutor> solver;
solver.compute(A);
```

Full implementation (`src/SupernodalLUExecutorTBB.h`):

```cpp
class TBBExecutor {
 public:
  // maxThreads == 0 (default) leaves TBB's ambient concurrency untouched. A
  // positive value installs a tbb::global_control capping TBB's arena to that
  // many threads for the lifetime of this TBBExecutor (oneTBB's documented
  // way to bound concurrency; there is no per-call thread-count argument).
  // The control block is held by shared_ptr (global_control has a
  // user-declared destructor but no move/copy semantics of its own) so
  // TBBExecutor stays cheaply copyable/assignable -- needed because
  // SupernodalLU's only executor-reconfiguration hook is assigning through
  // the mutable executor() accessor: `solver.executor() = TBBExecutor(n);`.
  explicit TBBExecutor(int maxThreads = 0) {
    if (maxThreads > 0)
      m_control = std::make_shared<oneapi::tbb::global_control>(
          oneapi::tbb::global_control::max_allowed_parallelism, static_cast<std::size_t>(maxThreads));
  }

  // active_value(), NOT info::default_concurrency(): the latter is the
  // platform's static default and does not reflect a currently active
  // global_control cap, which is what callers actually want to know.
  int concurrency() const {
    return static_cast<int>(
        oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism));
  }

  template <typename F>
  void parallelFor(Index begin, Index end, F&& f) const {
    if (end <= begin) return;
    // grainsize 1 + simple_partitioner: always split down to one index per
    // task, so TBB's work-stealing scheduler can balance the same
    // non-uniform per-supernode costs OpenMPExecutor handles with
    // schedule(dynamic).
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
```

Like `OpenMPExecutor`, this does not own a private pool: oneTBB maintains one process-wide
worker arena shared by every `TBBExecutor` and any other TBB-based code linked into the
application (e.g. oneMKL built with the TBB threading layer), so it composes without
oversubscribing the machine.

**Build** (needs the oneTBB headers and `tbb12`/`tbb` linked):

```sh
clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src -I <tbb>/include \
    your_code.cpp -o your_binary -L <tbb>/lib -ltbb12
```

with `<tbb>` your oneTBB install root (e.g. the oneAPI base toolkit's
`tbb/<version>/` directory). At runtime, `tbb12.dll` (Windows) / `libtbb.so.12` (Linux) must be
locatable (`PATH` / `LD_LIBRARY_PATH`). Verified against oneTBB 2022.0: bit-exact-vs-serial
agreement (`~1.7e-15`, same reassociation caveat as above) and a 1.4x speedup on the same
14400-row Laplacian/32-thread case; `solver.executor() = TBBExecutor(4)` correctly reconfigures
`concurrency()` afterward (confirmed 4, then 2, across two reassignments).

## Parallel scaling (measured)

`bench_parallel` sweeps thread counts and times each phase separately, with each
solver's intra-supernode mechanism toggled on and off, so every row isolates one
mechanism. Each row is warmed up before timing — see the note after the table for
why that is not optional here.

```sh
./build/bench_parallel                            # the built-in scaling set
./build/bench_parallel --threads 1,4,16 --reps 5
./build/bench_parallel --quick                    # synthetic matrices only
```

Measured 2026-08-22 on an AMD Ryzen 9 5950X (16 physical cores, 32 logical),
`StdThreadExecutor` via `PooledExecutor`, clang 22 `-O3` at the default x86-64
ISA, best of 5 after a discarded warm-up, AMD ordering. Times in ms; "speedup"
is 1t → 32t.

The `lap*` names are the synthetic Laplacians from `test/testing/TestMatrices.h`,
and the superscript is the grid exponent, not a footnote marker: `lap3d_30³` is
the 30×30×30 3D Laplacian (27000 rows, `lap3d_30x30x30` in the benchmark output)
and `lap2d_300²` the 300×300 2D one (90000 rows, `lap2d_300x300`). `laoss_1` and
`laoss_2` are real 3D FEM systems from `testdata/`.

Each solver contributes two rows, one per mechanism: for `SupernodalLU`,
elimination-tree levels alone and levels plus intra-supernode chunking; for
`LeftRightLU`, the barrier-free DAG alone and the DAG plus the chunked tail
sweep. The second row of each pair is the shipping default.

| matrix | phase | 1t | 8t | 16t | 32t | speedup |
|---|---|---:|---:|---:|---:|---:|
| `laoss_1` (251k) | analyze (symbolic) | 643 | 634 | 620 | 634 | 1.02x  (serial) |
| | SNLU factor, levels only | 2102 | 1391 | 1346 | 1348 | 1.56x |
| | SNLU factor, levels+intra | 2104 | 958 | 921 | 808 | **2.60x** |
| | LRLU factor, DAG only | 2110 | 1254 | 1225 | 1207 | 1.75x |
| | LRLU factor, DAG+intra | 2119 | 838 | 770 | 714 | **2.97x** |
| | solve, 1 rhs | 72.1 | 41.3 | 37.2 | 37.8 | **1.91x** |
| | solve, 8 rhs | 210 | 118 | 108 | 107 | **1.96x** |
| `laoss_2` (100k) | SNLU factor, levels only | 531 | 339 | 326 | 320 | 1.66x |
| | SNLU factor, levels+intra | 530 | 262 | 224 | 212 | **2.50x** |
| | LRLU factor, DAG only | 533 | 296 | 287 | 297 | 1.79x |
| | LRLU factor, DAG+intra | 536 | 228 | 215 | 222 | **2.41x** |
| `lap3d_30³` | SNLU factor, levels only | 603 | 528 | 527 | 523 | 1.15x |
| | SNLU factor, levels+intra | 599 | 258 | 217 | 187 | **3.21x** |
| | LRLU factor, DAG only | 621 | 529 | 527 | 526 | 1.18x |
| | LRLU factor, DAG+intra | 627 | 261 | 244 | 204 | **3.08x** |
| `lap2d_300²` | SNLU factor, levels only | 109 | 66.8 | 63.0 | 62.5 | 1.75x |
| | SNLU factor, levels+intra | 108 | 57.7 | 52.5 | 53.1 | **2.04x** |
| | LRLU factor, DAG only | 116 | 59.9 | 52.7 | 50.7 | **2.28x** |
| | LRLU factor, DAG+intra | 116 | 58.4 | 52.8 | 61.1 | 1.90x |

Five things this says, none of them visible from a single-thread-count timing:

1. **`solve()` parallelizes too** (~1.9x) — see [Parallel triangular
   solve](#parallel-triangular-solve). Even so it is only 3-7% of a
   factor+solve here, so this matters most when you factor once and solve many
   times.
2. **`analyzePattern()` is serial and is often the *largest* remaining term.**
   At 32 threads it is 43% of a single factor+solve on `laoss_1`, 48% on
   `laoss_2`, and 56% on `lap2d_300²` — more than the factorization it feeds.
   Improving factorization scaling further buys little until this moves.
3. **Parallelism INSIDE a supernode is the mechanism that pays, in both
   solvers.** On the 3D Laplacian, levels alone give 1.15x and the DAG alone
   1.18x; adding intra-supernode chunking takes them to 3.21x and 3.08x. Neither
   across-supernode schedule exceeded 1.79x on any matrix here. See [Chunk
   sizing](#chunk-sizing) for how the chunk extent is picked.
4. **The two solvers now land in the same place on 3D**, because `LeftRightLU`
   gained a chunked tail sweep of its own — the earlier version of this table
   showed it stuck at 1.19x on `lap3d_30³` for exactly the reason it no longer
   is. On the wide 2D tree the tail sweep is a *cost*, not a gain (1.90x with it
   against 2.28x without at 32 threads): there the levels it carves had real
   inter-supernode parallelism to give up, and the carve is a hard phase
   boundary in an otherwise barrier-free schedule.
5. **Peak is at 16 threads about as often as at 32.** Half the rows above are
   flat or slightly worse from 16t to 32t — the second SMT thread on each core
   adds no memory bandwidth, and bandwidth is what binds (see [The machine
   ceiling](#the-machine-ceiling)). Most of the available gain arrives by 8
   threads.

**On reproducing these numbers.** `bench_parallel` discards a warm-up run before
timing each row, and that is load-bearing rather than hygiene: on this machine a
factorization measured after earlier heavy work in the same process runs 9% (32
lanes) to 49% (2 lanes) slower than the first one in a fresh process, and stays
slow — it is not thermal, and a 10-second idle gap does not restore it. Without
the warm-up the row measured first reports a time no later row can reach, which
silently flatters whichever mechanism the sweep happens to try first. Run-to-run
spread with the warm-up is 1-3% on the synthetic matrices and up to ~15% on
`laoss_1`'s 32-thread cells.

### The machine ceiling

In-solver speedup confounds two very different limits: how much parallelism the
schedule exposes, and how much the machine can deliver. `bench_ceiling`
separates them by running **K independent single-threaded factorizations
concurrently** — they share no lock, no queue and no data, so their parallelism
is perfect by construction and the only contended resource is memory. Whatever
scaling that reaches is an upper bound on what *any* scheduler could achieve.

Measured 2026-08-22 on an AMD Ryzen 9 5950X (**16 physical cores**, 32 logical,
dual-channel DDR4):

| matrix | factor size | K=1 | K=8 | K=16 | K=32 |
|---|---|---:|---:|---:|---:|
| `lap3d_30³` | 92 MB | 1.00x | 4.33x | **7.69x** | 11.40x |
| `laoss_2` | 154 MB | 1.00x | 4.62x | **7.65x** | 10.46x |
| `laoss_1` | 463 MB | 1.00x | 4.56x | **7.71x** | 10.68x |

**Roughly half of the cores' throughput is already gone before our code does
anything.** With perfect, embarrassingly-parallel work, 16 cores deliver only
~7.7x. The absolute rate also falls with footprint — 16.1 GFLOP/s per core solo
on the 92 MB case versus 12.9 on the 154 MB one and 12.3 on the 463 MB one —
which is the signature of a memory-bound workload, as expected for a factor that
does not fit in L3. The K=32 column shows what the extra SMT thread per core is
worth here: 16 → 32 buys 1.35-1.48x, not 2x, and only because two threads on one
core cover each other's memory stalls.

So `laoss_1`'s 2.60x (`SupernodalLU`) and 2.97x (`LeftRightLU`) should be read
against **~7.7x, not against 32**. Two multiplicative limits produce them:

1. **Hardware**: 16 cores behave like ~7.7 for this workload (48% efficiency).
2. **Schedule**: the elimination tree simply does not offer 16 independent
   lanes' worth of work near the root, which is what the chunked tail sweep
   exists to patch and only partly can.

In absolute terms, `laoss_1` factors 26.6 GFLOP in 714 ms at 32 threads
(`LeftRightLU`) — 37.2 GFLOP/s, against the 130.9 GFLOP/s the same machine
delivers on 32 independent copies of that factorization and the 94.5 GFLOP/s it
delivers on 16. Read the shortfall as schedule, not as kernel: the kernels are
the same Eigen GEMMs in both measurements.

The practical consequences:

- **There is real headroom on `laoss_1` — about 3x, not 12x.** Anyone planning
  around these solvers should size expectations to the ceiling table, not to the
  core count.
- **Further gains must come from moving less memory**, not from more threads:
  better cache blocking and reuse in the update GEMMs. More scheduling
  sophistication cannot beat 7.7x.
- Fork-join dispatch costs **25-86 µs at 8-32 threads**, which is expensive in
  absolute terms, but a factorization issues only a few hundred dispatches
  (359 for `laoss_1`), so it accounts for ~2% here. Worth fixing eventually, not
  the bottleneck.
- Only **84 of `laoss_1`'s 49928 supernodes** run in inner (chunked) mode at 32
  lanes — and they carry **76% of the factorization time**. The other 24% sits in
  outer levels whose balance is whatever the level schedule gives. The same
  measurement at 16 lanes: 24 supernodes, 41% of the time; at 8 lanes, 22
  supernodes, 35%. That the chunked share *grows* with the lane count is the
  mechanism working as intended — more lanes make more levels too narrow to fill
  outer-mode, so more of them switch.

```sh
./build/bench_ceiling            # full: synthetic + the laoss matrices
./build/bench_ceiling --quick    # synthetic only
```

Re-run it on your own hardware before reading anything into a speedup number;
the ceiling is machine-specific and a dual-socket server with more memory
channels will land somewhere quite different.

### Parallel triangular solve

Both triangular sweeps dispatch over elimination-tree levels
(`setParallelSolve`, on by default) rather than running on the calling thread.

The two sweeps are not symmetric, which is the whole difficulty:

- The **backward** sweep is already a *gather*: supernode `s` reads rows owned by
  higher-numbered supernodes and writes only its own head. Levels visited from
  the root down parallelize with no restructuring.
- The **forward** sweep is a *scatter*: `s` pushes its solved head into its
  ancestors' rows. Two supernodes in one level can own row blocks facing a
  common ancestor, so running a level concurrently would race on the same rows
  of `y`. The parallel path therefore uses the equivalent **gather** form —
  each supernode pulls from its already-finished sources via the same
  `m_updateSources` structure the factorization uses, and writes only its own
  rows.

Because sources are stored in ascending order, the gather applies them in the
same order the scatter did, so **every element accumulates identically and the
result is bit-identical to the serial sweep** — `test_parallel_lu` asserts
exactly that (`maxDiff == 0.0`, not a tolerance), since a reformulation that
quietly reassociated would still pass a tolerance check.

| matrix | phase | 1t | 8t | 16t | 32t | speedup |
|---|---|---:|---:|---:|---:|---:|
| `laoss_1` | solve, 1 rhs | 72.1 | 41.3 | 37.2 | 37.8 | **1.91x** |
| | solve, 8 rhs | 210 | 118 | 108 | 107 | **1.96x** |

Scaling stops near 16 threads: the elimination tree narrows towards the root, so
the last levels hold one supernode and run inline. Below `rows × nrhs = 200000`
the sweeps stay serial regardless — a fork-join dispatch costs tens of
microseconds and a sweep issues one per level, so on a small system the
dispatches would cost more than the substitution they parallelize.

Note the honest scale: solve is only **3-7%** of a factor+solve on `laoss_1`, so
this is worth ~2-4% end-to-end. It matters when you factor once and solve
repeatedly, which is the case the solver is built for.

### Chunk sizing

The chunk extent is `clamp(ceil(offDiagonalRows / lanes), 32, 128)` rows — about
one chunk per lane, floored so a chunk stays thick enough to amortize the BLAS
call and the per-chunk walk over the target's update sources, and capped at 128
so tall panels still produce several chunks for load balance.

Scaling the extent with the lane count is what makes the mechanism pay. A fixed
128-row chunk would yield `ceil(offDiagonalRows / 128)` chunks **regardless of
how many threads exist**: on this project's matrices the heaviest supernodes
carry ~1400-1650 off-diagonal rows, so at 32 lanes they would split into only
11-13 chunks and 20 of 32 threads would idle through precisely the supernodes
that dominate the factorization — measured at 32 lanes, the chunked supernodes
account for 80% of `lap3d_30³`'s factorize time and 76% of `laoss_1`'s.

| matrix | factor (levels+intra) at 32t | speedup 1→32 |
|---|---:|---|
| `lap3d_30³` | 187 ms | **3.21x** |
| `laoss_2` (100k) | 212 ms | **2.50x** |
| `lap2d_300²` | 53.1 ms | 2.04x |
| `laoss_1` (251k) | 808 ms | **2.60x** |

`lap2d_300²` is where this pays least: its elimination tree is wide enough that
level parallelism already fills the lanes, so the chunked path has little left to
add (and for `LeftRightLU`, carving the tail actively costs — see [Work-stealing
ready queue](#work-stealing-ready-queue)). On the 3D matrices the chunked path
carries the majority of the factorization: at 32 lanes it accounts for 76% of
`laoss_1`'s factorize time and 80% of `lap3d_30³`'s, in 84 and 31 supernodes
respectively.

### Work-stealing ready queue

`LeftRightLU`'s dynamic scheduler holds its ready nodes in **per-worker deques
with work stealing**: a worker pushes the consumers it readied onto its own back
and pops from its own back, so the node whose data is hot in that core's cache
is the node it takes next; only when its deque runs dry does it steal, from the
*front* of a victim — the entry furthest from the victim's hot end, so steals
rarely collide with the owner and tend to move a coarse subtree rather than a
leaf. Idle workers park on a shared condition variable with a bounded wait, and
producers skip the notify entirely unless someone is actually parked.

The obvious alternative — **one** global mutex-guarded ready stack with a
`notify_all()` per completed supernode — serializes every task acquisition
behind a single lock, wakes all *P* workers when at most a couple can proceed,
and defeats the depth-first subtree affinity the LIFO exists to provide (a
worker's freshly readied children land on the shared stack, where any other
worker takes them immediately). Measured, it is slow enough at high thread
counts to make `lap2d_300²` and `lap3d_30³` run *worse* at 32 threads than at
16.

Measured 2026-08-22 (same setup as the table above), LRLU factorization with
the tail sweep OFF, so this row isolates the DAG and its queue:

| matrix | 16t | 32t | speedup 1→32 |
|---|---:|---:|---|
| `lap2d_300²` | 52.7 ms | **50.7 ms** | **2.28x** |
| `lap3d_30³` | 527 ms | 526 ms | 1.18x |
| `laoss_2` (100k) | 287 ms | 297 ms | 1.79x |
| `laoss_1` (251k) | 1225 ms | 1207 ms | 1.75x |

Read this honestly: **queue design decides the outcome only where tasks are fine
grained.** `lap2d_300²` factors in ~116 ms across 32919 supernodes, so per-task
queue overhead is a large fraction of task cost and queue throughput dominates;
`laoss_1` spends 2.1 s over 49928 supernodes, so its tasks are far coarser and
the queue is not the limiter there.

The `lap3d_30³` row's 1.18x is therefore **not** a queue problem, and the fix
was not a queue change. Near the root the DAG narrows to a chain of separator
supernodes, so a scheduler that only parallelizes ACROSS supernodes runs out of
work: VTune's threading analysis of this configuration measures 1.18 of the 8
lanes given and 1.32 of 32 — adding 24 lanes buys 0.14 of a lane — and puts the
idle time in **starvation**, not contention (the per-worker deque locks take
0.04 s of wait across ~8500 acquisitions in a 4.6 s factorization).

`LeftRightLU` therefore now carves those narrow top levels out of the DAG phase
and sweeps them afterwards with the pool applied *inside* each supernode, under
the same `setIntraSupernodeParallelism` switch `SupernodalLU` uses. The split
into two phases is what makes it possible at all: the executor's `parallelFor`
is fork-join and **not nestable**, and during the DAG phase every lane is
already inside one, so ending the parallel region is what frees the pool. With
it on, the same measurement reaches 2.72 of 8 lanes and 6.39 of 32, and
`lap3d_30³` goes 1.18x → **3.08x**.

The cost is real and shows up on the wide 2D tree, where the carved levels *did*
have inter-supernode parallelism worth having and the tail sweep is a hard phase
boundary in an otherwise barrier-free schedule: `lap2d_300²` at 32 threads is
50.7 ms with the sweep off and 61.1 ms with it on. `setIntraSupernodeParallelism(false)`
is the switch if your matrices look like that one.

