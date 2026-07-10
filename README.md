# SupernodalLU

A header-only, supernodal sparse **direct LU** solver for [Eigen](https://eigen.tuxfamily.org),
built as a template in the style of `Eigen::SparseLU`, but using the algorithmic design of
[PaStiX](https://gitlab.inria.fr/solverstack/pastix) rather than SuperLU: a fully precomputed
*static* symbolic block structure, **static pivoting** with iterative refinement instead of
partial pivoting, and BLAS-3 supernodal kernels that parallelize cleanly over the elimination
tree. See `pastix_algorithms.md` (repository root) for the algorithmic background this design
is based on.

```cpp
#include <SupernodalLU.h>

Eigen::SupernodalLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);                 // A: general values, SYMMETRIC nonzero pattern
Eigen::VectorXd x = solver.solve(b);
```

> **Sibling solver: [`LeftRightLU`](#leftrightlu--pardiso-style-sibling-solver).** This
> directory also ships `Eigen::LeftRightLU`, a same-interface solver built on the
> algorithmic design of **PARDISO** instead of PaStiX: a *left-right-looking* numeric
> factorization driven by a **barrier-free dynamic scheduler**, with in-block **complete
> pivoting**. It reuses this module's analysis/solve pipeline and shared headers; see the
> [LeftRightLU section](#leftrightlu--pardiso-style-sibling-solver) below and
> `pardiso_algorithms.md` for the design background.

## Scope — read this first

SupernodalLU factors **`A = P^T L U P`** for matrices with **general (unsymmetric) values but a
symmetric nonzero pattern** — i.e. `A(i,j) != 0 <=> A(j,i) != 0`, in *pattern* only, not value.
If your matrix's pattern isn't already symmetric, symmetrize it with explicit structural zeros
first (`compare_testdata.cpp`'s `symmetrizePattern` shows how); this does not change the
operator, since a genuine zero contributes nothing to the sum. Cholesky/LDLᵀ are **not**
implemented (LU only, for now).

The whole design rests on one decision: **no row interchanges**. Once `analyzePattern()` has
built the supernode/block structure it never changes again. Tiny or zero pivots are instead
*bumped* to a threshold (static pivoting) and the resulting error is cleaned up by refinement
during `solve()`. This is what keeps the factorization static-structure, BLAS-3, and
tree-parallel — the same trade-off PaStiX makes, and the reason this solver is not a drop-in
replacement for `Eigen::SparseLU`'s partial-pivoting robustness on badly unsymmetric matrices
without help (see [Matching & diagonal pivoting](#matching--diagonal-pivoting-robustness) below,
which recovers most of that robustness without giving up staticness).

**When it's a good fit:** symmetric-pattern matrices (physical PDE discretizations, circuit
matrices, many FEM/FVM systems) where you want direct-solver accuracy, need `solve()` for many
right-hand sides against one factorization, and can benefit from tree/BLAS parallelism.

**When it isn't:** matrices whose pattern is fundamentally unsymmetric with no natural
symmetrization (SupernodalLU forces one, which adds fill `Eigen::SparseLU`'s column-pivoted
COLAMD path wouldn't need); or huge (hundreds of thousands of rows) irregular matrices, where
static symmetric-pattern fill can become very large very fast — `Eigen::SparseLU` or an
external solver (PARDISO, PaStiX itself) generally wins there. Concretely, on this project's
`testdata/` benchmarks SupernodalLU is typically 2-3x slower and denser than `Eigen::SparseLU`
(the gap is fill, driven by symmetric-pattern elimination, not solver overhead) — but it is
**more robust** on some structurally-difficult matrices (e.g. `bcsstm13`, a singular mass
matrix `Eigen::SparseLU` fails outright on) thanks to matching + static pivoting + refinement.
Benchmark your own matrices with `DirectLUSolvers/test/compare_testdata.cpp` before committing.

## Files

| File | Contents |
|---|---|
| `src/SupernodalLU.h` | The solver itself. No dependencies beyond Eigen — always safe to include. |
| `src/SupernodalLU` | Eigen-style umbrella header, `#include <SupernodalLU>` (no extension), forwards to `SupernodalLU.h`. |
| `src/LeftRightLU.h` | The PARDISO-style sibling solver (see [below](#leftrightlu--pardiso-style-sibling-solver)). Reuses the shared support/matching/executor headers; self-contained otherwise. |
| `src/LeftRightLU` | Umbrella header for `LeftRightLU`, `#include <LeftRightLU>`. |
| `src/SupernodalLUSupport.h` | Plain data structures shared by the analysis/factorization phases (`Supernode`, `RowBlock`, `UpdateSource`). |
| `src/SupernodalLUMatching.h` | The maximum-transversal matching + permutation-sign helpers used by `setMatching()`. |
| `src/SupernodalLUExecutor.h` | The `Executor` concept, plus the bundled `SerialExecutor` and `StdThreadExecutor` backends. No dependency beyond `<thread>`. |
| `src/SupernodalLUExecutorOpenMP.h` | `OpenMPExecutor` — optional, requires an OpenMP-enabled build (see [below](#openmpexecutor)). |
| `src/SupernodalLUExecutorTBB.h` | `TBBExecutor` — optional, requires oneAPI Threading Building Blocks (see [below](#tbbexecutor)). |
| `src/SupernodalLUMetis.h` | `SupernodalLUMetis<Mat[,Executor]>` alias wiring in METIS nested dissection. Optional, requires METIS + GKlib. |
| `src/SupernodalLUAutoOrdering.h` | `SupernodalLUAuto<Mat[,Executor]>` alias: tries AMD and several METIS restarts, keeps the least-fill one. Optional, requires METIS + GKlib. |
| `test/test_supernodal_lu.cpp` | Correctness tests (dependency-free — only needs Eigen). |
| `test/test_leftright_lu.cpp` | `LeftRightLU` correctness tests (dependency-free; `-pthread` for the parallel-vs-serial test). |
| `test/test_parallel_lu.cpp` | Parallel-vs-serial agreement + speedup, using `StdThreadExecutor`. |
| `test/compare_testdata.cpp` | Benchmark harness comparing SupernodalLU (AMD/METIS/Auto) against `Eigen::SparseLU` and, optionally, MKL PARDISO, on the matrices in `testdata/`. |

## Requirements

The core solver (`SupernodalLU.h`, `SupernodalLUExecutor.h`, `SupernodalLUMatching.h`,
`SupernodalLUSupport.h`) needs only Eigen and a C++17 compiler — no external dependencies, no
linking beyond your usual Eigen setup. Everything else in the table above is an **opt-in**
header that pulls in one extra dependency, listed per-header below. This mirrors Eigen's own
`*Support` module convention: the base solver stays dependency-free so you only pay for what
you use.

## Quick start

```cpp
#include <Eigen/SparseCore>
#include <SupernodalLU.h>

using Eigen::SparseMatrix;

SparseMatrix<double> A = /* symmetric-pattern, general values */;
Eigen::VectorXd b = /* right-hand side */;

Eigen::SupernodalLU<SparseMatrix<double>> solver;
solver.compute(A);                 // analyzePattern(A) + factorize(A)
if (solver.info() != Eigen::Success) {
  // factorization broke down (structurally singular with matching off, etc.)
  std::cerr << solver.lastErrorMessage() << "\n";
}
Eigen::VectorXd x = solver.solve(b);
if (solver.info() != Eigen::Success) {
  // solve() itself measures the true residual and downgrades info() if it's
  // bad (see solveFailureThreshold below) -- never trust a solve blindly.
}
```

`compute()` is just `analyzePattern(A)` followed by `factorize(A)`; call them separately when
you need to refactor the same pattern with new values (skips the symbolic analysis):

```cpp
solver.analyzePattern(A);
solver.factorize(A);               // ... later, same pattern, new values:
solver.factorize(A2);
```

## The three-phase pipeline

1. **`analyzePattern(A)`** — *symbolic, values-free.* Runs maximum-transversal matching (moves
   large entries onto the diagonal), the fill-reducing ordering, elimination-tree + postorder
   computation, supernode detection (with amalgamation), and the block symbolic factorization.
   Everything here depends only on the sparsity pattern.
2. **`factorize(A)`** — *numeric.* Computes row/column equilibration, scatters `A`'s values into
   the (now fixed) supernode panels, and runs the left-looking supernodal numeric sweep:
   diagonal-block LU with static + restricted in-block pivoting, then the off-diagonal panel
   solves, dispatched over elimination-tree levels through the `Executor`.
3. **`solve(b)`** — block forward/backward substitution against the stored factors, followed by
   automatic iterative refinement (stationary or Krylov) to recover the accuracy lost to static
   pivoting, then an honesty check that measures the true residual and flags `NumericalIssue` if
   it's still too large.

## Template parameters

```cpp
template <typename MatrixType_,
          typename OrderingType_ = AMDOrdering<typename MatrixType_::StorageIndex>,
          typename Executor_ = supernodal_lu::SerialExecutor>
class SupernodalLU;
```

- **`MatrixType_`** — a column-major `Eigen::SparseMatrix<Scalar, ColMajor, StorageIndex>`. Any
  scalar Eigen supports (`double`, `float`, `std::complex<double>`, ...) and any `StorageIndex`
  (`int` is what this project's METIS/matching code is tested against).
- **`OrderingType_`** — the fill-reducing ordering functor, applied to the (matched) pattern
  during `analyzePattern()`. Default `Eigen::AMDOrdering<StorageIndex>`. Alternatives:
  - `Eigen::MetisOrdering<StorageIndex>` (needs `<Eigen/MetisSupport>` + METIS/GKlib) — or use
    the ready-made alias `Eigen::SupernodalLUMetis<Mat[,Executor]>` from
    `SupernodalLUMetis.h`, nested-dissection ordering. Often much better than AMD on large
    well-separated meshes; can also be *worse* than AMD on small/irregular matrices (measured on
    this project's `testdata/`: up to +200% fill on one matrix, -5% on another) — there is no
    universally correct choice.
  - `Eigen::AutoOrdering<StorageIndex>` (needs the same METIS dependency) — or the alias
    `Eigen::SupernodalLUAuto<Mat[,Executor]>` from `SupernodalLUAutoOrdering.h`. Tries AMD plus
    several deterministic METIS restarts, predicts each candidate's fill with a real (but
    values-free) symbolic pass, and keeps the cheapest. Never worse than plain AMD in testing,
    and substantially better on some matrices — at the cost of running the symbolic analysis up
    to ~4 extra times, which can dominate `analyzePattern()` on small/fast-to-factor matrices
    (measured 3-4x slower there) but is a small fraction of total time once numeric
    factorization dominates. See the header's comment for the exact seed/size policy.
- **`Executor_`** — the parallel-execution backend driving the numeric factorization's
  elimination-tree-level and intra-supernode parallelism (see [Parallelism](#parallelism)
  below). Default `supernodal_lu::SerialExecutor` (no threading at all).

## Full option reference

All setters below can be called any time before `factorize()` runs (most also work between
`compute()` calls to refactor with different settings); all queries are valid after the
operation they describe has run at least once.

### Pivoting & static-pivot threshold

- **`setStaticPivotThreshold(RealScalar t)`** / **`setPivotThreshold(t)`** (alias, matches
  `Eigen::SparseLU`'s name) — any diagonal pivot smaller in magnitude than `t` is replaced by
  a same-sign value of magnitude `t` (or exactly `t` if the pivot was zero) during
  `factorize()`. By default the threshold is chosen **automatically**, each `factorize()` call,
  as `sqrt(eps) * max|A_ij|` (of the equilibrated matrix): small enough to leave well-conditioned
  pivots untouched, large enough to step over exact zeros. Call this setter to override with a
  fixed value; pass `0` to disable replacement entirely (pure unpivoted LU — will fail on a
  genuinely singular/zero pivot).
- **`replacedPivots() -> Index`** — how many pivots were bumped by the last `factorize()`. A
  count near `n` (every pivot bumped) signals the diagonal is essentially unusable for this
  ordering and static pivoting alone can't fix it — that's what `setMatching(true)` is for.

### Refinement (recovers the accuracy lost to static pivoting)

- **`setRefinementMethod(supernodal_lu::Refinement method)`** — one of
  `Eigen::supernodal_lu::Refinement::{None, IterativeRefinement, BiCGStab}`. `None` returns the
  raw factor solve. `IterativeRefinement` is stationary refinement `x += M⁻¹(b - Ax)`.
  `BiCGStab` (**default**) is a Krylov method preconditioned by the LU factors — strictly more
  robust than stationary refinement (which can stall or diverge when many pivots were bumped)
  at negligible extra cost when the direct solve is already accurate (it returns after one
  matrix-vector product in that case).
- **`setMaxIterativeRefinements(Index iters)`** (default 5) / **`setRefinementTolerance(RealScalar tol)`**
  (default machine epsilon) — refinement stops early once the relative residual
  `‖b - Ax‖ / ‖b‖` meets `tol`, or after `iters` steps, or (for stationary refinement) if the
  residual starts growing, whichever comes first. Set `iters = 0` to disable refinement outright.
- **`iterativeRefinements() -> Index`** — steps actually taken by the *last* `solve()` call.
- **`setSolveFailureThreshold(RealScalar tol)`** (default `1e-6`) — after refinement, `solve()`
  measures the *true* relative residual against the original (unscaled) `A` and sets `info()` to
  `NumericalIssue` — instead of silently returning a bad answer — if that residual exceeds `tol`
  or the solution is non-finite. `solveResidual() -> RealScalar` reports the measured value. A
  subsequent `solve()` with a well-conditioned right-hand side against the *same* factorization
  restores `info() == Success`; use `isFactorized()`, not `info()`, to check whether the factors
  themselves are still usable (a bad solve doesn't poison later ones).

### Matching & diagonal pivoting (robustness)

- **`setMatching(bool on)`** (default **on**) — before the symbolic analysis, permutes rows so
  that large-magnitude entries land on the diagonal (MC64-style maximum-transversal matching,
  à la SuperLU_DIST/MUMPS). This is a pure row permutation — it does not touch the static block
  structure — and is the key fix for matrices with a numerically weak or structurally zero
  diagonal (unsymmetric circuit matrices, near-singular systems): without it, static pivoting
  alone can bump *every* pivot and the solve degenerates. The matched pattern is symmetrized
  internally, as this solver requires.
  `matchingIsPerfect() -> bool` reports whether the last matching found a fully zero-free
  diagonal (`false` indicates the matrix is structurally singular).
- **`setDiagonalPivoting(bool on)`** (default **on**) — factors each supernode's dense diagonal
  block with partial pivoting *confined to that block* (row swaps never leave the
  already-allocated dense panel, so the global symbolic structure — and therefore BLAS-3 shape
  and tree parallelism — is untouched). Combined with matching, this removes the "dead diagonal"
  failure mode that static pivoting alone cannot handle, without the dynamic structure growth
  true partial pivoting would require.

### Ordering & fill (amalgamation, blocking)

- **`setAmalgamation(Index relaxedSize, Index maxZeroRows)`** (default `(4, 4)`) — merges
  adjacent fundamental supernodes along elimination-tree paths into larger dense panels (better
  BLAS-3 efficiency) at the cost of a bounded number of explicit structural zeros. A merge is
  accepted when the supernode being closed is narrower than `relaxedSize` columns, **or** the
  merge adds at most `maxZeroRows` extra zero rows per column. `(1, 0)` recovers the pure
  fundamental-supernode partition (no amalgamation).
- **`setAmalgamationFillFraction(double fraction)`** (default `0.3`) — an additional *relative*
  merge rule: also accept a merge when the extra zero rows it introduces are at most `fraction`
  of the rows the supernode already carries. Matters mainly for dense-ish factorizations (a wide
  panel with hundreds of off-diagonal rows can absorb a few more essentially for free); barely
  affects sparse matrices, where the absolute rule already governs. `0` disables this rule.
- **`setAmalgamationCostModel(bool enable, double tolerance = 0.0)`** (default **off**) — an
  alternative to the two heuristics above: a machine-calibrated BLAS time model (ported from
  PaStiX's `cblk_time_fact`) predicts whether merging is actually *faster* to factor, not just
  "cheap in fill", and merges iff it is (or would be at most `tolerance` relatively slower, for
  coarser panels / better load balance at a small serial cost). Measured to *not* beat the
  tuned default heuristics on this project's matrices (the model doesn't know about our
  per-supernode bookkeeping overhead) — provided as a principled alternative, not a better
  default.
- **`setMaxBlockSize(Index maxBlockSize)`** (default `128`; `0` = unlimited) — caps supernode
  width by forcing extra boundaries. Adds **no fill** (entries beyond the cap just relocate into
  off-diagonal panels) and keeps dense panels cache-friendly; also measurably improves parallel
  load balance (finer, more uniform per-supernode tasks).

### Parallelism

- **`Executor& executor()`** / **`const Executor& executor() const`** — access the configured
  backend, e.g. to change a stateful executor's thread count:
  `solver.executor() = supernodal_lu::OpenMPExecutor(8);`. See [Parallelism](#parallelism).
- **`setIntraSupernodeParallelism(bool on)`** (default **on**) — elimination-tree levels with
  fewer supernodes than worker threads (typically the few huge separator supernodes near the
  root, which otherwise serialize) run their supernodes one at a time but parallelize *inside*
  each one instead: the Schur-update GEMMs and off-diagonal TRSMs are split into disjoint chunks
  dispatched across the executor. No effect with `SerialExecutor`. **Caveat:** where this
  triggers, the parallel result is no longer *bit-identical* to the serial one (differs at
  ~1e-14 relative, from floating-point reassociation across chunk boundaries) — it is still
  fully deterministic for a given thread count, and the true residual is unaffected.

### Scaling

- **`setEquilibration(bool on)`** (default **on**) — Ruiz row/column scaling
  `Ã = Dr·A·Dc` so every row and column has comparable magnitude, before factorization. Improves
  conditioning, reduces how often static pivoting fires, and improves backward stability. Fully
  transparent: `solve()`, `transpose()`/`adjoint()`, and `determinant()` all operate in terms of
  the *original* `A`. `rowScaling()`/`colScaling() -> const std::vector<RealScalar>&` expose the
  computed `Dr`/`Dc` (original numbering), valid after `factorize()`.

### Diagnostics & queries

- **`info() -> ComputationInfo`** — `Success`, or `NumericalIssue` if factorization broke down
  (a static pivot of exactly zero even after all robustness measures) or the last `solve()`
  failed its honesty check.
- **`isFactorized() -> bool`** — true once a numeric factorization has succeeded; unlike
  `info()`, unaffected by a subsequently failed `solve()`. Use this to decide whether the
  factors are still usable.
- **`lastErrorMessage() -> const std::string&`** — human-readable detail for the last failure.
- **`rows()` / `cols() -> Index`** — matrix dimension.
- **`rowsPermutation()` / `colsPermutation() -> const PermutationType&`** — the row and column
  permutations between original and internal numbering. They differ whenever matching reorders
  rows (equal when matching is off or found the identity).
- **`nnzL()` / `nnzU() -> Index`** — nonzero counts of the stored `L`/`U` factors (including
  amalgamation's structural zeros).
- **`predictedFactorNonzeros() -> Index`** — total scalars the `L`/`U` arenas will occupy,
  computed from the symbolic structure after `analyzePattern()` and **before** `factorize()`
  allocates them (memory ≈ this × `sizeof(Scalar)`). It equals `nnzL() + nnzU() - n` once
  factored. Use it to gauge cost up front — a symmetric-pattern factorization can predict a
  hundreds-of-GB factor on matrices that lack good vertex separators (some 3D FEM systems) where
  an unsymmetric solver stays sub-GB.
- **`setMaxFactorNonzeros(Index limit)`** (default `0` = off) — fail-fast fill guard. When set,
  `factorize()` compares `predictedFactorNonzeros()` against `limit` and, if exceeded, **aborts
  before allocating**, setting `info() == NumericalIssue` with a diagnostic `lastErrorMessage()`
  (naming the predicted size and pointing to iterative/unsymmetric solvers) — instead of
  attempting a doomed multi-hundred-GB allocation. Off by default (behavior unchanged); recommended
  when factoring matrices of unknown structure. `maxFactorNonzeros()` returns the current limit.
- **`supernodeCount() -> Index`** — number of supernodes after amalgamation/splitting.
- **`levelCount() -> Index`** — number of elimination-tree levels used for scheduling.
- **`widestLevel() -> Index`** — supernodes in the widest level, an upper bound on the useful
  concurrency of plain level-parallelism (see [Parallelism](#parallelism)).
- **`determinant() -> Scalar`** — `det(A)`, correctly divides out the equilibration scaling and
  folds in the sign of the matching permutation and every in-block pivot swap.

### Factor access & transposed solves (`Eigen::SparseLU`-compatible)

- **`matrixL()` / `matrixU()`** — return proxy objects exposing `solveInPlace(x)` against the
  packed `L`/`U` factors, in the solver's *internal* numbering:
  ```cpp
  Eigen::VectorXd y = solver.rowsPermutation() * b;
  solver.matrixL().solveInPlace(y);
  solver.matrixU().solveInPlace(y);
  Eigen::VectorXd x = solver.colsPermutation().transpose() * y;   // P A P^T = L U
  ```
- **`transpose()`** / **`adjoint()`** — return a solve-only view for `Aᵀx = b` / `Aᴴx = b`,
  reusing the existing factorization (no re-factorization needed):
  ```cpp
  Eigen::VectorXd x = solver.transpose().solve(b);   // solver must be non-const
  ```

## Parallelism

Numeric factorization parallelizes two ways, both driven by the same `Executor`:

1. **Elimination-tree level parallelism.** Independent supernodes within one elimination-tree
   level factor concurrently (`levelCount()` levels total; `widestLevel()` supernodes at the
   widest — an upper bound on how much this alone can use).
2. **Intra-supernode parallelism** (`setIntraSupernodeParallelism`, on by default). Chunks a
   single big supernode's GEMM/TRSM work across the executor when a level is too narrow to keep
   the machine busy on its own — this is what breaks the "serial tail" of the few huge
   root-separator supernodes and is responsible for most of the speedup on well-separated
   matrices (measured 3x+ on large 2D Laplacians with 32 threads, vs. ~1.5x from level
   parallelism alone).

The `Executor` concept (`SupernodalLUExecutor.h`) is two methods:

```cpp
template <class F> void parallelFor(Index begin, Index end, F&& f) const;  // run f(i) for every i in [begin,end)
int concurrency() const;                                                    // worker lanes, >= 1
```

Four backends are provided:

| Executor | Header | Dependency | Notes |
|---|---|---|---|
| `supernodal_lu::SerialExecutor` | `SupernodalLU.h` (bundled) | none | Default. No threading. |
| `supernodal_lu::StdThreadExecutor` | `SupernodalLUExecutor.h` (bundled) | `<thread>` | Persistent `std::thread` pool, fork-join, dynamic work-stealing. Thread count fixed at construction (default `hardware_concurrency()`); the instance is non-copyable, so it **cannot** be reconfigured via `solver.executor() = ...` after construction — build a custom executor wrapping a shared pool of the size you want if you need that. |
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
see above) — pick whichever backend fits how the rest of your application is threaded.

### `OpenMPExecutor`

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

### `TBBExecutor`

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

## Examples

### 1. Basic solve, multiple right-hand sides

```cpp
Eigen::SupernodalLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);
Eigen::MatrixXd X = solver.solve(B);   // B: n x k, any number of columns
```

### 2. Checking for failure honestly

```cpp
solver.compute(A);
if (solver.info() != Eigen::Success) {
  std::cerr << "factorize failed: " << solver.lastErrorMessage() << "\n";
  return;
}
Eigen::VectorXd x = solver.solve(b);
if (solver.info() != Eigen::Success) {
  // solve() measured ||b - Ax||/||b|| > solveFailureThreshold() (default 1e-6),
  // or got a non-finite answer -- treat x as untrustworthy.
  std::cerr << "solve reported residual " << solver.solveResidual() << "\n";
} else {
  use(x);
}
```

### 3. A structurally difficult (near-singular / weak-diagonal) matrix

```cpp
Eigen::SupernodalLU<Eigen::SparseMatrix<double>> solver;
// matching() and diagonalPivoting() are on by default -- this is usually
// enough on its own. Shown explicitly for matrices that still misbehave:
solver.setMatching(true);
solver.setDiagonalPivoting(true);
solver.setRefinementMethod(Eigen::supernodal_lu::Refinement::BiCGStab);
solver.compute(A);
if (!solver.matchingIsPerfect())
  std::cerr << "warning: matrix is structurally singular\n";
std::cout << "pivots replaced: " << solver.replacedPivots() << "\n";
```

### 4. Determinant

```cpp
solver.compute(A);
double detA = solver.determinant();   // correct sign & magnitude, scaling divided out
```

### 5. Transposed / adjoint solve without refactoring

```cpp
Eigen::VectorXd x  = solver.solve(b);              // A x = b
Eigen::VectorXd xt = solver.transpose().solve(b);   // A^T x = b, same factors
Eigen::VectorXd xh = solver.adjoint().solve(b);      // A^H x = b (== transpose for real scalars)
```

### 6. Direct access to L and U

```cpp
Eigen::VectorXd y = solver.rowsPermutation() * b;
solver.matrixL().solveInPlace(y);
solver.matrixU().solveInPlace(y);
Eigen::VectorXd x = solver.colsPermutation().transpose() * y;
```

### 7. Parallel factorization (bundled `std::thread` pool)

```cpp
#include <SupernodalLU.h>

Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>,
                    Eigen::supernodal_lu::StdThreadExecutor> solver;   // uses hardware_concurrency()
solver.compute(A);
std::cout << solver.supernodeCount() << " supernodes, "
          << solver.levelCount() << " levels, widest " << solver.widestLevel()
          << ", " << solver.executor().concurrency() << " threads\n";
```

### 8. Parallel factorization with OpenMP or TBB, with a chosen thread count

```cpp
#include <SupernodalLUExecutorOpenMP.h>

Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>,
                    Eigen::supernodal_lu::OpenMPExecutor> solver;
solver.executor() = Eigen::supernodal_lu::OpenMPExecutor(8);   // override the OpenMP default
solver.compute(A);
```

```cpp
#include <SupernodalLUExecutorTBB.h>

Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>,
                    Eigen::supernodal_lu::TBBExecutor> solver;
solver.executor() = Eigen::supernodal_lu::TBBExecutor(8);
solver.compute(A);
```

### 9. Nested-dissection (METIS) ordering, and automatic ordering selection

```cpp
#include <SupernodalLUMetis.h>
Eigen::SupernodalLUMetis<Eigen::SparseMatrix<double>> solver;   // = SupernodalLU<Mat, MetisOrdering<int>>
solver.compute(A);
```

```cpp
#include <SupernodalLUAutoOrdering.h>
Eigen::SupernodalLUAuto<Eigen::SparseMatrix<double>> solver;    // tries AMD + several METIS restarts
solver.compute(A);
std::cout << "nnz(L) = " << solver.nnzL() << "\n";
```

Both can be combined with a parallel executor via the alias's second template parameter:
`Eigen::SupernodalLUMetis<Eigen::SparseMatrix<double>, Eigen::supernodal_lu::StdThreadExecutor>`.

### 10. Tuning amalgamation for a very fragmented matrix

```cpp
solver.setAmalgamation(/*relaxedSize=*/16, /*maxZeroRows=*/16);
solver.setAmalgamationFillFraction(0.3);
solver.setMaxBlockSize(128);
solver.compute(A);
```

## Performance notes (honest summary)

- SupernodalLU is typically **2-3x slower and denser** than `Eigen::SparseLU` on general
  matrices — almost entirely because a *symmetric-pattern* elimination fundamentally fills more
  than `Eigen::SparseLU`'s unsymmetric COLAMD + partial pivoting. Static pivoting removes
  pivot-search overhead but does not reduce flops; flops are set by fill.
- It can be **more robust** than `Eigen::SparseLU` on structurally awkward matrices (matching +
  restricted diagonal pivoting + BiCGStab refinement handles some singular/near-singular cases
  `Eigen::SparseLU` fails outright on).
- Ordering matters more than any kernel-level tuning: `SupernodalLUAuto` never loses to plain
  AMD in testing and can win substantially (up to ~1000x tighter solve error observed on one
  near-singular matrix), at the cost of extra `analyzePattern()` time on small matrices.
- Parallel scaling benefits the most from `setIntraSupernodeParallelism` (on by default) on
  matrices with a wide, well-separated elimination tree (e.g. 2D/3D discretizations) — 3x+
  speedup measured at 32 threads, versus ~1.5x from level-parallelism alone.
- All of the above is measured on `DirectLUSolvers/test/compare_testdata.cpp`'s matrix set —
  benchmark your own matrices before drawing conclusions for your workload.

## Testing

```sh
# Core correctness suite (no external dependencies)
clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src \
    DirectLUSolvers/test/test_supernodal_lu.cpp -o build/test_supernodal_lu
./build/test_supernodal_lu

# Parallel-vs-serial agreement + speedup (StdThreadExecutor)
clang++ -std=c++17 -O2 -pthread -I eigen -I DirectLUSolvers/src \
    DirectLUSolvers/test/test_parallel_lu.cpp -o build/test_parallel_lu
./build/test_parallel_lu

# Benchmark vs Eigen::SparseLU (add -DHAVE_METIS / -DHAVE_PARDISO for those columns;
# see compare_testdata.cpp's own header comment for the exact linker flags needed)
clang++ -std=c++17 -O2 -I eigen -I DirectLUSolvers/src \
    DirectLUSolvers/test/compare_testdata.cpp -o build/compare_testdata
./build/compare_testdata
```

## LeftRightLU — PARDISO-style sibling solver

`Eigen::LeftRightLU` (`src/LeftRightLU.h`, `#include <LeftRightLU>`) is a second sparse
direct LU solver in this directory with the **same `Eigen::SparseLU`-compatible interface**
as `SupernodalLU`, but built on the algorithmic design of **PARDISO** (see
`pardiso_algorithms.md`) rather than PaStiX. It **reuses `SupernodalLU`'s symbolic analysis
and solve pipeline verbatim** — matching, ordering, elimination tree, supernode detection
with amalgamation/splitting, block symbolic factorization, Ruiz equilibration, and the
block triangular solve with iterative/Krylov refinement (it shares the same
`SupernodalLUSupport.h` / `SupernodalLUMatching.h` / `SupernodalLUExecutor.h` headers) — and
replaces only the **numeric core** with PARDISO's two distinctive ideas.

```cpp
#include <LeftRightLU.h>

Eigen::LeftRightLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);                 // A: general values, SYMMETRIC nonzero pattern
Eigen::VectorXd x = solver.solve(b);
```

### What's different from SupernodalLU

1. **Left-right-looking, barrier-free dynamic scheduling** (the single largest
   architectural gap `pardiso_algorithms.md` §7.2 identifies against `SupernodalLU`).
   `SupernodalLU` factors bulk-synchronously: one `parallelFor` per elimination-tree level
   with a **hard barrier between levels**, so the big root separators serialize a whole
   level. `LeftRightLU` instead gives each supernode an **atomic count of unfinished update
   sources** (its in-degree in the assembly DAG). A worker takes a *ready* supernode (count
   0), **gathers** its updates from the already-finished sources (left-looking), factors it,
   then **pushes** readiness to its consumers (right-looking) — decrementing their counters
   and enqueuing any that reach zero. No worker ever blocks on a dependency; it always takes
   the next ready node, so there are **no level barriers**. A LIFO ready-stack gives
   depth-first subtree affinity (PARDISO's cooperative subtree ownership, minus the NUMA
   placement, which is out of scope). This runs as a single `parallelFor(0, P, worker)` over
   the same pluggable `Executor` — each worker is itself a complete sequential scheduler, so
   even a serial or fork-join executor drives it correctly (verified with the serial,
   `StdThreadExecutor`, and `OpenMPExecutor` backends).

2. **In-block complete pivoting** (PARDISO `DGETC2`, unsymmetric `MTYPE=11/13`). Where
   `SupernodalLU` does row-only restricted pivoting, `LeftRightLU` factors each supernode's
   dense diagonal block with **both row and column interchanges** confined to that block,
   giving per-supernode row (`P_s`) and column (`Q_s`) permutations. Because the search never
   leaves the block, the precomputed symbolic structure is never invalidated (the Schur
   update a supernode sends is invariant under `P_s`/`Q_s` — the local permutations cancel).
   The column permutation is folded transparently through `solve()`, `transpose()`/
   `adjoint()`, and `determinant()`. Selectable via `setPivoting(Pivoting::{None, Partial,
   Complete})`; default **Complete**.

3. **Refinement gated on perturbation** (PARDISO `IPARM(8)=0`). By default `solve()` runs
   refinement **only if the factorization actually bumped a static pivot**
   (`replacedPivots() > 0`); an un-perturbed factorization is already backward-stable, so
   refinement is skipped. Toggle with `setRefineOnlyIfPerturbed(false)` to always refine.

4. **Log-determinant** (PARDISO `IPARM(33)`): `logAbsDeterminant()` returns `log|det(A)|` as
   a sum of logs (stays finite where `determinant()` would overflow), paired with
   `determinantSign()`.

**Excluded by design** (project scope): NUMA-aware data placement, out-of-core
factorization, MPI, and the symmetric-indefinite Bunch-Kaufman `LDLᵀ` path (a documented
follow-up — this first version is the unsymmetric LU path). Bit-reproducibility across thread
counts is **not** a goal: the dynamic scheduler reassociates floating-point updates, so the
parallel result may differ from the serial one at the ~1e-14 level (the true residual is
unaffected and refinement cleans up the rest).

### Option reference (deltas from SupernodalLU)

`LeftRightLU` shares SupernodalLU's full option surface (static-pivot threshold, refinement
method/tolerance, matching, equilibration, amalgamation, `setMaxBlockSize`, the `Executor`
accessor, `transpose()`/`adjoint()`, `matrixL()`/`matrixU()`, `determinant()`, the honest
`info()`/`solveResidual()` failure check, …). The differences:

- **`setPivoting(left_right_lu::Pivoting mode)`** — `None` / `Partial` (row-only) / `Complete`
  (row + column, **default**). `setDiagonalPivoting(true|false)` remains as a convenience
  alias mapping to `Complete` / `None`.
- **`setRefineOnlyIfPerturbed(bool)`** (default **true**) — PARDISO-style refinement gating.
- **`logAbsDeterminant() -> RealScalar`** and **`determinantSign() -> Scalar`** — the
  log-determinant pair.
- **`predictedFactorNonzeros()`** and **`setMaxFactorNonzeros(Index)`** — the shared fail-fast
  fill guard (see the SupernodalLU [Diagnostics & queries](#diagnostics--queries) section). Both
  solvers factor a symmetric pattern, so both can predict an infeasible factor on matrices without
  good separators; the guard turns that into a clean `NumericalIssue` before allocating.
- **`setMaxBlockSize`** (default 128) doubles as PARDISO's ~80-column panel cap and keeps the
  complete-pivoting search on a small dense block; the intra-supernode chunking `SupernodalLU`
  exposes is **not** present (the async scheduler can't nest a fork-join inside a worker, and
  the dynamic schedule already shrinks the serial tail to a single node).
- `levelCount()`/`widestLevel()` remain as **diagnostics only** — the scheduler does not use
  levels to schedule (there are no level barriers).

### Testing

```sh
clang++ -std=c++17 -O2 -pthread -I eigen -I DirectLUSolvers/src \
    DirectLUSolvers/test/test_leftright_lu.cpp -o build/test_leftright_lu
./build/test_leftright_lu
```

`test/test_leftright_lu.cpp` covers direct/multi-RHS solves, factor accessors, transpose/
adjoint, all three pivoting modes, the forced column-swap path (matching off + weak diagonal)
with its solve/transpose/determinant folding, log-determinant, equilibration, honest failure
reporting, and parallel(dynamic-scheduler)-vs-serial agreement plus a deadlock-stress loop.

## License

Mozilla Public License 2.0 (`LICENSE`), matching the surrounding Eigen code this solver
integrates with.
