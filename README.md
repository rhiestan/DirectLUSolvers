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
| `CMakeLists.txt` | Builds and registers every suite with CTest. See [Testing](#testing). |
| `test/test_supernodal_lu.cpp` | Correctness tests (dependency-free — only needs Eigen). |
| `test/test_leftright_lu.cpp` | `LeftRightLU` correctness tests (dependency-free; `-pthread` for the parallel-vs-serial test). |
| `test/test_parallel_lu.cpp` | Parallel-vs-serial agreement + speedup, using `StdThreadExecutor`. |
| `test/test_matrixmarket.cpp` | Unit tests for the shared MatrixMarket reader and the pattern helpers. |
| `test/test_scalar_types.cpp` | `float` and `std::complex<double>` coverage, including the `adjoint()`/`transpose()` distinction that only exists for complex. |
| `test/test_edge_cases.cpp` | Degenerate sizes (n = 0/1/2, diagonal-only, single dense supernode), the refactorize workflow, zero right-hand side, and a cross-solver differential. |
| `test/test_regression.cpp` | Fill/accuracy regression suite, checked against `test/baselines/testdata.baseline`. See [Fill regression baselines](#fill-regression-baselines). |
| `test/test_suitesparse.cpp` | Correctness sweep over the curated SuiteSparse corpus, including matrices these solvers cannot handle. See [The SuiteSparse corpus](#the-suitesparse-corpus). |
| `test/matrices/fetch_suitesparse.py` | Downloads the corpus named by `suitesparse.manifest` into a git-ignored `cache/`. No third-party dependency. |
| `test/matrices/suitesparse.manifest` | The checked-in, human-curated corpus definition. |
| `test/compare_testdata.cpp` | Benchmark harness comparing SupernodalLU (AMD/METIS/Auto) against `Eigen::SparseLU` and, optionally, MKL PARDISO, on the matrices in `testdata/`. |
| `test/bench_parallel.cpp` | Thread-count scaling sweep with per-phase timing (analyze / factor / solve), per mechanism. See [Parallel scaling](#parallel-scaling-measured). |
| `test/bench_ceiling.cpp` | What the *machine* can deliver, via independent concurrent factorizations — the upper bound any scheduler could reach. See [The machine ceiling](#the-machine-ceiling). |
| `test/testing/Check.h` | Shared PASS/FAIL reporting and timing used by every suite. |
| `test/testing/MatrixMarket.h` | MatrixMarket reader: coordinate + array formats, real/integer/complex/pattern fields, general/symmetric/skew-symmetric/hermitian symmetries. |
| `test/testing/TestMatrices.h` | Deterministic matrix generators (2D/3D Laplacians, random symmetric-pattern, weak-diagonal) and the `symmetrizePattern`/`patternIsSymmetric` helpers. |
| `test/testing/TestData.h` | The benchmark-matrix registry: one list of `testdata/` matrices, with size tiers, shared by every suite. |
| `test/testing/PooledExecutor.h` | A copy-assignable `Executor` wrapping a shared `StdThreadExecutor` pool, so a solver's thread count can be changed after construction (which `StdThreadExecutor` itself cannot do). |

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
  scalar Eigen supports and any `StorageIndex` (`int` is what this project's METIS/matching code is
  tested against). `double`, `float` and `std::complex<double>` are covered by
  `test/test_scalar_types.cpp`; for complex scalars `adjoint()` is a genuine conjugate-transpose
  solve, verified distinct from `transpose()`, and `determinant()` carries the right complex phase.
  Note that `determinant()` overflows `double` on moderately sized systems (a diagonally dominant
  n=150 matrix already gives |det| ~ 1e326) — use `LeftRightLU::logAbsDeterminant()` there.
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

  > **Matching is not always an improvement — try turning it off if a solve fails.**
  > Measured over the [SuiteSparse corpus](#the-suitesparse-corpus): of 8 matrices whose
  > solve failed, **4 are fixed outright by `setMatching(false)`** (`Chebyshev3`
  > 6e+94 → 7e-17, `CAG_mat1916` 4e+23 → 5e-16, `cavity10` 4.6e-04 → 3.6e-16,
  > `nnc1374` 5.4e-01 → 4.2e-10) — and `Eigen::SparseLU` solves all four, so the
  > matrices are not at fault. Matching remains *essential* on others
  > (`meg1` 5.9e-16 with, 1e+300 without; `gemat12` 1.6e-16 with, 7e+76 without),
  > which is why it stays on by default.
  >
  > The cause is that this is a maximum *transversal* that prefers large entries,
  > not a true maximum-weight (MC64) assignment — the header says so. The
  > augmenting-path phase optimizes only for completing the transversal, so it can
  > displace good diagonal entries, and nothing checks the result against the
  > un-permuted diagonal. On `Chebyshev3` it takes a diagonal where 4096 of 4101
  > entries are within 1e-3 of their column maximum down to 181.
  >
  > There is no cheap a-priori test for which case you are in: diagonal-quality
  > scores predict only `Chebyshev3`, and fill does not correlate either
  > (`cavity10` gets *less* fill with matching and a worse answer). The reliable
  > move is empirical — if `info()` reports a bad solve, refactor with
  > `setMatching(false)` and compare.
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
  dispatched across the executor. Chunk extent scales with the executor's
  `concurrency()` so a big supernode produces about one chunk per lane (see
  [Chunk sizing](#chunk-sizing)); this is the single largest contributor to
  parallel factorization speedup measured on this project's matrices. No effect
  with `SerialExecutor`. **Caveat:** where this
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
   matrices (measured 3.20x on a 30³ 3D Laplacian at 32 threads, versus 1.13x
   from level parallelism alone; see [Parallel scaling](#parallel-scaling-measured)).

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
- Ordering matters more than any kernel-level tuning: `SupernodalLUAuto` never loses to plain AMD
  in testing and can win, though the size of the win is ordering-seed- and matrix-dependent — e.g.
  on `bayer05` (a near-singular pathological matrix; resid stays ~1e-16..1e-18 regardless of
  ordering, so *err* is the meaningful metric there) plain AMD gives solve error ~2.2, METIS/Auto
  bring it to ~1.5 — a real but modest improvement (re-measure before quoting a specific factor;
  this number moves as the solver evolves and has shifted in this project's history — see the
  ordering-direction note below), at the cost of extra `analyzePattern()` time on small matrices.
  For **large, well-separated 3D FEM systems, use `MetisOrdering`** (nested dissection) — on this
  project's `laoss_1` benchmark (251k rows, 3.5M nnz, 3D FEM), METIS ordering gives **~11.9x
  fill** (41.8M scalars, ~330 MB), matching MKL PARDISO's ~12x/~0.5GB and beating
  `Eigen::SparseLU`'s ~24.2x/85.1M-scalar/~680MB factor by roughly 2x. Even the *default* AMD
  ordering alone already beats SparseLU on fill here (~16.5x, 57.8M scalars) and is **faster in
  absolute wall-clock time**: factor+solve 3.0s vs SparseLU's 10.1s (3.4x faster) vs PARDISO's
  1.6s (SupernodalLU ~1.9x behind PARDISO). `laoss_2` (100k rows, 1.4M nnz) shows the same shape:
  0.86s vs SparseLU's 2.6s (3.0x faster) vs PARDISO's 0.49s. `LeftRightLU` tracks these numbers
  closely (3.0s / 0.88s on laoss_1/laoss_2, single-threaded) since it reuses the same analysis
  pipeline and only its numeric core differs — see its own
  [Performance notes](#performance-notes-honest-summary-1) below. (Measured 2026-07-14 with
  `DirectLUSolvers/test/compare_testdata.cpp`; the remaining gap to PARDISO is factorization
  *speed*, not fill.)
- **Get the ordering direction right.** These solvers consume the fill-reducing permutation as the
  *inverse* of what `Eigen`'s `AMDOrdering`/`MetisOrdering` put in `indices()` (see the note in
  `analyzePattern`). This was a bug until 2026-07: the ordering was applied backwards, which is
  nearly invisible on near-symmetric orderings but inflates fill 250-350x on strongly directional
  3D matrices. If you write a custom `OrderingType`, return the same convention as Eigen's
  built-in orderings.
- Parallel scaling benefits the most from `setIntraSupernodeParallelism` (on by default) on
  matrices with a wide, well-separated elimination tree (e.g. 2D/3D discretizations) — 3.20x
  measured at 32 threads on a 30³ 3D Laplacian, versus 1.13x from level-parallelism alone.
  Note that on the *whole* pipeline the serial `analyzePattern()` then dominates (~41% of
  factor+solve on `laoss_1` at 32 threads), so total speedup is well below the factorization
  figure. See [Parallel scaling](#parallel-scaling-measured) for the per-phase breakdown.
- All of the above is measured on `DirectLUSolvers/test/compare_testdata.cpp`'s matrix set —
  benchmark your own matrices before drawing conclusions for your workload.

## Testing

The suites build with CMake and run under CTest. From the `DirectLUSolvers`
directory:

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

`ctest -L quick` runs only the fast subset (synthetic matrices, seconds); the
unlabelled remainder reads the benchmark matrices and takes substantially
longer. Optional dependencies are independent switches, all default `OFF`:

```sh
cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON -DDLU_WITH_PARDISO=ON
```

`DLU_EIGEN_DIR`, `DLU_TESTDATA_DIR`, `DLU_METIS_DIR` and `DLU_MKL_DIR` default
to this project's layout (siblings of `DirectLUSolvers/`); override them if
yours differs. **An in-tree Eigen is preferred over an installed one on
purpose** — `find_package(Eigen3)` resolves against the user's CMake package
registry, which is frequently an unrelated version, and the fill baselines below
were recorded against the Eigen that ships beside these solvers.

The build defaults to **Release**, and the default is set *before* `project()`
deliberately. On a toolchain targeting MSVC (including `clang++` with a
`*-windows-msvc` triple) `Platform/Windows-MSVC.cmake` sets
`CMAKE_BUILD_TYPE_INIT` to `Debug`, so the usual `if(NOT CMAKE_BUILD_TYPE)`
guard placed *after* `project()` never fires and you silently get `-O0`. For
this project that is a 50-100x timing error — enough to make every benchmark
number meaningless. `cmake` prints the resolved type at configure time; check it
before quoting a measurement.

### Fill regression baselines

`test_regression` is the suite that guards the failure mode the others cannot
see. Every other check gates on the residual — but the ordering-direction bug
fixed in 2026-07 left every residual at machine precision while inflating 3D
factors 250-350x. Fill is a deterministic function of the pattern and the
ordering, so `test_regression` pins `nnzL + nnzU` per (matrix, solver) against
`test/baselines/testdata.baseline` and fails on drift beyond 5%.

```sh
ctest --test-dir build -R test_regression --output-on-failure
./build/test_regression --synthetic-only     # no testdata/ needed
./build/test_regression --tier small         # skip the large 3D FEM systems
./build/test_regression --update             # re-record the baselines
```

Re-baseline only once you understand why the fill moved: `--update` rewrites
every entry, so read the diff before committing it. A fill change is a real
change.

### The SuiteSparse corpus

`testdata/` holds a handful of matrices this project happened to encounter, all
of which these solvers handle. `test/matrices/` adds a curated corpus from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu), **stratified on
pattern symmetry** and deliberately including matrices the solvers should *not*
handle well — because the interesting question is not "does it solve" but "does
it behave correctly when it cannot".

```sh
python test/matrices/fetch_suitesparse.py     # download (~59 MB, once)
ctest --test-dir build -R test_suitesparse --output-on-failure
```

The fetch script needs **no third-party package** — the SuiteSparse URL scheme
is stable, so plain `urllib` suffices and reproducing the corpus never depends
on a `pip install`. (`ssgetpy` is consulted only by `--propose`, and only if
installed.) `suitesparse.manifest` is checked in and human-curated; the matrices
themselves land in a git-ignored `cache/`. Because SuiteSparse matrices are
immutable, a manifest line always denotes the same matrix.

```sh
python test/matrices/fetch_suitesparse.py --list        # what's in the corpus
python test/matrices/fetch_suitesparse.py --verify      # manifest vs live index
python test/matrices/fetch_suitesparse.py --propose 20  # candidates to adopt
```

**The contract being tested.** A solver may solve accurately, may refuse to
factor, or may return a bad answer *it has itself flagged* — but must never
quietly return a wrong one. `test_suitesparse` judges exactly that, reading
`info()` **after** `solve()`, and additionally confirms that a flagged solve
really was bad (flagging a good one would be its own defect).

Results on the 23-matrix quick tier: **15 solved** to machine precision, **8
returned a bad answer the solver flagged itself**, 0 tripped the fill guard.
No unflagged wrong answers — the honesty machinery
(`solveFailureThreshold`, the post-solve residual check) is now tested against
matrices that genuinely defeat the solvers, which nothing previously did.

**Why the 8 failures fail.** Diagnosed by comparing against `Eigen::SparseLU`
(real partial pivoting) on the same systems, then sweeping the solver options:

| matrix | psym | SparseLU | diagnosis |
|---|---:|---|---|
| `Chebyshev3` | 0.50 | solves 4e-20 | **matching**; `setMatching(false)` → 7e-17 |
| `CAG_mat1916` | 0.30 | solves 1e-15 | **matching**; → 5e-16 |
| `cavity10` | 0.94 | solves 2e-15 | **matching**; → 3.6e-16 |
| `nnc1374` | 0.82 | solves 7e-16 | **matching**; → 4.2e-10 |
| `lhr10c` | 0.01 | solves 5e-16 | **block size**; `setMaxBlockSize(0)` → 2.4e-16 |
| `shyy41` | 0.72 | **fails** 1e-06 | the matrix |
| `rw5151` | 0.49 | **fails** 7e-02 | the matrix |
| `foldoc` | 0.48 | **fails** inf | structurally singular |

So **5 of 8 are ours, not the matrix**, and a single setting recovers each. See
the [`setMatching`](#matching--diagonal-pivoting-robustness) note for the
mechanism and for why there is no cheap way to pick the right setting up front.
`lhr10c` is the one case where the 128-column `setMaxBlockSize` cap is the
binding constraint: widening it gives in-block pivoting enough candidate rows.

**A finding worth knowing: pattern symmetry does not predict success.** The
intuition that `psym == 1.00` is safe and `psym < 0.5` is doomed is wrong in
both directions. Three matrices with a *completely* unsymmetric pattern
(`HB/gemat12`, `Grund/meg1`, `Simon/raefsky5`, all psym 0.00) solved to ~1e-16,
while four in the *partial* band failed, including `DRIVCAV/cavity10` at
psym 0.94. Fill ratio tracks symmetry as expected (`Pajek/foldoc` 282x,
`HB/gemat12` 160x), but whether the answer is usable is governed by
conditioning, not by pattern. Do not use `psym` to decide whether these solvers
suit your matrix — run it and check `info()`.

### Parallel scaling (measured)

`bench_parallel` sweeps thread counts and times each phase separately, with the
intra-supernode mechanism toggled on and off:

```sh
./build/bench_parallel                            # the built-in scaling set
./build/bench_parallel --threads 1,4,16 --reps 5
./build/bench_parallel --quick                    # synthetic matrices only
```

Measured 2026-08-07, 32 hardware threads, `StdThreadExecutor` via
`PooledExecutor`, best of 3, AMD ordering. Times in ms; "32t" is the speedup
from 1 to 32 threads.

| matrix | phase | 1t | 8t | 32t | speedup |
|---|---|---:|---:|---:|---:|
| `laoss_1` (251k) | analyze (symbolic) | 650 | 637 | 626 | **1.04x** |
| | factor, levels only | 2069 | 1394 | 1336 | 1.55x |
| | factor, levels+intra | 2071 | 972 | 834 | **2.48x** |
| | factor, LRLU DAG | 2150 | 1303 | 1223 | 1.76x |
| | solve, 1 rhs | 71.3 | 71.4 | 71.5 | **1.00x** |
| | solve, 8 rhs | 207 | 209 | 208 | **1.00x** |
| `lap3d_30³` | factor, levels only | 600 | 544 | 533 | 1.13x |
| | factor, levels+intra | 604 | 263 | 189 | **3.20x** |
| | factor, LRLU DAG | 645 | 539 | 543 | 1.19x |

Four things this says, none of them visible from a single-thread-count timing:

1. **`solve()` does not parallelize at all** — 1.00x on every matrix, by
   construction: `solveTriangular` walks the supernodes sequentially and never
   touches the `Executor`. For the "many right-hand sides against one
   factorization" use case this README advertises as the sweet spot, that is a
   hard ceiling.
2. **`analyzePattern()` is serial and is often the *largest* remaining term.**
   At 32 threads it is 41% of a single factor+solve on `laoss_1`, 44% on
   `laoss_2`, and 57% on `lap2d_300²` — more than the factorization it feeds.
   Improving factorization scaling further buys little until this moves.
3. **Intra-supernode chunking is the mechanism that pays, not level
   parallelism.** On the 3D Laplacian, levels alone give 1.13x while adding
   intra-supernode chunking gives 3.20x. Level parallelism on its own never
   exceeded 1.65x on any matrix here. See [Chunk sizing](#chunk-sizing) for the
   cap that used to hold this back.
4. **`LeftRightLU`'s barrier-free scheduler still trails `SupernodalLU`'s
   levels + chunking on 3D** (1.19x vs 2.33x on `lap3d_30³`; 1.76x vs 2.53x on
   `laoss_1`) — see [Work-stealing ready queue](#work-stealing-ready-queue)
   below for what that residual gap is and is not.

Across the board, most of the available gain arrives by 8 threads; 8 → 32
adds little.

#### The machine ceiling

In-solver speedup confounds two very different limits: how much parallelism the
schedule exposes, and how much the machine can deliver. `bench_ceiling`
separates them by running **K independent single-threaded factorizations
concurrently** — they share no lock, no queue and no data, so their parallelism
is perfect by construction and the only contended resource is memory. Whatever
scaling that reaches is an upper bound on what *any* scheduler could achieve.

Measured on an AMD Ryzen 9 5950X (**16 physical cores**, 32 logical,
dual-channel DDR4):

| matrix | factor size | K=1 | K=8 | K=16 | K=32 |
|---|---|---:|---:|---:|---:|
| `lap3d_30³` | 92 MB | 1.00x | 4.93x | **7.76x** | 11.36x |
| `laoss_2` | 154 MB | 1.00x | 4.27x | **7.30x** | 10.28x |
| `laoss_1` | 463 MB | 1.00x | 4.59x | **7.84x** | — |

**Roughly half of the cores' throughput is already gone before our code does
anything.** With perfect, embarrassingly-parallel work, 16 cores deliver only
~7.3-7.9x. The absolute rate also falls with footprint — 16.0 GFLOP/s per core
solo on the 92 MB case versus 12.4 on the 154 MB one — which is the signature of
a memory-bound workload, as expected for a factor that does not fit in L3.

So `laoss_1`'s 2.48x should be read against **~7.8x, not against 32**. Two
multiplicative limits produce it:

1. **Hardware**: 16 cores behave like ~7.8 for this workload (49% efficiency).
2. **Schedule**: a flop-accurate model of the exact level/chunk schedule —
   counting the real Schur-update flops per supernode, the serial diagonal-block
   part, and an LPT bound on each outer level — predicts only **6.31x** for
   `laoss_1` even with perfect hardware.

Against a hardware-corrected structural prediction (~6.3 lanes at the ~8 GFLOP/s
per-lane rate the machine sustains at that concurrency, ≈ 50 GFLOP/s) the solver
achieves 37.4 GFLOP/s, or about **75%**. The remaining shortfall is dispatch
overhead and load imbalance beyond the LPT bound.

The practical consequences:

- **There is real headroom on `laoss_1` — about 3x, not 12x.** Anyone planning
  around these solvers should size expectations to the ceiling table, not to the
  core count.
- **Further gains must come from moving less memory**, not from more threads:
  better cache blocking and reuse in the update GEMMs. More scheduling
  sophistication cannot beat 7.8x.
- Fork-join dispatch costs **33-77 µs at 8-32 threads**, which is expensive in
  absolute terms, but a factorization issues only a few hundred dispatches
  (359 for `laoss_1`), so it accounts for ~2% here. Worth fixing eventually, not
  the bottleneck.
- Only **84 of `laoss_1`'s 49350 supernodes** run in inner (chunked) mode — but
  they carry 77% of the work. The other 22.6% sits in outer levels whose balance
  is whatever LPT gives.

```sh
./build/bench_ceiling            # full: synthetic + the laoss matrices
./build/bench_ceiling --quick    # synthetic only
```

Re-run it on your own hardware before reading anything into a speedup number;
the ceiling is machine-specific and a dual-socket server with more memory
channels will land somewhere quite different.

#### Chunk sizing

Intra-supernode chunking originally split a panel into fixed 128-row chunks, so
a supernode yielded `ceil(offDiagonalRows / 128)` chunks **regardless of how
many threads existed**. On this project's matrices at 32 lanes the heaviest
supernodes carry ~1400-1650 off-diagonal rows and therefore got only 11-13
chunks: 20 of 32 threads idled through precisely the supernodes that dominate
the factorization. Weighted by work, 52% (`lap3d_30³`) and 44% (`laoss_1`) of
all factorization work sat in supernodes that could not fill the machine.

The chunk extent is now `clamp(ceil(total / lanes), 32, 128)` — one chunk per
lane, floored so a chunk stays thick enough to amortize the BLAS call and the
per-chunk walk over the target's update sources, and ceilinged at the old 128 so
tall panels still produce many chunks for load balance. It is **never coarser
than before**, so it cannot reduce parallelism, and at low lane counts it
reproduces the old behaviour exactly.

| matrix | factor (levels+intra) at 32t | before | after | speedup 1→32 |
|---|---|---:|---:|---|
| `lap3d_30³` | | 265 ms | **189 ms** | 2.33x → **3.20x** |
| `laoss_2` (100k) | | 261 ms | **214 ms** | 2.01x → **2.47x** |
| `lap2d_300²` | | 59.1 ms | **51.3 ms** | 1.84x → **2.13x** |
| `laoss_1` (251k) | | 835 ms | 834 ms | 2.35x → 2.48x |

**`laoss_1` did not move**, despite the diagnostic predicting 44% idle there.
That was investigated separately — see [The machine ceiling](#the-machine-ceiling).

#### Work-stealing ready queue

The first version of the dynamic scheduler kept **one** global mutex-guarded
ready stack and called `notify_all()` after every completed supernode. That
serialized every task acquisition behind a single lock and woke all *P* workers
per supernode when at most a couple could proceed. It also defeated the
depth-first subtree affinity the LIFO existed to provide: a worker pushed its
freshly-readied children onto the shared stack, where any other worker took them
immediately.

It was replaced (2026-08-07) with **per-worker deques and work stealing**: a
worker pushes the consumers it readied onto its own back and pops from its own
back, so the node whose data is hot in that core's cache is the node it takes
next; only when its deque runs dry does it steal, from the *front* of a victim —
the entry furthest from the victim's hot end, so steals rarely collide with the
owner and tend to move a coarse subtree rather than a leaf. Idle workers park on
a shared condition variable with a bounded wait, and producers skip the notify
entirely unless someone is actually parked.

Measured effect (same setup as the table above), LRLU factorization:

| matrix | before, 16t → 32t | after, 16t → 32t | speedup 1→32 |
|---|---|---|---|
| `lap2d_300²` | 56.3 → **72.6** ms *(regressed)* | 58.1 → **53.9** ms | 1.69x → **2.30x** |
| `lap3d_30³` | 542 → **634** ms *(regressed)* | 536 → 543 ms | 1.02x → **1.19x** |
| `laoss_2` (100k) | 311 → 312 ms | 298 → 301 ms | 1.82x → 1.85x |
| `laoss_1` (251k) | 1262 → 1251 ms | 1259 → 1227 ms | 1.75x → 1.76x |

Read this honestly: **the change removes the high-thread-count regressions and
is a large win on `lap2d_300²`, but is roughly neutral on the two real FEM
matrices.** That split is consistent — `lap2d_300²` factors in ~120 ms across
32919 supernodes, so per-task queue overhead is a large fraction of task cost
and queue throughput dominates; `laoss_1` spends 2.1 s over 49350 supernodes, so
its tasks are far coarser and the queue was never the limiter there.

The remaining `lap3d_30³` gap to `SupernodalLU` (1.19x vs 2.33x) is therefore
**not** a queue problem. It is the absence of intra-supernode parallelism:
`LeftRightLU` deliberately has no equivalent of `setIntraSupernodeParallelism`
(the async scheduler cannot nest a fork-join inside a worker), so the few huge
root-separator supernodes remain single tasks that one thread must chew through
— and the measurements above show that mechanism, not level/DAG parallelism, is
what carries 3D scaling.

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
   the next ready node, so there are **no level barriers**. Each worker owns a **LIFO deque
   and steals from the front of a victim's** when it runs dry, which gives genuine
   depth-first subtree affinity (PARDISO's cooperative subtree ownership, minus the NUMA
   placement, which is out of scope) — see [Work-stealing ready
   queue](#work-stealing-ready-queue) for the measurements that motivated it. This runs as a single `parallelFor(0, P, worker)` over
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

### Performance notes (honest summary)

Measured 2026-07-14 with `DirectLUSolvers/test/compare_testdata.cpp`, single-threaded
(`SerialExecutor`) — the numbers below do **not** yet exercise the barrier-free dynamic
scheduler's headline advantage (that requires a parallel executor and hasn't been separately
benchmarked in this doc; see [Parallelism](#parallelism) for the mechanism).

- On this project's real-world `testdata/` set, `LeftRightLU` tracks `SupernodalLU`'s
  factor+solve time closely (both reuse the same analysis pipeline and static-pivoting numeric
  design — only in-block pivoting and scheduling differ): e.g. gemat11 1573ms vs 1496ms,
  bayer05 329ms vs 304ms, dendrimer 14.4ms vs 13.5ms. Same story on the large 3D FEM matrices —
  laoss_1 (251k rows) 3.0s vs SupernodalLU's 3.0s, laoss_2 (100k rows) 0.88s vs 0.86s — both well
  ahead of `Eigen::SparseLU` there (see the [SupernodalLU performance
  notes](#performance-notes-honest-summary) above for the SparseLU/PARDISO comparison).
- One measured, mechanistic difference: on a couple of already well-conditioned matrices
  (tomography, YaleB_10NN) `LeftRightLU` lands on a visibly looser — but still safely
  small — residual than `SupernodalLU` (tomography resid 2.2e-12 vs 2.8e-16; YaleB 1.0e-13 vs
  2.0e-16; both far under the 1e-6 `solveFailureThreshold()`). This is the documented
  **`setRefineOnlyIfPerturbed`** default at work: `LeftRightLU` skips refinement entirely when
  `replacedPivots()==0`, while `SupernodalLU`'s default BiCGStab refinement always runs at least
  one matvec check (and polishes further) even on an already-accurate direct solve. Not a bug —
  call `setRefineOnlyIfPerturbed(false)` if you want the tighter residual at the cost of that
  extra matvec.
- Complete (row+col) pivoting costs nothing extra over `SupernodalLU`'s row-only restricted
  pivoting in these measurements — both are confined to the small dense diagonal block
  (`setMaxBlockSize`, default 128), so the search cost is the same order regardless of how many
  interchange directions it considers.

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
ctest --test-dir build -R test_leftright_lu --output-on-failure
```

`test/test_leftright_lu.cpp` covers direct/multi-RHS solves, factor accessors, transpose/
adjoint, all three pivoting modes, the forced column-swap path (matching off + weak diagonal)
with its solve/transpose/determinant folding, log-determinant, equilibration, honest failure
reporting, and parallel(dynamic-scheduler)-vs-serial agreement plus a deadlock-stress loop.

## License

Mozilla Public License 2.0 (`LICENSE`), matching the surrounding Eigen code this solver
integrates with.
