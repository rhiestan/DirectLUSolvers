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

> **This restriction applies to `SupernodalLU` only.** The sibling
> [`LeftRightLU`](#leftrightlu--pardiso-style-sibling-solver) takes an **unsymmetric nonzero
> pattern** directly and symmetrizes internally, at the one point in the pipeline where it is
> cheapest — pre-symmetrizing its input instead costs 102x the fill on `gemat11`. If your
> pattern is unsymmetric, reach for `LeftRightLU`; see [Unsymmetric nonzero
> patterns](#unsymmetric-nonzero-patterns).

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
| `src/PointBlockLU.h` | The unsymmetric-pattern solver with refactorization replay (see [below](#pointblocklu--unsymmetric-pattern-solver-for-sparse-factors)). |
| `src/PointBlockLU` | Umbrella header for `PointBlockLU`, `#include <PointBlockLU>`. |
| `src/PointBlockOrdering.h` | `PointBlockOrdering` — fill-reducing ordering on the node graph, for matrices with several unknowns per grid point (see [below](#pointblockordering--ordering-the-node-graph)). Dependency-free. |
| `src/SupernodalLUSymbolic.h` | Shared symbolic helpers: the A+Aᵀ adjacency graph and the fill estimate used to rank candidate orderings. No METIS dependency, unlike `SupernodalLUAutoOrdering.h`, which uses it. |
| `src/SupernodalLUSupport.h` | Plain data structures shared by the analysis/factorization phases (`Supernode`, `RowBlock`, `UpdateSource`). |
| `src/SupernodalLUMatching.h` | The maximum-transversal matching + permutation-sign helpers (`MatchingMethod::Transversal`). |
| `src/SupernodalLUMC64.h` | Exact maximum-product matching with dual scaling (`MatchingMethod::MC64`). Eigen + standard library only. |
| `src/SupernodalLUExecutor.h` | The `Executor` concept, plus the bundled `SerialExecutor` and `StdThreadExecutor` backends. No dependency beyond `<thread>`. |
| `src/SupernodalLUExecutorOpenMP.h` | `OpenMPExecutor` — optional, requires an OpenMP-enabled build (see [below](#openmpexecutor)). |
| `src/SupernodalLUExecutorTBB.h` | `TBBExecutor` — optional, requires oneAPI Threading Building Blocks (see [below](#tbbexecutor)). |
| `src/SupernodalLUMetis.h` | `SupernodalLUMetis<Mat[,Executor]>` alias wiring in METIS nested dissection. Optional, requires METIS + GKlib. |
| `src/SupernodalLUAutoOrdering.h` | `SupernodalLUAuto<Mat[,Executor]>` alias: tries AMD and several METIS restarts, keeps the least-fill one. Optional, requires METIS + GKlib. |
| `src/HeaderOnlyMetis.h` | `Eigen::HeaderOnlyMetisOrdering<StorageIndex>` — a drop-in `MetisOrdering` replacement with **nothing to link**. Eigen only. |
| `src/HeaderOnlyMetis/` | The templated `METIS_NodeND` reimplementation behind it (coarsening, initial separator, FM refinement, nested-dissection driver, MT19937-64 RNG). |
| `CMakeLists.txt` | Builds and registers every suite with CTest. See [Testing](#testing). |
| `test/test_supernodal_lu.cpp` | Correctness tests (dependency-free — only needs Eigen). |
| `test/test_leftright_lu.cpp` | `LeftRightLU` correctness tests (dependency-free; `-pthread` for the parallel-vs-serial test). |
| `test/test_parallel_lu.cpp` | Parallel-vs-serial agreement + speedup, using `StdThreadExecutor`. |
| `test/test_matrixmarket.cpp` | Unit tests for the shared MatrixMarket reader and the pattern helpers. |
| `test/test_mc64.cpp` | MC64 optimality against a brute-force oracle, the dual-scaling property, and integration through both solvers. |
| `test/test_scalar_types.cpp` | `float` and `std::complex<double>` coverage, including the `adjoint()`/`transpose()` distinction that only exists for complex. |
| `test/test_executors.cpp` | One shared contract for every `Executor` backend — `StdThread`, `OpenMP`, `TBB` — checked against `SerialExecutor`. See [Testing the executor backends](#testing-the-executor-backends). |
| `test/test_edge_cases.cpp` | Degenerate sizes (n = 0/1/2, diagonal-only, single dense supernode), the refactorize workflow, zero right-hand side, and a cross-solver differential. |
| `test/test_regression.cpp` | Fill/accuracy regression suite, checked against `test/baselines/testdata.baseline`. See [Fill regression baselines](#fill-regression-baselines). |
| `test/test_suitesparse.cpp` | Correctness sweep over the curated SuiteSparse corpus, including matrices these solvers cannot handle. See [The SuiteSparse corpus](#the-suitesparse-corpus). |
| `test/matrices/fetch_suitesparse.py` | Downloads the corpus named by `suitesparse.manifest` into a git-ignored `cache/`. No third-party dependency. |
| `test/matrices/suitesparse.manifest` | The checked-in, human-curated corpus definition. |
| `test/compare_testdata.cpp` | Benchmark harness comparing SupernodalLU (AMD/METIS/Auto) against `Eigen::SparseLU` and, optionally, MKL PARDISO, on the matrices in `testdata/`. |
| `test/bench_parallel.cpp` | Thread-count scaling sweep with per-phase timing (analyze / factor / solve), per mechanism. See [Parallel scaling](#parallel-scaling-measured). |
| `test/bench_ceiling.cpp` | What the *machine* can deliver, via independent concurrent factorizations — the upper bound any scheduler could reach. See [The machine ceiling](#the-machine-ceiling). |
| `test/bench_solvers.cpp` | Per-matrix solver/**ordering** shootout: warm-up, best-of-N, per-phase timing, against `Eigen::SparseLU` and optionally MKL PARDISO. See [Choosing a configuration for one matrix](#choosing-a-configuration-for-one-matrix). |
| `test/profile_driver.cpp` | Per-phase driver for a profiler (not a test, not built by default). See [Profiling: where the time actually goes](#profiling-where-the-time-actually-goes). |
| `test/test_pointblock_lu.cpp` | `PointBlockLU` correctness: orderings and their permutation conventions, unsymmetric patterns, the replay path against fresh factorizations, degenerate sizes, structural singularity. |
| `test/test_parallel_consistency.cpp` | Serial-vs-parallel agreement for the chunked intra-supernode paths of both solvers: fill must match exactly, and the parallel solve must be no less accurate. |
| `test/test_header_only_metis.cpp` | Full-corpus gate for the header-only METIS port: `perm`/`iperm` must be byte-identical to the linked C `METIS_NodeND` on every test matrix. Passes trivially without METIS. |
| `test/test_header_only_metis_internal.cpp` | Per-module white-box comparison against METIS internals (`libmetis__*`), so a mismatch localizes to one algorithm instead of one permutation. Uses `test/metis_internal_bridge.cpp`. |
| `test/test_header_only_metis_ordering.cpp` | The `Eigen::HeaderOnlyMetisOrdering` wiring: permutation parity with `MetisOrdering`, identical solver fill, and — when built without METIS — that it works with nothing linked. |
| `test/testing/Check.h` | Shared PASS/FAIL reporting and timing used by every suite. |
| `test/testing/MatrixMarket.h` | MatrixMarket reader: coordinate + array formats, real/integer/complex/pattern fields, general/symmetric/skew-symmetric/hermitian symmetries. |
| `test/testing/TestMatrices.h` | Deterministic matrix generators (2D/3D Laplacians, random symmetric-pattern, weak-diagonal) and the `symmetrizePattern`/`patternIsSymmetric` helpers. |
| `test/testing/TestData.h` | The benchmark-matrix registry: one list of `testdata/` matrices, with size tiers, shared by every suite. |

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
  - `Eigen::HeaderOnlyMetisOrdering<StorageIndex>` (`HeaderOnlyMetis.h`, **no library to
    link**) — a templated reimplementation of `METIS_NodeND` that produces *bit-identical*
    permutations to `Eigen::MetisOrdering`, verified against the linked C library across this
    project's whole test corpus. Use it wherever `MetisOrdering` would go when you would rather
    not depend on METIS + GKlib at build time. Bit-identity is defined against a reference METIS
    built with the default 32-bit `idx_t`/`real_t` and GKlib's `GKRAND=ON` (its portable
    MT19937-64 rather than the platform `rand()`).
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
  > (`cavity10` gets *less* fill with matching and a worse answer).
  > **`setMatchingMethod(MatchingMethod::MC64)` fixes all of them properly** —
  > see below.

- **`setMatchingMethod(supernodal_lu::MatchingMethod m)`** — `None`,
  `Transversal` (default), or `MC64`.

  `MC64` is the exact **maximum-product assignment** (Duff & Koster), solved as a
  linear assignment problem by shortest augmenting paths. Two things follow that
  the transversal cannot offer:

  1. It maximizes `∏|a_ij|` over *all* permutations, so **it can never choose a
     worse diagonal than leaving the matrix alone** — exactly the guarantee whose
     absence lets the transversal break otherwise-solvable matrices.
  2. Its dual variables give scaling factors `Dr`, `Dc` making every matched
     diagonal entry exactly 1 and no entry larger than 1. These seed
     equilibration (Ruiz then composes on top), and are much of why MC64 works in
     MUMPS/SuperLU_DIST. The transversal returns a permutation and nothing else.

  Measured over the [SuiteSparse corpus](#the-suitesparse-corpus), MC64 **fixes
  every one of the five failures that were the solver's fault rather than the
  matrix's** — `Chebyshev3`, `CAG_mat1916`, `cavity10`, `nnc1374` and `lhr10c`
  (the last of which otherwise needed `setMaxBlockSize(0)`) — and breaks nothing.
  It also quietly improves others: `cavity17` goes from 197 bumped pivots to 0.
  The three remaining failures are matrices `Eigen::SparseLU` cannot solve either.

  **Why it is not the default.** Cost is O(n) shortest-path searches rather than
  one greedy pass. On this project's `testdata/` that is free or better —
  `laoss_1` analyze 703 → 659 ms, `laoss_2` 235 → 232 ms, and fill *drops* on
  several (`tomography` 0.59x, `bayer05` 0.95x). But the worst case is bad:
  `Pajek/foldoc`, a directed graph, goes 190 ms → 4.1 s. Since it can slow some
  inputs down by an order of magnitude it stays opt-in — though on PDE/FEM-shaped
  matrices there is no measured reason not to enable it.

  ```cpp
  solver.setMatchingMethod(Eigen::supernodal_lu::MatchingMethod::MC64);
  ```

  `setMatching(true|false)` is an alias for `Transversal`/`None`.
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

  > **Banded matrices are the case to watch.** Neither rule carries a *cumulative* budget:
  > each merge is judged on its own, so on a **chain elimination tree** — what AMD produces
  > for a banded matrix, tridiagonal being the extreme — every step adds exactly one zero
  > row, always passes `maxZeroRows`, and the chain amalgamates until `setMaxBlockSize`
  > stops it. On `testdata/setfos` (1015 rows, bandwidth 2) the defaults give 9 supernodes
  > and a 131422-scalar factor, against 4060 for `Eigen::SparseLU`. Two things fix it, both
  > measured on that matrix: `setAmalgamation(1, 0)` → 4074 scalars (and 1.8x faster), or
  > `setAmalgamation(4, 0)` → 7108 and faster still; alternatively `MetisOrdering`, whose
  > nested dissection does not produce a chain tree in the first place, → 8130 with the
  > defaults untouched. If your matrix is banded or otherwise chain-structured, set
  > `maxZeroRows` to `0` or order it with METIS.
- **`setAmalgamationFillFraction(double fraction)`** (default `0.3`) — an additional *relative*
  merge rule: also accept a merge when the extra zero rows it introduces are at most `fraction`
  of the rows the supernode already carries. Matters mainly for dense-ish factorizations (a wide
  panel with hundreds of off-diagonal rows can absorb a few more essentially for free); barely
  affects sparse matrices, where the absolute rule already governs. `0` disables this rule.
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
  triggers, the parallel result is not *bit-identical* to the serial one — a chunk is a
  differently-shaped GEMM, so Eigen may pick a different kernel and sum each dot product in a
  different order. Usually that shows up around 1e-14 relative, but on an ill-conditioned
  matrix it can be far larger (`Bai/cryg10000` moves 0.88 relative at an unchanged residual of
  5e-15): the factorization is exactly as good, but it is a different one. It is still fully
  deterministic for a given thread count, fill is unaffected, and `test_parallel_consistency`
  checks the invariants that really are thread-independent.

- **`setParallelSolve(bool on)`** (default **on**) — dispatch `solve()`'s forward and
  backward triangular sweeps across the `Executor`, over elimination-tree levels. Results are
  **bit-identical** to the serial sweeps (the forward sweep switches to an equivalent gather
  formulation that preserves each element's accumulation order — see
  [Parallel triangular solve](#parallel-triangular-solve)). Systems below
  `rows × nrhs = 200000` stay serial, because a per-level fork-join dispatch would cost more
  than the substitution. No effect with `SerialExecutor`. Measured ~1.9x on `laoss_1`.

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
- **`logAbsDeterminant() -> RealScalar`** / **`determinantSign() -> Scalar`** — `log|det(A)|`
  accumulated as a sum of logs, plus the sign (±1 for real scalars, a unit-modulus phase for
  complex, `0` on a zero pivot). **Prefer these over `determinant()` above a few hundred rows:**
  `det` scales like the product of the pivots, so a diagonally dominant 150×150 system already
  gives `|det| ~ 1e326` — `inf` in `double`, and any comparison against it is vacuous.
  `determinantSign() * exp(logAbsDeterminant())` reconstructs the value where it is
  representable. (`LeftRightLU` exposes the same pair.)

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
checked rather than asserted; see [Testing the executor backends](#testing-the-executor-backends).

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

Both of the above link METIS + GKlib. For the same nested-dissection ordering with **nothing to
link**, use `HeaderOnlyMetisOrdering` — a templated reimplementation of `METIS_NodeND` whose
permutations are bit-identical to the C library's:

```cpp
#include <HeaderOnlyMetis.h>
Eigen::SupernodalLU<Eigen::SparseMatrix<double>,
                    Eigen::HeaderOnlyMetisOrdering<int>> solver;
solver.compute(A);
```

It is the ordering functor on its own rather than a solver alias, so it drops into `LeftRightLU`
and `PointBlockLU` the same way. Being bit-identical, it gives exactly the fill and timings the
`MetisOrdering` rows report throughout this README — the only thing that changes is the build.

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
  bring it to ~1.5 — a real but modest improvement (re-measure before quoting a specific factor),
  at the cost of extra `analyzePattern()` time on small matrices.
  For **large, well-separated 3D FEM systems, use `MetisOrdering`** (nested dissection) — on this
  project's `laoss_1` benchmark (251k rows, 3.5M nnz, 3D FEM), METIS ordering gives **~11.9x
  fill** (41.9M scalars, ~335 MB), matching MKL PARDISO's 41.1M and beating
  `Eigen::SparseLU`'s ~24.2x/85.1M-scalar/~680MB factor by roughly 2x. Even the *default* AMD
  ordering alone already beats SparseLU on fill here (~16.5x, 58.0M scalars) and is **faster in
  absolute wall-clock time**: factor+solve 2.9s vs SparseLU's 8.7s (3.0x faster) vs PARDISO's
  1.4s, all single-threaded (SupernodalLU ~2.1x behind PARDISO). `laoss_2` (100k rows, 1.4M nnz)
  shows the same shape: 0.82s vs SparseLU's 2.1s (2.6x faster) vs PARDISO's 0.47s. `LeftRightLU`
  tracks these numbers closely (2.7s / 0.74s on laoss_1/laoss_2, single-threaded) since it reuses
  the same analysis pipeline and only its numeric core differs — see its own
  [Performance notes](#performance-notes-honest-summary-1) below. **Give every solver its
  threads before comparing**: at 16 threads PARDISO does `laoss_1` factor+solve in 0.80s against
  `LeftRightLU`'s 1.31s and `SupernodalLU`'s 1.66s, so the gap narrows to ~1.6x rather than
  widening. (Measured 2026-08-22 with `DirectLUSolvers/test/compare_testdata.cpp` and
  `test/bench_solvers.cpp`; the remaining gap to PARDISO is factorization *speed*, not fill.)
- **Get the ordering direction right.** These solvers consume the fill-reducing permutation as the
  *inverse* of what `Eigen`'s `AMDOrdering`/`MetisOrdering` put in `indices()` (see the note in
  `analyzePattern`). Applying it backwards is nearly invisible on near-symmetric orderings but
  inflates fill 250-350x on strongly directional 3D matrices, at unchanged residuals. If you write
  a custom `OrderingType`, return the same convention as Eigen's built-in orderings.
- Parallel scaling benefits the most from `setIntraSupernodeParallelism` (on by default) on
  matrices with a wide, well-separated elimination tree (e.g. 2D/3D discretizations) — 3.21x
  measured at 32 threads on a 30³ 3D Laplacian, versus 1.15x from level-parallelism alone.
  Note that on the *whole* pipeline the serial `analyzePattern()` then dominates (~43% of
  factor+solve on `laoss_1` at 32 threads), so total speedup is well below the factorization
  figure. Expect the peak somewhere around 16 threads on a 16-core part, not at 32: the
  workload is memory-bound and the second SMT thread per core adds no bandwidth. See
  [Parallel scaling](#parallel-scaling-measured) for the per-phase breakdown.
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

`ctest -L quick` runs only the fast subset — synthetic matrices, no external
data, a few seconds — while the remainder reads the benchmark matrices and takes
substantially longer. That split is what makes CI possible: **Eigen and
`testdata/` live outside this repository**, so `.github/workflows/ci.yml` runs
the `quick` label on every push (gcc, clang, MSVC) and a scheduled job fetches
the SuiteSparse corpus for a real-matrix sweep.

CI **pins Eigen to an exact commit**, deliberately. The fill baselines below
depend on the fill-reducing ordering, which comes from Eigen's AMD
implementation; a different Eigen can legitimately produce a different
permutation and therefore different fill. Bump the pin and re-record the
baselines together, never separately. Optional dependencies are independent switches, all default `OFF`:

```sh
cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON -DDLU_WITH_PARDISO=ON
```

`DLU_EIGEN_DIR`, `DLU_TESTDATA_DIR`, `DLU_METIS_DIR` and `DLU_MKL_DIR` default
to this project's layout (siblings of `DirectLUSolvers/`); override them if
yours differs. On Windows the METIS and MKL builds also need `tbb12.dll` /
`mkl_rt.dll` at *run* time, which CTest gets handed automatically — you only
need `<mkl root>/bin` on your own `PATH` when running those binaries by hand. **An in-tree Eigen is preferred over an installed one on
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

### Testing the executor backends

`test_executors` holds all four backends to one shared contract, checked against
`SerialExecutor`. `StdThreadExecutor` is always covered; the other two are
opt-in, and when their switch is off that backend is reported as skipped rather
than silently omitted:

```sh
cmake -S . -B build -G Ninja -DDLU_WITH_OPENMP=ON -DDLU_WITH_TBB=ON
ctest --test-dir build -R test_executors --output-on-failure
```

All three multithreaded backends run in **one binary against the same checks**,
deliberately — separate per-backend tests drift, and the property worth testing
is agreement *between* them. Each must match `SerialExecutor` on fill, solution,
residual and determinant; keep the parallel triangular solve bit-identical (a
different dispatch path per backend, so the claim has to hold for all of them);
drive `LeftRightLU`'s DAG scheduler, which asks something much stranger of an
executor than a plain loop — one `parallelFor` whose body is an entire
work-stealing scheduler; and survive repeated factorizations without deadlock or
drift.

**Discovery.** `DLU_WITH_TBB=ON` locates oneTBB automatically via its
`TBBConfig.cmake`, but note a trap in the oneAPI layout: the `latest` symlink can
point at a version that installed libraries **without headers** while a complete
older version sits beside it. The search therefore prefers a config whose
`include/oneapi/tbb.h` actually exists rather than trusting `latest`. Override
with `-DDLU_TBB_DIR=<dir containing TBBConfig.cmake>`. CTest is also handed
TBB's `bin` directory on `PATH`, so `tbb12.dll` resolves without any manual
environment setup — running the binary directly still needs it on `PATH`.
`DLU_WITH_OPENMP=ON` uses CMake's own `find_package(OpenMP)`.

Verified on this project's setup: clang 22 with `-fopenmp=libomp`, and oneTBB
2022.0. `TBBExecutor`'s documented reconfiguration (`solver.executor() =
TBBExecutor(n)` re-capping concurrency across successive assignments) is checked
explicitly, as is `OpenMPExecutor`'s thread-count override.

### Fill regression baselines

`test_regression` is the suite that guards the failure mode the others cannot
see. Every other check gates on the residual — but an ordering-direction mistake
leaves every residual at machine precision while inflating 3D factors 250-350x.
Fill is a deterministic function of the pattern and the
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
(`solveFailureThreshold`, the post-solve residual check) is exercised against
matrices that genuinely defeat the solvers.

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

So **5 of 8 are ours, not the matrix** — and **`setMatchingMethod(MatchingMethod::MC64)`
fixes all five at once**, including `lhr10c`. See
[Matching & diagonal pivoting](#matching--diagonal-pivoting-robustness) for the
mechanism and the cost trade-off.

**A finding worth knowing: pattern symmetry does not predict success.** The
intuition that `psym == 1.00` is safe and `psym < 0.5` is doomed is wrong in
both directions. Three matrices with a *completely* unsymmetric pattern
(`HB/gemat12`, `Grund/meg1`, `Simon/raefsky5`, all psym 0.00) solved to ~1e-16,
while four in the *partial* band failed, including `DRIVCAV/cavity10` at
psym 0.94. Fill ratio tracks symmetry as expected (`Pajek/foldoc` 282x,
`HB/gemat12` 160x), but whether the answer is usable is governed by
conditioning, not by pattern. Do not use `psym` to decide whether these solvers
suit your matrix — run it and check `info()`.

### Choosing a configuration for one matrix

`compare_testdata` answers "does every solver get the right answer across the corpus, and
roughly how fast". `bench_solvers` answers the question that follows it: **given this matrix,
which configuration should I actually use?** It sweeps solvers × orderings × thread counts on
one matrix at a time, warming each solver up before timing and reporting analyze / factor /
solve separately.

```sh
./build/bench_solvers                                   # the Tier::Small corpus
./build/bench_solvers --quick                           # synthetic only, no testdata needed
./build/bench_solvers --threads 1,2,16 --reps 5 path/to/A.mtx
./build/bench_solvers --no-matching path/to/A.mtx       # with setMatching(false)
```

Three things it shows that a single cold factor+solve number cannot:

- **Cold-start cost is excluded.** MKL's first `pardiso()` call spins up its thread pool; on a
  1015-row matrix that is ~500 ms against ~1 ms of real work, which makes an unwarmed PARDISO
  measurement meaningless.
- **Which phase costs.** `analyzePattern` is a third of wall clock for METIS on `setfos_2` —
  and it is exactly the phase you skip when refactorizing an unchanged pattern. Note the shape
  of the table below at 16 threads: for METIS (78.8 ms analyze against 65.4 ms factor) and for
  PARDISO (117.6 against 45.3) the symbolic phase is now the *larger* half.
- **The ordering**, which on an unsymmetric pattern moves the result further than the choice of
  solver does. Measured on `setfos_2` (n=3048, 238 nnz/row, symmetry 0.44), best of 5:

  | configuration | thr | analyze | factor | solve | total | fill |
  |---|--:|--:|--:|--:|--:|--:|
  | `LeftRightLU` AMD | 1 | 33.2 | 189.1 | 2.5 | 224.8 | 3,933,570 |
  | `LeftRightLU` AMD | 16 | 34.6 | 92.0 | 2.6 | 129.1 | 3,933,570 |
  | `LeftRightLU` COLAMD | 1 | 34.1 | 97.4 | 2.8 | 134.3 | 2,360,714 |
  | **`LeftRightLU` COLAMD** | 16 | 33.5 | 60.5 | 2.6 | **96.7** | 2,360,714 |
  | `LeftRightLU` METIS | 1 | 76.3 | 112.7 | 1.2 | 190.1 | 1,609,832 |
  | `LeftRightLU` METIS | 16 | 78.8 | 65.4 | 1.2 | 145.3 | 1,609,832 |
  | `SupernodalLU` AMD (on `Asym`) | 1 | 88.9 | 191.3 | 6.2 | 286.3 | 3,927,774 |
  | `SupernodalLU` AMD (on `Asym`) | 16 | 88.7 | 91.5 | 5.6 | 185.9 | 3,927,774 |
  | `Eigen::SparseLU` | 1 | 8.6 | 105.7 | 1.0 | 115.3 | 1,935,897 |
  | MKL PARDISO | 1 | 94.4 | 110.2 | 4.4 | 209.0 | 1,563,528 |
  | MKL PARDISO | 16 | 117.6 | 45.3 | 4.3 | 167.2 | 1,563,528 |

  Two results worth reading twice. COLAMD carries 47% more fill than METIS and still factors
  faster (39 wide supernodes against METIS's 321 narrow ones — the fatter dense blocks win the
  difference back in BLAS-3 efficiency), so fill is a first-order proxy for cost and not more
  than that. And with only 29 supernodes there is almost no assembly-DAG parallelism to find,
  so what scaling either solver gets on this matrix comes from the chunked
  intra-supernode path rather than from the schedule.

Fill is printed as each solver reports it: ours and `Eigen::SparseLU` count the diagonal in
both factors, PARDISO's `IPARM(18)` counts it once, so those columns are comparable only up to
an offset of `n`. The exit code counts only *our* solvers failing `resid < 1e-6` — the
benchmark is not a bug report against Eigen or MKL.

### Parallel scaling (measured)

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

#### The machine ceiling

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

#### Parallel triangular solve

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

#### Chunk sizing

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

#### Work-stealing ready queue

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

#### Profiling: where the time actually goes

`test/profile_driver.cpp` exists so a profiler sees `analyzePattern` / `factorize`
/ `solve` as three separately-attributable phases rather than one `compute()`
blob. It is not a test and is not built by default:

```sh
cmake -S . -B build -G Ninja -DDLU_BUILD_PROFILE_DRIVER=ON \
      -DDLU_WITH_ITT=ON -DDLU_ITT_DIR="<VTune>/sdk"      # ITT markers optional
cmake --build build

# the whole SuiteSparse corpus, both solvers, all three phases
vtune -collect hotspots -knob sampling-mode=sw -r r_hs -- \
      ./build/profile_driver --reps 3

# one scheduler, one matrix shape, one thread count
vtune -collect threading -knob sampling-and-waits=sw -r r_thr -- \
      ./build/profile_driver --solver lrlu --synthetic lap3d --threads 32 \
                             --reps 12 --phase factorize --no-intra
```

`--no-intra` turns `setIntraSupernodeParallelism` off in both solvers, which is
what isolates the scheduler under a threading profile: with it on, the narrow
top levels are chunked across the pool and the starvation the level/DAG schedule
suffers there no longer appears in the timeline.

Hotspots over the corpus (both solvers, `--reps 3`, 195 s of CPU time,
re-collected 2026-08-22 — build with `-g -gcodeview` or every frame comes back
as a hex address):

| source | CPU time | share |
|---|---:|---:|
| Eigen `PacketMath.h` (GEMM inner loops) | 153.2 s | 79% |
| Eigen `GeneralBlockPanelKernel.h` | 13.9 s | 7.1% |
| Eigen `AssignEvaluator.h` | 5.0 s | 2.6% |
| Eigen `Amd.h` (ordering, in analyze) | 1.8 s | 0.9% |
| **`SupernodalLU.h`** | **1.2 s** | **0.6%** |
| **`LeftRightLU.h`** | **0.9 s** | **0.5%** |

**That is the shape a finished solver should have**: ~89% of the time is Eigen's
dense kernels doing the arithmetic, and the solvers' own bookkeeping is ~1%. It
is also the check that the three profile-guided fixes stuck — the symbolic
`set_union` that was once the hottest line in the analyze phase now costs 0.33 s
(0.2%), `rowPanelPosition` 0.60 s (0.3%), and `LeftRightLU`'s complete-pivot
search, once ~44% of its factorize, is down to 0.11 s (0.06%) across
`pabs`/`predux`/`find_coeff_loop` combined. Reading the whole 26-matrix corpus
off disk takes 0.83 s in total — it used to be the single largest entry in this
report, back when the MatrixMarket reader built an `istringstream` per stored
nonzero.

If your own profile does not look like this, that is the interesting result —
these solvers are memory-bound in the kernels and everything else is noise.

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
solver.compute(A);                 // A: any square matrix -- values AND pattern
Eigen::VectorXd x = solver.solve(b);   //    may both be unsymmetric
```

Unlike `SupernodalLU`, `LeftRightLU` takes an **unsymmetric nonzero pattern** directly, and
you should *not* pre-symmetrize its input — see [Unsymmetric nonzero
patterns](#unsymmetric-nonzero-patterns) below, where doing so costs 102x the fill on
`gemat11`.

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
   queue](#work-stealing-ready-queue) for the measurements behind it. This runs as a single `parallelFor(0, P, worker)` over
   the same pluggable `Executor` — each worker is itself a complete sequential scheduler, so
   even a serial or fork-join executor drives it correctly (verified with the serial,
   `StdThreadExecutor`, and `OpenMPExecutor` backends). The one exception to "no barriers" is
   the tail: the narrow top levels are carved out of the DAG phase and swept after it, so the
   pool can be applied *inside* those few enormous separator supernodes (`parallelFor` is not
   nestable, so the DAG phase has to end first). That boundary is deliberate and measured —
   see [Work-stealing ready queue](#work-stealing-ready-queue).

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
factorization, MPI, and the symmetric-indefinite Bunch-Kaufman `LDLᵀ` path (this solver
implements the unsymmetric LU path only). Bit-reproducibility across thread
counts is **not** a goal: the dynamic scheduler reassociates floating-point updates, and so
does the chunked tail sweep, which splits one GEMM into several differently-shaped ones and
lets Eigen pick a different kernel for each. Fill is unchanged everywhere and residuals stay
at machine precision, but the computed **solution** can differ substantially on an
ill-conditioned matrix — `Bai/cryg10000` moves 0.88 relative at an unchanged residual of
5e-15. That is a different, equally valid factorization, not a worse one. Callers who need
run-to-run reproducibility should set `setIntraSupernodeParallelism(false)`, which restores the
previous behaviour exactly; `test_parallel_consistency` asserts what really is
thread-independent (fill exactly, and residual no worse than serial's).

### Unsymmetric nonzero patterns

`LeftRightLU` accepts **any square sparse matrix**: unsymmetric values *and* an unsymmetric
nonzero pattern. Nothing has to be done to `A` first.

The symbolic phase runs on the pattern of `B + Bᵀ`, where `B` is `A` with the matching row
permutation applied. Because the numeric phase never moves a row or column outside its own
diagonal block, the fill pattern of `chol(B + Bᵀ)` is a valid superset of the structure of
both `L` and `U`, so every entry of `A` lands in a slot that already exists. An unsymmetric
pattern costs **fill, never correctness**. (This is also what PARDISO does for its
unsymmetric matrix types, `MTYPE=11/13`.)

**Do not pad the pattern with explicit structural zeros.** That was the documented workaround
for `SupernodalLU`, and on `LeftRightLU` it is not merely redundant — it is actively harmful,
because *when* you symmetrize decides how much structure you get:

| gemat11 (n=4929) | input nnz | `patternSymmetry()` | `nnzL` | factor time |
|---|---|---|---|---|
| raw `A` | 33,185 | 0.81 | **51,728** | **2.6 ms** |
| `symmetrizePattern(A)` | 66,313 | 0.38 | 5,281,259 | 1421 ms |

Both give a machine-precision residual; the padded input just costs **102x the fill and 546x
the factorization time**. The reason is ordering: the solver symmetrizes *after* the matching
row permutation, on `P·A`, while a caller symmetrizing up front does it *before*, on `A` — and
the pattern `P·(A ∪ Aᵀ)` then gets symmetrized a second time into
`P·(A ∪ Aᵀ) ∪ (A ∪ Aᵀ)·Pᵀ`, which is far larger than `P·A ∪ Aᵀ·Pᵀ`. On `bayer05` the same
effect is 8.3x fill / 24x time. Matrices whose pattern is *already* symmetric
(`sherman1`, `rdb2048_noL`, `dendrimer`) are bit-identical either way — padding only matters
when there is something to pad.

Practical notes:

- **`patternSymmetry()`** (in `[0,1]`) reports the fraction of off-diagonal nonzeros with a
  structural mirror, measured on `B` — the graph actually eliminated, i.e. *after* matching.
  A symmetric-pattern input can therefore still report below 1: matching permutes rows.
  **`structurallySymmetric()`** is the exact `== 1` case.
- **Try the other orderings** — the default `AMDOrdering` is a reasonable start (it minimizes
  degree in exactly the `A + Aᵀ` graph this factorization eliminates), but the symmetrized
  graph it sees on an unsymmetric pattern can be far denser than the one AMD was designed for,
  and the gap between orderings is much wider here than on symmetric-pattern matrices. On
  `setfos_2` (n=3048, 238 nnz/row, symmetry 0.44):

  | ordering | fill (nnzL+nnzU) | factor ms | supernodes |
  |---|--:|--:|--:|
  | `AMDOrdering` (default) | 3,933,570 | 201 | 29 |
  | `MetisOrdering` | **1,609,832** | 127 | 321 |
  | `COLAMDOrdering` | 2,360,714 | **116** | 39 |

  COLAMD wins on *time* while losing on fill: it leaves 39 wide supernodes against METIS's 321
  narrow ones, and the fatter dense blocks pay back the extra flops in BLAS-3 efficiency. Fill
  is the right first-order proxy for cost, but on a matrix this dense it is not the whole
  story — measure. (On sparse structured matrices the ordering ranking is the usual one:
  AMD 13,679 `nnzL` against COLAMD's 15,181 on a 900-column upwind grid.)
- A custom ordering functor that returns the **direct** permutation (rather than the inverse,
  as AMD and METIS do) needs a one-line `left_right_lu::OrderingConvention` specialization;
  Eigen's own AMD/METIS/COLAMD/Natural functors are all handled already. Getting this
  direction wrong is invisible in the residual and shows up only as fill.
- **Structural singularity** — a column whose nonzero rows are all claimed elsewhere — is the
  failure mode unsymmetric patterns introduce. `matchingIsPerfect()` reports it right after
  `analyzePattern()`; `solve()` then refuses to call the result a success via
  `info()`/`solveResidual()`.
- On a strongly unsymmetric pattern with no good vertex separators, fill can still exceed what
  a partial-pivoting solver needs by orders of magnitude. `setMaxFactorNonzeros()` turns that
  into a clean error before any factor memory is touched, and its message now names the
  pattern symmetry as a contributing cause.

### Performance notes (honest summary)

Measured 2026-08-22 with `DirectLUSolvers/test/compare_testdata.cpp`, single-threaded
(`SerialExecutor`) — the numbers below do **not** exercise the barrier-free dynamic
scheduler's headline advantage, which requires a parallel executor (see
[Parallel scaling](#parallel-scaling-measured) for the threaded numbers).

- On **symmetric-pattern** matrices from this project's real-world `testdata/` set,
  `LeftRightLU` tracks `SupernodalLU`'s factor+solve time closely or beats it modestly (both
  reuse the same analysis pipeline and static-pivoting numeric design — only in-block pivoting
  and scheduling differ): e.g. dendrimer 9.8ms vs 14.7ms, laoss_3 17.5ms vs 33.1ms. Same story
  on the large 3D FEM matrices — laoss_1 (251k rows) 2.7s vs SupernodalLU's 3.0s, laoss_2 (100k
  rows) 0.74s vs 0.82s — both well ahead of `Eigen::SparseLU` there (see the [SupernodalLU
  performance notes](#performance-notes-honest-summary) above for the SparseLU/PARDISO
  comparison).
- On **unsymmetric-pattern** matrices the two diverge sharply, because `SupernodalLU` must
  be handed a pre-symmetrized matrix and `LeftRightLU` symmetrizes internally *after* matching
  (see [Unsymmetric nonzero patterns](#unsymmetric-nonzero-patterns)): gemat11 **9.1ms vs
  1415ms**, bayer05 **22.8ms vs 317ms**, setfos_2 229ms vs 291ms. gemat11 and bayer05 are also
  where `SupernodalLU` loses accuracy outright (bayer05 err 1.6e+00 vs `LeftRightLU`'s
  5.3e-03), so this is not only a speed difference.
- One measured, mechanistic difference: on a couple of already well-conditioned matrices
  (tomography, YaleB_10NN) `LeftRightLU` lands on a visibly looser — but still safely
  small — residual than `SupernodalLU` (tomography resid 3.2e-12 vs 2.6e-16; YaleB 1.2e-13 vs
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
  (row + column, **default**). `setDiagonalPivoting(true|false)` is a convenience alias
  mapping to `Complete` / `None`.
- **`setRefineOnlyIfPerturbed(bool)`** (default **true**) — PARDISO-style refinement gating.
- **`logAbsDeterminant() -> RealScalar`** and **`determinantSign() -> Scalar`** — the
  log-determinant pair. Not actually a delta: `SupernodalLU` exposes the same pair, since
  `determinant()` overflows on both solvers for the same reason. See its
  [Diagnostics & queries](#diagnostics--queries).
- **`predictedFactorNonzeros()`** and **`setMaxFactorNonzeros(Index)`** — the shared fail-fast
  fill guard (see the SupernodalLU [Diagnostics & queries](#diagnostics--queries) section). Both
  solvers factor a symmetric pattern, so both can predict an infeasible factor on matrices without
  good separators; the guard turns that into a clean `NumericalIssue` before allocating.
- **`setMaxBlockSize`** (default 128) doubles as PARDISO's ~80-column panel cap and keeps the
  complete-pivoting search on a small dense block.
- **`setIntraSupernodeParallelism(bool on)`** (default **on**) — present, but it works
  differently here. The async scheduler cannot nest a fork-join inside a worker, so the narrow
  top levels are carved out of the DAG phase entirely and swept afterwards, sequentially, with
  the pool applied inside each supernode. It therefore engages only where the DAG has run dry;
  turning it off leaves the root-separator tail serial. Same numerical caveat as
  `SupernodalLU`'s (see [Work-stealing ready queue](#work-stealing-ready-queue)).
- `levelCount()`/`widestLevel()` are **diagnostics only** — the scheduler does not use
  levels to schedule (there are no level barriers).

### Testing

```sh
ctest --test-dir build -R test_leftright_lu --output-on-failure
```

`test/test_leftright_lu.cpp` covers direct/multi-RHS solves, factor accessors, transpose/
adjoint, all three pivoting modes, the forced column-swap path (matching off + weak diagonal)
with its solve/transpose/determinant folding, log-determinant, equilibration, honest failure
reporting, and parallel(dynamic-scheduler)-vs-serial agreement plus a deadlock-stress loop.

An `Unsymmetric nonzero patterns` block covers that input class specifically: accuracy on
random unsymmetric patterns and on upwind advection grids, `patternSymmetry()` against a
reference computed from `A` and `Aᵀ`, equality of fill and answer between a raw and a
pre-symmetrized input, the AMD/COLAMD/Natural ordering conventions pinned by fill (a misread
permutation is invisible in the residual), structural-singularity reporting, and unsymmetric
patterns combined with multi-RHS, transpose, determinant and the parallel scheduler.

## PointBlockLU — unsymmetric-pattern solver for sparse factors

`Eigen::PointBlockLU` (`src/PointBlockLU.h`, `#include <PointBlockLU>`) is the third solver
here, and the only one that does **not** symmetrize the pattern. It factors `A` as it is —
left-looking Gilbert–Peierls with partial pivoting, column by column, scalar kernels, no
supernodes and no BLAS-3 — and it records the pattern and pivot sequence of the first
factorization so every later one on the same pattern is a pure numeric **replay**.

```cpp
#include <PointBlockLU.h>

Eigen::PointBlockLU<Eigen::SparseMatrix<double>> solver;
solver.analyzePattern(A);            // once per pattern
for (/* each Newton step */) {
  solver.factorize(A);               // replays: no search, no pivot choice, no allocation
  x = solver.solve(b);
}
```

### When to use it

**When the factor stays sparse.** That is the whole envelope, and it is narrower than "small
matrices". Symmetrizing an unsymmetric pattern can cost enormous fill, and avoiding that is
what this solver buys; but a scalar column algorithm runs at roughly a fifth of the throughput
of a supernodal one, so once the factor densifies the fill advantage is spent and
`LeftRightLU` wins. Measured single-threaded, best of 5, COLAMD ordering, `bench_solvers`
(PointBlockLU timed on its **replay** — the call the target workload actually makes; its
`analyze` column is paid once):

| matrix | n | PointBlockLU fill / replay+solve | LeftRightLU fill / factor+solve | `Eigen::SparseLU` |
|---|---:|---:|---:|---:|
| `setfos` | 1015 | **4,080** / **0.03 ms** | 116,786 / 2.06 ms | 4,080 / 0.13 ms |
| `bayer05` | 3268 | **77,462** / **1.25 ms** | 456,036 / 13.41 ms | 126,396 / 4.96 ms |
| `gemat11` | 4929 | **79,614** / **1.52 ms** | 131,156 / 3.32 ms | 86,476 / 3.92 ms |
| `tomography` | 500 | **46,540** / **2.39 ms** | 180,154 / 4.52 ms | 91,650 / 5.18 ms |
| `sherman1` | 1000 | 32,916 / 0.84 ms | 40,884 / 0.84 ms | 31,900 / 1.05 ms |
| `laoss_3` | 4180 | 731,852 / 44.0 ms | 1,210,476 / **24.9 ms** | 731,852 / 30.2 ms |
| `YaleB_10NN` | 2414 | 1,232,024 / 258.1 ms | 1,638,482 / **80.2 ms** | 1,226,238 / 110.7 ms |
| `setfos_2` | 3048 | 1,935,546 / 452.8 ms | 2,360,714 / **100.7 ms** | 1,935,897 / 106.0 ms |

**The crossover sits near 100k stored scalars in the factor.** Below it `PointBlockLU` is the
fastest solver in this project — on `setfos`, `gemat11`, `sherman1`, `tomography` and
`bayer05` it beats `Eigen::SparseLU` and MKL PARDISO outright. Above it, use `LeftRightLU`.

It is often more *accurate* too, because it never perturbs a pivot: on `gemat11` its solve
error is 8.9e-13 against `LeftRightLU`'s 4.3e-08, on `tomography` 6.8e-14 against 7.1e-09,
and on the near-singular `bayer05` 1.1e-03 against `Eigen::SparseLU`'s 8.4e+00.

Equilibration iterates to convergence rather than a fixed sweep count, which matters at this
scale: the sweep is O(nnz) and runs on every replay, so a fixed eight sweeps was 80% of
`setfos`'s entire replay (98 µs against 19 µs with scaling off). It now costs one or two
sweeps on a well-scaled matrix, and the replay is 20 µs.

### Why PointBlockLU is not parallel

A parallel version was built, measured and removed. The finding is worth recording, because
"add threads" is the obvious next request and it does not work here.

The **replay** is the only half that could be scheduled: by then the pattern and pivots are
fixed, so the column dependency DAG is known (column *k* needs every *j* < *k* appearing in
`U(:,k)`, and writes only its own entries). The first factorization cannot be scheduled even in
principle — with partial pivoting the choice made in column *k* determines the structure of every
later column. The implementation was a DAG scheduler over the replay: one fork-join dispatch for
the whole factorization, per-lane scratch rows, per-worker deques with stealing, and results
bit-identical to the serial replay. Measured replay times, microseconds:

| matrix | ideal (work / critical path) | 1 lane | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| `setfos` | 1.25x | 24 | 70 | 163 | 219 | 459 |
| `bayer05` | 3.32x | 1129 | 1069 | 1015 | **988** | 1294 |
| `gemat11` | 1.78x | 1380 | 1336 | 1263 | **1241** | 1640 |
| `setfos_2` | 1.00x | 437610 | **401312** | 471920 | 491068 | 502278 |

The best result anywhere was **1.14x**; past 8 lanes every case lost, and `setfos` degraded 19x.
Two structural reasons, either sufficient on its own:

1. **The DAG has no width.** "ideal" above is total work over critical path — the ceiling on
   *any* schedule, before a single thread exists. It is 1.00x on `setfos_2`, 1.02x on
   `YaleB_10NN`, 1.01x on `tomography`, 1.25x on `setfos`, and reaches only 3.3x on `bayer05`.
   A 2-D Laplacian control scores just **1.4x**, so this is not a quirk of these matrices: it is
   the same fact the [parallel scaling](#parallel-scaling-measured) section records for the
   supernodal solvers, where level/DAG parallelism alone never exceeded 1.79x and all the real
   scaling came from chunking *inside* dense panels. PointBlockLU has no dense panels to
   chunk — having none is the point of it.
2. **The tasks are too small to schedule.** A column of `bayer05`'s replay costs ~350 ns, while
   its plan has ~37k DAG edges each needing an atomic decrement to release a consumer. The
   bookkeeping costs what the arithmetic costs.

If you need threads on a matrix in this class, the lever is `LeftRightLU`, whose supernodal
panels are coarse enough to schedule — and by the time the factor is dense enough for threads to
matter, the crossover above has been passed anyway.

### Refactorization

`analyzePattern()` only chooses the column ordering — with partial pivoting the pattern of
`L` and `U` is not knowable until the values are seen, so the first `factorize()` does the
symbolic search (a depth-first reachability pass per column) alongside the numeric work and
records what it found. Every later `factorize()` replays that record. `refactorizations()`
reports how many replays have happened since the last full factorization.

A replay is **rejected** — and a full factorization redone automatically — when a pivot falls
below `setMinPivotRatio()` (default 1e-8) of the magnitude it had when the plan was recorded.
That check is what makes replaying safe as the caller's values drift;
`setForceFullFactorization(true)` disables replaying altogether.

### Deltas from the other two solvers

- **Structurally singular input is declined, not patched.** An unsymmetric LU needs a pivot in
  every column, so `testdata/bcsstm13` (762 numerically empty columns out of 2003) returns
  `NumericalIssue` naming the column — exactly as `Eigen::SparseLU` does. The supernodal
  siblings appear to succeed there only because symmetrizing fills those columns in from the
  transpose; every solver's answer on that matrix carries a relative error of 0.62.
- **Single-threaded, with no `Executor` template parameter** — see [Why it is not
  parallel](#why-pointblocklu-is-not-parallel) below, which is a measurement rather than an
  omission.
- **No matching and no static pivoting.** Partial pivoting does that job; Ruiz equilibration
  is on by default (`setEquilibration`) and matters — `setfos_2` spans 4e48 in magnitude,
  where an unscaled pivot comparison is meaningless.
- **`setPivotThreshold(t)`** (default 1.0 = strict partial pivoting) keeps the structural
  diagonal as pivot when `|a_kk| >= t * max|a_ik|`. Lower values mean less fill and a pivot
  sequence that survives refactorization better, at some stability cost.
- The default ordering is **COLAMD**, not `PointBlockOrdering` — see the note below.

## PointBlockOrdering — ordering the node graph

`Eigen::PointBlockOrdering` (`src/PointBlockOrdering.h`) is a standalone ordering functor for
matrices with `nv` unknowns per grid point. It collapses the pattern onto the **node** graph,
orders that with AMD, and expands the result keeping each node's unknowns adjacent. It
auto-detects `nv` and the layout by scoring candidates, always keeping plain scalar AMD in the
running, so it can only be chosen when it predicts less fill.

Use it with **`SupernodalLU` or `LeftRightLU`**. Measured end to end on `testdata/setfos_2`
(3 variables × 1016 nodes), stored scalars in `LeftRightLU`'s factor:

| ordering | fill | factor |
|---|---:|---:|
| Eigen AMD on the 3048-node scalar graph | 3,933,570 | 188 ms |
| COLAMD on the scalar graph | 2,360,714 | 96 ms |
| **AMD on the 1016-node node graph** | **1,651,762** | **75 ms** |

The mechanism is more mundane than "a node's unknowns belong together": AMD is a heuristic and
simply does a better job on the 3× smaller graph. Ordering the node graph lands within a few
percent of the best symmetric-pattern ordering known for this matrix, where Eigen's scalar AMD
is 2.6x off it.

Two things it is **not**:

- **Not a good ordering for `PointBlockLU`.** It ranks candidates by *symmetric*-pattern fill,
  which is the right objective for the supernodal solvers and the wrong one for an
  unsymmetric-pattern LU. On `gemat11` it gives `PointBlockLU` 6,742,455 stored scalars against
  COLAMD's 79,479.
- **Not a way to make a conservative union pattern cheap.** `setfos_2` ships 724,732 stored
  entries of which only 46,449 are numerically nonzero, and that costs **16x in factor flops**
  (4.02e8 against 2.45e7) under every ordering tried, node-major and variable-major alike. If
  the pattern can be tightened, tighten it — no ordering will do it for you. (Beware measuring
  this with a sparse permutation product: it silently prunes explicit zeros and makes the
  penalty appear to vanish.)

Its auto-detection costs several symbolic passes, so `analyzePattern()` gets slower — 36 ms to
241 ms on `setfos_2`. That is amortized to nothing in a Newton loop with a fixed pattern, but
set `setVariablesPerNode(nv)` explicitly to skip the search.

## License

Mozilla Public License 2.0 (`LICENSE`), matching the surrounding Eigen code this solver
integrates with.

`THIRD-PARTY-NOTICES.md` records the external work these solvers build on, and
distinguishes **algorithmic lineage** (published algorithms reimplemented from
their descriptions — PaStiX's supernodal design, PARDISO's scheduler, Duff &
Koster's MC64) from **code derivation**. No third-party source is incorporated:
everything under `src/` is original code. Note in particular that PaStiX is
CeCILL v2, a copyleft license incompatible with MPL-2.0 redistribution, which is
why its design is reimplemented rather than translated.
