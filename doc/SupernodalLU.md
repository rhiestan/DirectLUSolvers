# SupernodalLU

*[← DirectLUSolvers](../README.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockLU](PointBlockLU.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md) · [Parallelism](Parallelism.md) · [Testing](Testing.md)*

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

## Scope — read this first

SupernodalLU factors **`A = P^T L U P`** for matrices with **general (unsymmetric) values but a
symmetric nonzero pattern** — i.e. `A(i,j) != 0 <=> A(j,i) != 0`, in *pattern* only, not value.
If your matrix's pattern isn't already symmetric, symmetrize it with explicit structural zeros
first (`compare_testdata.cpp`'s `symmetrizePattern` shows how); this does not change the
operator, since a genuine zero contributes nothing to the sum. Cholesky/LDLᵀ are **not**
implemented (LU only, for now).

> **This restriction applies to `SupernodalLU` only.** The sibling
> [`LeftRightLU`](LeftRightLU.md) takes an **unsymmetric nonzero
> pattern** directly and symmetrizes internally, at the one point in the pipeline where it is
> cheapest — pre-symmetrizing its input instead costs 102x the fill on `gemat11`. If your
> pattern is unsymmetric, reach for `LeftRightLU`; see [Unsymmetric nonzero
> patterns](LeftRightLU.md#unsymmetric-nonzero-patterns).

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
  - [`Eigen::HeaderOnlyMetisOrdering<StorageIndex>`](HeaderOnlyMetis.md) (`HeaderOnlyMetis.h`,
    **no library to link**) — a templated reimplementation of `METIS_NodeND` that produces *bit-identical*
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
  elimination-tree-level and intra-supernode parallelism (see
  [Parallelism](Parallelism.md)). Default `supernodal_lu::SerialExecutor` (no threading at all).

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
  > Measured over the [SuiteSparse corpus](Testing.md#the-suitesparse-corpus): of 8 matrices whose
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

  Measured over the [SuiteSparse corpus](Testing.md#the-suitesparse-corpus), MC64 **fixes
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
  `solver.executor() = supernodal_lu::OpenMPExecutor(8);`. See [Parallelism](Parallelism.md).
- **`setIntraSupernodeParallelism(bool on)`** (default **on**) — elimination-tree levels with
  fewer supernodes than worker threads (typically the few huge separator supernodes near the
  root, which otherwise serialize) run their supernodes one at a time but parallelize *inside*
  each one instead: the Schur-update GEMMs and off-diagonal TRSMs are split into disjoint chunks
  dispatched across the executor. Chunk extent scales with the executor's
  `concurrency()` so a big supernode produces about one chunk per lane (see
  [Chunk sizing](Parallelism.md#chunk-sizing)); this is the single largest contributor to
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
  [Parallel triangular solve](Parallelism.md#parallel-triangular-solve)). Systems below
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
  concurrency of plain level-parallelism (see [Parallelism](Parallelism.md)).
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
`MetisOrdering` rows report throughout these documents — the only thing that changes is the
build. See [HeaderOnlyMetis](HeaderOnlyMetis.md) for what parity is defined against, the module
map, and the deterministic parallel variant.

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
  [Performance notes](LeftRightLU.md#performance-notes-honest-summary). **Give every solver its
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
  [Parallel scaling](Parallelism.md#parallel-scaling-measured) for the per-phase breakdown.
- All of the above is measured on `DirectLUSolvers/test/compare_testdata.cpp`'s matrix set —
  benchmark your own matrices before drawing conclusions for your workload.

