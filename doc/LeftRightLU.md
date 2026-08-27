# LeftRightLU — PARDISO-style sibling solver

*[← DirectLUSolvers](../README.md) · [RobustLU](RobustLU.md) · [SupernodalLU](SupernodalLU.md) · [PointBlockLU](PointBlockLU.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md) · [Parallelism](Parallelism.md) · [Testing](Testing.md)*

`Eigen::LeftRightLU` (`src/LeftRightLU.h`, `#include <LeftRightLU>`) is a second sparse
direct LU solver in this project with the **same `Eigen::SparseLU`-compatible interface**
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

## What's different from SupernodalLU

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
   queue](Parallelism.md#work-stealing-ready-queue) for the measurements behind it. This runs as a single `parallelFor(0, P, worker)` over
   the same pluggable `Executor` — each worker is itself a complete sequential scheduler, so
   even a serial or fork-join executor drives it correctly (verified with the serial,
   `StdThreadExecutor`, and `OpenMPExecutor` backends). The one exception to "no barriers" is
   the tail: the narrow top levels are carved out of the DAG phase and swept after it, so the
   pool can be applied *inside* those few enormous separator supernodes (`parallelFor` is not
   nestable, so the DAG phase has to end first). That boundary is deliberate and measured —
   see [Work-stealing ready queue](Parallelism.md#work-stealing-ready-queue).

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

5. **Block triangular form** (Dulmage–Mendelsohn; this is what `KLU` is built around, and
   neither `SupernodalLU` nor `Eigen::SparseLU` has it). Many unsymmetric matrices —
   circuit, chemical-process, economic, web — are *reducible*: a symmetric permutation puts
   them in block upper triangular form, and only the **diagonal blocks** need factoring.
   `analyzePattern()` finds that form by following the maximum transversal it already
   computes with a strongly-connected-components pass (`src/LeftRightLUBlockTriangular.h`),
   then factors each diagonal block independently — its own fill-reducing ordering,
   elimination tree and supernodes. The off-diagonal blocks are **never eliminated**: they
   are applied once each in a block back-substitution, so they generate no fill and
   contribute exactly.

   On by default, and self-cancelling when it cannot help. An irreducible matrix yields one
   block and takes exactly the path it took before — bit-identically, which
   `test_btf` asserts — for the cost of one `O(n + nnz)` sweep. That sweep is
   **5–100× cheaper than the matching that precedes it** (measured: `n=251k` → 14 ms SCC
   vs 122 ms matching vs 108 ms AMD; `pre2` `n=659k` → 52 ms vs 6.8 s). A structurally
   symmetric pattern is always irreducible in this sense — its SCCs are just connected
   components — so every PDE/FEM system pays only that sweep.

   Where it *does* apply the gain is not proportional to the block sizes, because fill is
   superlinear in them. Measured on this project's fill baselines (`LeftRightLU`, AMD and
   COLAMD columns; 46 of 50 changed rows fell):

   | matrix | largest block | fill before → after |
   |---|---|---|
   | `Simon/raefsky6` | 1 (0.0% of n) | 2,187,054 → **6,804** (0.003×) |
   | `Simon/raefsky5` | 1 (0.0%) | 2,604,252 → **12,632** (0.005×) |
   | `Pajek/SmaGri` | 9 (0.8%) | 432,284 → **2,212** (0.005×) |
   | `upwind2d_120x120` | 1 (0.0%) | 849,770 → **28,800** (0.034×) |
   | `TSOPF/TSOPF_RS_b9_c6` | 18 (0.2%) | 1,052,700 → **43,478** (0.041×) |
   | `bayer05` | 97 (3.0%) | 374,866 → **37,548** (0.100×) |
   | `Mallya/lhr10c` | 3,658 (34.3%) | 2,807,074 → **2,117,404** (0.754×) |

   `raefsky5`/`raefsky6` and the upwind grids reduce to **blocks of size 1** — fully
   triangular after matching, so `L` and `U` are the diagonal, there is nothing to factor,
   and the solve is pure substitution.

   Two consequences worth knowing. **Pivoting is confined to a diagonal block**: without
   BTF, partial pivoting on a column could legally take a row from an earlier block, which
   would destroy the structure. In practice this is a net stability gain — far fewer
   operations means far less accumulated rounding — and it shows: `Mallya/lhr10c` under
   COLAMD goes from a *flagged* 4.1e-4 to **2.8e-13 (`Success`)**, and `Hohn/fd12` from
   1.0e-7 to **7.2e-16**. On a matrix that is genuinely singular the residual can instead get
   much *louder* (`Pajek/SmaGri`, structurally singular with 563 of 1059 pivots bumped, goes
   from a meaningless-but-finite 2.6e-3 to 1.5e+29) — both answers are flagged, and BTF
   stops 195× of spurious fill from masking the singularity. **And fill is not monotone**:
   each block is ordered separately and a fill-reducing ordering is a heuristic, so a matrix
   that barely splits can come out marginally worse — worst case on the corpus +1.3%
   (`Bai/rw5151` under COLAMD: 6 blocks, the largest still 99.9% of `n`).

   BTF needs the zero-free diagonal the matching provides, so it is skipped when
   `setMatching(false)` removes it. Note that `matrixL()`/`matrixU()` then expose the
   **block-diagonal** factors, whose product is not `A` — `solve()`, `determinant()` and
   `logAbsDeterminant()` account for the off-diagonal blocks, direct factor access does not.

6. **Condition estimation and error bounds** (`src/LeftRightLUConditionEstimate.h`). A
   residual answers "does this x solve *a* system near the one I asked about?" and nothing
   else. It cannot distinguish an answer good to 13 digits from one good to none, because
   both have the same tiny residual on a matrix that is merely ill-conditioned. Two
   quantities close that gap, and **neither runs unless you ask for it**:

   - `conditionEstimate()` — Hager-Higham estimate of `kappa_1(A) = ||A||_1 ||A^-1||_1`,
     computed from the factors already in hand. `||A^-1||_1` is never formed: Hager's
     algorithm needs only products `A^-1 v` and `A^-H v`, so the whole thing costs **4-5
     triangular solves** — measured over this project's corpus, against an algorithmic ceiling
     of 10 — once, cached until the next `factorize()`. `conditionEstimateSolves()` reports
     what it actually spent.
   - `componentwiseBackwardError(b, x)` — Oettli-Prager,
     `max_i |b - Ax|_i / (|A||x| + |b|)_i`. Unlike the condition number this is **exact, not
     an estimate**, and costs one O(nnz) pass. Near machine epsilon means the solver did
     everything a backward-stable method could; whatever inaccuracy remains belongs to the
     matrix.

   Their product bounds the forward error, which is what `estimatedForwardError(b, x)` and
   `estimatedCorrectDigits(b, x)` report. Measured on an upper bidiagonal matrix with a
   superdiagonal of -1.7, where `n` dials the conditioning and nothing else changes:

   | n | kappa | backward error | forward bound | digits | **true error** | **residual** |
   |---|---|---|---|---:|---|---|
   | 8 | 2.7e+02 | 1.3e-16 | 3.6e-14 | 13 | 4.8e-16 | 3.2e-16 |
   | 40 | 6.4e+09 | 1.5e-16 | 9.2e-07 | 6 | 4.0e-08 | 3.5e-16 |
   | 60 | 2.6e+14 | 1.4e-16 | 3.7e-02 | 1 | 1.5e-03 | 3.3e-16 |
   | 80 | 1.1e+19 | 1.6e-16 | 1.0 | 0 | 7.2e+01 | 1.6e-14 |

   Read the last two columns together. The **residual is pinned at ~1e-16 in every row**, so
   a caller with only `info()` and `solveResidual()` sees four identical successes — while
   the true error walks from 5e-16 to 72. The forward-error column is the only one that tells
   them apart, and the solver is behaving impeccably throughout: the backward error never
   leaves machine epsilon.

   On this project's own benchmark matrices the estimate puts numbers on things the
   documentation could previously only call "near-singular": `bayer05` reports
   `kappa = 8.9e+26`, `bcsstm13` (the singular mass matrix) `1.7e+10`, `setfos_2` `1.2e+10`,
   against `1.1e+02` for a 3D Laplacian and `3.0` for an upwind grid.

   `setErrorBounds(true)` wires this into `solve()` itself — it then fills
   `lastBackwardError()`/`lastForwardError()`/`lastCorrectDigits()` and downgrades `info()` to
   `NumericalIssue` when the estimate leaves *no* digit standing, which is the n=80 row above.
   It is **off by default** so the ordinary solve path is exactly the work it always was.

   Three caveats, none cosmetic. The condition estimate is a **lower bound** (Hager maximises
   over a subset of the unit ball) — usually exact on this project's test matrices, and
   documented as almost always within a factor of 3, but it errs towards calling a matrix
   *better* conditioned than it is. It describes **the operator this factorization inverts**,
   which under static pivoting is a perturbed `A` — a perturbation regularises, so read it
   next to `replacedPivots()`. And the forward bound is **first order**, hence meaningless as
   it approaches 1; it is clamped there, which reads as "no digits are guaranteed" rather than
   a claim about the size of the error.

7. **Extended-precision residuals in refinement** (`src/LeftRightLUExtendedResidual.h`).
   Refinement's ceiling is the precision of `r = b - Ax`, and the consequence is sharper than
   it sounds: with `r` in working precision, refinement drives the **backward** error to
   `O(eps)` and stops — the forward error settles at about `kappa(A) * eps` and no number of
   further steps moves it, because `r` is already dominated by the rounding in forming it.
   Computing `r` in doubled precision drives the **forward** error to `O(eps)` too (Skeel
   1980; the basis of LAPACK's `xGESVXX`).

   `setExtendedPrecisionResidual(true)`, off by default. Measured on an integer matrix with
   an integer solution — so `b = A x` is exact and there is a real answer to converge to —
   with six refinement steps and only the residual precision differing:

   | kappa | no refinement | working-precision residual | **extended residual** |
   |---|---|---|---|
   | 3.2e+07 | 2.1e-10 | 2.1e-10 | **0** |
   | 6.4e+09 | 2.2e-08 | 2.2e-08 | **0** |
   | 1.3e+12 | 2.3e-06 | 2.3e-06 | **0** |
   | 2.6e+14 | 7.2e-04 | 7.2e-04 | **0** |

   **`long double` is deliberately not used.** On MSVC — and therefore on every Windows
   toolchain targeting it — it is a 64-bit alias of `double` (measured on this project's own
   toolchain: 8 bytes, 53 mantissa bits), so code written against it would silently do nothing
   on Windows while working on Linux. What runs instead is **double-double** arithmetic built
   from error-free transformations (`twoSum`, `twoProduct`), which is pure software and
   behaves identically everywhere, including for `float` and complex scalars.

   Three things worth knowing before turning it on:

   - **It changes the stopping rule, and has to.** The usual test — stop once
     `||r|| <= eps*||b||` — is *already satisfied* after a backward-stable direct solve, so it
     would exit before applying a single correction. With extended residuals,
     `Refinement::IterativeRefinement` instead watches the correction: keep going while
     `||dx||/||x||` is still halving, stop at `eps`. That is what LAPACK's `xGERFSX` does, for
     the same reason. Under `BiCGStab` the Krylov recurrence updates its residual from
     previous ones, which extended precision cannot reach — so the same stationary loop runs
     afterwards as a polish, and the flag means one thing under either method.
   - **It costs O(nnz) per refinement step**, against the correction solve's O(fill).
     Measured on this project's matrices: 1-13% of one triangular solve with hardware FMA,
     2-25% without it. On x86-64 that means building with `-mfma` or `-march=native`; the
     baseline target has no FMA and falls back to Dekker splitting, which is about 2x slower
     but computes exactly the same thing.
   - **`-ffast-math` / `/fp:fast` silently disable it.** Error-free transformations are
     algebraically trivial — their correction terms *are* zero in exact arithmetic — so a
     compiler allowed to reason about floating point as real arithmetic deletes them, and the
     extended residual quietly becomes the plain one. The `test_extended_residual` target pins
     strict FP for exactly this reason.

   It pairs with item 6: `estimatedForwardError()` is what says whether extended precision can
   help at all. Past `kappa * eps ~ 1` nothing recovers the answer, and the estimate is what
   tells you which regime you are in.

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

## Unsymmetric nonzero patterns

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

## Performance notes (honest summary)

Measured 2026-08-26 with `DirectLUSolvers/test/compare_testdata.cpp`, single-threaded
(`SerialExecutor`) — the numbers below do **not** exercise the barrier-free dynamic
scheduler's headline advantage, which requires a parallel executor (see
[Parallel scaling](Parallelism.md#parallel-scaling-measured) for the threaded numbers).

- On **symmetric-pattern** matrices from this project's real-world `testdata/` set,
  `LeftRightLU` tracks `SupernodalLU`'s factor+solve time closely or beats it modestly (both
  reuse the same analysis pipeline and static-pivoting numeric design — only in-block pivoting
  and scheduling differ): e.g. dendrimer 9.7ms vs 13.8ms, laoss_3 17.4ms vs 25.9ms. Same story
  on the large 3D FEM matrices — laoss_1 (251k rows) 2.9s vs SupernodalLU's 3.3s, laoss_2 (100k
  rows) 0.79s vs 0.87s — both well ahead of `Eigen::SparseLU` there (see the [SupernodalLU
  performance notes](SupernodalLU.md#performance-notes-honest-summary) for the SparseLU/PARDISO
  comparison).
- On **unsymmetric-pattern** matrices the two diverge sharply, because `SupernodalLU` must
  be handed a pre-symmetrized matrix and `LeftRightLU` symmetrizes internally *after* matching
  (see [Unsymmetric nonzero patterns](#unsymmetric-nonzero-patterns)): gemat11 **10.7ms vs
  1440ms**, bayer05 **8.1ms vs 290ms**, setfos_2 254ms vs 304ms. gemat11 and bayer05 are also
  where `SupernodalLU` loses accuracy outright (bayer05 err 1.7e+00 vs `LeftRightLU`'s
  6.3e-04), so this is not only a speed difference. bayer05 is the one row where the block
  triangular form does the work rather than the symmetrization point: it is reducible into 2461
  blocks, which is worth a further 2.8x on top (22.8ms before BTF existed).
- One measured, mechanistic difference: on a couple of already well-conditioned matrices
  (tomography, YaleB_10NN) `LeftRightLU` lands on a visibly looser — but still safely
  small — residual than `SupernodalLU` (tomography resid 4.3e-12 vs 2.6e-16; YaleB 1.2e-13 vs
  1.8e-16; both far under the 1e-6 `solveFailureThreshold()`). This is the documented
  **`setRefineOnlyIfPerturbed`** default at work: `LeftRightLU` skips refinement entirely when
  `replacedPivots()==0`, while `SupernodalLU`'s default BiCGStab refinement always runs at least
  one matvec check (and polishes further) even on an already-accurate direct solve. Not a bug —
  call `setRefineOnlyIfPerturbed(false)` if you want the tighter residual at the cost of that
  extra matvec.
- Complete (row+col) pivoting costs nothing extra over `SupernodalLU`'s row-only restricted
  pivoting in these measurements — both are confined to the small dense diagonal block
  (`setMaxBlockSize`, default 128), so the search cost is the same order regardless of how many
  interchange directions it considers.

## Option reference (deltas from SupernodalLU)

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
  [Diagnostics & queries](SupernodalLU.md#diagnostics--queries).
- **`predictedFactorNonzeros()`** and **`setMaxFactorNonzeros(Index)`** — the shared fail-fast
  fill guard (see the SupernodalLU [Diagnostics & queries](SupernodalLU.md#diagnostics--queries) section). Both
  solvers factor a symmetric pattern, so both can predict an infeasible factor on matrices without
  good separators; the guard turns that into a clean `NumericalIssue` before allocating.
- **`setMaxBlockSize`** (default 128) doubles as PARDISO's ~80-column panel cap and keeps the
  complete-pivoting search on a small dense block.
- **`setIntraSupernodeParallelism(bool on)`** (default **on**) — present, but it works
  differently here. The async scheduler cannot nest a fork-join inside a worker, so the narrow
  top levels are carved out of the DAG phase entirely and swept afterwards, sequentially, with
  the pool applied inside each supernode. It therefore engages only where the DAG has run dry;
  turning it off leaves the root-separator tail serial. Same numerical caveat as
  `SupernodalLU`'s (see [Work-stealing ready queue](Parallelism.md#work-stealing-ready-queue)).
- **`setBlockTriangularForm(bool on)`** (default **on**) — permute to block upper triangular
  form and factor only the diagonal blocks (see item 5 above). Requires `matching()` and is
  silently skipped without it. `btfBlockCount()` and `largestBtfBlock()` report what was
  found (`1` and `n` mean the matrix is irreducible and BTF cost only its `O(n + nnz)`
  sweep); `btfBlockPointers()` gives the block boundaries in the internal numbering, which
  is how you locate *which* block is singular; `btfOffDiagonalNonzeros()` counts the entries
  that are applied during the solve but never eliminated.
- **`setErrorBounds(bool on)`** (default **off**) — compute the condition estimate and the
  componentwise backward error inside every `solve()`, and let `info()` act on them (see item
  6 above). Off by default because the default solve path is otherwise exactly the factor
  solve and nothing else. The accessors `conditionEstimate()`,
  `componentwiseBackwardError(b, x)`, `estimatedForwardError(b, x)` and
  `estimatedCorrectDigits(b, x)` work regardless of this switch — it only controls whether
  `solve()` computes them for you and records them in `lastBackwardError()` /
  `lastForwardError()` / `lastCorrectDigits()`.
- **`setExtendedPrecisionResidual(bool on)`** (default **off**) — form refinement's residual
  in double-double arithmetic, which is what converts refinement's guarantee from a small
  backward error into a small forward one (see item 7 above). Off by default: it costs O(nnz)
  per refinement step, and buys nothing when refinement does not run at all — which by default
  is whenever no static pivot was perturbed.
- Error bounds are **not** computed for `transpose()`/`adjoint()` solves: `kappa_1(A^T)` is
  `kappa_inf(A)`, not `kappa_1(A)`, and reporting the wrong one would be worse than reporting
  none. Those solves leave the recorded values NaN and `lastCorrectDigits()` at `-1` rather
  than letting an earlier solve's numbers stand in.
- `levelCount()`/`widestLevel()` are **diagnostics only** — the scheduler does not use
  levels to schedule (there are no level barriers).

## Testing

```sh
ctest --test-dir build -R "test_leftright_lu|test_btf" --output-on-failure
```

`test/test_leftright_lu.cpp` covers direct/multi-RHS solves, factor accessors, transpose/
adjoint, all three pivoting modes, the forced column-swap path (matching off + weak diagonal)
with its solve/transpose/determinant folding, log-determinant, equilibration, honest failure
reporting, and parallel(dynamic-scheduler)-vs-serial agreement plus a deadlock-stress loop.

An `Unsymmetric nonzero patterns` block covers that input class specifically: accuracy on
random unsymmetric patterns and on upwind advection grids, `patternSymmetry()` against a
reference computed from `A` and `Aᵀ`, equality of fill and answer between a raw and a
pre-symmetrized input, the AMD/COLAMD/Natural ordering conventions pinned by fill (a misread
permutation is invisible in the residual — pinned with BTF *off*, since an upwind grid is
fully triangular after matching and would otherwise leave nothing to order),
structural-singularity reporting, and unsymmetric patterns combined with multi-RHS,
transpose, determinant and the parallel scheduler.

`test/test_btf.cpp` covers the block triangular form in two layers. The decomposition
itself is checked on graphs whose block structure is known by construction (a cycle is
irreducible; a triangular matrix is `n` singletons; a chain of cycles has exactly as many
blocks as cycles), including the structural invariant the design rests on — that Tarjan's
pop order really does produce a block *upper* triangular form, verified directly rather
than argued. The solver layer runs **BTF on against BTF off on the same matrix**: same
solution, same transposed solution, same `log|det|`, no meaningful fill increase. That
comparison is the load-bearing one — a block solved out of order, an off-diagonal entry
applied to the wrong block, or a missed conjugation in the adjoint path all surface as a
wrong *answer*. Irreducible inputs are additionally required to be **bit-identical** with
BTF on and off, which is what makes the "free when it cannot help" claim testable rather
than rhetorical. Also covered: fully reducible input (zero fill), dense off-diagonal
coupling, complex adjoint solves, refactorization against a reused block structure,
structural singularity, and BTF correctly switching itself off when matching is off.

`test/test_condition_estimate.cpp` covers the error bounds in three layers: the estimator
against matrices whose exact `kappa_1` is known in closed form (an upper bidiagonal with a
superdiagonal of -2 has `kappa_1 = 3(2^n - 1)`, so the conditioning is dialled purely by `n`)
and against a dense inverse on real matrices; the backward error against its defining
properties, including invariance under row scaling, which is what distinguishes it from a
normwise residual; and the contract itself -- that a default `solve()` spends **zero**
estimator solves, that enabling the bounds leaves the solution bit-identical, and that the
digit count tracks conditioning while the residual does not. The `-2` and `-1.7` bidiagonals
are both there on purpose: with `-2` every operation is exact in binary floating point, so
the backward error is identically zero and no amount of conditioning costs a digit -- correct,
and useless as a test of the bound.

`test/test_extended_residual.cpp` covers the error-free transformations against their
defining property (`p + e` reproduces `a*b` exactly, including where the product needs more
than 53 bits), the residual routine, and the forward-vs-backward claim itself. Its matrices
are integer-valued on purpose, and that is the subtle part: if `b` is formed as a
floating-point product `A*xTrue`, then `xTrue` is *not* the exact solution of the stored
system — `b` carries its own rounding, worth roughly `kappa*eps` of forward error, and no
residual precision can see past it. An experiment built that way shows the feature doing
nothing at all, for reasons that have nothing to do with the feature.

Whether BTF actually pays on a given matrix is a separate question from whether it is correct,
and `bench_btf` answers it by running the solver both ways — see
[Does the block triangular form pay?](Testing.md#does-the-block-triangular-form-pay).

