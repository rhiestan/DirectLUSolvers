# PointBlockLU — unsymmetric-pattern solver for sparse factors

*[← DirectLUSolvers](../README.md) · [RobustLU](RobustLU.md) · [SupernodalLU](SupernodalLU.md) · [SupernodalLDLT](SupernodalLDLT.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockOrdering](PointBlockOrdering.md) · [Testing](Testing.md)*

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

## When to use it

**When the factor stays sparse.** That is the whole envelope, and it is narrower than "small
matrices". Symmetrizing an unsymmetric pattern can cost enormous fill, and avoiding that is
what this solver buys; but a scalar column algorithm runs at roughly a fifth of the throughput
of a supernodal one, so once the factor densifies the fill advantage is spent and
`LeftRightLU` wins. Measured 2026-09-13, single-threaded, best of 5, COLAMD ordering,
`bench_solvers` (PointBlockLU timed on its **replay** — the call the target workload actually
makes; its `analyze` column is paid once):

| matrix | n | PointBlockLU fill / replay+solve | LeftRightLU fill / factor+solve | `Eigen::SparseLU` |
|---|---:|---:|---:|---:|
| `setfos` | 1015 | **4,080** / **0.04 ms** | 116,602 / 1.76 ms | 4,080 / 0.14 ms |
| `bayer05` | 3268 | 77,462 / **1.26 ms** | **58,036** / 4.33 ms | 126,396 / 5.24 ms |
| `gemat11` | 4929 | **79,614** / **1.54 ms** | 121,294 / 2.90 ms | 86,476 / 4.33 ms |
| `tomography` | 500 | **46,540** / **2.13 ms** | 164,836 / 4.55 ms | 91,650 / 5.50 ms |
| `sherman1` | 1000 | **32,916** / **0.73 ms** | 40,884 / 0.87 ms | 31,900 / 1.11 ms |
| `laoss_3` | 4180 | 731,852 / 37.4 ms | 1,210,476 / **23.8 ms** | 731,852 / 32.5 ms |
| `YaleB_10NN` | 2414 | 1,232,024 / 229.1 ms | 1,638,482 / **82.4 ms** | 1,226,238 / 111.8 ms |
| `setfos_2` | 3048 | 1,935,546 / 402.6 ms | 2,349,388 / **102.1 ms** | 1,935,897 / 111.2 ms |

Every row is the shipping configuration, so the `solve` half of each includes the residual check
each solver runs by default — see [What it costs](#what-it-costs) for what that is worth here.

`bayer05` is the one row where `LeftRightLU` carries *less* fill than `PointBlockLU`: it is
reducible, so [block triangular form](LeftRightLU.md#whats-different-from-supernodallu) leaves
it 2461 blocks whose largest is 97 columns, and there is almost nothing left to fill in. It is
still the slower of the two here — a supernodal solver has more per-block overhead than a
scalar one has arithmetic on blocks that small.

**The crossover sits near 100k stored scalars in the factor.** Below it `PointBlockLU` is the
fastest solver in this project — on `setfos`, `gemat11`, `sherman1`, `tomography` and
`bayer05` it beats `Eigen::SparseLU` and MKL PARDISO outright. Above it, use `LeftRightLU`.

It is often more *accurate* too, because it never perturbs a pivot: on `gemat11` its solve
error is 8.8e-13 against `LeftRightLU`'s 4.3e-08, on `tomography` 6.8e-14 against 7.2e-09,
and on the near-singular `bayer05` 1.1e-03 against `Eigen::SparseLU`'s 8.3e+00.

Equilibration iterates to convergence rather than a fixed sweep count, which matters at this
scale: the sweep is O(nnz) and runs on every replay, so a fixed eight sweeps was 80% of
`setfos`'s entire replay (98 µs against 19 µs with scaling off). It now costs one or two
sweeps on a well-scaled matrix, and the replay is 24 µs.

## Why PointBlockLU is not parallel

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
   the same fact the [parallel scaling](Parallelism.md#parallel-scaling-measured) section records for the
   supernodal solvers, where level/DAG parallelism alone never exceeded 1.79x and all the real
   scaling came from chunking *inside* dense panels. PointBlockLU has no dense panels to
   chunk — having none is the point of it.
2. **The tasks are too small to schedule.** A column of `bayer05`'s replay costs ~350 ns, while
   its plan has ~37k DAG edges each needing an atomic decrement to release a consumer. The
   bookkeeping costs what the arithmetic costs.

If you need threads on a matrix in this class, the lever is `LeftRightLU`, whose supernodal
panels are coarse enough to schedule — and by the time the factor is dense enough for threads to
matter, the crossover above has been passed anyway.

## Refactorization

`analyzePattern()` only chooses the column ordering — with partial pivoting the pattern of
`L` and `U` is not knowable until the values are seen, so the first `factorize()` does the
symbolic search (a depth-first reachability pass per column) alongside the numeric work and
records what it found. Every later `factorize()` replays that record. `refactorizations()`
reports how many replays have happened since the last full factorization.

A replay is **rejected** — and a full factorization redone automatically — when a pivot falls
below `setMinPivotRatio()` (default 1e-8) of the magnitude it had when the plan was recorded.
That check is what makes replaying safe as the caller's values drift;
`setForceFullFactorization(true)` disables replaying altogether.

## How much of the answer you may believe

Declining a singular matrix is the coarse half of that question. The fine half is that partial
pivoting can find a pivot in every column and still return an answer with **no correct digit in
it** — because the matrix was ill-conditioned rather than singular, and a residual cannot see
the difference. The bidiagonal `[1, -1.7]` at n=80 is the clean demonstration, and it is pinned
in `test_pointblock_lu.cpp`:

| | value |
|---|---:|
| κ₁(A), `conditionEstimate()` | 1.05e+19 |
| relative residual, `solveResidual()` | 2.2e-14 |
| backward error, `lastBackwardError()` | 2.0e-16 |
| **true relative error** | **73** |

The answer is backward stable — it is the exact solution of a system a rounding error away from
`A` — and it is wrong by a factor of 70. Three accessors close that gap, in increasing cost:

**`solveResidual()`, on by default.** `solve()` measures `‖b − Ax‖ / ‖b‖` against the original
`A` and downgrades `info()` to `NumericalIssue` past `setSolveFailureThreshold()` (default 1e-6,
the same as `LeftRightLU`; `0` disables the check and the work with it). This is the contract the
[README](../README.md) states for all three solvers.

**`setErrorBounds(true)`, off by default.** Adds the Oettli–Prager backward error and a
Hager–Higham condition estimate to every `solve()`, and downgrades `info()` when the forward-error
estimate reaches 1 — no digit supported — *even though the residual check passed*. That is the
n=80 row above, and it is the only setting that catches it. `lastCorrectDigits()` reports the
count directly.

**`conditionEstimate()`, `componentwiseBackwardError()`, `growthFactor()`, à la carte.** All lazy
and cached until the next `factorize()`; each costs nothing until called.
`conditionEstimate()` is exact on the closed-form yardstick `κ₁ = 3·(2ⁿ−1)` and a strict lower
bound elsewhere, as Hager's algorithm guarantees. Unlike `LeftRightLU`'s it describes **`A`
itself** rather than a statically perturbed stand-in, because this solver never replaces a pivot.

`growthFactor()` is what earns a relaxed `setPivotThreshold()`. That knob buys less fill by
allowing element growth, and the price was previously invisible — measured on `weakDiagonal(200)`:

| `setPivotThreshold()` | growth | fill | forward error |
|---|---:|---:|---:|
| 1.0 (strict partial pivoting) | 1.9 | 2141 | 1.4e-15 |
| 1e-8 | 5.4e+05 | 1698 | 4.5e-11 |

### What it costs

The factorization algorithm is untouched — the crossover table above still holds. Two things are
added: an O(nnz) copy of `A` per `factorize()`, and an O(nnz) matrix-vector product per `solve()`.
Measured 2026-09-13, min of 5, as a percentage of the phase each is added to:

| | `setfos` | `bayer05` | `gemat11` | `tomography` | `sherman1` | `laoss_3` | `YaleB_10NN` | `setfos_2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| copy of `A` per `factorize()` | 2.1% | 0.5% | 0.6% | 0.3% | 0.1% | 0.1% | 0.0% | 0.1% |
| residual check per `solve()` | 29% | 23% | 27% | 30% | 13% | 8.5% | 2.2% | 20% |
| **replay + solve together** | **+11.8%** | **+2.0%** | **+2.2%** | **+0.9%** | **+0.7%** | **+0.2%** | **+0.0%** | **+0.2%** |

The replay needs a copy of `A` because `solve()` is handed neither the matrix nor anything
equivalent — the factors describe the *equilibrated* `A~`, not the matrix the caller asked about.
The copy is a value `memcpy` whenever the pattern is the one already held, which in a Newton loop
is every call after the first. The worst row is `setfos`, the smallest and sparsest matrix, where
the whole Newton iteration goes from 38 µs to 43 µs; everywhere else the cost is under 3%.
`setSolveFailureThreshold(0)` removes the per-solve half of it.

**How those numbers were taken, because the obvious method gives the wrong answer.** Each column
above compares two code paths compiled into the *same* executable — the residual check against
`setSolveFailureThreshold(0)` in the same binary, and the copy timed in place. That is deliberate.
Timing this phase in a build *before* the change and a build *after* it does not work here: adding
~400 lines to this header relocates the hot replay loop, and the resulting layout shift is larger
than the thing being measured. Interleaved before/after runs of `bench_solvers` put the replay
12-15% **slower**; the same interleaved experiment in a small single-solver probe put it 12-13%
**faster**; the true added work is the sub-1% copy in the table. Both cross-binary answers are
layout artifacts. Where a measurement has to span two builds, check the rows the change cannot
have touched first — here `Eigen::SparseLU` and `SupernodalLU` stayed within ±1%, which is what
says the machine was steady and the movement was in the compiled layout.

## Deltas from the other two solvers

- **Structurally singular input is declined, not patched.** An unsymmetric LU needs a pivot in
  every column, so `testdata/bcsstm13` (762 numerically empty columns out of 2003) returns
  `NumericalIssue` naming the column — exactly as `Eigen::SparseLU` does. The supernodal
  siblings appear to succeed there only because symmetrizing fills those columns in from the
  transpose; every solver's answer on that matrix carries a relative error of 0.62.
- **Single-threaded, with no `Executor` template parameter** — see [Why it is not
  parallel](#why-pointblocklu-is-not-parallel), which is a measurement rather than an
  omission.
- **No matching and no static pivoting.** Partial pivoting does that job; Ruiz equilibration
  is on by default (`setEquilibration`) and matters — `setfos_2` spans 4e48 in magnitude,
  where an unscaled pivot comparison is meaningless.
- **`setPivotThreshold(t)`** (default 1.0 = strict partial pivoting) keeps the structural
  diagonal as pivot when `|a_kk| >= t * max|a_ik|`. Lower values mean less fill and a pivot
  sequence that survives refactorization better, at some stability cost — `growthFactor()` is
  what makes that cost visible, see [How much of the answer you may
  believe](#how-much-of-the-answer-you-may-believe).
- **`conditionEstimate()` describes `A`, not a perturbed stand-in.** `LeftRightLU`'s estimate
  describes the operator static pivoting actually inverts, so it can read *better* than the true
  κ(A) on a matrix whose pivots were bumped, and `replacedPivots()` is what says whether that
  happened. This solver never replaces a pivot, so the question does not arise.
- The default ordering is **COLAMD**, not `PointBlockOrdering` — see the note in
  [PointBlockOrdering](PointBlockOrdering.md).

