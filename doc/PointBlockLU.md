# PointBlockLU — unsymmetric-pattern solver for sparse factors

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockOrdering](PointBlockOrdering.md) · [Testing](Testing.md)*

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
  sequence that survives refactorization better, at some stability cost.
- The default ordering is **COLAMD**, not `PointBlockOrdering` — see the note in
  [PointBlockOrdering](PointBlockOrdering.md).

