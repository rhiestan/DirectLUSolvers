# SupernodalLDLT — symmetric positive definite

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [RobustLU](RobustLU.md) · [PointBlockLU](PointBlockLU.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md) · [Parallelism](Parallelism.md) · [Testing](Testing.md)*

`Eigen::SupernodalLDLT` (`src/SupernodalLDLT.h`, `#include <SupernodalLDLT>`) factors a
**symmetric positive definite** matrix as `A = P^T (L D L^T) P`, with `L` unit lower
triangular and `D` diagonal.

```cpp
#include <SupernodalLDLT.h>

Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>> solver;   // Lower triangle by default
solver.compute(A);
Eigen::VectorXd x = solver.solve(b);
```

It shares its entire symbolic analysis with the LU solvers
(`SupernodalLUSymbolic.h`: elimination tree, postorder, supernode partition with
amalgamation, block structure, update lists, scheduling levels) and replaces only the
numeric core and the solve.

## Why it exists

Handed a symmetric matrix, `SupernodalLU` and `LeftRightLU` still compute both factors, and
`U` is the transpose of what `L` already holds. This solver keeps one of them. The saving is
structural — it is the redundant half of an LU of a symmetric matrix, not overhead — and it
shows up in both storage and time.

Measured on this project's generated Laplacians (`clang 22, -O3`, single-threaded, best of 3;
`SupernodalLU` with `setMatching(false)`, since an SPD matrix needs no matching):

| matrix | n | factor: LDLT / LU / Simplicial (ms) | speedup vs LU | arena ratio LU/LDLT |
|---|--:|---|--:|--:|
| `lap2d 200²` | 40,000 | 30.8 / 41.4 / 38.2 | 1.34x | 1.77x |
| `lap3d 20³` | 8,000 | 31.4 / 54.8 / 86.0 | 1.75x | 1.82x |
| `lap3d 30³` | 27,000 | 306 / 605 / 1426 | 1.97x | 1.89x |
| `lap3d 40³` | 64,000 | 2558 / 5280 / 16640 | **2.06x** | **1.93x** |

The arena ratio is the point, and it is not a tuning result. Per supernode the LU arenas hold
`w² + 2·w·r` scalars against this solver's `w² + w·r`, so the ratio rises towards 2 as the
off-diagonal panels grow relative to the supernode width, and factorization time tracks it.
The 2D row is lower on both counts for the same reason, read the other way: its panels are
shorter, so the shared `w²` term is a larger share.

Against `Eigen::SimplicialLDLT`, which factors the same matrix class, the gap grows to 6.5x on
`lap3d 40³` — that is the supernodal BLAS-3 kernels against scalar simplicial code, and it
widens with problem size. `SimplicialLDLT` wins on **analyze** time throughout (roughly 2-3x),
and on small 2D problems its cheaper analysis can make it the better total choice.

## Scope — read this first

The matrix must be symmetric (Hermitian for complex scalars) **and positive definite**.

**Only one triangle is read**, selected by the `UpLo` template parameter, so a caller holding
half the matrix pays nothing to use it. Passing a fully populated symmetric matrix works
equally well — the redundant half is ignored, not added twice.

**An indefinite matrix is declined, not approximated.** `LDL^T` with 1x1 pivots is unstable on
an indefinite matrix; the stable factorization needs 2x2 pivot blocks (Bunch–Kaufman), which
this solver does not implement. `factorize()` reports `NumericalIssue` and names the offending
column via `notPositiveDefiniteColumn()`, rather than returning a factor that exists and means
nothing. Saddle-point and KKT systems are the common case — symmetric and indefinite — and
belong in [`LeftRightLU`](LeftRightLU.md) until the 2x2 path exists.

Positive definiteness is not something you have to assert up front. The pivot test *is* the
check, it costs nothing, and it is exact: a matrix that factors here was positive definite in
the arithmetic actually used.

**No matching, and that is not a gap.** The LU siblings permute rows to put large entries on
the diagonal (MC64/transversal). That is an *unsymmetric* row permutation: applying it to a
symmetric matrix destroys the symmetry this solver exploits, and an SPD matrix does not need
it — its diagonal already dominates its row and column, and no pivoting is required for
stability. Symmetric weighted matching (Duff–Pralet) is what an *indefinite* path would need,
alongside the 2x2 pivots.

## Not implemented yet

Stated so it is not mistaken for an oversight:

- **2x2 pivots** (symmetric indefinite / Bunch–Kaufman), and with them the inertia that would
  fall out of the pivot signs for free.
- **Intra-supernode parallelism.** Factorization is dispatched over elimination-tree levels
  only, so the few enormous root separators run on one lane. [Parallelism](Parallelism.md)
  records that intra-supernode chunking is where most of `SupernodalLU`'s parallel speedup
  comes from, so expect this solver to *scale* worse than that one until it is ported, even
  though its serial work is smaller.
- **`matrixL()` / `vectorD()`** factor accessors.

## Storage

One contiguous panel per supernode, column-major, `(width + offDiag)` rows by `width` columns:

```
 ┌─────────────┐
 │ D on diag   │  w x w : L_kk strictly below the diagonal, D ON it
 │  L_kk below │        (L_kk's unit diagonal is implicit — the LAPACK
 ├─────────────┤         packed convention)
 │   L_sk      │  offDiag x w : the off-diagonal panel
 └─────────────┘
```

Only the lower triangle of the diagonal block is ever read or written. The backward sweep
reads the same `L` panel transposed, so there is no second factor anywhere.

## Options

The option surface is a subset of [`SupernodalLU`](SupernodalLU.md)'s, and the shared ones
behave identically because they drive the same code:

- **`setEquilibration(bool)`** (default **on**) — symmetric Ruiz scaling `Ã = S A S`. One
  scaling applied on *both* sides, because the two-sided `Dr A Dc` the unsymmetric siblings
  use would not preserve symmetry. Transparent to `solve()` and `logAbsDeterminant()`.
- **`setAmalgamation(relaxedSize, maxZeroRows)`** (default `(4, 4)`),
  **`setAmalgamationFillFraction(double)`** (default `0.3`), **`setMaxBlockSize(Index)`**
  (default `128`) — the partition pass is literally the same code as `SupernodalLU`'s, so its
  [documentation](SupernodalLU.md#ordering--fill-amalgamation-blocking) applies unchanged,
  including the banded-matrix caveat.
- **`setMaxFactorNonzeros(Index)`** (default `0` = off) — fail-fast fill guard; aborts before
  allocating the arena.
- **`setMaxIterativeRefinements(Index)`** (default **0**, off) — stationary refinement. Unlike
  the LU siblings there is nothing here for refinement to repair: with no pivoting and no
  static-pivot perturbation the factorization is backward stable, so the first solve is
  already as good as the arithmetic allows. Turn it on for an ill-conditioned system where the
  extra accuracy is worth a matvec plus a triangular solve per step. Note that the natural
  Krylov method for an SPD operator is conjugate gradients, not the BiCGStab the unsymmetric
  siblings default to; the stationary loop here is for small corrections.
- **`setSolveFailureThreshold(RealScalar)`** (default `1e-6`) — `solve()` measures the true
  relative residual against the original operator and downgrades `info()` rather than return a
  bad answer silently. `solveResidual()` reports the measured value.
- **`setParallelSolve(bool)`** (default **on**) — dispatch the triangular sweeps across the
  `Executor` over elimination-tree levels. Systems below `rows × nrhs = 200000` stay serial.

Diagnostics: `info()`, `isFactorized()`, `lastErrorMessage()`,
`notPositiveDefiniteColumn()`, `nnzL()`, `predictedFactorNonzeros()`, `supernodeCount()`,
`levelCount()`, `permutation()`, `solveResidual()`, `iterativeRefinements()`,
`determinant()`, `logAbsDeterminant()`.

`determinant()` overflows on moderately sized systems exactly as it does on the LU solvers —
prefer `logAbsDeterminant()`, which accumulates a sum of logs. There is no sign to carry: `D`
is positive throughout or the factorization would have failed.

## Examples

### Only half the matrix

```cpp
// Assemble just the lower triangle -- the upper is never read.
Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver;
solver.compute(Alower);
Eigen::VectorXd x = solver.solve(b);
```

### Detecting indefiniteness

```cpp
Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>> solver;
solver.compute(A);
if (solver.info() != Eigen::Success) {
  // Not positive definite: the message names the internal column whose pivot
  // was not positive, and points at LeftRightLU for the indefinite case.
  std::cerr << solver.lastErrorMessage() << "\n";
}
```

### Many right-hand sides against one factorization

```cpp
solver.analyzePattern(Alower);   // once per pattern
solver.factorize(Alower);        // once per set of values
Eigen::MatrixXd X = solver.solve(B);   // B: n x k
```

### Parallel factorization

```cpp
#include <SupernodalLDLT.h>

Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                      Eigen::AMDOrdering<int>,
                      Eigen::supernodal_lu::StdThreadExecutor> solver;
solver.compute(Alower);
```

### Nested dissection, nothing to link

```cpp
#include <HeaderOnlyMetis.h>
Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                      Eigen::HeaderOnlyMetisOrdering<int>> solver;
```

Eigen's AMD and METIS functors symmetrize whatever they are given, so passing a single
triangle orders the same graph the factorization eliminates. See
[HeaderOnlyMetis](HeaderOnlyMetis.md).

## Testing

```sh
ctest --test-dir build -R test_supernodal_ldlt --output-on-failure
```

`test/test_supernodal_ldlt.cpp` is dependency-free — every matrix is generated, and the
references are `Eigen::SimplicialLDLT` and a dense determinant. Beyond accuracy and the
determinant, it pins the three things specific to a *supernodal* `LDL^T` that could each be
wrong without moving a residual:

- **Only one triangle is read.** Lower, Upper and a fully populated input must all produce the
  same fill and the same answer. If the redundant half were added rather than ignored, the
  factor would be of `2A` off the diagonal — a wrong answer, not merely a different one.
- **Only one triangle of the factor is computed.** The arena is compared against
  `SupernodalLU`'s on the same matrix and required to be 1.5-2x smaller; at or under 1x would
  mean the second factor never went away.
- **The backward sweep reads the same `L` the forward sweep wrote**, including the conjugation
  on complex input. Dropping it makes the solver factor the (non-Hermitian) *symmetric* matrix
  instead, which still runs and gives a wrong answer — so there is a complex Hermitian case
  with an exact right-hand side.

Also covered: indefinite input is declined and names a column, equilibration independence of
the determinant, multi-RHS agreement with single solves, refactorization against a reused
symbolic structure, parallel-vs-serial agreement, the fill guard firing before allocation, and
the `n = 0` / `n = 1` / diagonal-matrix degenerate shapes.
