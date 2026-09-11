# SupernodalLDLT — symmetric, definite or indefinite

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [RobustLU](RobustLU.md) · [PointBlockLU](PointBlockLU.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md) · [Parallelism](Parallelism.md) · [Testing](Testing.md)*

`Eigen::SupernodalLDLT` (`src/SupernodalLDLT.h`, `#include <SupernodalLDLT>`) factors a
**symmetric** matrix as `A = P^T (L D L^T) P`, with `L` unit lower triangular and `D` block
diagonal — 1×1 and 2×2 blocks. The 2×2 blocks are what make it work on an **indefinite**
matrix, so saddle-point and KKT systems are in scope, not only positive definite ones.

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
| `lap2d 200²` | 40,000 | 31 / 42 / 39 | 1.35x | 1.77x |
| `lap3d 20³` | 8,000 | 32 / 55 / 88 | 1.72x | 1.82x |
| `lap3d 30³` | 27,000 | 310 / 617 / 1546 | 1.99x | 1.89x |
| `lap3d 40³` | 64,000 | 2640 / 5130 / 17724 | 1.94x | **1.93x** |

**The arena ratio is the reproducible half of this table, and it is the point.** Per supernode
the LU arenas hold `w² + 2·w·r` scalars against this solver's `w² + w·r`, so the ratio rises
towards 2 as the off-diagonal panels grow relative to the supernode width. It is exact and
deterministic. Factorization time tracks it but is not deterministic — the 3D rows move
between roughly 1.85x and 2.2x run to run, so read them as "about 2x", not as the third
decimal. The 2D row is lower on both counts for the same reason, read the other way: its
panels are shorter, so the shared `w²` term is a larger share.

Against `Eigen::SimplicialLDLT`, which factors the same matrix class, the gap grows to 6.5x on
`lap3d 40³` — that is the supernodal BLAS-3 kernels against scalar simplicial code, and it
widens with problem size. `SimplicialLDLT` wins on **analyze** time throughout (roughly 2-3x),
and on small 2D problems its cheaper analysis can make it the better total choice.

## Scope — read this first

The matrix must be symmetric (Hermitian for complex scalars). It need **not** be positive
definite.

**Only one triangle is read**, selected by the `UpLo` template parameter, so a caller holding
half the matrix pays nothing to use it. Passing a fully populated symmetric matrix works
equally well — the redundant half is ignored, not added twice.

### Indefinite matrices, and why the interchanges stay local

A symmetric matrix can have an arbitrarily small diagonal and still be perfectly well
conditioned, so "divide by the diagonal" is not available the way it is for a positive
definite matrix. `[[0,1],[1,0]]` is the smallest example: nothing on the diagonal to pivot on,
yet its eigenvalues are ±1. The fix is a 2×2 pivot block, chosen by the bounded
**Bunch–Kaufman** criterion, which bounds the growth of `L` either way.

A 2×2 pivot needs a *symmetric* interchange, and the interesting property is that it stays
confined to the dense diagonal block — so the precomputed symbolic structure is never
invalidated. The Schur complement a supernode sends out is unchanged by it:

```
L_R D L_R^H = A_R Pᵀ (L_kk D L_kk^H)⁻¹ P A_R^H
            = A_R Pᵀ (P A_kk Pᵀ)⁻¹ P A_R^H
            = A_R A_kk⁻¹ A_R^H          ← P cancels
```

`P` is therefore invisible outside the block, and the solve applies it locally: `P` before the
forward diagonal solve, `Pᵀ` after the backward one, with panel contributions untouched.

**The one case where the bargain costs something** is a 2×2 pivot wanted at the *last* column
of a supernode, where there is no second column to pair with. Growing the structure to find
one — a "delayed pivot", as MUMPS and PARDISO do — is exactly what this design refuses. A
perturbed 1×1 pivot is taken instead, counted in `straddlingPivots()`, and refinement cleans
up after it. A count large relative to `supernodeCount()` suggests a smaller `setMaxBlockSize()`
is ending blocks in awkward places.

### Inertia

The signs of `D` give the counts of positive, negative and zero eigenvalues (Sylvester's law
of inertia), free once the factorization exists. `inertia()` reports them. A Bunch–Kaufman 2×2
block always has a negative determinant, so it contributes exactly one of each sign.

This is checkable rather than merely plausible: a KKT matrix `[[H, Bᵀ], [B, 0]]` with `H`
positive definite and `B` of full row rank has inertia exactly `(n₁ positive, n₂ negative)`,
and the test suite pins that against dense eigenvalues.

Where static pivoting fired, the inertia is that of the perturbed matrix — read it next to
`replacedPivots()`.

### Positive definiteness

You do not have to assert it. `setPivoting(Pivoting::None)` is the fast path that assumes it,
and its pivot test *is* the check: it reports a non-positive pivot rather than stepping over
one. The default handles either case, and `inertia()` then says exactly which it was.

On this project's SPD benchmarks Bunch–Kaufman chooses **no** 2×2 blocks and costs between
−9% and +2% against `Pivoting::None` — i.e. nothing measurable, which is why it is the
default. `Pivoting::None` is there for a caller who knows the matrix is SPD and has measured
that the search costs them something.

**No unsymmetric matching, and that is not a gap.** The LU siblings permute rows to put large
entries on the diagonal (MC64/transversal). That is an *unsymmetric* row permutation: applying
it to a symmetric matrix destroys the symmetry this solver exploits. The symmetric analogue is
Duff–Pralet matching, noted below, not MC64 as the LU solvers use it.

## Not implemented yet

Stated so it is not mistaken for an oversight:

- **Intra-supernode parallelism.** Factorization is dispatched over elimination-tree levels
  only, so the few enormous root separators run on one lane. [Parallelism](Parallelism.md)
  records that intra-supernode chunking is where most of `SupernodalLU`'s parallel speedup
  comes from, so expect this solver to *scale* worse than that one until it is ported, even
  though its serial work is smaller.
- **Symmetric weighted matching (Duff–Pralet).** Bunch–Kaufman chooses pivots *inside* a
  supernode; a matching would choose a better diagonal before the ordering ever runs, which is
  what an indefinite matrix with a genuinely awkward diagonal wants.
- **A blocked Bunch–Kaufman kernel.** The dense diagonal-block factorization is unblocked,
  deliberately: the block is capped by `setMaxBlockSize` (128) and the BLAS-3 work that matters
  is the off-diagonal panel, so a worse constant on a small term does not show up in the
  measurements above.
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

- **`setPivoting(Pivoting)`** (default **`BunchKaufman`**) — `BunchKaufman` for 1×1 and 2×2
  pivot blocks with symmetric interchanges, handling indefinite input; `None` for the positive
  definite fast path, which does no pivot search, carries no permutation through the solve, and
  reports a non-positive pivot rather than handling it.
- **`setStaticPivotThreshold(RealScalar)`** — magnitude below which a 1×1 pivot is replaced by
  a same-sign value of this magnitude. By default chosen each `factorize()` as
  `sqrt(eps) · max|Ã_ij|` of the equilibrated matrix. Pass `0` to make a zero pivot a reported
  failure instead. Only `BunchKaufman` perturbs; `None` reports.
- **`setRefineOnlyIfPerturbed(bool)`** (default **true**) — run refinement only when
  `replacedPivots() > 0`. An unperturbed `LDL^T` is backward stable and has nothing to repair,
  so a positive definite solve costs exactly the factor solve; a perturbed one factored a
  slightly different matrix, and refinement recovers the difference.
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
- **`setMaxIterativeRefinements(Index)`** (default `5`, gated by `setRefineOnlyIfPerturbed`) —
  stationary refinement. The natural Krylov method for a symmetric operator is conjugate
  gradients, or MINRES when it is indefinite, not the BiCGStab the unsymmetric siblings default
  to; the stationary loop here is for small corrections.
- **`setSolveFailureThreshold(RealScalar)`** (default `1e-6`) — `solve()` measures the true
  relative residual against the original operator and downgrades `info()` rather than return a
  bad answer silently. `solveResidual()` reports the measured value.
- **`setParallelSolve(bool)`** (default **on**) — dispatch the triangular sweeps across the
  `Executor` over elimination-tree levels. Systems below `rows × nrhs = 200000` stay serial.

Diagnostics: `info()`, `isFactorized()`, `lastErrorMessage()`,
`notPositiveDefiniteColumn()`, `replacedPivots()`, `pivotBlocks2x2()`, `straddlingPivots()`,
`inertia()`, `nnzL()`, `predictedFactorNonzeros()`, `supernodeCount()`, `levelCount()`,
`permutation()`, `solveResidual()`, `iterativeRefinements()`, `determinant()`,
`determinantSign()`, `logAbsDeterminant()`.

`determinant()` overflows on moderately sized systems exactly as it does on the LU solvers —
prefer `logAbsDeterminant()` with `determinantSign()`. `L` is unit triangular and the
interchanges are symmetric, so neither contributes a sign: `det(A)` has exactly the sign of
`det(D)`, and each 2×2 block contributes `-1`.

## Examples

### Only half the matrix

```cpp
// Assemble just the lower triangle -- the upper is never read.
Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver;
solver.compute(Alower);
Eigen::VectorXd x = solver.solve(b);
```

### A saddle-point system, and reading its inertia

```cpp
// [[H, B^T], [B, 0]] -- symmetric, indefinite, and exactly what the 2x2 pivots
// are for. Nothing special has to be requested.
Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>> solver;
solver.compute(kktLower);
Eigen::VectorXd x = solver.solve(b);

const auto in = solver.inertia();       // (positive, negative, zero)
std::cout << in.positive << " / " << in.negative << "\n";
std::cout << solver.pivotBlocks2x2() << " 2x2 pivots, "
          << solver.replacedPivots()  << " perturbed\n";
```

### Asserting positive definiteness

```cpp
Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>> solver;
solver.setPivoting(Eigen::supernodal_ldlt::Pivoting::None);
solver.compute(A);
if (solver.info() != Eigen::Success) {
  // The pivot test IS the check: this matrix is not positive definite, and the
  // message names the internal column whose pivot was not.
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
references are `Eigen::SimplicialLDLT`, a dense determinant, and `SelfAdjointEigenSolver` for
the inertia. Beyond accuracy and the determinant, it pins the three things specific to a
*supernodal* `LDL^T` that could each be wrong without moving a residual:

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

For the indefinite path specifically:

- **A matrix with a zero diagonal**, where every pivot must be 2×2 and there is nothing to
  divide by. `[[0,1],[1,0]]` is solved exactly, and a 120×120 random symmetric matrix with an
  empty diagonal to machine precision.
- **Inertia against dense eigenvalue signs**, on shifted Laplacians and on a KKT system whose
  `(n₁, n₂)` inertia is known in closed form. The shifts are deliberately non-integer: a
  `g × g` Laplacian has 4 as an *exact* eigenvalue whenever `p + q = g + 1`, so `A − 4I` is
  singular by construction and would be testing conditioning rather than inertia.
- **Complex Hermitian indefinite**, which is where the conjugation inside the 2×2 arithmetic
  has to be right.
- **`Pivoting::None` still declines** what Bunch–Kaufman accepts, with the same matrix passing
  under the default — the contrast is what makes the declining meaningful.
- **A genuinely singular matrix is flagged**, not returned as a success. Static pivoting will
  happily step over a zero pivot and produce a factorization of a nearby matrix; the check is
  that `solve()` then refuses to call the result trustworthy.

Also covered: equilibration independence of the determinant, the determinant *sign* on an
indefinite matrix, multi-RHS agreement with single solves, refactorization against a reused
symbolic structure, parallel-vs-serial agreement, the fill guard firing before allocation, and
the `n = 0` / `n = 1` / diagonal-matrix degenerate shapes.
