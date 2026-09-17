# MultifrontalQR — rank-revealing sparse QR

*[← DirectLUSolvers](../README.md) · [RobustLU](RobustLU.md) · [SupernodalLU](SupernodalLU.md) · [Parallelism](Parallelism.md) · [Testing](Testing.md)*

`Eigen::MultifrontalQR` (`src/MultifrontalQR.h`, `#include <MultifrontalQR>`) computes

    As · P · diag(I, V) = Q · R        As = diag(rowScaling) · A · diag(colScaling)

for a sparse matrix of any shape, **decides its numerical rank and checks that decision**, and
solves square, least-squares, underdetermined and rank-deficient systems. It answers the
questions the LU solvers stop at: *what is the rank of A, which columns are dependent, and which
of the many solutions of a singular system should come back?*

```cpp
#include <MultifrontalQR>

Eigen::MultifrontalQR<Eigen::SparseMatrix<double>> qr(A);   // A: m x n, real or complex
Eigen::Index r = qr.rank();
bool trusted   = qr.rankIsVerified();
Eigen::VectorXd x = qr.solve(b);   // minimum-norm least-squares solution by default
```

Two factorization engines sit behind the one interface — a multifrontal one with dense
block kernels and a scalar left-looking one — and `Engine::Auto` picks between them per matrix
(see [Two engines](#two-engines)). Dependency-free (Eigen + C++17), parallel through the same
[`Executor`](Parallelism.md) as the other solvers.

## Why not `Eigen::SparseQR`

Two defects of `Eigen::SparseQR`, both measured on this project's matrices, decide it:

- **Fill.** `SparseQR` never permutes rows. Its column elimination tree assumes a structural
  diagonal (`firstRowElt(k) = k`) and every reflector is forced to include row `k`; after the
  column ordering that fictitious entry couples unrelated columns and R goes nearly dense. On a
  4737-column, 14k-nonzero setfos matrix R has **5.6M** entries where 14k suffice. Putting the
  row permutation back (CSparse's `cs_vcount`) brings R to within 1–9% of the Cholesky bound of
  `|A|ᵀ|A|` and the factorization from 7944 ms to 3 ms. The multifrontal method never has this
  problem: a row enters the front of its leftmost column, by construction.
- **Rank on badly scaled matrices.** `SparseQR`'s threshold is `20 (m+n) eps · max column
  norm` of the *raw* matrix. On the complex DriftDiffusion matrices (entries up to 1e53) it
  reports rank **1 or 2** of 303–2025 while every LU solver solves them to 1e-13. Column
  scaling alone does not fix it (error 1e35 on the 2025 case); row *and* column equilibration
  does. Numerical rank is a property of a scaled matrix, so this solver scales first.

## How it works

**Analysis** (pattern only). Column ordering (COLAMD, AMD on the pattern of AᴴA, or natural;
`Auto` tries COLAMD and AMD and keeps the smaller predicted R — skipping AMD when COLAMD's R is
already within 2× of nnz(A), and stopping AMD's count once it passes COLAMD's), the column elimination tree
straight from A (never forming AᴴA), a postorder, and the front partition. The partition is the
shared one in `SupernodalLUSymbolic.h`, fed a *pseudo-adjacency* (each column's own rows are the
rows whose leftmost column it is — everything else arrives through the tree), with the
**cumulative-zero amalgamation rule**: the per-step rules the LU solvers use would merge an entire
chain elimination tree (every banded matrix) into one dense front, which only their width cap
prevents; a QR front is one dense block and wants no cap, so here the zero fraction of the whole
merged front is bounded instead.

**Numeric.** Each front assembles its original rows and its children's contribution blocks,
sorted by leftmost column so the front has a staircase shape, and is factored with blocked
compact-WY Householder transformations. A front with more remaining rows than columns compresses
its contribution block to at most `noff` rows before passing it up — an orthogonal
transformation among non-pivot rows is part of Q like any other. Fronts in the same tree level
run concurrently; a level with fewer fronts than lanes (the root separators, where most of the
work is) instead splits each trailing update into fixed-width column chunks across the executor.
The chunk width does not depend on the thread count, so **parallel results are bit-identical to
serial ones**.

### Two engines

The multifrontal engine above pays per front: assembly, sorting, a dense block with its
amalgamation zeros. While R stays sparse that bookkeeping costs more than the arithmetic, so a
second engine computes the same factorization without it — a **scalar left-looking Householder
QR** in compact sparse storage (Davis's `cs_qr`, with its `cs_vcount` row assignment, which is
exactly what `Eigen::SparseQR` lacks). Scaling, rank verification, the deferred block, the solves
and refinement are shared; only the loop that produces R and Q differs.

Its structure is static: every column has a fixed pivot row. So it cannot do what Heath's rule
does in a front — let a dead column's row serve a later column — and instead **defers** any
column whose pivot is at or below the threshold. The SVD of the deferred block then decides
those columns exactly. That is cheap for a few dependent columns and expensive for many
(`Pajek/SmaGri`'s 548 took 75 ms against 4.7 ms multifrontal), so `Auto` hands a matrix to the
multifrontal engine as soon as its deferred block would exceed `setScalarDeferralLimit` (64).

What decides between the engines is not the size of R but **how full its columns are**.
Measured over 77 matrices (both engines, best of three):

| nnz(R) per column | scalar vs multifrontal | examples |
|---|---|---|
| any, with nnz(R) ≤ 25k | **1.2–3.1× faster** | every setfos sample, `bwm2000` 2.9×, `b2_ss` 2.1× |
| ≤ ~38, nnz(R) up to ~250k | **1.1–2.9× faster** | `spmsrtls` (7/col) 2.3×, `ted_B_unscaled` (24/col) 2.4×, `TSOPF_RS_b9_c6` 1.5× |
| ≥ ~40 | 0.15–1.0× | `lap3d_8` (69/col, only 35k) 0.85×, `cell2` 0.42×, `sit100` 0.15× |
| 20M-entry R | does not finish in 60 s | `foldoc`, `as-caida`, `wiki-RfA` |

`Auto` therefore takes the scalar engine when nnz(R) ≤ 25k, or when nnz(R) ≤ 500k and at most
38 per column (the symbolic count stops as soon as it passes that budget, so asking is cheap).
Over the 77 matrices that choice is **1.56× faster than multifrontal-only** (geometric mean,
0.91–4.3×), 1.29× faster than scalar-only, and within 2% of the better engine per matrix; it
picked the scalar engine for 42 of them. Every rank was identical under all three settings.

The scalar engine is serial; the multifrontal one is the parallel path.

**Q is stored**, as per-front Householder vectors. It costs memory (nnzH is comparable to nnzR)
but is what makes the least-squares solve backward stable without relying on the semi-normal
equations.

## How rank is decided — and checked

1. **Heath's rule while factoring** (as SuiteSparseQR): a pivotal column whose remaining norm is
   at most `tol = rankTolerance() · max column norm` is *dead* — its entries above the current
   row stay (they are R12), the rest is dropped and accumulated in `droppedNorm()`. This catches
   exact and structural deficiency.
2. **Verification.** Heath's rule cannot see a dependency spread over several columns none of
   which is small on its own; then R11 is ill-conditioned while the rank looks full. After
   factoring, the smallest singular values of R11 are computed — by a dense SVD for a tiny R11,
   otherwise by block inverse iteration on (R11ᴴR11)⁻¹ with Rayleigh–Ritz. If any lies at or
   below `tol`:
   - the columns the near-null vectors lean on most are chosen (a pivoted QR of the vectors,
     then the heaviest columns by weight, growing geometrically with each repair, since a null
     vector spread over many columns is not isolated by deferring a few);
   - they are moved to the **end** of the column order and the matrix is refactored. They form
     one final front, which gets a plain QR followed by an **SVD of its triangular factor**:
     `R_D = U Σ Vᴴ`. U joins Q, V is the `deferredRotation()`, and the rank decision there is
     exact. Column pivoting is *not* good enough for this block: a column's residual norm is
     never below the singular value it hides, so any column-norm criterion overestimates rank
     when singular values cluster near `tol` (see `nnc1374` below).

   This repeats until the check passes (`setMaxRepairs`, default 8).

   The check has a cheap first stage. σ_min(R11) ≥ 1/(√r · ‖R11⁻¹‖₁), and a Hager–Higham
   estimate of that 1-norm costs a handful of solves (it is also what `conditionEstimate()`
   needs). When the bound clears the threshold by 1000× the iteration is skipped and
   `smallestSingularValues()` holds that single bound (`singularValuesAreBound()`). A matrix
   near its rank threshold never passes, so Kahan, `nnc1374` and `rw5151` still get the full
   iteration; on a well-conditioned matrix the check drops from ~25 solves to ~6.
   `setThoroughVerification(true)` always runs the full iteration.

What the decision rests on is reported rather than hidden:

| accessor | meaning |
|---|---|
| `rank()` | order of R11 |
| `smallestSingularValues()` | smallest singular values of R11, ascending — estimates of σ_r, σ_(r-1), … of the scaled matrix |
| `droppedNorm()` | Frobenius norm of everything dropped — an upper bound on σ_(r+1) up to rounding |
| `rankIsVerified()` | verification ran and no singular value of R11 is at or below the threshold |
| `absoluteRankThreshold()` | the `tol` actually used, in the scaled matrix |
| `deadColumns()` / `deferredNullity()` | where the deficiency was found: Heath-dead columns, and null directions inside the deferred block |
| `repairIterations()`, `deferredColumns()` | what verification had to do |

The ratio of `smallestSingularValues()[0]` to `droppedNorm()` is the size of the gap the rank
rests on. When it is small the rank is ill-determined *by the matrix* — no algorithm can do
better than report that.

**Measured against a dense SVD at the same threshold, in the same scaling:**

| matrix | SVD rank | Heath only | verified | repairs |
|---|---:|---:|---:|---:|
| Kahan, n = 100 (test) | 99 | 100 | **99** | 1 |
| `HB/nnc1374` | 1371 | 1374 | **1371** | 6 |
| `Bai/rw5151` | 5128 | 5146 | **5128** | 7 |
| `Shyy/shyy41` | 4712 | 4717 | **4712** | 4 |
| 450 random 0/1 graph matrices (test) | — | — | **all equal** | — |

`nnc1374` is the instructive one: its singular values around the threshold are 2.1e-11, 1.5e-11,
8.3e-12, 3.3e-12 against `tol` = 1.2e-11 — no gap at all. Column pivoting on the same deferred
block stops at 1373, and the repair loop can then only report failure; the SVD reaches 1371.

## What `solve()` returns

| case | answer |
|---|---|
| full rank, m = n | the solution, refined |
| full rank, m > n | the least-squares solution, refined on the augmented system (Björck) |
| rank r < n, `Solution::MinimumNorm` (default) | the **minimum-norm** least-squares solution — `pinv(A) b` up to the rank decision |
| rank r < n, `Solution::Basic` | the basic solution: dead columns and null directions zero |

The minimum-norm answer is the basic one projected orthogonally off the null space, in the
*original* (unscaled) inner product. The null-space basis `[-R11⁻¹R12; I]` is formed once per
factorization with all right-hand sides in one block solve per front, orthonormalized, and kept
in factored form; `nullSpace()` materializes it on request. Its size is capped by
`setMaxNullSpaceScalars` (default 5e7); beyond it the basic solution is returned and
`lastSolveMessage()` says so.

Refinement residuals are computed in **double-double** ([`LeftRightLUExtendedResidual.h`]
(../src/LeftRightLUExtendedResidual.h)). That is exact on the scaled matrix only because every
scaling factor is a power of two. On a κ ≈ 1e9 system with a binary-exact right-hand side, the
unrefined error is 5e-9, double-precision refinement leaves 2e-9, and extended-precision
refinement reaches machine precision. (The test uses an *exact* right-hand side on purpose: with
a rounded `b = A * xTrue`, xTrue is not the solution of the stored system, and extended
refinement correctly moves away from it.)

**Row scaling and least squares.** Scaling rows turns a least-squares problem into a
*weighted* one — the solver minimizes `‖Dr (A x − b)‖`. For a consistent system the two agree,
so `Scaling::Auto` scales rows only for a square matrix — and the one case where a square
system can be inconsistent, a rank-deficient matrix, is handled when it happens: when `solve()`
finds the residual clearly above rounding level, it factors the matrix once more with column
scaling alone (cached for later solves) and answers that column from it. On a 200×200 matrix
with an empty column and `b = 1`, the row-scaled answer has `‖Aᵀr‖ = 1.5`; the fallback's equals
`pinv(A) b` to 4e-15. Consistent systems — every benchmark here — never pay for it. When column
scaling alone decides a *different* rank (rows spanning many orders of magnitude), the fallback
is discarded, the row-scaled answer stands, and `isWeightedLeastSquares()` says so. For a
rectangular matrix `Auto` scales columns only; set `Scaling::RowsAndColumns` when the rank
decision matters more than the least-squares weights.

**Diagnostics.** `solveResidual()` is `‖b − Ax‖/‖b‖` in the original matrix; for an inconsistent
system that is the optimal residual, not an error. `leastSquaresOptimality()` is
`‖Aᴴr‖/(‖A‖_F ‖r‖)`, reported as 0 when `r` is at rounding level (there it is 0/0 noise —
a system solved to 1e-16 would otherwise read as far from optimal).
`conditionEstimate()` is κ₁(R11) of the scaled matrix (Hager–Higham).

## Performance, measured

Single thread unless noted; `err` is the forward error for a full-rank matrix.

**setfos samples** — every matrix, both solvers correct except where noted:

| matrix | n | Eigen::SparseQR factor | MultifrontalQR factor | nnz(R) Eigen → ours |
|---|---:|---:|---:|---:|
| DriftDiffusion `…n29841` | 4737 | 8974 ms | 0.86 ms | 5,588,544 → 14,219 |
| DriftDiffusion `…n722` | 3333 | 2348 ms | 3.1 ms | 5,277,925 → 70,946 |
| DriftDiffusion `…n12841` | 3006 | 1230 ms | 1.8 ms | 3,142,408 → 24,030 |
| Optics `…n26` | 1260 | 87 ms | 2.8 ms | 324,096 → 46,190 |
| complex AcSolver `…n1` | 2025 | 113 ms, **rank 2, err 1.0** | 2.6 ms, rank 2025, err 5e-09 | — |
| complex AcSolver `…n9` | 303 | 2.5 ms, **rank 2, err 1.0** | 0.17 ms, rank 303, err 3e-14 | — |

Factor times are `Engine::Auto` (the scalar engine, for all of these) and include rank
verification. With the cheap first stage verification costs about a third of the scalar
factorization on these matrices; `setRankVerification(false)` removes it.

**SuiteSparse corpus** — see [the comparison table below](#suitesparse-corpus-against-eigensparseqr).

**Parallel** (factor time, 1 → 16 lanes, `PooledExecutor`):

| matrix | nnz(R) | 1 lane | 8 lanes | 16 lanes |
|---|---:|---:|---:|---:|
| `Pajek/foldoc` | 19.8M | 19.7 s | — | 3.27 s (6.0×) |
| `SNAP/as-caida` | 9.8M | 15.9 s | — | 4.17 s (3.8×) |
| `SNAP/wiki-RfA` | 4.7M | 4.46 s | 1.14 s | 1.09 s (4.1×) |
| `Simon/raefsky2` | 3.6M | 706 ms | 328 ms | 302 ms (2.3×) |
| `Muite/Chebyshev3` | 8.4M | 572 ms | 409 ms | 430 ms (1.3×, one verification refactor) |

## SuiteSparse corpus, against `Eigen::SparseQR`

Both solvers single-threaded, `b = A·xTrue`; Eigen had 300 s per run. `rank` is each solver's
own decision; MultifrontalQR's was verified on every matrix. Our factor time is
`Engine::Auto`, best of three, and includes verification. Over the 34 matrices both finished,
MultifrontalQR is **5–6400× faster, median 57×**.

| matrix | n | Eigen factor | Eigen rank | Eigen resid | ours factor | engine | our rank | our resid |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `nasa1824` | 1824 | 535 ms | 1824 | 3e-15 | 19 ms | multifrontal | 1824 | 3e-16 |
| `bwm2000` | 2000 | 282 ms | 2000 | 1e-14 | 1.0 ms | scalar | 2000 | 6e-16 |
| `swang2` | 3169 | 3182 ms | 3169 | 4e-15 | 7.7 ms | multifrontal | 3169 | 1e-16 |
| `raefsky2` | 3242 | 3095 ms | 3242 | 5e-15 | 622 ms | multifrontal | 3242 | 4e-16 |
| `cell2` | 7055 | 66.6 s | 7054 | 5e-15 | 20 ms | multifrontal | 7054 | 3e-16 |
| `nemeth01` | 9506 | 92.4 s | 9506 | 2e-15 | 37 ms | multifrontal | 9506 | 3e-16 |
| `cryg10000` | 10000 | 210 s | 9999 | **1e-8** | 81 ms | multifrontal | 9999 | 1e-14 |
| `sit100` | 10262 | 216 s | 10261 | 5e-15 | 222 ms | multifrontal | 10261 | 1e-15 |
| `ted_B_unscaled` | 10605 | 8.4 s | **9183** | 3e-11 | 8.2 ms | scalar | 10605 (err 4e-16) | 2e-23 |
| `Trefethen_20000b` | 19999 | timeout |  |  | timeout | | | |
| `spmsrtls` | 29995 | 117 ms | 29995 | 3e-16 | 18 ms | scalar | 29995 | 1e-16 |
| `as-caida` | 31379 | timeout |  |  | 12.5 s | multifrontal | 7360 | 5e-16 |
| `nnc1374` | 1374 | 243 ms | **1093** | 6e-12 | 43 ms | multifrontal | 1371 | 3e-15 |
| `adder_dcop_11` | 1813 | 569 ms | **1793** | 3e-12 | 78 ms | multifrontal | 1813 (err 1e-11) | 1e-16 |
| `cavity10` | 2597 | 943 ms | 2597 | 4e-15 | 18 ms | multifrontal | 2597 | 3e-16 |
| `Chebyshev3` | 4101 | 3818 ms | 4100 | 2e-11 | 400 ms | multifrontal | 4099 | 1e-13 |
| `cavity17` | 4562 | 4194 ms | 4562 | 5e-15 | 39 ms | multifrontal | 4562 | 4e-16 |
| `shyy41` | 4720 | 7921 ms | 4715 | **4e-6** | 69 ms | multifrontal | 4712 | 2e-16 |
| `tols1090` | 1090 | 16 ms | 1090 | 7e-17 | 0.28 ms | scalar | 1090 | 7e-17 |
| `CAG_mat1916` | 1916 | 503 ms | 1916 | 1e-15 | 41 ms | multifrontal | 1916 | 4e-16 |
| `meg1` | 2904 | 1716 ms | **2890** | 1e-10 | 45 ms | multifrontal | 2904 (err 7e-12) | 7e-16 |
| `gemat12` | 4929 | 8173 ms | 4929 | 3e-16 | 7.8 ms | scalar | 4929 | 3e-16 |
| `rw5151` | 5151 | 25.4 s | 5149 | **3e-7** | 331 ms | multifrontal | 5128 | 3e-13 |
| `raefsky5` | 6316 | 2601 ms | **622** | 6e-10 | 51 ms | multifrontal | 6316 (err 3e-16) | 6e-17 |
| `lhr10c` | 10672 | 134 s | 10672 | 2e-15 | 134 ms | multifrontal | 10670 | 3e-14 |
| `foldoc` | 13356 | timeout |  |  | 16.4 s | multifrontal | 12919 | 3e-14 |
| `circuit204` | 1020 | 121 ms | 1020 | 1e-15 | 2.7 ms | scalar | 1020 | 1e-16 |
| `SmaGri` | 1059 | 42 ms | 511 | 3e-16 | 3.7 ms | multifrontal | 511 | 6e-16 |
| `b2_ss` | 1089 | 56 ms | 1089 | 4e-16 | 0.74 ms | scalar | 1089 | 2e-16 |
| `gre_1107` | 1107 | 180 ms | 1107 | 1e-15 | 8.8 ms | multifrontal | 1107 | 8e-17 |
| `mahindas` | 1258 | 59 ms | 1258 | 6e-16 | 3.1 ms | multifrontal | 1258 | 4e-16 |
| `cz1268` | 1268 | 46 ms | 1268 | 4e-15 | 1.5 ms | scalar | 1268 | 4e-16 |
| `extr1b` | 2836 | 796 ms | **2834** | 9e-13 | 2.3 ms | scalar | 2836 (err 1e-11) | 9e-17 |
| `raefsky6` | 3402 | 1129 ms | **690** | 7e-11 | 91 ms | multifrontal | 3402 (err 2e-14) | 1e-16 |
| `TSOPF_RS_b9_c6` | 7224 | 32.5 s | 7224 | 4e-15 | 5.1 ms | scalar | 7224 | 7e-17 |
| `fd12` | 7500 | 55.2 s | 7500 | 3e-15 | 29 ms | multifrontal | 7500 | 3e-16 |
| `wiki-RfA` | 11380 | 173 s | 3474 | 2e-15 | 3925 ms | multifrontal | 3474 | 4e-16 |
| `onetone2` | 36057 | timeout |  |  | 95 ms | multifrontal | 36057 | 1e-16 |

Bold marks an Eigen answer that is wrong: a rank far below the one at which the system solves
to a forward error of 1e-11 or better (`raefsky5`: 622 of 6316), or a residual on a consistent
system that shows the rank decision left an ill-conditioned R11 behind (`shyy41`, `rw5151`,
`cryg10000`). The largest ratios come from the missing row permutation (`TSOPF_RS_b9_c6`:
16.8M entries in Eigen's R against 87k), the smallest from matrices where Eigen's R is already
close to the true one (`spmsrtls`, 6.6×).

## Options

| option | default | effect |
|---|---|---|
| `setEngine(Engine)` | `Auto` | `Multifrontal`, `Scalar`, or `Auto` (see [Two engines](#two-engines)) |
| `setScalarEngineThreshold(n)` | 25000 | `Auto` takes the scalar engine up to this nnz(R) regardless of density |
| `setScalarEngineDensity(perCol, cap)` | 38, 500000 | … and beyond it while nnz(R) ≤ min(perCol · n, cap) |
| `setScalarDeferralLimit(k)` | 64 | `Auto` abandons the scalar engine when its deferred block would exceed k columns |
| `setOrdering(Ordering)` | `Auto` | `COLAMD`, `AMD` (on AᴴA), `Natural`, or `Auto` (both, smaller predicted R; AMD skipped when AᴴA is too large to form — `setMaxAtAPattern`) |
| `setScaling(Scaling)` | `Auto` | `RowsAndColumns`, `Columns`, `None`; `Auto` = rows+columns when square, with a column-scaled fallback for inconsistent least-squares solves (see [Row scaling](#what-solve-returns)) |
| `setRankTolerance(tau)` | 20 (m+n) eps | relative threshold; `0` keeps every nonzero pivot |
| `setRankVerification(bool)` | on | the check-and-repair loop |
| `setMaxRepairs(n)` | 8 | refactorizations verification may spend |
| `setDenseVerificationLimit(r)` | 24 | R11 order up to which verification uses a dense SVD |
| `setThoroughVerification(bool)` | off | always run the singular-value iteration, even when the condition-estimate bound already clears the threshold 1000× |
| `setSolution(Solution)` | `MinimumNorm` | or `Basic` |
| `setMaxNullSpaceScalars(x)` | 5e7 | largest null-space basis a minimum-norm solve forms |
| `setMaxRefinements(n)` | 3 | refinement steps per solve (stops early when the correction stalls) |
| `setExtendedPrecisionResidual(bool)` | on | double-double refinement residuals |
| `setAmalgamation(bool)` | on | relaxed front amalgamation |
| `setBlockSize(nb)` | 32 | Householder panel width |
| `setIntraFrontParallelism(bool)` | on | chunked trailing updates on narrow levels |
| `setMaxFactorNonzeros(n)` | off | refuse before allocating when predicted nnz(R) is larger |

## Limits

- **Fill is the fill of AᴴA's Cholesky factor.** `JGD_Trefethen/Trefethen_20000b` (entries at
  every power-of-two offset) does not finish in 300 s; no column ordering helps a matrix whose
  AᴴA is that dense. `setMaxFactorNonzeros` turns that into an immediate refusal.
- **Verification refactors.** Each repair is a full numeric refactorization; `rw5151` needs 7.
  The alternative — trusting Heath's rule — is wrong by 3 on `nnc1374`.
- **The minimum-norm solution needs the null space.** Its basis costs one factor-sized solve per
  null vector: 437 of them on `foldoc` make the first solve take ~2.3 s; later solves reuse it.
- **The scalar engine defers every dependent column.** With many of them the deferred block is
  a large dense SVD; `Auto` avoids it, but a forced `Engine::Scalar` does not.
- **The first inconsistent solve on a square rank-deficient matrix factors it again** (column
  scaling only, then cached), so that its least-squares answer is unweighted.
- **Weighted least squares under row scaling** — see above.
- `matrixR()` is upper trapezoidal only up to the placement of the deferred block (its rows come
  last, its columns after the dead ones); the identity it satisfies is the one at the top of
  this page.
