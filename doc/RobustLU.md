# RobustLU — the fallback ladder

*[← DirectLUSolvers](../README.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockLU](PointBlockLU.md) · [SupernodalLU](SupernodalLU.md) · [Testing](Testing.md)*

`Eigen::RobustLU` (`src/RobustLU.h`, `#include <RobustLU>`) is not a fourth factorization. It is a
**policy** over the ones that already exist: it runs `LeftRightLU`, measures how well that went,
and escalates to a different strategy only when the measurement says a different strategy would
help. Afterwards it tells you what it tried.

```cpp
#include <RobustLU.h>

Eigen::RobustLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);
Eigen::VectorXd x = solver.solve(b);
if (solver.info() != Eigen::Success) std::cerr << solver.report();
```

Escalation is **on by default** here — unlike every other option in this project. Choosing this
class *is* the opt-in; callers who want a single strategy keep using [`LeftRightLU`](LeftRightLU.md)
and pay none of this.

## What it buys, measured

Over this project's 33-matrix SuiteSparse quick tier:

| | count |
|---|---:|
| solved on the first rung, **one factorization, nothing else** | 19 |
| rescued by escalation that would otherwise have failed | 6 |
| stopped with a diagnosis (unsolvable in this precision, or past the fill guard) | 7 |

The first row is the one to watch. A ladder that quietly doubles everyone's work in order to
rescue a minority is a bad trade, and `test_robust_lu` asserts per matrix that a case the first
rung handles costs exactly one attempt — and returns a **bit-identical** answer to `LeftRightLU`
used alone.

## Diagnosis-directed, not cheapest-first

The obvious design climbs from cheap rungs to expensive ones. Measurement says that is wrong.
Extended-precision residuals need no refactorization at all — the cheapest possible rung — and
they rescue **nothing** on this corpus, because every failure here is a *backward*-error failure
and extended precision only converts an already-small backward error into a small forward one.

So the class does not climb. It measures, diagnoses, and jumps:

| observation | diagnosis | action |
|---|---|---|
| ω ≈ eps and residual small | fine | done, one attempt |
| `matchingIsPerfect() == false` | structurally singular | **rank-revealing QR** — no LU exists to try |
| ω, residual fine; κ·eps ∈ [1e-3, 1) | losing digits, recoverable | extended-residual **re-solve** (no refactor) |
| ω, residual fine; κ·eps ≥ 1 | the matrix, not the solver | **stop**, flagged |
| ω ≫ eps | the diagonal was unusable | **MC64** matching |
| ω ≫ eps after MC64 | in-block pivoting insufficient | **true partial pivoting** via `PointBlockLU` |
| every LU rung failed | numerical rank, not structure | **rank-revealing QR** — least squares + a rank |
| fill guard tripped | infeasible | **stop** |

Note where structural singularity now goes. Before the rank-revealing rung existed it was a dead
end; it is now the one diagnosis that reaches QR *without* an LU failure first, because there is
no point trying MC64 or partial pivoting on a matrix that has no LU at all.

MC64 is the workhorse — 4 of the 6 rescues. The partial-pivoting rung matters because
`LeftRightLU` confines pivoting to a diagonal block *by construction*; a failure that survives
MC64 needs a factorization that can take a pivot from anywhere, and
[`PointBlockLU`](PointBlockLU.md) already is one. Delegating costs a class member; building
dynamic threshold pivoting inside `LeftRightLU` would break its static-structure architecture.

## The rank-revealing rung

The terminal rung is `Eigen::SparseQR` with COLAMD ordering — genuinely rank-revealing, with a
SuiteSparseQR-style pivot threshold. It rescues two corpus matrices nothing else can:

| matrix | LU result | QR rank | QR residual | time |
|---|---|---|---|---|
| `Pajek/SmaGri` (structurally singular) | 1.6e-03 | **511 / 1059** | **2.7e-16** | 0.14 s |
| `Bai/rw5151` (every LU rung failed) | 2.1e+05 | **5150 / 5151** | **1.6e-08** | 26 s |

**The answer means something different, and that is reported rather than smoothed over.** QR
returns a *basic* least-squares solution — free variables set to zero — not the minimum-norm one,
which would need a complete orthogonal decomposition Eigen has no sparse version of. On `SmaGri`
(rank 511, so a 548-dimensional null space) the basic solution has norm 288 against the reference
solution's 35. Both satisfy `Ax = b` to machine precision; they differ by a null-space vector. So
`outcome()` becomes `RankDeficient`, `isLeastSquares()` returns true, `rank()` is exposed, and the
message says all of it. If your problem cares *which* solution it gets, this rung alone is not
enough.

**Its acceptance test has to be different from the LU rungs', in a way that is easy to get
backwards.** Demanding a small residual is wrong: on an inconsistent system a nonzero residual
*is* the answer. The right test is that the residual is orthogonal to the range of `A`,
`‖Aᵀr‖ ≤ tol·‖A‖·‖r‖`. But that test is meaningless when the residual is negligible — on `SmaGri`,
solved to 2.7e-16, the ratio reads 9.9e-02 because it is 0/0 noise, and using it alone would
reject a perfect answer. So: residual test when `r` is negligible, optimality test when it is not.

**The guard is on fill, not size.** This was measured after a first version guarded on rows and
would have let `Pajek/foldoc` run for over eight minutes at only 13,356 rows:

| matrix | LU fill | QR time | outcome |
|---|---:|---:|---|
| `SmaGri` | 348 k | 0.14 s | rescued |
| `shyy41` | 203 k | 7.8 s | rescued |
| `rw5151` | 583 k | 26 s | rescued |
| `lhr10c` | 57.1 M | 265 s | LU already had a better answer |
| `foldoc` | 51.6 M | **515 s** | correct, but 8.6 minutes |

Four clean orders of magnitude in LU fill separate the cases that pay from the ones that do not,
and the LU fill is already measured by an earlier rung. Default `setMaxRankRevealingFill(5e6)`.

What that default gives up is worth stating plainly: **`foldoc` is solvable this way** — 8.6
minutes for a correct answer where every LU strategy returns 1.8e+33 — and the default declines
it. A silent multi-minute stall is judged the worse failure mode, and the declined rung is logged
with its reason, so raising the guard is an informed choice rather than a discovery.

## Knowing when to stop is the hard half

Some corpus matrices cannot be rescued by any rung, and an MC64 attempt on `Mallya/lhr10c` costs
**11.9 seconds** while a QR attempt on `Pajek/foldoc` costs 515. Spending that to confirm a
failure is a bad trade, so the stopping rules carry as much weight as the escalation rules:

- **Structural singularity is checked before the acceptance test, not only on failure.** A
  singular matrix can still produce a small backward error on a consistent right-hand side —
  static pivoting bumps the zero pivot, the probe happens to lie in the range of `A`, and the
  answer looks fine. It is not: the factors are of a perturbed matrix, and another `b` exposes it.
- **A small backward error next to a huge condition number is a solver that did everything right
  on a problem with no answer in double.** No rung improves conditioning. The class says so and
  stops. Note the message does not overclaim: a condition number is a worst-case *bound*, so the
  answer may still be accurate — there is simply no way to tell from the residual, which is
  exactly why it is flagged.

This is why step 4 came after [condition estimation](LeftRightLU.md#whats-different-from-supernodallu):
"the solver did badly" and "the matrix is unsolvable" are indistinguishable from a residual, and
only the first is worth escalating.

## Two calibrations that were guesses first

**The acceptance threshold.** The first version rejected a rung whose backward error exceeded
`64·eps`, which sounded principled and escalated **8 matrices that were already fine**, at up to
11.9 seconds each. Measured properly: over the corpus, every matrix that solves acceptably has
ω ≤ 3.9e-08 (median 7.8e-16), while every matrix that does not starts at 1.2e-02. Six clear orders
of magnitude of gap. The default is now 1e-6, in the middle of it, matching the scale
`LeftRightLU::setSolveFailureThreshold` already uses.

**Acceptance needs the residual too, not just ω.** The componentwise backward error is relative to
`|A||x| + |b|`, so a solution whose magnitude has blown up makes ω look excellent while
`‖Ax−b‖/‖b‖` sits at 0.17. That is measured, not hypothetical — `Bai/rw5151` was accepted and
returned as a success until the residual joined the criterion. An unflagged wrong answer is the
one outcome this class exists to prevent, so acceptance is now the conjunction of both.

## The probe right-hand side

The awkward part of the design, stated plainly: a backward error needs a right-hand side, and
`compute()` has none. The ladder judges each rung against a deterministic probe — `b = A·x` with
`x_i = 1 + sin(i)/2` — on the reasoning that a factorization broken enough to matter is broken for
essentially any `b`. `setProbeRightHandSide()` supplies your own when you know it, and
`LeftRightLU`'s per-solve honesty check still measures the *real* right-hand side afterwards
either way.

## Option reference

- **`setMaxStrategy(Strategy)`** — how far the ladder may climb. `Strategy::Default` disables
  escalation entirely, leaving a thin wrapper over `LeftRightLU` plus the diagnosis.
- **`setBackwardErrorTolerance(RealScalar)`** (default `1e-6`) and
  **`setResidualTolerance(RealScalar)`** (default `1e-6`) — a rung must satisfy **both**.
- **`setMaxFactorNonzeros(Index)`** — the fill guard, applied to every rung.
- **`setMaxRankRevealingFill(long long)`** (default `5e6`) and
  **`setMaxRankRevealingSize(Index)`** (default `50000` rows, a backstop for when no LU rung
  produced a fill figure) — what the terminal QR rung is allowed to attempt.
- **`rank()`** and **`isLeastSquares()`** — meaningful only after the rank-revealing rung; `-1`
  and `false` otherwise, so a caller cannot mistake an unmeasured rank for a full one.
- **`setProbeRightHandSide(const DenseVector&)`** — see above.
- **`strategy()`, `outcome()`, `attempts()`, `report()`** — what was tried, what was accepted, and
  why it stopped. Each `Attempt` carries its backward error, condition estimate, growth factor,
  replaced-pivot count and wall time, whether it was accepted or not.
- **`backwardError()`, `conditionEstimate()`, `nnzL()`, `nnzU()`** — diagnostics from the accepted
  rung.

## A caveat worth knowing before you pin behaviour

Which rung rescues a given matrix is **not** always reproducible across builds. `Hohn/fd12` and
`Mallya/lhr10c` have rung-0 growth factors of 1e+15 and 1e+24 respectively, so their backward
error at that rung is essentially noise — and the noise moves with compiler flags. Built with
`-mfma`, `fd12` is rescued by MC64; built without it, `fd12` squeaks past the first rung outright.
Both are correct behaviour. `test_robust_lu` therefore asserts that the ladder reaches a
trustworthy answer, never which rung it took to get there: pinning that would pin the rounding,
not the ladder.

## Testing

```sh
ctest --test-dir build -R test_robust_lu --output-on-failure
```

Four properties, tested separately because they fail separately: **cost** (one factorization when
the first rung works, and a bit-identical answer), **escalation** (a failing matrix reaches a
strategy that works), **stopping** (structural singularity and hopeless conditioning halt after a
single attempt), and **honesty** (never a claimed success that is not real, and `report()` matches
the attempt log line for line). The corpus provides the real cases; constructed matrices cover the
stopping rules so the suite still means something without a download.

The ill-conditioned test matrix uses a superdiagonal of `-1.7` rather than `-2` deliberately —
with `-2` the whole factorization is exact in binary floating point, the answer comes out perfect
however enormous κ is, and flagging it would be a false alarm rather than a test.
