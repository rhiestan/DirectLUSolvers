# HeaderOnlyMetis — nested dissection with nothing to link

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockLU](PointBlockLU.md) · [Parallelism](Parallelism.md) · [Testing](Testing.md)*

`Eigen::HeaderOnlyMetisOrdering<StorageIndex>` (`src/HeaderOnlyMetis.h`) is a drop-in
replacement for `Eigen::MetisOrdering` that needs **no library on the link line**. It is a
header-only, templated reimplementation of METIS's `METIS_NodeND` — the one entry point
`Eigen::MetisOrdering` actually calls — and it produces **bit-identical** permutations to the C
library across this project's entire test corpus.

```cpp
#include <HeaderOnlyMetis.h>

Eigen::LeftRightLU<Eigen::SparseMatrix<double>,
                   Eigen::HeaderOnlyMetisOrdering<int>> solver;
solver.compute(A);
```

It is the **ordering functor alone**, deliberately not paired with a solver alias the way
`SupernodalLUMetis.h` is, so all three solvers can take it without the header dragging one
solver's definition in behind it:

```cpp
Eigen::SupernodalLU<Eigen::SparseMatrix<double>, Eigen::HeaderOnlyMetisOrdering<int>> solver;
Eigen::PointBlockLU<Eigen::SparseMatrix<double>, Eigen::HeaderOnlyMetisOrdering<int>> pb;
```

Being bit-identical, it gives exactly the fill and timings the `MetisOrdering` rows report
throughout these documents — the only thing that changes is the build.

## What "bit-identical" is defined against

Nested dissection is randomized, so the claim is only meaningful against a *specified*
reference. Parity holds against a METIS built with:

- **`IDXTYPEWIDTH=32`, `REALTYPEWIDTH=32`** — METIS's own defaults. The port's `RealT` is
  pinned to `float` for exactly this reason: the balance thresholds on the ND path are integer
  truncations of products involving `real_t`, so a wider type silently moves them.
- **GKlib's `GKRAND=ON`** — its portable MT19937-64, rather than the fallback to the platform
  CRT's `rand()`. With `GKRAND=OFF` (the CMake default) even the *reference* is not portable
  across compilers, and "bit-identical" would mean "identical to whatever this CRT does today".
  Flipping this switch changes METIS's random stream, its orderings, and therefore the fill of
  every `+METIS` row in [the regression baselines](Testing.md#fill-regression-baselines).
- **Strict floating point** — `-ffp-contract=off` on Clang/GCC, `/fp:strict` on MSVC, applied
  to the parity test targets in `CMakeLists.txt`. Without it a compiler may contract `a*b+c`
  into a single-rounding FMA wherever one side's source happens to keep the expression in one
  statement and the other's does not. That is a source-shape accident, not an algorithmic
  difference, but it can flip a borderline `float` comparison and cascade into a completely
  different — still valid — separator choice.

Everything is checked at these settings by `test_header_only_metis`; see
[Testing](#testing) below.

## Permutation direction

`HeaderOnlyMetisOrdering` mirrors `MetisOrdering`'s contract exactly, **including the direction
of the permutation it returns**: row (column) `i` of `A` is the `matperm(i)` row (column) of the
permuted matrix — METIS's `iperm`, not `perm`.

That is the convention `left_right_lu::OrderingConvention` and `point_block::OrderingConvention`
already declare in their primary template (`returnsInverse = true`), so this class needs **no**
specialization of those traits — and must not grow one. Reading the permutation the wrong way
round stays a valid permutation with a machine-precision residual and shows up only as fill,
measured elsewhere in this project at 250-350x on 3D FEM systems. It is not something a
numerical check would catch.

## Scope of the port

Only what `METIS_NodeND` reaches with `vwgt = NULL` and `options = NULL`, which is what
`Eigen::MetisOrdering` passes: unweighted vertices, `adjwgt ≡ 1` throughout, single-threaded,
default options. Concretely that is `SetupCtrl`'s defaults, `CompressGraph`, `SetupGraph`,
`MlevelNestedDissection`, `MlevelNodeBisectionMultiple/L2/L1`, `CoarsenGraph` (RM/SHEM matching
plus the 2-hop fallback), `InitSeparator`, `Refine2WayNode` (1-sided and 2-sided FM, plus
balancing), `SplitGraphOrder`, `MMDOrder` as the small-graph base case, and GKlib's MT19937-64
and counting sort.

Deliberately **not** ported, each confirmed unreachable on that path: `numflag`/Fortran
numbering (Eigen never sets it), `PruneGraph` (`ctrl->pfactor` defaults to 0),
`MlevelNestedDissectionCC`/`SplitGraphOrderCC` (`ctrl->ccorder` defaults to 0), and all of
k-way partitioning, meshes, ParMETIS, multi-constraint weights and the Fortran bindings.

| File | Contents |
|---|---|
| `src/HeaderOnlyMetis.h` | `Eigen::HeaderOnlyMetisOrdering` and `Eigen::HeaderOnlyMetisParallelOrdering` — the Eigen-facing functors, including the `A + Aᵀ` graph construction. |
| `src/HeaderOnlyMetis/Random.h` | MT19937-64 and the `irandArrayPermute` family, including the non-obvious 4-way blocked-swap shuffle that must be copied exactly rather than improved. RNG state lives in a `RandomState` owned by `Ctrl`, not in process globals. |
| `src/HeaderOnlyMetis/Graph.h`, `Ctrl.h` | Graph and control state; `Ctrl` also owns the RNG and the scratch workspace. |
| `src/HeaderOnlyMetis/Workspace.h` | A bump/slot allocator standing in for `ctrl->mcore` + `iwspacemalloc`, handing back **uninitialized** reusable scratch. |
| `src/HeaderOnlyMetis/Compress.h` | `CompressGraph` — on by default (`ctrl->compress = 1`). |
| `src/HeaderOnlyMetis/MinimumDegree.h` | `genmmd`/`MMDOrder`, the small-graph base case. No RNG involved. |
| `src/HeaderOnlyMetis/Sorting.h` | `BucketSortKeysInc` and the `ikvsort` family reachable on this path. |
| `src/HeaderOnlyMetis/Coarsen.h` | `CoarsenGraph`/`CoarsenGraphNlevels`, RM/SHEM matching, 2-hop fallback, coarse-graph construction. |
| `src/HeaderOnlyMetis/PQueue.h` | The FM priority queue. Templated, so it inlines — see the performance note below. |
| `src/HeaderOnlyMetis/InitialSeparator.h` | `InitSeparator` (`initpart.c` + `separator.c`). |
| `src/HeaderOnlyMetis/SeparatorRefinement.h` | `Refine2WayNode` — 1-sided and 2-sided FM, plus balancing. |
| `src/HeaderOnlyMetis/NestedDissection.h` | The driver: `MlevelNestedDissection`, the bisection trials, `SplitGraphOrder`, and the `nodeND()` entry point matching `METIS_NodeND`'s array signature. |
| `src/HeaderOnlyMetis/NestedDissectionParallel.h` | The level-synchronous parallel driver, `nodeNDParallel()` — a different, deterministic ordering (see below). |
| `src/HeaderOnlyMetis/Executor.h` | The duck-typed executor concept the parallel driver dispatches through. Deliberately *not* an include of `SupernodalLUExecutor.h`, so the port keeps its no-dependency property. |

## Performance against the C library

`test/profile_header_only_metis.cpp` times the port head-to-head against linked METIS on the
same graph and asserts the permutations still match, so a speed run is also a correctness run:

```sh
cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON -DDLU_BUILD_PROFILE_DRIVER=ON
./build/profile_header_only_metis --reps 8               # head-to-head table
./build/profile_header_only_metis --synthetic --side port  # clean profile target
```

Measured 2026-08-25: **1.06x the reference's ordering time**, bit-identical throughout; per
matrix 1.02-1.09 with two matrices *faster* than the reference, and `dendrimer` / `setfos_2` the
remaining ~1.2x outliers. It started at 1.17x, and what closed the gap was not algorithmic:

- The port's **FM refinement is faster than the reference's**, because `PQueue` is templated and
  inlines where METIS's `rpq*` functions are out-of-line calls (0.44 s of
  `rpqUpdate`/`rpqGetTop`/`rpqInsert` in the reference profile). Do not "optimize" `PQueue` into
  something that stops inlining.
- The entire real gap was allocator traffic and value-initialization of scratch that the
  reference gets uninitialized from `iwspacemalloc`. Three changes closed it — building each
  contracted vertex's edge list directly into `cadjncy` instead of a resize-and-append scratch
  vector (the largest single win), the `Workspace.h` bump allocator, and a `PQueue` heap that
  stays trivially constructible. All three are pure memory behaviour: no iteration order and no
  arithmetic changes, so bit-identity survives them structurally rather than by luck.

## Parallel ordering

`Eigen::HeaderOnlyMetisParallelOrdering<StorageIndex, Executor>` walks the dissection tree
**level by level** and dispatches each level through an `Executor`:

```cpp
#include <HeaderOnlyMetis.h>
#include <SupernodalLUExecutor.h>

using Exec = Eigen::supernodal_lu::PooledExecutor;
Exec pool(8);

Eigen::HeaderOnlyMetisParallelOrdering<int, Exec> ordering(pool);  // executor is borrowed
Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
ordering(A, perm);
```

The solvers construct their ordering functor themselves and expose no accessor to it, so used
directly as an `OrderingType_` this class runs on its built-in serial executor — you get the
deterministic parallel-path *ordering*, but not the threading. To get both, wrap it in a functor
that installs a pool of your own:

```cpp
struct PooledMetisOrdering : Eigen::HeaderOnlyMetisParallelOrdering<int, Exec> {
  PooledMetisOrdering() { setExecutor(sharedPool()); }   // sharedPool() outlives the solver
};
Eigen::SupernodalLU<Eigen::SparseMatrix<double>, PooledMetisOrdering> solver;
```

The lowest level, `header_only_metis::nodeNDParallel(n, xadj, adjncy, vwgt, perm, iperm, exec)`,
takes the executor as a plain argument and is what the tests drive.

**Its output is not the METIS ordering, and cannot be.** The reference draws every random number
of the whole recursion from one stream, so the *order tree nodes are visited in* is itself an
input to the result; a breadth-first frontier would hand each node different numbers, and under
threads the interleaving would not even be repeatable. Each node therefore seeds its own
generator from its **position in the tree** (depth and `lastvtx`, both fixed by the input). A
node's result is then a function of its subgraph and its position only.

That makes the path **fully deterministic**: the permutation depends on the matrix alone, not on
the executor, the thread count, or the scheduling. Which is what gives it a testable oracle — its
own serial run — and what lets its fill be pinned absolutely in the baselines.

**Quality**, measured over the 42 matrices in `test/baselines/testdata.baseline` that both
orderings factor: **mean +0.6% fill against METIS, median +0.3%, and lower fill on 17 of 42.**
The one real outlier is `lap3d_16x16x16` at +25.3% — and `lap3d_20x20x20` is *better*, so this is
not a "3D is worse" pattern. On quality grounds it is a fine default.

**Scaling** on `laoss_1` (251k rows): 1.46x at 2 threads, 2.19x at 4, 2.73x at 8, 3.11x at 16,
3.74x at 32 — plateauing near 230 ms. Across the whole corpus, 8 threads gives 2.58x (2.7-2.85x
on the big matrices).

The plateau is **not** idle lanes and not memory bandwidth. Summing `level_ms / min(level_width, 16)`
over the measured per-level costs puts the schedule's own structural ideal at 190 ms on 16 cores
against a measured 215 ms — 88% efficiency, almost no scheduling slack. The binding constraint is
that **level 0 is a single node costing 96 ms**, 45% of the measured time, irreducible without
parallelism *inside* a bisection — which coarsening's sequential level dependencies,
order-dependent matching and inherently serial FM refinement all rule out.

A second parallel axis (flattening `MlevelNodeBisectionL2`'s five speculative trials into the
level dispatch) was implemented **twice**, measured, and removed both times: 222.7 vs 222.8 vs
223.5 ms on `laoss_1` at 16 physical cores, inside a ~3% noise band, for ~120 lines of scheduling,
five graph clones per split node and two extra barriers per level. `NestedDissectionParallel.h`
records what to measure first if anyone tries a third time.

Note that a breadth-first frontier holds a whole level's subgraphs at once, so peak memory is
higher than the depth-first recursion's.

## Testing

```sh
ctest --test-dir build -R test_header_only_metis --output-on-failure
```

Four suites, each answering a different question:

- **`test_header_only_metis`** — the full-corpus oracle. `perm`/`iperm` must be element-wise
  identical to linked `METIS_NodeND` on every matrix: synthetic grids at sizes chosen to straddle
  the algorithm's internal thresholds, the `testdata/` benchmark set, and the SuiteSparse corpus.
  112 bit-identical assertions. Without `DLU_WITH_METIS` it reports that there is nothing to
  check and passes, so a fresh checkout is never broken by a missing dependency.
- **`test_header_only_metis_internal`** — per-module white-box comparison against METIS's own
  internals (`libmetis__CoarsenGraph`, `libmetis__CompressGraph`, `libmetis__InitSeparator`, …)
  via `test/metis_internal_bridge.cpp`, a separate translation unit that exists to dodge
  `rename.h`/`gklib_rename.h` macro collisions. This is what makes a mismatch localize to one
  algorithm instead of one permutation — a single early tie-break divergence otherwise cascades
  into a completely different separator tree with no useful diff at the top.
- **`test_header_only_metis_ordering`** — the Eigen wiring: permutation parity with
  `MetisOrdering` element-wise, identical solver fill (`nnzL`/`nnzU`) through both `SupernodalLU`
  and `LeftRightLU`, and — built with `DLU_WITH_METIS=OFF` — that it links and passes with no
  METIS or GKlib on the link line at all, producing byte-identical residuals to the METIS build.
- **`test_header_only_metis_parallel`** — the determinism gate for the parallel path, which links
  no METIS because its oracle is its own serial run. It pins thread-count invariance (N threads
  must give the byte-identical permutation to serial, for every N — this is the race detector),
  run-to-run reproducibility, and validity of the result as a permutation. Ordering *quality* is
  deliberately not checked here as a ratio against the exact path: a relative check cannot see
  both orderings degrading together, which is the likeliest failure mode given how much code they
  share. Fill is pinned absolutely instead, per matrix, as the `SupernodalLU+HOMetisPar` rows of
  `test/baselines/testdata.baseline`. Those rows are **not** guarded by `HAVE_METIS` — the
  ordering links nothing, so they stay pinned and reproduce identical fill in a no-METIS build.
