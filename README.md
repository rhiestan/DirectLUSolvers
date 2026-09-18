# DirectLUSolvers

Header-only sparse **direct** solvers for [Eigen](https://eigen.tuxfamily.org): three LU
factorizations, a symmetric `LDL^T`, a rank-revealing QR and a fallback ladder over them, plus a
header-only reimplementation of METIS's nested-dissection ordering. The three LU solvers are
templates in the style of `Eigen::SparseLU` and share its interface (`compute`/`analyzePattern`/
`factorize`/`solve`, `matrixL()`/`matrixU()`, `transpose()`/`adjoint()`, `determinant()`,
`info()`), so they are interchangeable at the call site — they differ in what kind of matrix they
are good at. `SupernodalLDLT` and `MultifrontalQR` follow the same `compute`/`solve`/`info` shape.

```cpp
#include <LeftRightLU.h>

Eigen::LeftRightLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);
Eigen::VectorXd x = solver.solve(b);
if (solver.info() != Eigen::Success) { /* solve() measured the residual and said so */ }
```

The core of every solver needs **only Eigen and C++17** — nothing to link. Everything with an
external dependency (METIS, oneTBB, OpenMP, MKL) is an opt-in header or an opt-in CMake switch.

## Documentation

| Document | What it covers |
|---|---|
| [SupernodalLU](doc/SupernodalLU.md) | PaStiX-style supernodal solver: static symbolic structure, static pivoting + refinement, BLAS-3 tree-parallel kernels. **The full option reference lives here** — the other solvers document only their deltas from it. |
| [LeftRightLU](doc/LeftRightLU.md) | PARDISO-style sibling: barrier-free dynamic scheduler, in-block complete pivoting, block triangular form, and direct support for unsymmetric nonzero patterns. |
| [PointBlockLU](doc/PointBlockLU.md) | Scalar left-looking Gilbert–Peierls with partial pivoting and refactorization replay — no symmetrization at all. Fastest of the LU solvers while the factor stays sparse. |
| [SupernodalLDLT](doc/SupernodalLDLT.md) | Supernodal `LDL^T` for **symmetric** matrices, definite or indefinite (Bunch–Kaufman 2×2 pivots): reads one triangle, stores one factor, and reports inertia. Half the arena and about twice the factorization speed of running an LU on the same matrix. |
| [MultifrontalQR](doc/MultifrontalQR.md) | Rank-revealing sparse QR for any shape, with a multifrontal and a scalar engine chosen per matrix: equilibrated rank decision, **verified** by the smallest singular values of R11 and repaired with an SVD-decided deferred block; least-squares, minimum-norm and double-double-refined solutions. 5–6900x faster than `Eigen::SparseQR` on the SuiteSparse corpus (median 59x), and correct where `SparseQR` reports rank 622 of a full-rank 6316 or rank 2 of 2025. |
| [RobustLU](doc/RobustLU.md) | The fallback ladder: one solver that escalates through strategies when the first fails, and reports what it tried and why it stopped. |
| [PointBlockOrdering](doc/PointBlockOrdering.md) | Fill-reducing ordering on the *node* graph, for matrices with several unknowns per grid point. |
| [HeaderOnlyMetis](doc/HeaderOnlyMetis.md) | `Eigen::HeaderOnlyMetisOrdering` — nested dissection, bit-identical to `METIS_NodeND`, with nothing to link; plus a deterministic parallel variant. |
| [Parallelism](doc/Parallelism.md) | The `Executor` concept and its four backends, the two parallel mechanisms, and every scaling measurement (including the machine ceiling that bounds them). |
| [Testing](doc/Testing.md) | Building, the CTest suites, the fill regression baselines, the SuiteSparse corpus, and the benchmark/profiling drivers. |

Algorithmic background for the two supernodal solvers is in `pastix_algorithms.md` and
`pardiso_algorithms.md` at the repository root.

## Choosing a solver

| | [`SupernodalLU`](doc/SupernodalLU.md) | [`LeftRightLU`](doc/LeftRightLU.md) | [`PointBlockLU`](doc/PointBlockLU.md) |
|---|---|---|---|
| Input pattern | **symmetric only** (symmetrize first) | any | any |
| Pivoting | static + row-only, confined to the diagonal block | static + **complete** (row+col), confined to the diagonal block | true **partial** pivoting |
| Scheduling | elimination-tree levels, barrier per level | barrier-free assembly DAG, work stealing | single-threaded by design |
| Kernels | supernodal, BLAS-3 | supernodal, BLAS-3 | scalar columns |
| Block triangular form | no | **yes**, on by default | no |
| Refactorization | re-run `factorize()` | re-run `factorize()` | **numeric replay** of a recorded pivot sequence |

The table above covers the three **LU** solvers. [`SupernodalLDLT`](doc/SupernodalLDLT.md)
sits outside it: it takes a symmetric matrix (one triangle of it) and computes `L D L^T`.

Practical guidance:

- **If your matrix is symmetric, use [`SupernodalLDLT`](doc/SupernodalLDLT.md).** An LU of a
  symmetric matrix computes a `U` that is the transpose of the `L` it already has; dropping it
  is worth 1.9x the arena (exactly — it is structural) and about 2x the factorization time on
  3D Laplacians, and 6.5x against `Eigen::SimplicialLDLT` at `n = 64000`. It reads one
  triangle, so you need
  only assemble half. **Indefinite is fine** — 2×2 Bunch–Kaufman pivots cover saddle-point and
  KKT systems, and `inertia()` then reports the eigenvalue signs for free. Positive
  definiteness needs no declaring; `setPivoting(None)` is a fast path that assumes it and
  reports if it was wrong. If `replacedPivots()` comes back large, the diagonal is the problem:
  `setMatching(true)` pairs columns into 2×2 candidates before the ordering and solves matrices
  that are otherwise out of reach (a zero-diagonal system goes from a flagged `2e17` residual to
  machine precision, in less time and less fill). It is off by default because it is a 2×-fill
  regression on a matrix that does not need it.
- **Start with `LeftRightLU`** for anything else. It takes any pattern, matches `SupernodalLU`
  on symmetric-pattern matrices, and is far ahead of it on unsymmetric ones (`gemat11` 9.8 ms
  against 1117 ms).
- **Use `PointBlockLU` when the factor stays sparse** — below roughly 100k stored scalars it is
  the fastest solver here and often the most accurate, because it never perturbs a pivot. Above
  that its scalar kernels lose to supernodal ones.
- **`SupernodalLU` is the reference implementation** of the shared analysis pipeline and the
  option surface; reach for it when your pattern is symmetric anyway and you want tree-parallel
  BLAS-3 factorization.
- **Use [`RobustLU`](doc/RobustLU.md) when you cannot inspect the matrix yourself** — it takes the `LDL^T` shortcut when the matrix turns out symmetric and the factorization is unambiguously good, and otherwise runs `LeftRightLU`, measures the backward error, and escalates to MC64 matching, true partial pivoting, or a rank-revealing QR only when the diagnosis calls for it. On this project's SuiteSparse tier it costs 19 of 33 matrices exactly one factorization, rescues 6 that would otherwise fail — two of them by reporting a **rank** and a least-squares answer where no LU exists — and stops with a diagnosis on the rest rather than thrashing.
- **Use [`MultifrontalQR`](doc/MultifrontalQR.md) when the question is the rank, or the system is
  not square or not regular** — least squares, underdetermined, singular, or so ill-conditioned
  that the numerically meaningful answer is a truncated one. It decides the rank on an
  equilibrated matrix, *checks* the decision against the smallest singular values of R11 (and
  repairs it when Heath's column-norm rule is fooled), and returns the minimum-norm solution by
  default. On this project's corpus its rank matches a dense SVD at the same threshold. Small and very sparse problems get a scalar
  engine automatically, the PointBlockLU of the two. For a regular square system an LU solver is
  cheaper.
- **Ordering usually matters more than the solver.** On large well-separated 3D systems nested
  dissection is worth ~2x the fill of AMD; use [`HeaderOnlyMetisOrdering`](doc/HeaderOnlyMetis.md)
  if you would rather not link METIS.
- **Do not pre-symmetrize** input to `LeftRightLU` or `PointBlockLU`. It is not merely redundant,
  it costs 102x the fill on `gemat11` — see
  [Unsymmetric nonzero patterns](doc/LeftRightLU.md#unsymmetric-nonzero-patterns).

All three LU solvers, and `SupernodalLDLT`, report failure honestly: `solve()` measures the true
residual against the original `A` and downgrades `info()` to `NumericalIssue` rather than
returning a bad answer quietly. That contract is itself tested, against matrices that genuinely
defeat the solvers — see
[The SuiteSparse corpus](doc/Testing.md#the-suitesparse-corpus).

## Requirements

The core solvers (`SupernodalLU.h`, `LeftRightLU.h`, `PointBlockLU.h`, `SupernodalLDLT.h`,
`MultifrontalQR.h`, `RobustLU.h`, and the shared `SupernodalLUExecutor.h` /
`SupernodalLUMatching.h` / `SupernodalLUSupport.h`) need only Eigen and a C++17 compiler — no
external dependencies, no linking beyond your usual Eigen setup. Everything else in the table
below is an **opt-in** header that pulls in one extra dependency, listed per-header. This mirrors Eigen's own `*Support` module convention: the base solvers stay
dependency-free so you only pay for what you use.

## Quick start

```cpp
#include <Eigen/SparseCore>
#include <LeftRightLU.h>

Eigen::SparseMatrix<double> A = /* any square matrix */;
Eigen::VectorXd b = /* right-hand side */;

Eigen::LeftRightLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);                 // analyzePattern(A) + factorize(A)
if (solver.info() != Eigen::Success) std::cerr << solver.lastErrorMessage() << "\n";

Eigen::VectorXd x = solver.solve(b);
if (solver.info() != Eigen::Success) std::cerr << "resid " << solver.solveResidual() << "\n";
```

`compute()` is just `analyzePattern(A)` followed by `factorize(A)`; call them separately to
refactor the same pattern with new values, which skips the symbolic analysis:

```cpp
solver.analyzePattern(A);
solver.factorize(A);               // ... later, same pattern, new values:
solver.factorize(A2);
```

## Files

| File | Contents |
|---|---|
| `src/SupernodalLU.h` | The [supernodal solver](doc/SupernodalLU.md). No dependencies beyond Eigen — always safe to include. |
| `src/SupernodalLU` | Eigen-style umbrella header, `#include <SupernodalLU>` (no extension), forwards to `SupernodalLU.h`. |
| `src/LeftRightLU.h` | The [PARDISO-style sibling solver](doc/LeftRightLU.md). Reuses the shared support/matching/executor headers; self-contained otherwise. |
| `src/LeftRightLU` | Umbrella header for `LeftRightLU`, `#include <LeftRightLU>`. |
| `src/LeftRightLUBlockTriangular.h` | Block triangular form: strongly connected components of the matched matrix, the second half of the Dulmage–Mendelsohn decomposition (see [What's different from SupernodalLU](doc/LeftRightLU.md#whats-different-from-supernodallu)). Eigen only. |
| `src/LeftRightLUConditionEstimate.h` | Hager-Higham 1-norm condition estimation and the Oettli-Prager componentwise backward error — what lets a caller tell a 13-digit answer from a 2-digit one (see [What's different from SupernodalLU](doc/LeftRightLU.md#whats-different-from-supernodallu)). Shared by `LeftRightLU`, `PointBlockLU`, `SupernodalLDLT` and `RobustLU` despite the name. Eigen only. |
| `src/LeftRightLUExtendedResidual.h` | Double-double (compensated) residuals for iterative refinement — what turns refinement's small *backward* error into a small *forward* one. Portable software arithmetic, not `long double`. Eigen only. |
| `src/PointBlockLU.h` | The [unsymmetric-pattern solver](doc/PointBlockLU.md) with refactorization replay. |
| `src/PointBlockLU` | Umbrella header for `PointBlockLU`, `#include <PointBlockLU>`. |
| `src/PointBlockOrdering.h` | [`PointBlockOrdering`](doc/PointBlockOrdering.md) — fill-reducing ordering on the node graph, for matrices with several unknowns per grid point. Dependency-free. |
| `src/RobustLU.h` | [`Eigen::RobustLU`](doc/RobustLU.md) — the fallback ladder over `LeftRightLU` and `PointBlockLU`, with an attempt log. |
| `src/RobustLU` | Umbrella header, `#include <RobustLU>`. |
| `src/MultifrontalQR.h` | [`Eigen::MultifrontalQR`](doc/MultifrontalQR.md) — rank-revealing sparse QR (multifrontal and scalar engines) with verified rank, least-squares and minimum-norm solutions. Eigen only. |
| `src/MultifrontalQR` | Umbrella header, `#include <MultifrontalQR>`. |
| `src/SupernodalLDLT.h` | [`Eigen::SupernodalLDLT`](doc/SupernodalLDLT.md) — supernodal `LDL^T` for symmetric matrices, definite or indefinite. Reads one triangle, stores one factor, reports inertia. Eigen only. |
| `src/SupernodalLDLT` | Umbrella header, `#include <SupernodalLDLT>`. |
| `src/SupernodalLDLTMatching.h` | Symmetric weighted matching (Duff–Pralet): reads a maximum transversal as a set of 2×2 pivot candidates rather than as a row permutation (`SupernodalLDLT::setMatching`). |
| `src/SupernodalLUSymbolic.h` | The symbolic analysis shared by every supernodal solver here: the A+Aᵀ adjacency graph, elimination tree, postorder, supernode partition with amalgamation, block structure, update lists, scheduling levels — plus the fill estimate used to rank candidate orderings. Free functions over plain vectors, so a solver passes its own state in and inherits nothing. No METIS dependency, unlike `SupernodalLUAutoOrdering.h`, which uses it. |
| `src/SupernodalLUSupport.h` | Plain data structures shared by the analysis/factorization phases (`Supernode`, `RowBlock`, `UpdateSource`). |
| `src/SupernodalLUMatching.h` | The maximum-transversal matching + permutation-sign helpers (`MatchingMethod::Transversal`). |
| `src/SupernodalLUMC64.h` | Exact maximum-product matching with dual scaling (`MatchingMethod::MC64`). Eigen + standard library only. |
| `src/SupernodalLUExecutor.h` | The [`Executor` concept](doc/Parallelism.md), plus the bundled `SerialExecutor` and `StdThreadExecutor` backends. No dependency beyond `<thread>`. |
| `src/SupernodalLUExecutorOpenMP.h` | [`OpenMPExecutor`](doc/Parallelism.md#openmpexecutor) — optional, requires an OpenMP-enabled build. |
| `src/SupernodalLUExecutorTBB.h` | [`TBBExecutor`](doc/Parallelism.md#tbbexecutor) — optional, requires oneAPI Threading Building Blocks. |
| `src/SupernodalLUMetis.h` | `SupernodalLUMetis<Mat[,Executor]>` alias wiring in METIS nested dissection. Optional, requires METIS + GKlib. |
| `src/SupernodalLUAutoOrdering.h` | `SupernodalLUAuto<Mat[,Executor]>` alias: tries AMD and several METIS restarts, keeps the least-fill one. Optional, requires METIS + GKlib. |
| `src/HeaderOnlyMetis.h` | [`Eigen::HeaderOnlyMetisOrdering<StorageIndex>`](doc/HeaderOnlyMetis.md) — a drop-in `MetisOrdering` replacement with **nothing to link**. Eigen only. |
| `src/HeaderOnlyMetis/` | The templated `METIS_NodeND` reimplementation behind it (coarsening, initial separator, FM refinement, nested-dissection driver, MT19937-64 RNG, scratch workspace). |
| `CMakeLists.txt` | Builds and registers every suite with CTest. See [Testing](doc/Testing.md). |
| `test/test_supernodal_lu.cpp` | Correctness tests (dependency-free — only needs Eigen). |
| `test/test_leftright_lu.cpp` | `LeftRightLU` correctness tests (dependency-free; `-pthread` for the parallel-vs-serial test). |
| `test/test_supernodal_ldlt.cpp` | `SupernodalLDLT` against `Eigen::SimplicialLDLT` and dense `LDLT`: that only one triangle of the input is read and only one triangle of the factor computed, 2×2 pivots, inertia, matching, and the transposed backsolve. |
| `test/test_btf.cpp` | Block triangular form: the decomposition on graphs whose block structure is known by construction, and the solver with BTF on against BTF off. See [LeftRightLU testing](doc/LeftRightLU.md#testing). |
| `test/test_condition_estimate.cpp` | Condition estimation and error bounds: the estimator against closed-form and dense references, the backward error against its defining properties, and the promise that a default solve pays nothing for either. |
| `test/test_extended_residual.cpp` | Error-free transformations, the compensated residual, and the forward-vs-backward error claim — on integer systems, so there is an exact answer to converge to. |
| `test/test_robust_lu.cpp` | The ladder's four properties: it costs one factorization when the first rung works, escalates when it does not, stops immediately when no rung can help, and never claims a success it cannot back up. |
| `test/test_multifrontal_qr.cpp` | `MultifrontalQR` against dense Eigen references (SVD, pivoted QR): square/tall/wide, real and complex, exact and hidden (Kahan) rank deficiency, 450 random graph matrices whose rank must equal the SVD's, bad scaling, extended-precision refinement on an exact right-hand side, and serial/parallel bit-identity — all of it once per engine, plus the `Auto` engine choice. |
| `test/bench_multifrontal_qr.cpp` | One matrix, one solver (`eigen`, or `mfqr` with `-mf`/`-sc`/`-noverify`) per run, best of N, CSV out — so a driver loop can time-limit `Eigen::SparseQR`. |
| `test/test_parallel_lu.cpp` | Parallel-vs-serial agreement + speedup, using `StdThreadExecutor`. |
| `test/test_matrixmarket.cpp` | Unit tests for the shared MatrixMarket reader and the pattern helpers. |
| `test/test_mc64.cpp` | MC64 optimality against a brute-force oracle, the dual-scaling property, and integration through both solvers. |
| `test/test_scalar_types.cpp` | `float` and `std::complex<double>` coverage, including the `adjoint()`/`transpose()` distinction that only exists for complex. |
| `test/test_executors.cpp` | One shared contract for every `Executor` backend — `StdThread`, `OpenMP`, `TBB` — checked against `SerialExecutor`. See [Testing the executor backends](doc/Testing.md#testing-the-executor-backends). |
| `test/test_edge_cases.cpp` | Degenerate sizes (n = 0/1/2, diagonal-only, single dense supernode), the refactorize workflow, zero right-hand side, and a cross-solver differential. |
| `test/test_regression.cpp` | Fill/accuracy regression suite, checked against `test/baselines/testdata.baseline`. See [Fill regression baselines](doc/Testing.md#fill-regression-baselines). |
| `test/test_suitesparse.cpp` | Correctness sweep over the curated SuiteSparse corpus, including matrices these solvers cannot handle. See [The SuiteSparse corpus](doc/Testing.md#the-suitesparse-corpus). |
| `test/matrices/fetch_suitesparse.py` | Downloads the corpus named by `suitesparse.manifest` into a git-ignored `cache/`. No third-party dependency. |
| `test/matrices/suitesparse.manifest` | The checked-in, human-curated corpus definition. |
| `test/compare_testdata.cpp` | Benchmark harness comparing SupernodalLU (AMD/METIS/Auto) and LeftRightLU against `Eigen::SparseLU` and, optionally, MKL PARDISO, on the matrices in `testdata/`. |
| `test/bench_setfos_samples.cpp` | Every solver here plus Eigen's direct and iterative ones over the `setfosmatrices_samples/` category corpus; raw rows to `analysis/benchmark_results.csv`. See [Which solver per matrix family](doc/Testing.md#which-solver-per-matrix-family). |
| `test/bench_parallel.cpp` | Thread-count scaling sweep with per-phase timing (analyze / factor / solve), per mechanism. See [Parallel scaling](doc/Parallelism.md#parallel-scaling-measured). |
| `test/bench_ceiling.cpp` | What the *machine* can deliver, via independent concurrent factorizations — the upper bound any scheduler could reach. See [The machine ceiling](doc/Parallelism.md#the-machine-ceiling). |
| `test/bench_btf.cpp` | Block triangular form on against off, the one comparison no other benchmark makes — every other runs the shipping configuration, where BTF is simply on. See [Does the block triangular form pay?](doc/Testing.md#does-the-block-triangular-form-pay). |
| `test/bench_solvers.cpp` | Per-matrix solver/**ordering** shootout: warm-up, best-of-N, per-phase timing, against `Eigen::SparseLU` and optionally MKL PARDISO. See [Choosing a configuration for one matrix](doc/Testing.md#choosing-a-configuration-for-one-matrix). |
| `test/profile_driver.cpp` | Per-phase driver for a profiler (not a test, not built by default). See [Profiling](doc/Testing.md#profiling-where-the-time-actually-goes). |
| `test/test_pointblock_lu.cpp` | `PointBlockLU` correctness: orderings and their permutation conventions, unsymmetric patterns, the replay path against fresh factorizations, degenerate sizes, structural singularity, and the reliability accessors — the condition estimate against a closed-form κ, the ill-conditioned system whose residual is tiny and whose answer is worthless, and the promise that a default solve pays for none of it. |
| `test/test_parallel_consistency.cpp` | Serial-vs-parallel agreement for the chunked intra-supernode paths of both solvers: fill must match exactly, and the parallel solve must be no less accurate. |
| `test/test_header_only_metis.cpp` | Full-corpus gate for the header-only METIS port: `perm`/`iperm` must be byte-identical to the linked C `METIS_NodeND` on every test matrix. Passes trivially without METIS. |
| `test/test_header_only_metis_internal.cpp` | Per-module white-box comparison against METIS internals (`libmetis__*`), so a mismatch localizes to one algorithm instead of one permutation. Uses `test/metis_internal_bridge.cpp`. |
| `test/test_header_only_metis_ordering.cpp` | The `Eigen::HeaderOnlyMetisOrdering` wiring: permutation parity with `MetisOrdering`, identical solver fill, and — when built without METIS — that it works with nothing linked. |
| `test/test_header_only_metis_parallel.cpp` | Determinism gate for the parallel ordering: byte-identical output at every thread count, with its own serial run as the oracle. See [Parallel ordering](doc/HeaderOnlyMetis.md#parallel-ordering). |
| `test/profile_header_only_metis.cpp` | VTune driver for the ordering: times the port head-to-head against the linked C library and checks the permutations still match. Built with `-DDLU_BUILD_PROFILE_DRIVER=ON`; not a CTest target. |
| `test/testing/Check.h` | Shared PASS/FAIL reporting and timing used by every suite. |
| `test/testing/MatrixMarket.h` | MatrixMarket reader: coordinate + array formats, real/integer/complex/pattern fields, general/symmetric/skew-symmetric/hermitian symmetries. |
| `test/testing/TestMatrices.h` | Deterministic matrix generators (2D/3D Laplacians, random symmetric-pattern, weak-diagonal) and the `symmetrizePattern`/`patternIsSymmetric` helpers. |
| `test/testing/MetisGraph.h` | The symmetrized graph exactly as `Eigen::MetisOrdering` builds it, so the METIS-comparison suites feed both sides identical input. |
| `test/testing/TestData.h` | The benchmark-matrix registry: one list of `testdata/` matrices, with size tiers, shared by every suite. |

## Building and testing

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure     # ctest -L quick for the fast subset
```

Optional dependencies are independent switches, all default `OFF`:
`-DDLU_WITH_METIS=ON`, `-DDLU_WITH_PARDISO=ON`, `-DDLU_WITH_OPENMP=ON`, `-DDLU_WITH_TBB=ON`.
The build defaults to **Release**, deliberately set before `project()` — on an MSVC-targeting
toolchain the usual guard placed after it never fires and you silently get `-O0`, which is a
50-100x timing error here. See [Testing](doc/Testing.md) for the full story, the baseline
workflow, and the benchmark drivers.

## License

Mozilla Public License 2.0 (`LICENSE`), matching the surrounding Eigen code these solvers
integrate with.

`THIRD-PARTY-NOTICES.md` records the external work these solvers build on, and
distinguishes **algorithmic lineage** (published algorithms reimplemented from
their descriptions — PaStiX's supernodal design, PARDISO's scheduler, Duff &
Koster's MC64) from **code derivation**. No third-party source is incorporated:
everything under `src/` is original code. Note in particular that PaStiX is
CeCILL v2, a copyleft license incompatible with MPL-2.0 redistribution, which is
why its design is reimplemented rather than translated.
