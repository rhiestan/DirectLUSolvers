# DirectLUSolvers

Three header-only sparse **direct LU** solvers for [Eigen](https://eigen.tuxfamily.org), plus a
header-only reimplementation of METIS's nested-dissection ordering. Each solver is a template in
the style of `Eigen::SparseLU` and shares its interface (`compute`/`analyzePattern`/`factorize`/
`solve`, `matrixL()`/`matrixU()`, `transpose()`/`adjoint()`, `determinant()`, `info()`), so they
are interchangeable at the call site — they differ in what kind of matrix they are good at.

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
| [SupernodalLU](doc/SupernodalLU.md) | PaStiX-style supernodal solver: static symbolic structure, static pivoting + refinement, BLAS-3 tree-parallel kernels. **The full option reference lives here** — the other two solvers document only their deltas from it. |
| [LeftRightLU](doc/LeftRightLU.md) | PARDISO-style sibling: barrier-free dynamic scheduler, in-block complete pivoting, block triangular form, and direct support for unsymmetric nonzero patterns. |
| [PointBlockLU](doc/PointBlockLU.md) | Scalar left-looking Gilbert–Peierls with partial pivoting and refactorization replay — no symmetrization at all. Fastest of the three while the factor stays sparse. |
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

Practical guidance:

- **Start with `LeftRightLU`.** It takes any pattern, matches `SupernodalLU` on symmetric-pattern
  matrices, and is far ahead of it on unsymmetric ones (`gemat11` 9.1 ms against 1415 ms).
- **Use `PointBlockLU` when the factor stays sparse** — below roughly 100k stored scalars it is
  the fastest solver here and often the most accurate, because it never perturbs a pivot. Above
  that its scalar kernels lose to supernodal ones.
- **`SupernodalLU` is the reference implementation** of the shared analysis pipeline and the
  option surface; reach for it when your pattern is symmetric anyway and you want tree-parallel
  BLAS-3 factorization.
- **Ordering usually matters more than the solver.** On large well-separated 3D systems nested
  dissection is worth ~2x the fill of AMD; use [`HeaderOnlyMetisOrdering`](doc/HeaderOnlyMetis.md)
  if you would rather not link METIS.
- **Do not pre-symmetrize** input to `LeftRightLU` or `PointBlockLU`. It is not merely redundant,
  it costs 102x the fill on `gemat11` — see
  [Unsymmetric nonzero patterns](doc/LeftRightLU.md#unsymmetric-nonzero-patterns).

All three report failure honestly: `solve()` measures the true residual against the original `A`
and downgrades `info()` to `NumericalIssue` rather than returning a bad answer quietly. That
contract is itself tested, against matrices that genuinely defeat the solvers — see
[The SuiteSparse corpus](doc/Testing.md#the-suitesparse-corpus).

## Requirements

The core solvers (`SupernodalLU.h`, `LeftRightLU.h`, `PointBlockLU.h`, and the shared
`SupernodalLUExecutor.h` / `SupernodalLUMatching.h` / `SupernodalLUSupport.h`) need only Eigen and
a C++17 compiler — no external dependencies, no linking beyond your usual Eigen setup. Everything
else in the table below is an **opt-in** header that pulls in one extra dependency, listed
per-header. This mirrors Eigen's own `*Support` module convention: the base solvers stay
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
| `src/LeftRightLUConditionEstimate.h` | Hager-Higham 1-norm condition estimation and the Oettli-Prager componentwise backward error — what lets a caller tell a 13-digit answer from a 2-digit one (see [What's different from SupernodalLU](doc/LeftRightLU.md#whats-different-from-supernodallu)). Eigen only. |
| `src/PointBlockLU.h` | The [unsymmetric-pattern solver](doc/PointBlockLU.md) with refactorization replay. |
| `src/PointBlockLU` | Umbrella header for `PointBlockLU`, `#include <PointBlockLU>`. |
| `src/PointBlockOrdering.h` | [`PointBlockOrdering`](doc/PointBlockOrdering.md) — fill-reducing ordering on the node graph, for matrices with several unknowns per grid point. Dependency-free. |
| `src/SupernodalLUSymbolic.h` | Shared symbolic helpers: the A+Aᵀ adjacency graph and the fill estimate used to rank candidate orderings. No METIS dependency, unlike `SupernodalLUAutoOrdering.h`, which uses it. |
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
| `test/test_btf.cpp` | Block triangular form: the decomposition on graphs whose block structure is known by construction, and the solver with BTF on against BTF off. See [LeftRightLU testing](doc/LeftRightLU.md#testing). |
| `test/test_condition_estimate.cpp` | Condition estimation and error bounds: the estimator against closed-form and dense references, the backward error against its defining properties, and the promise that a default solve pays nothing for either. |
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
| `test/compare_testdata.cpp` | Benchmark harness comparing SupernodalLU (AMD/METIS/Auto) against `Eigen::SparseLU` and, optionally, MKL PARDISO, on the matrices in `testdata/`. |
| `test/bench_parallel.cpp` | Thread-count scaling sweep with per-phase timing (analyze / factor / solve), per mechanism. See [Parallel scaling](doc/Parallelism.md#parallel-scaling-measured). |
| `test/bench_ceiling.cpp` | What the *machine* can deliver, via independent concurrent factorizations — the upper bound any scheduler could reach. See [The machine ceiling](doc/Parallelism.md#the-machine-ceiling). |
| `test/bench_btf.cpp` | Block triangular form on against off, the one comparison no other benchmark makes — every other runs the shipping configuration, where BTF is simply on. See [Does the block triangular form pay?](doc/Testing.md#does-the-block-triangular-form-pay). |
| `test/bench_solvers.cpp` | Per-matrix solver/**ordering** shootout: warm-up, best-of-N, per-phase timing, against `Eigen::SparseLU` and optionally MKL PARDISO. See [Choosing a configuration for one matrix](doc/Testing.md#choosing-a-configuration-for-one-matrix). |
| `test/profile_driver.cpp` | Per-phase driver for a profiler (not a test, not built by default). See [Profiling](doc/Testing.md#profiling-where-the-time-actually-goes). |
| `test/test_pointblock_lu.cpp` | `PointBlockLU` correctness: orderings and their permutation conventions, unsymmetric patterns, the replay path against fresh factorizations, degenerate sizes, structural singularity. |
| `test/test_parallel_consistency.cpp` | Serial-vs-parallel agreement for the chunked intra-supernode paths of both solvers: fill must match exactly, and the parallel solve must be no less accurate. |
| `test/test_header_only_metis.cpp` | Full-corpus gate for the header-only METIS port: `perm`/`iperm` must be byte-identical to the linked C `METIS_NodeND` on every test matrix. Passes trivially without METIS. |
| `test/test_header_only_metis_internal.cpp` | Per-module white-box comparison against METIS internals (`libmetis__*`), so a mismatch localizes to one algorithm instead of one permutation. Uses `test/metis_internal_bridge.cpp`. |
| `test/test_header_only_metis_ordering.cpp` | The `Eigen::HeaderOnlyMetisOrdering` wiring: permutation parity with `MetisOrdering`, identical solver fill, and — when built without METIS — that it works with nothing linked. |
| `test/test_header_only_metis_parallel.cpp` | Determinism gate for the parallel ordering: byte-identical output at every thread count, with its own serial run as the oracle. See [Parallel ordering](doc/HeaderOnlyMetis.md#parallel-ordering). |
| `test/profile_header_only_metis.cpp` | VTune driver for the ordering: times the port head-to-head against the linked C library and checks the permutations still match. Built with `-DDLU_BUILD_PROFILE_DRIVER=ON`; not a CTest target. |
| `test/testing/Check.h` | Shared PASS/FAIL reporting and timing used by every suite. |
| `test/testing/MatrixMarket.h` | MatrixMarket reader: coordinate + array formats, real/integer/complex/pattern fields, general/symmetric/skew-symmetric/hermitian symmetries. |
| `test/testing/TestMatrices.h` | Deterministic matrix generators (2D/3D Laplacians, random symmetric-pattern, weak-diagonal) and the `symmetrizePattern`/`patternIsSymmetric` helpers. |
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
