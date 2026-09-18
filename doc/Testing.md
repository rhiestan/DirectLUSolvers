# Testing

*[← DirectLUSolvers](../README.md) · [RobustLU](RobustLU.md) · [SupernodalLU](SupernodalLU.md) · [SupernodalLDLT](SupernodalLDLT.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockLU](PointBlockLU.md) · [MultifrontalQR](MultifrontalQR.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md) · [Parallelism](Parallelism.md)*

Every suite in `test/` — correctness, regression, and the benchmark drivers — is described
here, for every solver and the header-only METIS port.

The suites build with CMake and run under CTest. From the `DirectLUSolvers`
directory:

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

`ctest -L quick` runs only the fast subset — synthetic matrices, no external
data, a few seconds — while the remainder reads the benchmark matrices and takes
substantially longer. That split is what makes CI possible: **Eigen and
`testdata/` live outside this repository**, so `.github/workflows/ci.yml` runs
the `quick` label on every push (gcc, clang, MSVC) and a scheduled job fetches
the SuiteSparse corpus for a real-matrix sweep.

CI **pins Eigen to an exact commit**, deliberately. The fill baselines below
depend on the fill-reducing ordering, which comes from Eigen's AMD
implementation; a different Eigen can legitimately produce a different
permutation and therefore different fill. Bump the pin and re-record the
baselines together, never separately. Optional dependencies are independent switches, all default `OFF`:

```sh
cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON -DDLU_WITH_PARDISO=ON
```

`DLU_EIGEN_DIR`, `DLU_TESTDATA_DIR`, `DLU_METIS_DIR` and `DLU_MKL_DIR` default
to this project's layout (siblings of `DirectLUSolvers/`); override them if
yours differs. On Windows the METIS and MKL builds also need `tbb12.dll` /
`mkl_rt.dll` at *run* time, which CTest gets handed automatically — you only
need `<mkl root>/bin` on your own `PATH` when running those binaries by hand. **An in-tree Eigen is preferred over an installed one on
purpose** — `find_package(Eigen3)` resolves against the user's CMake package
registry, which is frequently an unrelated version, and the fill baselines below
were recorded against the Eigen that ships beside these solvers.

The build defaults to **Release**, and the default is set *before* `project()`
deliberately. On a toolchain targeting MSVC (including `clang++` with a
`*-windows-msvc` triple) `Platform/Windows-MSVC.cmake` sets
`CMAKE_BUILD_TYPE_INIT` to `Debug`, so the usual `if(NOT CMAKE_BUILD_TYPE)`
guard placed *after* `project()` never fires and you silently get `-O0`. For
this project that is a 50-100x timing error — enough to make every benchmark
number meaningless. `cmake` prints the resolved type at configure time; check it
before quoting a measurement.

## Testing the executor backends

`test_executors` holds all four backends to one shared contract, checked against
`SerialExecutor`. `StdThreadExecutor` is always covered; the other two are
opt-in, and when their switch is off that backend is reported as skipped rather
than silently omitted:

```sh
cmake -S . -B build -G Ninja -DDLU_WITH_OPENMP=ON -DDLU_WITH_TBB=ON
ctest --test-dir build -R test_executors --output-on-failure
```

All three multithreaded backends run in **one binary against the same checks**,
deliberately — separate per-backend tests drift, and the property worth testing
is agreement *between* them. Each must match `SerialExecutor` on fill, solution,
residual and determinant; keep the parallel triangular solve bit-identical (a
different dispatch path per backend, so the claim has to hold for all of them);
drive `LeftRightLU`'s DAG scheduler, which asks something much stranger of an
executor than a plain loop — one `parallelFor` whose body is an entire
work-stealing scheduler; and survive repeated factorizations without deadlock or
drift.

**Discovery.** `DLU_WITH_TBB=ON` locates oneTBB automatically via its
`TBBConfig.cmake`, but note a trap in the oneAPI layout: the `latest` symlink can
point at a version that installed libraries **without headers** while a complete
older version sits beside it. The search therefore prefers a config whose
`include/oneapi/tbb.h` actually exists rather than trusting `latest`. Override
with `-DDLU_TBB_DIR=<dir containing TBBConfig.cmake>`. CTest is also handed
TBB's `bin` directory on `PATH`, so `tbb12.dll` resolves without any manual
environment setup — running the binary directly still needs it on `PATH`.
`DLU_WITH_OPENMP=ON` uses CMake's own `find_package(OpenMP)`.

Verified on this project's setup: clang 22 with `-fopenmp=libomp`, and oneTBB
2022.0. `TBBExecutor`'s documented reconfiguration (`solver.executor() =
TBBExecutor(n)` re-capping concurrency across successive assignments) is checked
explicitly, as is `OpenMPExecutor`'s thread-count override.

## Fill regression baselines

`test_regression` is the suite that guards the failure mode the others cannot
see. Every other check gates on the residual — but an ordering-direction mistake
leaves every residual at machine precision while inflating 3D factors 250-350x.
Fill is a deterministic function of the pattern and the
ordering, so `test_regression` pins `nnzL + nnzU` per (matrix, solver) against
`test/baselines/testdata.baseline` and fails on drift beyond 5%.

```sh
ctest --test-dir build -R test_regression --output-on-failure
./build/test_regression --synthetic-only     # no testdata/ needed
./build/test_regression --tier small         # skip the large 3D FEM systems
./build/test_regression --update             # re-record the baselines
```

Re-baseline only once you understand why the fill moved: `--update` rewrites
every entry, so read the diff before committing it. A fill change is a real
change.

## The SuiteSparse corpus

`testdata/` holds a handful of matrices this project happened to encounter, all
of which these solvers handle. `test/matrices/` adds a curated corpus from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu), **stratified on
pattern symmetry** and deliberately including matrices the solvers should *not*
handle well — because the interesting question is not "does it solve" but "does
it behave correctly when it cannot".

```sh
python test/matrices/fetch_suitesparse.py     # download (~59 MB, once)
ctest --test-dir build -R test_suitesparse --output-on-failure
```

The fetch script needs **no third-party package** — the SuiteSparse URL scheme
is stable, so plain `urllib` suffices and reproducing the corpus never depends
on a `pip install`. (`ssgetpy` is consulted only by `--propose`, and only if
installed.) `suitesparse.manifest` is checked in and human-curated; the matrices
themselves land in a git-ignored `cache/`. Because SuiteSparse matrices are
immutable, a manifest line always denotes the same matrix.

```sh
python test/matrices/fetch_suitesparse.py --list        # what's in the corpus
python test/matrices/fetch_suitesparse.py --verify      # manifest vs live index
python test/matrices/fetch_suitesparse.py --propose 20  # candidates to adopt
```

**The contract being tested.** A solver may solve accurately, may refuse to
factor, or may return a bad answer *it has itself flagged* — but must never
quietly return a wrong one. `test_suitesparse` judges exactly that, reading
`info()` **after** `solve()`, and additionally confirms that a flagged solve
really was bad (flagging a good one would be its own defect).

Results on the 33-matrix quick tier, as the summary line counts them (by `SupernodalLU`):
**22 solved** to at most 1e-6, **11 returned a bad answer the solver flagged itself**, 0 tripped
the fill guard. `LeftRightLU` flags 7 of those 11 and solves the other four (`nnc1374`,
`cavity10`, `b2_ss`, `fd12`). No unflagged wrong answers — the honesty machinery
(`solveFailureThreshold`, the post-solve residual check) is exercised against matrices that
genuinely defeat the solvers.

**Why the 11 failures fail.** Diagnosed by comparing against `Eigen::SparseLU` (real partial
pivoting) on the same systems, then sweeping `SupernodalLU`'s options. Residuals, default
flags (no `-mavx2 -mfma`), measured 2026-09-17:

| matrix | psym | SparseLU | default | `setMatching(false)` | MC64 | `setMaxBlockSize(0)` |
|---|---:|---:|---:|---:|---:|---:|
| `nnc1374` | 0.82 | 8e-16 | 1.9e-01 | **4.2e-10** | **2.4e-13** | 1.9e-01 |
| `cavity10` | 0.94 | 2e-15 | 4.6e-04 | **3.5e-16** | **3.8e-16** | 3.1e-03 |
| `Chebyshev3` | 0.50 | 3e-16 | 6.1e+94 | **7.3e-17** | **1.7e-16** | **1.5e-16** |
| `CAG_mat1916` | 0.30 | 1e-15 | 9.3e+23 | **4.8e-16** | **4.7e-16** | 2.0e+20 |
| `lhr10c` | 0.01 | 1e-15 | 8.1e+14 | inf | **2.4e-16** | **2.4e-16** |
| `b2_ss` | 0.01 | 1e-16 | 5.8e+01 | 2.4e+69 | **2.0e-16** | **2.6e-16** |
| `fd12` | 0.00 | 3e-16 | 4.0e+02 | 8.2e+104 | 3.3e-01 | **4.7e-16** |
| `shyy41` | 0.72 | 4e-14 | 5.9e+07 | 7.9e+07 | 1.7e+07 | 5.9e+07 |
| `rw5151` | 0.49 | **fails** 4e-01 | 1.7e+05 | 8.2e+02 | 3.0e+15 | 6.4e+04 |
| `foldoc` | 0.48 | **fails** | 6.2e+85 | 2.3e+126 | 7.1e+59 | 1.1e+33 |
| `SmaGri` | 0.00 | **fails** | 9.1e+05 | inf | 3.9e+25 | 7.0e-02 |

So **8 of 11 are ours, not the matrix**, and **`setMatchingMethod(MatchingMethod::MC64)` fixes
6 of those 8** while leaving every matrix that already solved solved. Of the other two, `fd12`
needs `setMaxBlockSize(0)` (or `LeftRightLU`, which solves it by default), and `shyy41` yields
to nothing short of true partial pivoting — [`RobustLU`](RobustLU.md) reaches it through its
`PointBlockLU` rung. The remaining three are the matrix: `SparseLU` fails too, `foldoc` is
structurally singular, and `SmaGri` has numerical rank 511 of 1059
([`MultifrontalQR`](MultifrontalQR.md#suitesparse-corpus-against-eigensparseqr)). See
[Matching & diagonal pivoting](SupernodalLU.md#matching--diagonal-pivoting-robustness) for the
mechanism and the cost trade-off.

These residuals move with the floating-point code the compiler emits: built with
`-mavx2 -mfma`, the default `SupernodalLU` solves `b2_ss` to 4e-16 and the list above is ten
long. The classification — which of them are the solver's fault — does not change.

**A finding worth knowing: pattern symmetry does not predict success.** The
intuition that `psym == 1.00` is safe and `psym < 0.5` is doomed is wrong in
both directions. Three matrices with a *completely* unsymmetric pattern
(`HB/gemat12`, `Grund/meg1`, `Simon/raefsky5`, all psym 0.00) solved to ~1e-16,
while four in the *partial* band failed, including `DRIVCAV/cavity10` at
psym 0.94. Fill ratio tracks symmetry as expected (`Pajek/foldoc` 282x,
`HB/gemat12` 160x), but whether the answer is usable is governed by
conditioning, not by pattern. Do not use `psym` to decide whether these solvers
suit your matrix — run it and check `info()`.

## Choosing a configuration for one matrix

`compare_testdata` answers "does every solver get the right answer across the corpus, and
roughly how fast". `bench_solvers` answers the question that follows it: **given this matrix,
which configuration should I actually use?** It sweeps solvers × orderings × thread counts on
one matrix at a time, warming each solver up before timing and reporting analyze / factor /
solve separately.

```sh
./build/bench_solvers                                   # the Tier::Small corpus
./build/bench_solvers --quick                           # synthetic only, no testdata needed
./build/bench_solvers --threads 1,2,16 --reps 5 path/to/A.mtx
./build/bench_solvers --no-matching path/to/A.mtx       # with setMatching(false)
```

Three things it shows that a single cold factor+solve number cannot:

- **Cold-start cost is excluded.** MKL's first `pardiso()` call spins up its thread pool; on a
  1015-row matrix that is ~500 ms against ~1 ms of real work, which makes an unwarmed PARDISO
  measurement meaningless.
- **Which phase costs.** `analyzePattern` is more than half of wall clock for METIS on
  `setfos_2` — and it is exactly the phase you skip when refactorizing an unchanged pattern. At
  16 threads the symbolic phase is the *larger* half for METIS (101.9 ms analyze against 61.0 ms
  factor) and for PARDISO (122.6 against 40.4).
- **The ordering**, which on an unsymmetric pattern moves the result further than the choice of
  solver does. Measured 2026-09-17 on `setfos_2` (n=3048, 238 nnz/row, symmetry 0.44), best of 5:

  | configuration | thr | analyze | factor | solve | total | fill |
  |---|--:|--:|--:|--:|--:|--:|
  | `LeftRightLU` AMD | 1 | 55.8 | 137.7 | 2.0 | 195.4 | 3,844,854 |
  | `LeftRightLU` AMD | 16 | 59.0 | 85.1 | 1.8 | 145.9 | 3,844,854 |
  | `LeftRightLU` COLAMD | 1 | 56.6 | 74.7 | 3.4 | 134.7 | 2,349,388 |
  | **`LeftRightLU` COLAMD** | 16 | 58.0 | 57.4 | 2.1 | **117.6** | 2,349,388 |
  | `LeftRightLU` METIS | 1 | 101.9 | 80.6 | 1.1 | 183.6 | 1,629,952 |
  | `LeftRightLU` METIS | 16 | 101.9 | 61.0 | 1.2 | 164.1 | 1,629,952 |
  | `SupernodalLU` AMD (on `Asym`) | 1 | 94.5 | 146.1 | 4.9 | 245.5 | 3,927,774 |
  | `SupernodalLU` AMD (on `Asym`) | 16 | 92.7 | 83.7 | 4.6 | 180.9 | 3,927,774 |
  | `PointBlockLU` COLAMD (replay) | 1 | 6.2 | 392.6 | 2.5 | 401.3 | 1,935,546 |
  | **`Eigen::SparseLU`** | 1 | 8.9 | 96.8 | 1.0 | **106.7** | 1,935,897 |
  | MKL PARDISO | 1 | 99.4 | 116.3 | 4.6 | 220.2 | 1,563,528 |
  | MKL PARDISO | 16 | 122.6 | 40.4 | 4.3 | 167.3 | 1,563,528 |

  Three results worth reading twice. COLAMD carries 44% more fill than METIS and still factors
  faster (44 wide supernodes against METIS's 322 narrow ones — the fatter dense blocks win the
  difference back in BLAS-3 efficiency), so fill is a first-order proxy for cost and not more
  than that. With only 34 supernodes under AMD there is almost no assembly-DAG parallelism to
  find, so what scaling either solver gets on this matrix comes from the chunked
  intra-supernode path rather than from the schedule. And the fastest *cold* solve here is
  `Eigen::SparseLU`, because `LeftRightLU`'s analysis (matching, block triangular form,
  symmetrization, ordering, symbolic factorization) takes 57 ms where `SparseLU`'s takes 9. On a
  refactorization, which skips that phase, `LeftRightLU` COLAMD is ahead at 60 ms to 98.

Fill is printed as each solver reports it: ours and `Eigen::SparseLU` count the diagonal in
both factors, PARDISO's `IPARM(18)` counts it once, so those columns are comparable only up to
an offset of `n`. The exit code counts only *our* solvers failing `resid < 1e-6` — the
benchmark is not a bug report against Eigen or MKL.

## Which solver per matrix family

`bench_setfos_samples` asks the question one level up from `bench_solvers`: not "which
configuration for this matrix" but "which solver for this *kind* of matrix". It runs every solver
in this library — plus `Eigen::SparseLU`, `Eigen::SparseQR` and BiCGSTAB/GMRES with ILUT, and on
matrices verified numerically symmetric at run time also `SupernodalLDLT`, `SimplicialLDLT` and
CG with incomplete Cholesky — over `setfosmatrices_samples/`: five pattern categories of real
drift-diffusion and optics Jacobians, five examples each, one of them complex.

```sh
./build/bench_setfos_samples                        # ../setfosmatrices_samples
./build/bench_setfos_samples --reps 5 path/to/samples
./build/bench_setfos_samples --csv out.csv          # default analysis/benchmark_results.csv (git-ignored)
```

Like the other drivers it synthesizes `b = A·xTrue` rather than reading the samples' own
right-hand sides, so every row carries a forward error as well as a residual; it warms each
solver up and reports the best of N per phase. It is not a CTest target — the corpus lives
outside the repository (`DLU_SETFOS_SAMPLES_DIR`).

Measured 2026-09-17, best of 3, mean total time per category:

| category | fastest | runner-up | `MultifrontalQR` | `Eigen::SparseQR` |
|---|---|---|---:|---:|
| complex unsymmetric, general sparse | `PointBlockLU` 0.38 ms | BiCGSTAB+ILUT 0.48 ms | 1.41 ms | 32.0 ms |
| real symmetric tridiagonal | `PointBlockLU` 0.14 ms | BiCGSTAB+ILUT 0.20 ms | 0.43 ms | 1695 ms |
| real unsymmetric banded | `PointBlockLU` 0.14 ms | BiCGSTAB+ILUT 0.19 ms | 0.51 ms | 286 ms |
| real unsymmetric, general sparse | `PointBlockLU` 0.29 ms | `Eigen::SparseLU` 0.41 ms | 1.16 ms | 18.9 ms |
| real unsymmetric tridiagonal | BiCGSTAB+ILUT 0.32 ms | GMRES+ILUT 0.33 ms | 1.31 ms | 535 ms |

These matrices are small (n ≤ 4737), which is the regime where `PointBlockLU`'s scalar kernels win
— see [its crossover](PointBlockLU.md#when-to-use-it). Every solver in the table solved all 25,
with two exceptions that are reports rather than wrong answers: `RobustLU` declines two of the
complex matrices as too ill-conditioned to guarantee any digits, and `SupernodalLU` flags one
optics matrix's solve. `MultifrontalQR` picks its scalar engine on every one. The category labels
come from one representative per structural group, so individual files can differ from their
label: only one of the five "symmetric tridiagonal" examples is numerically symmetric at run
time, and so is one of the "unsymmetric tridiagonal" ones — those two are the only matrices the
symmetric solvers ran on.

## Does the block triangular form pay?

`bench_btf` runs `LeftRightLU` twice per matrix — `setBlockTriangularForm(true)` and `(false)` —
with everything else held fixed, so every difference in a row is BTF's doing. No other benchmark
here can make that comparison: they all run the shipping configuration, where BTF is simply on.

```sh
./build/bench_btf                     # testdata + SuiteSparse corpora
./build/bench_btf --quick             # three synthetic matrices, nothing to download
./build/bench_btf --reps 15 ted_B     # one matrix, tighter estimate
```

Measured 2026-09-17 over 44 matrices, best of 3 after a warm-up. The summary separates three
groups, because their answers have nothing to do with each other:

| group | n | speedup (median) | fill | what it means |
|---|---:|---|---|---|
| irreducible (1 block) | 14 | 0.99x / 1.00x | 1.000x | BTF found nothing and cost one `O(n + nnz)` sweep. Every symmetric-pattern matrix is here. |
| reducible, fill unchanged | 7 / 8 | 0.93x / 0.93x | 1.000x | The matrix split but the split bought no fill. **This group pays.** |
| reducible, fill reduced | 23 / 22 | **1.20x / 1.35x** | 0.75x / 0.72x | What BTF is for. |

(AMD / COLAMD.) Read the **median row** — 0.99x under AMD, 1.00x under COLAMD — alongside the
corpus total of 1.16x / 1.29x. They disagree because the win is concentrated, not broad: under
AMD 14 matrices gain 5% or more, 21 are unchanged within ±5%, 9 lose more than that, and the total is dominated by
whichever matrix is slowest (`Pajek/foldoc` alone is most of it, which is why the total moves
several points between runs while the medians do not). The big movers are
`TSOPF_RS_b9_c6` 3.4x / 7.7x, `raefsky5` 3.8x / 5.9x, `raefsky6` 3.8x / 5.7x, `SmaGri` 8.0x / 6.9x
and `bayer05` 2.8x / 2.4x.

**Where BTF costs, the cost is entirely in `analyzePattern`.** Ordering many small blocks
separately is more expensive than ordering one large graph, and `factorize` and `solve` are
untouched — confirmed at 15 repetitions, where the noise is well below the effect:

| matrix | blocks | analyze off→on | factor off→on | total |
|---|---:|---|---|---|
| `setfos_2` | 7 | 34.46 → **54.84** ms | 188.5 → 180.3 ms | 0.95x |
| `Bindel/ted_B_unscaled` | 4245 | 10.64 → **13.49** ms | 3.82 → 3.78 ms | 0.84x |
| `tomography` | 37 | 3.76 → **4.46** ms | 3.55 → 3.67 ms | 0.90x |
| `CPM/cz1268` | 2 | 1.97 → **2.29** ms | 1.02 → 1.01 ms | 0.91x |

**`setfos_2` is the shape that pays most, and it is worth understanding.** Its BTF finds six
singleton blocks and one block holding the other 3042 columns — so it is reducible by a hair,
buys 0.5% of fill, and still leaves `analyzePattern` 59% slower. The reason is that any
`nblocks > 1` takes the per-block path, which *re-extracts each block as its own matrix* before
ordering it: on a near-irreducible matrix that means rebuilding essentially the whole matrix
(18.5 ms of the 21 ms BTF adds here; the SCC sweep itself is 1.0 ms and the ordering is
unchanged at ~5.4 ms). The "one `O(n + nnz)` sweep" above is what a truly irreducible matrix
pays — one block, no extraction. A matrix that splits into one giant block plus a few singletons
pays the sweep *and* a full copy.

Worst confirmed cases are ~59% of the symbolic phase (`setfos_2`) and ~16% of a cold
factor+solve (`ted_B_unscaled`) — and
`analyzePattern` is exactly the phase a refactorization workflow skips, so in a Newton loop with
a fixed pattern it amortizes to nothing. Three repetitions is too few to trust the per-group
*extremes* (a single row's min swung 0.68x-1.91x between runs at `--reps 3`); the medians are
stable, and anything you intend to quote should be re-measured with `--reps 15` on that matrix.

The benchmark deliberately never fails: a residual that got worse is information, not a verdict.
Whether a solve is acceptable is [`test_suitesparse`](#the-suitesparse-corpus)'s judgement, and
whether the fill moved is [`test_regression`](#fill-regression-baselines)'s.

## Profiling: where the time actually goes

`test/profile_driver.cpp` exists so a profiler sees `analyzePattern` / `factorize`
/ `solve` as three separately-attributable phases rather than one `compute()`
blob. It is not a test and is not built by default:

```sh
cmake -S . -B build -G Ninja -DDLU_BUILD_PROFILE_DRIVER=ON \
      -DDLU_WITH_ITT=ON -DDLU_ITT_DIR="<VTune>/sdk"      # ITT markers optional
cmake --build build

# the whole SuiteSparse corpus, both solvers, all three phases
vtune -collect hotspots -knob sampling-mode=sw -r r_hs -- \
      ./build/profile_driver --reps 3

# one scheduler, one matrix shape, one thread count
vtune -collect threading -knob sampling-and-waits=sw -r r_thr -- \
      ./build/profile_driver --solver lrlu --synthetic lap3d --threads 32 \
                             --reps 12 --phase factorize --no-intra
```

`--no-intra` turns `setIntraSupernodeParallelism` off in both solvers, which is
what isolates the scheduler under a threading profile: with it on, the narrow
top levels are chunked across the pool and the starvation the level/DAG schedule
suffers there no longer appears in the timeline.

Hotspots over the corpus (both solvers, `--reps 3`, 195 s of CPU time,
re-collected 2026-08-22 — build with `-g -gcodeview` or every frame comes back
as a hex address):

| source | CPU time | share |
|---|---:|---:|
| Eigen `PacketMath.h` (GEMM inner loops) | 153.2 s | 79% |
| Eigen `GeneralBlockPanelKernel.h` | 13.9 s | 7.1% |
| Eigen `AssignEvaluator.h` | 5.0 s | 2.6% |
| Eigen `Amd.h` (ordering, in analyze) | 1.8 s | 0.9% |
| **`SupernodalLU.h`** | **1.2 s** | **0.6%** |
| **`LeftRightLU.h`** | **0.9 s** | **0.5%** |

**That is the shape a finished solver should have**: ~89% of the time is Eigen's
dense kernels doing the arithmetic, and the solvers' own bookkeeping is ~1%. It
is also the check that the three profile-guided fixes stuck — the symbolic
`set_union` that was once the hottest line in the analyze phase now costs 0.33 s
(0.2%), `rowPanelPosition` 0.60 s (0.3%), and `LeftRightLU`'s complete-pivot
search, once ~44% of its factorize, is down to 0.11 s (0.06%) across
`pabs`/`predux`/`find_coeff_loop` combined. Reading the whole 26-matrix corpus
off disk takes 0.83 s in total — it used to be the single largest entry in this
report, back when the MatrixMarket reader built an `istringstream` per stored
nonzero.

If your own profile does not look like this, that is the interesting result —
these solvers are memory-bound in the kernels and everything else is noise.

