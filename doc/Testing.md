# Testing

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockLU](PointBlockLU.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md) · [Parallelism](Parallelism.md)*

Every suite in `test/` — correctness, regression, and the benchmark drivers — is described
here, for all three solvers and the header-only METIS port.

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

Results on the 23-matrix quick tier: **15 solved** to machine precision, **8
returned a bad answer the solver flagged itself**, 0 tripped the fill guard.
No unflagged wrong answers — the honesty machinery
(`solveFailureThreshold`, the post-solve residual check) is exercised against
matrices that genuinely defeat the solvers.

**Why the 8 failures fail.** Diagnosed by comparing against `Eigen::SparseLU`
(real partial pivoting) on the same systems, then sweeping the solver options:

| matrix | psym | SparseLU | diagnosis |
|---|---:|---|---|
| `Chebyshev3` | 0.50 | solves 4e-20 | **matching**; `setMatching(false)` → 7e-17 |
| `CAG_mat1916` | 0.30 | solves 1e-15 | **matching**; → 5e-16 |
| `cavity10` | 0.94 | solves 2e-15 | **matching**; → 3.6e-16 |
| `nnc1374` | 0.82 | solves 7e-16 | **matching**; → 4.2e-10 |
| `lhr10c` | 0.01 | solves 5e-16 | **block size**; `setMaxBlockSize(0)` → 2.4e-16 |
| `shyy41` | 0.72 | **fails** 1e-06 | the matrix |
| `rw5151` | 0.49 | **fails** 7e-02 | the matrix |
| `foldoc` | 0.48 | **fails** inf | structurally singular |

So **5 of 8 are ours, not the matrix** — and **`setMatchingMethod(MatchingMethod::MC64)`
fixes all five at once**, including `lhr10c`. See
[Matching & diagonal pivoting](SupernodalLU.md#matching--diagonal-pivoting-robustness) for the
mechanism and the cost trade-off.

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
- **Which phase costs.** `analyzePattern` is a third of wall clock for METIS on `setfos_2` —
  and it is exactly the phase you skip when refactorizing an unchanged pattern. Note the shape
  of the table below at 16 threads: for METIS (78.8 ms analyze against 65.4 ms factor) and for
  PARDISO (117.6 against 45.3) the symbolic phase is now the *larger* half.
- **The ordering**, which on an unsymmetric pattern moves the result further than the choice of
  solver does. Measured on `setfos_2` (n=3048, 238 nnz/row, symmetry 0.44), best of 5:

  | configuration | thr | analyze | factor | solve | total | fill |
  |---|--:|--:|--:|--:|--:|--:|
  | `LeftRightLU` AMD | 1 | 33.2 | 189.1 | 2.5 | 224.8 | 3,933,570 |
  | `LeftRightLU` AMD | 16 | 34.6 | 92.0 | 2.6 | 129.1 | 3,933,570 |
  | `LeftRightLU` COLAMD | 1 | 34.1 | 97.4 | 2.8 | 134.3 | 2,360,714 |
  | **`LeftRightLU` COLAMD** | 16 | 33.5 | 60.5 | 2.6 | **96.7** | 2,360,714 |
  | `LeftRightLU` METIS | 1 | 76.3 | 112.7 | 1.2 | 190.1 | 1,609,832 |
  | `LeftRightLU` METIS | 16 | 78.8 | 65.4 | 1.2 | 145.3 | 1,609,832 |
  | `SupernodalLU` AMD (on `Asym`) | 1 | 88.9 | 191.3 | 6.2 | 286.3 | 3,927,774 |
  | `SupernodalLU` AMD (on `Asym`) | 16 | 88.7 | 91.5 | 5.6 | 185.9 | 3,927,774 |
  | `Eigen::SparseLU` | 1 | 8.6 | 105.7 | 1.0 | 115.3 | 1,935,897 |
  | MKL PARDISO | 1 | 94.4 | 110.2 | 4.4 | 209.0 | 1,563,528 |
  | MKL PARDISO | 16 | 117.6 | 45.3 | 4.3 | 167.2 | 1,563,528 |

  Two results worth reading twice. COLAMD carries 47% more fill than METIS and still factors
  faster (39 wide supernodes against METIS's 321 narrow ones — the fatter dense blocks win the
  difference back in BLAS-3 efficiency), so fill is a first-order proxy for cost and not more
  than that. And with only 29 supernodes there is almost no assembly-DAG parallelism to find,
  so what scaling either solver gets on this matrix comes from the chunked
  intra-supernode path rather than from the schedule.

Fill is printed as each solver reports it: ours and `Eigen::SparseLU` count the diagonal in
both factors, PARDISO's `IPARM(18)` counts it once, so those columns are comparable only up to
an offset of `n`. The exit code counts only *our* solvers failing `resid < 1e-6` — the
benchmark is not a bug report against Eigen or MKL.

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

