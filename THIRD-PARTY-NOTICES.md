# Third-party notices

DirectLUSolvers is licensed under the **Mozilla Public License 2.0** (`LICENSE`).

This file records the external work these solvers build on. The distinction it
draws throughout is between:

- **Algorithmic lineage** — published algorithms and designs, reimplemented from
  their descriptions. Algorithms are not copyrightable; this section is
  attribution and scholarship, not a licensing obligation.
- **Code derivation** — where source was translated or copied. That *is* a
  licensing matter.

**No third-party source code is incorporated in this project.** Everything under
`src/` is original code written for it. The single case of code derivation that
previously existed is documented under [Removed](#removed) below.

---

## Dependencies

| Project | License | Relationship |
|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | MPL-2.0 | Required. Used through its public API only (`SparseSolverBase`, `SparseCore`, `OrderingMethods`, dense kernels). No Eigen internals are copied or vendored; in particular no code is taken from `Eigen::SparseLU`, which these solvers are compared against but do not reuse. Same license as this project. |
| [METIS](https://github.com/KarypisLab/METIS) + GKlib | Apache-2.0 | **Optional.** Used only via `SupernodalLUMetis.h` / `SupernodalLUAutoOrdering.h`, through Eigen's own `MetisSupport` wrapper. Linked, never vendored. |
| [Intel oneMKL](https://www.intel.com/oneMKL) (PARDISO) | Intel proprietary | **Optional, test-only.** `compare_testdata.cpp` benchmarks against MKL PARDISO when `DLU_WITH_PARDISO=ON`. Not required to build or use the solvers. |
| [oneTBB](https://github.com/uxlfoundation/oneTBB) | Apache-2.0 | **Optional.** `SupernodalLUExecutorTBB.h` forwards `parallelFor` to `oneapi::tbb::parallel_for`. Linked, never vendored. |
| OpenMP runtime | per toolchain (LLVM: Apache-2.0 WITH LLVM-exception) | **Optional.** `SupernodalLUExecutorOpenMP.h` uses `#pragma omp` and `<omp.h>`. |
| [SuiteSparse Matrix Collection](https://sparse.tamu.edu) | per matrix; the collection is freely redistributable for research | **Test data, not distributed here.** `test/matrices/suitesparse.manifest` names matrices; `fetch_suitesparse.py` downloads them into a git-ignored cache. No matrix data is committed. |

## Algorithmic lineage

These are designs and published algorithms that shaped the implementation. Each
was reimplemented from its description; no source was consulted line-by-line
except where noted under [Removed](#removed).

**PaStiX** — <https://gitlab.inria.fr/solverstack/pastix>, license **CeCILL v2**.
The overall design of `SupernodalLU` follows PaStiX's approach, as documented in
`pastix_algorithms.md` at the repository root: a fully precomputed static
symbolic block structure, static pivoting with iterative refinement in place of
partial pivoting, BLAS-3 supernodal kernels, elimination-tree level scheduling,
contiguous per-supernode factor storage (PaStiX calls these `coeftab`/`ucoeftab`),
and capping supernode width (`MAX_BLOCKSIZE`). One further item — the traversal
that walks maximal runs of contiguous off-diagonal rows in
`SupernodalLU::cblkFactorTime` — was structurally derived from PaStiX's
`cblk_time_fact`; that function has since been removed entirely (see below).

> **CeCILL v2 is a copyleft license and is not compatible with redistributing
> derived work under MPL-2.0.** This is why the project holds itself to
> reimplementation from published descriptions rather than translation of PaStiX
> source, and why the one function that crossed that line was removed rather than
> kept.

**PARDISO** — the design of `LeftRightLU`, as documented in
`pardiso_algorithms.md`: left-right-looking numeric factorization driven by a
barrier-free dynamic scheduler, in-block complete pivoting (`DGETC2`-style),
refinement gated on whether a pivot was perturbed (`IPARM(8)`), and the
log-determinant (`IPARM(33)`). Reimplemented from the published papers in
`PardisoPapers/`; no PARDISO source was available or consulted.

**MC64 / maximum-product matching** — `src/SupernodalLUMC64.h` implements the
maximum-product assignment of I. S. Duff and J. Koster, *"On algorithms for
permuting large entries to the diagonal of a sparse matrix"* (SIAM J. Matrix
Anal. Appl., 2001), including the dual variables that yield the row/column
scaling. Written from the published formulation. **HSL's own MC64
implementation is proprietary and was not obtained, consulted, or copied.** The
implementation here is validated independently against a brute-force optimal
assignment (`test/test_mc64.cpp`).

**Other standard algorithms**, implemented from the general literature: Ruiz
row/column equilibration; Liu's elimination-tree and symbolic-factorization
algorithms; supernode amalgamation; Chase–Lev-style work-stealing deques;
BiCGStab; and the shortest-augmenting-path / Hungarian family of linear
assignment methods.

## Removed

`SupernodalLU::cblkFactorTime` and its `setAmalgamationCostModel()` option were
removed on 2026-08-08.

The function was a structural translation of PaStiX's `cblk_time_fact`
(`kass/amalgamate.c`) — same algorithm, same variable names (`L`, `G`, `H`), same
loop shape — and its coefficients were copied verbatim from PaStiX's
`perf/perf.h`. Every floating-point constant in this project's `src/` came from
that file; there were no others. Since PaStiX is CeCILL v2 and this project is
MPL-2.0, that was a genuine license incompatibility rather than merely an
attribution gap.

It was first re-derived honestly: `test/calibrate_cost_model.cpp` measured the
kernels this solver actually issues (LU of the diagonal block, panel TRSM, Schur
GEMM) on the host machine and least-squares fitted the same functional forms,
replacing the borrowed numbers with measured ones. The borrowed constants were
in any case wrong here — `perf.h` is labelled `PERF_MODEL "AMD 6180 MKL"`, an
Opteron with a decade-old MKL, and was fitted to a Cholesky factorization while
these solvers do LU.

With correct constants the model was then measured across roughly 30 matrices
(`testdata/` plus the SuiteSparse corpus) and **still did not help**: factor time
moved within noise on the matrices that matter (`laoss_1` −2% to −1%), while
being repeatably slower on others (`tomography` +9–10%, `gemat11` +4%). No matrix
showed a reproducible win. Rather than keep a feature that earns nothing and
carries a licensing question, both it and its calibration tool were deleted. The
absolute/fractional fill heuristics it could substitute for remain the default
and are unchanged.
