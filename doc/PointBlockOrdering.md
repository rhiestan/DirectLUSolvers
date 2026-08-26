# PointBlockOrdering — ordering the node graph

*[← DirectLUSolvers](../README.md) · [SupernodalLU](SupernodalLU.md) · [LeftRightLU](LeftRightLU.md) · [PointBlockLU](PointBlockLU.md) · [HeaderOnlyMetis](HeaderOnlyMetis.md)*

`Eigen::PointBlockOrdering` (`src/PointBlockOrdering.h`) is a standalone ordering functor for
matrices with `nv` unknowns per grid point. It collapses the pattern onto the **node** graph,
orders that with AMD, and expands the result keeping each node's unknowns adjacent. It
auto-detects `nv` and the layout by scoring candidates, always keeping plain scalar AMD in the
running, so it can only be chosen when it predicts less fill.

Use it with **`SupernodalLU` or `LeftRightLU`**. Measured end to end on `testdata/setfos_2`
(3 variables × 1016 nodes), stored scalars in `LeftRightLU`'s factor:

| ordering | fill | factor |
|---|---:|---:|
| Eigen AMD on the 3048-node scalar graph | 3,933,570 | 188 ms |
| COLAMD on the scalar graph | 2,360,714 | 96 ms |
| **AMD on the 1016-node node graph** | **1,651,762** | **75 ms** |

The mechanism is more mundane than "a node's unknowns belong together": AMD is a heuristic and
simply does a better job on the 3× smaller graph. Ordering the node graph lands within a few
percent of the best symmetric-pattern ordering known for this matrix, where Eigen's scalar AMD
is 2.6x off it.

Two things it is **not**:

- **Not a good ordering for `PointBlockLU`.** It ranks candidates by *symmetric*-pattern fill,
  which is the right objective for the supernodal solvers and the wrong one for an
  unsymmetric-pattern LU. On `gemat11` it gives `PointBlockLU` 6,742,455 stored scalars against
  COLAMD's 79,479.
- **Not a way to make a conservative union pattern cheap.** `setfos_2` ships 724,732 stored
  entries of which only 46,449 are numerically nonzero, and that costs **16x in factor flops**
  (4.02e8 against 2.45e7) under every ordering tried, node-major and variable-major alike. If
  the pattern can be tightened, tighten it — no ordering will do it for you. (Beware measuring
  this with a sparse permutation product: it silently prunes explicit zeros and makes the
  penalty appear to vanish.)

Its auto-detection costs several symbolic passes, so `analyzePattern()` gets slower — 36 ms to
241 ms on `setfos_2`. That is amortized to nothing in a Newton loop with a fixed pattern, but
set `setVariablesPerNode(nv)` explicitly to skip the search.

