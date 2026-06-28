// Optional wiring of the METIS nested-dissection fill-reducing ordering into
// SupernodalLU.
//
// SupernodalLU's ordering is a template parameter (default AMDOrdering), so
// METIS can be selected simply with
//     SupernodalLU<Mat, MetisOrdering<int>>.
// This header just provides a convenience alias and the required include.
//
// METIS nested dissection typically produces far sparser factors than AMD on
// matrices with good vertex separators (PDE meshes, etc.). Because this solver
// uses a STATIC symbolic structure (no row pivoting), fill set by the ordering
// directly determines flops and memory -- so a better ordering is the single
// biggest performance lever here (see pastix_algorithms.md).
//
// This header is kept SEPARATE from SupernodalLU.h on purpose: including it
// pulls in <Eigen/MetisSupport> and therefore requires linking the METIS
// library (and its GKlib dependency). The core SupernodalLU.h stays
// dependency-free for users who do not have METIS.
//
// Usage:
//     #include <SupernodalLUMetis.h>      // DirectLUSolvers/src on include path
//     Eigen::SupernodalLUMetis<Eigen::SparseMatrix<double>> solver;
//     solver.compute(A);
//     x = solver.solve(b);
// and link against metis + GKlib (e.g. -lmetis -lGKlib).

#ifndef SUPERNODAL_LU_METIS_H
#define SUPERNODAL_LU_METIS_H

// Eigen/MetisSupport uses std::cerr on a METIS error but does not include
// <iostream> itself; pull it in here so this header is self-contained.
#include <iostream>

#include <Eigen/MetisSupport>

#include "SupernodalLU.h"

namespace Eigen {

/** \brief SupernodalLU using METIS_NodeND (nested dissection) for the
 *         fill-reducing ordering instead of the default AMD.
 *
 * \tparam MatrixType_ a column-major Eigen::SparseMatrix.
 * \tparam Executor_   optional parallel-execution backend, matching
 *                     SupernodalLU (default SerialExecutor).
 */
template <typename MatrixType_,
          typename Executor_ = supernodal_lu::SerialExecutor>
using SupernodalLUMetis =
    SupernodalLU<MatrixType_, MetisOrdering<typename MatrixType_::StorageIndex>,
                 Executor_>;

}  // namespace Eigen

#endif  // SUPERNODAL_LU_METIS_H
