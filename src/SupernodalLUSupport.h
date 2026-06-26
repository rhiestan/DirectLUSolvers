// SupernodalLU - a sparse direct LU solver for Eigen, using the supernodal
// block algorithms popularized by PaStiX (see pastix_algorithms.md).
//
// This header defines the structural building blocks shared by the analysis
// (symbolic) and factorization (numeric) phases. Names are intentionally
// descriptive and do NOT follow the original PaStiX naming.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_SUPPORT_H
#define SUPERNODAL_LU_SUPPORT_H

#include <vector>

namespace Eigen {
namespace supernodal_lu {

// A supernode is a maximal range of consecutive columns (in the internal,
// fill-reducing + postordered numbering) that share the same below-diagonal
// row structure. It is factored and updated as one dense panel.
//
// Storage convention for a supernode of width w with `offDiagonalRowCount`
// rows below the diagonal:
//   * diagonal block : w x w   (packed LU: unit-lower L_kk and upper U_kk)
//   * lower panel    : offDiagonalRowCount x w   (the L entries below diag)
//   * upper panel    : w x offDiagonalRowCount   (the U entries to the right)
// The lower panel rows and the upper panel columns are indexed by the same
// ordered set of off-diagonal rows (the matrix pattern is symmetric).
template <typename StorageIndex>
struct Supernode {
  StorageIndex firstColumn = 0;        // inclusive, internal index
  StorageIndex lastColumn = 0;         // inclusive, internal index
  StorageIndex firstRowBlock = 0;      // index into the flattened row-block array
  StorageIndex rowBlockCount = 0;      // number of off-diagonal row blocks
  StorageIndex offDiagonalRowCount = 0;  // total rows below the diagonal

  StorageIndex width() const { return lastColumn - firstColumn + 1; }
};

// A contiguous range of off-diagonal rows of one supernode that all fall within
// the columns of a single other supernode ("the facing supernode"). This is the
// unit of dense BLAS work during the numeric update.
template <typename StorageIndex>
struct RowBlock {
  StorageIndex firstRow = 0;        // inclusive, internal index
  StorageIndex lastRow = 0;         // inclusive, internal index
  StorageIndex facingSupernode = 0;  // supernode whose columns contain these rows
  StorageIndex panelOffset = 0;      // start position within the owner's off-diagonal ordering

  StorageIndex height() const { return lastRow - firstRow + 1; }
};

// Identifies, for a target supernode, one source supernode that contributes a
// Schur-complement update to it, together with the source's row block that
// faces the target.
template <typename StorageIndex>
struct UpdateSource {
  StorageIndex sourceSupernode = 0;
  StorageIndex facingRowBlock = 0;  // index (global) of the facing block in the source
};

}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_SUPPORT_H
