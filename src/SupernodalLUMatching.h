// SupernodalLU - maximum-transversal matching for static-pivoting robustness.
//
// PaStiX-style solvers keep a STATIC symbolic block structure and never do row
// interchanges, so a numerically weak or structurally zero diagonal forces the
// static-pivot threshold to fire on every column and the solve degenerates
// (this is what happened on unsymmetric matrices such as bayer05 / gemat11).
//
// The standard fix (SuperLU_DIST / MUMPS use MC64) is to permute large-magnitude
// entries onto the diagonal BEFORE the symbolic factorization, so the diagonal
// is zero-free and well-scaled and threshold-bumping rarely triggers. This is a
// pure (unsymmetric) row permutation that does not touch the static structure;
// the matched pattern is symmetrized afterwards by the solver.
//
// We compute a maximum transversal that prefers large entries: a greedy pass
// (each column grabs its largest free row) seeds an augmenting-path completion
// (Kuhn's algorithm, iterative, candidate rows tried in descending magnitude).
// This is "MC64-style" in spirit; it is not the exact maximum-product assignment
// but, combined with Ruiz scaling and restricted in-block pivoting, it removes
// the dead-diagonal failure mode. Numerical scaling stays the solver's job
// (Ruiz equilibration), which composes with the permutation.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef SUPERNODAL_LU_MATCHING_H
#define SUPERNODAL_LU_MATCHING_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace Eigen {
namespace supernodal_lu {

// Compute a row permutation that places large-magnitude entries on the diagonal.
//
// On return matchRowForCol[j] is the original row index assigned to column j
// (so that entry lands on the diagonal of column j). The result is always a
// valid permutation of 0..n-1. The return value is true when every column was
// matched to a distinct nonzero row (a zero-free diagonal / perfect transversal);
// it is false when the matrix is structurally singular, in which case the
// unmatched columns are completed with leftover rows to keep a valid permutation.
template <typename MatrixType>
bool maximumWeightMatching(const MatrixType& A,
                           std::vector<typename MatrixType::StorageIndex>& matchRowForCol) {
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const StorageIndex n = static_cast<StorageIndex>(A.cols());

  matchRowForCol.assign(static_cast<std::size_t>(n), StorageIndex(-1));
  std::vector<StorageIndex> matchColForRow(static_cast<std::size_t>(n), StorageIndex(-1));
  if (n == 0) return true;

  // Per-column candidate rows, sorted by descending |value| (nonzeros only). This
  // both drives the greedy seeding and orders the augmenting-path search so that
  // large entries are preferred on the diagonal.
  std::vector<std::vector<StorageIndex>> candidates(static_cast<std::size_t>(n));
  for (StorageIndex j = 0; j < n; ++j) {
    std::vector<std::pair<RealScalar, StorageIndex>> rows;
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      const RealScalar mag = numext::abs(it.value());
      if (mag > RealScalar(0)) rows.emplace_back(mag, static_cast<StorageIndex>(it.index()));
    }
    std::sort(rows.begin(), rows.end(),
              [](const std::pair<RealScalar, StorageIndex>& a,
                 const std::pair<RealScalar, StorageIndex>& b) { return a.first > b.first; });
    auto& dst = candidates[static_cast<std::size_t>(j)];
    dst.reserve(rows.size());
    for (const auto& pr : rows) dst.push_back(pr.second);
  }

  // Greedy seed: each column takes its largest-magnitude row if still free.
  StorageIndex matched = 0;
  for (StorageIndex j = 0; j < n; ++j)
    for (StorageIndex i : candidates[static_cast<std::size_t>(j)])
      if (matchColForRow[i] == StorageIndex(-1)) {
        matchRowForCol[j] = i;
        matchColForRow[i] = j;
        ++matched;
        break;
      }

  // Augmenting-path completion (Kuhn's algorithm), iterative to bound stack use.
  // For each still-unmatched column we search for an alternating path to a free
  // row, trying candidate rows in descending magnitude.
  std::vector<StorageIndex> visitMark(static_cast<std::size_t>(n), StorageIndex(-1));
  std::vector<std::size_t> nextCand(static_cast<std::size_t>(n));  // DFS cursor per column
  std::vector<StorageIndex> colStack;
  colStack.reserve(static_cast<std::size_t>(n));

  for (StorageIndex start = 0; start < n; ++start) {
    if (matchRowForCol[start] != StorageIndex(-1)) continue;

    colStack.clear();
    colStack.push_back(start);
    nextCand[static_cast<std::size_t>(start)] = 0;
    StorageIndex freeRow = StorageIndex(-1);  // free row that ends the augmenting path

    while (!colStack.empty()) {
      const StorageIndex col = colStack.back();
      auto& cursor = nextCand[static_cast<std::size_t>(col)];
      const auto& cand = candidates[static_cast<std::size_t>(col)];

      bool advanced = false;
      while (cursor < cand.size()) {
        const StorageIndex row = cand[cursor++];
        if (visitMark[static_cast<std::size_t>(row)] == start) continue;  // already in this tree
        visitMark[static_cast<std::size_t>(row)] = start;
        const StorageIndex owner = matchColForRow[row];
        if (owner == StorageIndex(-1)) {
          // Found an augmenting path: rematch back along the stack. `row` is the
          // free endpoint reached from `col` (= colStack.back()).
          freeRow = row;
          advanced = true;
          break;
        }
        // Row is taken; descend into its current owner column to look for a swap.
        colStack.push_back(owner);
        nextCand[static_cast<std::size_t>(owner)] = 0;
        advanced = true;
        break;
      }

      if (freeRow != StorageIndex(-1)) break;  // augmenting path complete
      if (!advanced) colStack.pop_back();       // dead end: backtrack
    }

    if (freeRow != StorageIndex(-1)) {
      // Walk the stack from the deepest column up, reassigning each column the row
      // it reached. The deepest column (colStack.back()) takes freeRow; each
      // shallower column takes the row that its child column just vacated.
      StorageIndex row = freeRow;
      for (std::size_t k = colStack.size(); k-- > 0;) {
        const StorageIndex col = colStack[k];
        const StorageIndex prevRow = matchRowForCol[col];  // -1 for the start column
        matchRowForCol[col] = row;
        matchColForRow[row] = col;
        row = prevRow;
      }
      ++matched;
    }
  }

  const bool perfect = (matched == n);

  // Complete any unmatched columns with leftover free rows so the result is a
  // valid permutation even for a structurally singular matrix.
  if (!perfect) {
    std::vector<StorageIndex> freeRows;
    for (StorageIndex i = 0; i < n; ++i)
      if (matchColForRow[i] == StorageIndex(-1)) freeRows.push_back(i);
    std::size_t f = 0;
    for (StorageIndex j = 0; j < n; ++j)
      if (matchRowForCol[j] == StorageIndex(-1)) {
        const StorageIndex row = freeRows[f++];
        matchRowForCol[j] = row;
        matchColForRow[row] = j;
      }
  }

  return perfect;
}

// Parity (+1 / -1) of a permutation given as perm[j] = image of j. Used to track
// the determinant sign contributed by the matching row permutation.
template <typename StorageIndex>
int permutationSign(const std::vector<StorageIndex>& perm) {
  const std::size_t n = perm.size();
  std::vector<char> seen(n, 0);
  int sign = 1;
  for (std::size_t i = 0; i < n; ++i) {
    if (seen[i]) continue;
    std::size_t j = i;
    std::size_t len = 0;
    while (!seen[j]) {
      seen[j] = 1;
      j = static_cast<std::size_t>(perm[j]);
      ++len;
    }
    if ((len & 1u) == 0u) sign = -sign;  // even-length cycle flips parity
  }
  return sign;
}

}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_MATCHING_H
