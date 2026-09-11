// SupernodalLDLT -- symmetric weighted matching, for a symmetric matrix whose
// diagonal is unusable.
//
// WHY THE LU SOLVERS' MATCHING IS THE WRONG TOOL HERE
//
// SupernodalLU and LeftRightLU put large entries on the diagonal with a maximum
// transversal: a ROW permutation P, applied as P A. On a symmetric matrix that
// destroys the symmetry the LDL^T solver exists to exploit -- P A is not
// symmetric for any interesting P. So the matching cannot be applied as the LU
// solvers apply it, and the answer is not to skip matching but to change what is
// done with it.
//
// WHAT IS DONE WITH IT INSTEAD (Duff-Pralet)
//
// The matching is still computed the same way: a maximum transversal preferring
// large entries, so that sigma(j) is a row whose entry in column j is big. What
// changes is the reading. Write sigma as a product of cycles:
//
//   * a 1-cycle, sigma(i) = i, says a_ii is large -- a good 1x1 pivot.
//   * a 2-cycle, sigma(i) = j and sigma(j) = i, says a_ij is large while neither
//     diagonal entry need be -- a good 2x2 PIVOT BLOCK, and exactly the object
//     Bunch-Kaufman would otherwise have to discover for itself.
//   * a longer cycle is broken into consecutive pairs. Every such pair (c_k,
//     c_k+1) satisfies sigma(c_k) = c_k+1, so its off-diagonal entry is one the
//     matching already judged large; an odd cycle leaves one singleton over.
//
// The result is a set of disjoint PAIRS, and those pairs are then kept together
// through the ordering by ordering the QUOTIENT graph -- each pair contracted to
// a single vertex -- and expanding afterwards. The permutation this produces is
// symmetric, which is the whole point: it is applied as P A P^T and the matrix
// stays symmetric.
//
// WHAT IT DOES NOT DO
//
// It does not force a 2x2 pivot. A pair arrives at the numeric phase as two
// ADJACENT columns, and Bunch-Kaufman then decides, on the actual values after
// all Schur updates have landed, whether to take them as a 2x2 block, as two 1x1
// pivots, or to pair either of them with something else in the block. The
// matching's job is only to make the good choice AVAILABLE -- pivots it cannot
// reach are the ones this solver refuses to chase, since reaching outside a
// supernode is what would invalidate the static structure.
//
// Nor does adjacency guarantee availability: a supernode boundary can fall
// between a pair, and then that 2x2 is out of reach exactly as a straddling
// pivot is. Nothing is wrong when it happens -- the pivot is simply not taken --
// and it is one reason matchedPairs() exceeds pivotBlocks2x2().
//
// Nor does it scale. Duff-Pralet also derive a symmetric scaling from the
// matching's dual variables; here the scaling stays the solver's symmetric Ruiz
// equilibration, which composes with the permutation and is computed from the
// matrix rather than from this.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef SUPERNODAL_LDLT_MATCHING_H
#define SUPERNODAL_LDLT_MATCHING_H

#include <Eigen/SparseCore>

#include <vector>

#include "SupernodalLUMatching.h"

namespace Eigen {
namespace supernodal_ldlt {

/** Pair up columns into 2x2 pivot candidates, from a maximum transversal.
 *
 * \param full     the FULL symmetric matrix (both triangles). The matching reads
 *                 rows and columns alike, so a single stored triangle would
 *                 describe a different, unsymmetric problem.
 * \param partner  on return, partner[i] is the index i is paired with, or -1 if
 *                 i is a singleton. Symmetric and involutive by construction:
 *                 partner[partner[i]] == i wherever it is not -1.
 * \returns the number of pairs formed.
 *
 * Every pair is CONNECTED -- a_ij is a matched entry, hence structurally nonzero
 * -- which the ordering step downstream relies on: contracting two vertices that
 * share no edge would be a claim about the graph that is not true.
 */
template <typename MatrixType>
Index symmetricMatchingPairs(const MatrixType& full,
                             std::vector<typename MatrixType::StorageIndex>& partner) {
  typedef typename MatrixType::StorageIndex StorageIndex;
  const StorageIndex n = static_cast<StorageIndex>(full.cols());
  partner.assign(static_cast<std::size_t>(n), StorageIndex(-1));
  if (n == 0) return 0;

  std::vector<StorageIndex> match;
  supernodal_lu::maximumWeightMatching(full, match);

  // Walk each cycle of sigma once. `seen` is what keeps a cycle from being
  // entered twice and, with it, a vertex from being paired twice.
  std::vector<char> seen(static_cast<std::size_t>(n), 0);
  std::vector<StorageIndex> cycle;
  Index pairs = 0;
  for (StorageIndex start = 0; start < n; ++start) {
    if (seen[static_cast<std::size_t>(start)]) continue;
    cycle.clear();
    StorageIndex v = start;
    while (!seen[static_cast<std::size_t>(v)]) {
      seen[static_cast<std::size_t>(v)] = 1;
      cycle.push_back(v);
      v = match[static_cast<std::size_t>(v)];
    }
    // A 1-cycle is a usable diagonal entry and needs no pair. Everything longer
    // pairs off along the cycle, which is where the large entries are; an odd
    // cycle leaves its last vertex a singleton.
    for (std::size_t k = 0; k + 1 < cycle.size(); k += 2) {
      const StorageIndex a = cycle[k], b = cycle[k + 1];
      partner[static_cast<std::size_t>(a)] = b;
      partner[static_cast<std::size_t>(b)] = a;
      ++pairs;
    }
  }
  return pairs;
}

}  // namespace supernodal_ldlt
}  // namespace Eigen

#endif  // SUPERNODAL_LDLT_MATCHING_H
