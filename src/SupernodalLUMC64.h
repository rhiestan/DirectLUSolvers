// SupernodalLU - MC64 maximum-product matching with dual scaling.
//
// SupernodalLUMatching.h computes a maximum TRANSVERSAL that prefers large
// entries: a magnitude-greedy seed completed by augmenting paths. Completing the
// transversal is all the augmenting phase optimizes for, so it can displace good
// diagonal entries with poor ones, and nothing compares the result against the
// un-permuted diagonal. Measured over this project's SuiteSparse corpus, that
// made four matrices unsolvable that Eigen::SparseLU handles -- most starkly
// Muite/Chebyshev3, where it took a diagonal with 4096 of 4101 entries within
// 1e-3 of their column maximum down to 181.
//
// This is the real thing (Duff & Koster, HSL MC64 job 5): the exact maximum
// product assignment, plus the dual variables that come with it.
//
// FORMULATION. Maximizing prod |a_ij| over perfect matchings is a linear
// assignment problem under
//
//     c_ij = log(colmax_j) - log|a_ij|   >= 0,
//
// minimizing sum c_ij. It is solved by shortest augmenting paths (Dijkstra on
// reduced costs) maintaining row duals u_i and column duals v_j with
//
//     c_ij - u_i - v_j >= 0   for every entry,  = 0 on matched entries.
//
// WHY THE DUALS MATTER. Substituting c_ij back into that inequality gives
//
//     |a_ij| * exp(u_i) * exp(v_j)/colmax_j  <=  1,  with equality when matched.
//
// So Dr_i = exp(u_i), Dc_j = exp(v_j)/colmax_j scales the matrix so that every
// matched diagonal entry has magnitude exactly 1 and no entry exceeds 1. That
// scaling is a large part of why MC64 works in MUMPS/SuperLU_DIST, and it is
// precisely what the transversal version cannot provide -- it returns a
// permutation and leaves scaling entirely to Ruiz afterwards.
//
// COST. O(n) shortest-path searches, each O(nnz log n) worst case, versus the
// transversal's near-linear greedy pass. It is therefore OFF BY DEFAULT and
// selected with setMatchingMethod(MatchingMethod::MC64); see the README.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0.

#ifndef SUPERNODAL_LU_MC64_H
#define SUPERNODAL_LU_MC64_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace Eigen {
namespace supernodal_lu {

// Exact maximum-product matching with the MC64 dual scaling.
//
// On return:
//   matchRowForCol[j]  original row placed on the diagonal of column j
//   rowScale[i]        Dr_i = exp(u_i)
//   colScale[j]        Dc_j = exp(v_j) / colmax_j
//
// so that |Dr_i * A_ij * Dc_j| <= 1 everywhere and == 1 on the matched entries.
//
// Returns true when a perfect matching exists. When it does not, the matrix is
// structurally singular: the columns that could be matched keep their optimal
// assignment, the rest are completed with leftover rows so the result is still a
// valid permutation, and the scaling entries for the unmatched part are left at
// 1 (there is no finite dual for them).
template <typename MatrixType>
bool mc64Matching(const MatrixType& A,
                  std::vector<typename MatrixType::StorageIndex>& matchRowForCol,
                  std::vector<typename NumTraits<typename MatrixType::Scalar>::Real>& rowScale,
                  std::vector<typename NumTraits<typename MatrixType::Scalar>::Real>& colScale) {
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  const StorageIndex n = static_cast<StorageIndex>(A.cols());
  matchRowForCol.assign(static_cast<std::size_t>(n), StorageIndex(-1));
  rowScale.assign(static_cast<std::size_t>(n), RealScalar(1));
  colScale.assign(static_cast<std::size_t>(n), RealScalar(1));
  if (n == 0) return true;

  const RealScalar kInf = std::numeric_limits<RealScalar>::infinity();

  // --- build the cost graph, column by column ------------------------------
  // c_ij = log(colmax_j) - log|a_ij| >= 0. Structural zeros (which pattern
  // symmetrization introduces in quantity) carry no information and are dropped;
  // a zero entry can never be a useful diagonal.
  std::vector<StorageIndex> colStart(static_cast<std::size_t>(n) + 1, 0);
  std::vector<StorageIndex> edgeRow;
  std::vector<RealScalar> edgeCost;
  std::vector<RealScalar> logColMax(static_cast<std::size_t>(n), RealScalar(0));
  edgeRow.reserve(static_cast<std::size_t>(A.nonZeros()));
  edgeCost.reserve(static_cast<std::size_t>(A.nonZeros()));

  for (StorageIndex j = 0; j < n; ++j) {
    RealScalar colMax(0);
    for (typename MatrixType::InnerIterator it(A, j); it; ++it)
      colMax = numext::maxi(colMax, numext::abs(it.value()));
    logColMax[static_cast<std::size_t>(j)] =
        colMax > RealScalar(0) ? numext::log(colMax) : RealScalar(0);
    colStart[static_cast<std::size_t>(j)] = static_cast<StorageIndex>(edgeRow.size());
    if (colMax > RealScalar(0)) {
      for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
        const RealScalar mag = numext::abs(it.value());
        if (mag <= RealScalar(0)) continue;
        edgeRow.push_back(static_cast<StorageIndex>(it.index()));
        edgeCost.push_back(logColMax[static_cast<std::size_t>(j)] - numext::log(mag));
      }
    }
  }
  colStart[static_cast<std::size_t>(n)] = static_cast<StorageIndex>(edgeRow.size());

  // --- state ---------------------------------------------------------------
  std::vector<RealScalar> u(static_cast<std::size_t>(n), RealScalar(0));  // row duals
  std::vector<RealScalar> v(static_cast<std::size_t>(n), RealScalar(0));  // column duals
  std::vector<StorageIndex> rowMatch(static_cast<std::size_t>(n), StorageIndex(-1));

  std::vector<RealScalar> dist(static_cast<std::size_t>(n), RealScalar(0));
  std::vector<StorageIndex> pred(static_cast<std::size_t>(n), StorageIndex(-1));
  std::vector<StorageIndex> stamp(static_cast<std::size_t>(n), StorageIndex(-1));
  std::vector<char> scanned(static_cast<std::size_t>(n), 0);
  std::vector<StorageIndex> scannedRows;
  scannedRows.reserve(static_cast<std::size_t>(n));

  typedef std::pair<RealScalar, StorageIndex> HeapItem;  // (dist, row)
  std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;

  StorageIndex matched = 0;

  // --- one shortest augmenting path per unmatched column -------------------
  for (StorageIndex jStart = 0; jStart < n; ++jStart) {
    if (matchRowForCol[static_cast<std::size_t>(jStart)] != StorageIndex(-1)) continue;

    // `stamp` avoids an O(n) clear of dist/scanned per search.
    for (StorageIndex i : scannedRows) scanned[static_cast<std::size_t>(i)] = 0;
    scannedRows.clear();
    while (!heap.empty()) heap.pop();

    RealScalar delta(0);
    StorageIndex j = jStart;
    StorageIndex sink = StorageIndex(-1);

    for (;;) {
      // relax every unscanned row of column j
      const StorageIndex begin = colStart[static_cast<std::size_t>(j)];
      const StorageIndex end = colStart[static_cast<std::size_t>(j) + 1];
      for (StorageIndex e = begin; e < end; ++e) {
        const StorageIndex i = edgeRow[static_cast<std::size_t>(e)];
        if (scanned[static_cast<std::size_t>(i)]) continue;
        const RealScalar reduced = edgeCost[static_cast<std::size_t>(e)] -
                                   u[static_cast<std::size_t>(i)] - v[static_cast<std::size_t>(j)];
        const RealScalar nd = delta + reduced;
        if (stamp[static_cast<std::size_t>(i)] != jStart) {
          stamp[static_cast<std::size_t>(i)] = jStart;
          dist[static_cast<std::size_t>(i)] = nd;
          pred[static_cast<std::size_t>(i)] = j;
          heap.push(HeapItem(nd, i));
        } else if (nd < dist[static_cast<std::size_t>(i)]) {
          dist[static_cast<std::size_t>(i)] = nd;
          pred[static_cast<std::size_t>(i)] = j;
          heap.push(HeapItem(nd, i));  // lazy decrease-key
        }
      }

      // nearest unscanned row
      StorageIndex iStar = StorageIndex(-1);
      while (!heap.empty()) {
        const HeapItem top = heap.top();
        heap.pop();
        const StorageIndex i = top.second;
        if (scanned[static_cast<std::size_t>(i)]) continue;              // stale
        if (top.first > dist[static_cast<std::size_t>(i)]) continue;     // superseded
        iStar = i;
        delta = top.first;
        break;
      }
      if (iStar == StorageIndex(-1)) break;  // no augmenting path: singular

      scanned[static_cast<std::size_t>(iStar)] = 1;
      scannedRows.push_back(iStar);
      if (rowMatch[static_cast<std::size_t>(iStar)] == StorageIndex(-1)) {
        sink = iStar;
        break;
      }
      j = rowMatch[static_cast<std::size_t>(iStar)];  // continue through its column
    }

    if (sink == StorageIndex(-1)) continue;  // leave jStart unmatched

    // --- dual update ------------------------------------------------------
    // For scanned rows u_i += dist_i - delta (<= 0, so every inequality into a
    // scanned row only slackens). For each column already on the tree,
    // v_j += delta - dist_{rowMatch[j]}, which exactly cancels the change in its
    // matched row's dual and so preserves equality there; and v_jStart += delta.
    // Every edge along the augmenting path then has zero reduced cost, so
    // augmenting keeps complementary slackness intact.
    for (StorageIndex i : scannedRows) {
      const StorageIndex jm = rowMatch[static_cast<std::size_t>(i)];
      if (jm != StorageIndex(-1))
        v[static_cast<std::size_t>(jm)] += delta - dist[static_cast<std::size_t>(i)];
      u[static_cast<std::size_t>(i)] += dist[static_cast<std::size_t>(i)] - delta;
    }
    v[static_cast<std::size_t>(jStart)] += delta;

    // --- augment along pred ------------------------------------------------
    StorageIndex i = sink;
    for (;;) {
      const StorageIndex jj = pred[static_cast<std::size_t>(i)];
      const StorageIndex previousRow = matchRowForCol[static_cast<std::size_t>(jj)];
      matchRowForCol[static_cast<std::size_t>(jj)] = i;
      rowMatch[static_cast<std::size_t>(i)] = jj;
      if (jj == jStart) break;
      i = previousRow;  // the row jj used to hold, now displaced
    }
    ++matched;
  }

  const bool perfect = (matched == n);

  // --- scaling -----------------------------------------------------------
  // Dr_i = exp(u_i), Dc_j = exp(v_j)/colmax_j.
  for (StorageIndex i = 0; i < n; ++i)
    rowScale[static_cast<std::size_t>(i)] = numext::exp(u[static_cast<std::size_t>(i)]);
  for (StorageIndex j = 0; j < n; ++j)
    colScale[static_cast<std::size_t>(j)] =
        numext::exp(v[static_cast<std::size_t>(j)] - logColMax[static_cast<std::size_t>(j)]);

  // --- complete an imperfect matching into a valid permutation ------------
  if (!perfect) {
    std::vector<char> rowUsed(static_cast<std::size_t>(n), 0);
    for (StorageIndex j = 0; j < n; ++j)
      if (matchRowForCol[static_cast<std::size_t>(j)] != StorageIndex(-1))
        rowUsed[static_cast<std::size_t>(matchRowForCol[static_cast<std::size_t>(j)])] = 1;
    StorageIndex next = 0;
    for (StorageIndex j = 0; j < n; ++j) {
      if (matchRowForCol[static_cast<std::size_t>(j)] != StorageIndex(-1)) continue;
      while (next < n && rowUsed[static_cast<std::size_t>(next)]) ++next;
      matchRowForCol[static_cast<std::size_t>(j)] = next;
      rowUsed[static_cast<std::size_t>(next)] = 1;
      // No finite dual for an unmatched column; leave its scaling neutral.
      colScale[static_cast<std::size_t>(j)] = RealScalar(1);
    }
  }

  return perfect;
}

}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_MC64_H
