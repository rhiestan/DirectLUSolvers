// Shared symbolic-analysis helpers: the A+A^T adjacency graph and the
// fill estimate used to RANK candidate fill-reducing orderings.
//
// Extracted from SupernodalLUAutoOrdering.h, which needed exactly this to
// choose between AMD and METIS restarts, and which cannot be included by
// anything that must stay dependency-free -- it pulls in METIS and GKlib. The
// machinery below has no such dependency, so ordering functors that only need
// to score a permutation (PointBlockOrdering) can include this instead.
//
// estimateFillFromPermutation() mirrors SupernodalLU::analyzePattern's
// ordering -> elimination tree -> postorder -> recompute -> column structures
// pipeline on the graph alone: no values, no numeric work, and no amalgamation
// (amalgamation adds roughly the same relative overhead to any base ordering,
// so comparing pre-amalgamation fill is enough to RANK candidates -- the number
// it returns is not the fill the solver will report).
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_SYMBOLIC_H
#define SUPERNODAL_LU_SYMBOLIC_H

#include <Eigen/SparseCore>

#include <algorithm>
#include <vector>

namespace Eigen {
namespace supernodal_lu {
namespace symbolic {

// A+A^T pattern (no diagonal) as CSR: indexPtr size n+1, innerIndices size
// indexPtr(n). Identical in spirit to Eigen::MetisOrdering's own
// get_symmetrized_graph (protected there, so not reusable directly).
template <typename StorageIndex, typename MatrixType>
void buildSymmetrizedGraph(const MatrixType& A, Matrix<StorageIndex, Dynamic, 1>& indexPtr, Matrix<StorageIndex, Dynamic, 1>& innerIndices) {
  const StorageIndex n = internal::convert_index<StorageIndex>(A.cols());
  MatrixType At = A.transpose();
  Matrix<StorageIndex, Dynamic, 1> visited(n);
  visited.setConstant(-1);
  Index totalNz = 0;
  for (StorageIndex j = 0; j < n; ++j) {
    visited(j) = j;  // exclude the diagonal
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      const StorageIndex idx = static_cast<StorageIndex>(it.index());
      if (visited(idx) != j) {
        visited(idx) = j;
        ++totalNz;
      }
    }
    for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
      const StorageIndex idx = static_cast<StorageIndex>(it.index());
      if (visited(idx) != j) {
        visited(idx) = j;
        ++totalNz;
      }
    }
  }

  indexPtr.resize(n + 1);
  innerIndices.resize(totalNz);
  visited.setConstant(-1);
  StorageIndex cur = 0;
  for (StorageIndex j = 0; j < n; ++j) {
    indexPtr(j) = cur;
    visited(j) = j;
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      const StorageIndex idx = static_cast<StorageIndex>(it.index());
      if (visited(idx) != j) {
        visited(idx) = j;
        innerIndices(cur++) = idx;
      }
    }
    for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
      const StorageIndex idx = static_cast<StorageIndex>(it.index());
      if (visited(idx) != j) {
        visited(idx) = j;
        innerIndices(cur++) = idx;
      }
    }
  }
  indexPtr(n) = cur;
}

// --- symbolic fill estimate: mirrors SupernodalLU::analyzePattern's
//     ordering -> elimination tree -> postorder -> recompute -> column
//     structures pipeline, on the graph alone (no values, no amalgamation:
//     amalgamation adds roughly the same relative overhead to any base
//     ordering, so comparing pre-amalgamation fill is enough to rank
//     candidates). Cost is one real symbolic analysis, paid per candidate.

template <typename StorageIndex>
void adjacencyForPermutation(StorageIndex n, const Matrix<StorageIndex, Dynamic, 1>& indexPtr, const Matrix<StorageIndex, Dynamic, 1>& innerIndices,
                                    const std::vector<StorageIndex>& toNew,
                                    std::vector<std::vector<StorageIndex>>& adjacency) {
  adjacency.assign(n, std::vector<StorageIndex>());
  for (StorageIndex j = 0; j < n; ++j) {
    const StorageIndex nj = toNew[j];
    for (StorageIndex k = indexPtr(j); k < indexPtr(j + 1); ++k) adjacency[nj].push_back(toNew[innerIndices(k)]);
  }
  for (auto& row : adjacency) std::sort(row.begin(), row.end());
}

// Liu's elimination-tree algorithm with path compression (verbatim copy of
// SupernodalLU::computeEliminationTree's body, parameterized on n instead
// of reading a class member).
template <typename StorageIndex>
void computeEliminationTreeOf(StorageIndex n, const std::vector<std::vector<StorageIndex>>& adjacency,
                                     std::vector<StorageIndex>& parent) {
  parent.assign(n, StorageIndex(-1));
  std::vector<StorageIndex> ancestor(n, StorageIndex(-1));
  for (StorageIndex j = 0; j < n; ++j) {
    for (StorageIndex neighbor : adjacency[j]) {
      if (neighbor >= j) continue;
      StorageIndex r = neighbor;
      while (ancestor[r] != StorageIndex(-1) && ancestor[r] != j) {
        StorageIndex next = ancestor[r];
        ancestor[r] = j;
        r = next;
      }
      if (ancestor[r] == StorageIndex(-1)) {
        ancestor[r] = j;
        parent[r] = j;
      }
    }
  }
}

// Verbatim copy of SupernodalLU::computePostorder's body.
template <typename StorageIndex>
void computePostorderOf(StorageIndex n, const std::vector<StorageIndex>& parent,
                               std::vector<StorageIndex>& postorder) {
  std::vector<StorageIndex> childHead(n, StorageIndex(-1));
  std::vector<StorageIndex> childNext(n, StorageIndex(-1));
  for (StorageIndex j = n - 1; j >= 0; --j) {
    if (parent[j] != StorageIndex(-1)) {
      childNext[j] = childHead[parent[j]];
      childHead[parent[j]] = j;
    }
    if (j == 0) break;  // avoid unsigned underflow if StorageIndex is unsigned
  }
  postorder.clear();
  postorder.reserve(n);
  std::vector<StorageIndex> stack;
  std::vector<StorageIndex> nextChild = childHead;
  for (StorageIndex root = 0; root < n; ++root) {
    if (parent[root] != StorageIndex(-1)) continue;
    stack.push_back(root);
    while (!stack.empty()) {
      StorageIndex node = stack.back();
      StorageIndex child = nextChild[node];
      if (child != StorageIndex(-1)) {
        nextChild[node] = childNext[child];
        stack.push_back(child);
      } else {
        postorder.push_back(node);
        stack.pop_back();
      }
    }
  }
}

// Verbatim copy of SupernodalLU::computeColumnStructures's body (parent[]
// must already be in postorder, i.e. parent[j] > j).
template <typename StorageIndex>
void computeColumnStructuresOf(StorageIndex n, const std::vector<std::vector<StorageIndex>>& adjacency,
                                      const std::vector<StorageIndex>& parent,
                                      std::vector<std::vector<StorageIndex>>& columnStructure) {
  columnStructure.assign(n, std::vector<StorageIndex>());
  std::vector<std::vector<StorageIndex>> children(n);
  for (StorageIndex j = 0; j < n; ++j)
    if (parent[j] != StorageIndex(-1)) children[parent[j]].push_back(j);

  std::vector<StorageIndex> markedAt(n, StorageIndex(-1));
  std::vector<StorageIndex> scratch;
  for (StorageIndex j = 0; j < n; ++j) {
    scratch.clear();
    scratch.push_back(j);
    markedAt[j] = j;
    for (StorageIndex neighbor : adjacency[j]) {
      if (neighbor > j && markedAt[neighbor] != j) {
        markedAt[neighbor] = j;
        scratch.push_back(neighbor);
      }
    }
    for (StorageIndex c : children[j]) {
      for (StorageIndex r : columnStructure[c]) {
        if (r > j && markedAt[r] != j) {
          markedAt[r] = j;
          scratch.push_back(r);
        }
      }
    }
    std::sort(scratch.begin(), scratch.end());
    columnStructure[j] = scratch;
  }
}

template <typename StorageIndex>
double estimateFillFromPermutation(StorageIndex n, const Matrix<StorageIndex, Dynamic, 1>& indexPtr,
                                          const Matrix<StorageIndex, Dynamic, 1>& innerIndices, const PermutationMatrix<Dynamic, Dynamic, StorageIndex>& matperm) {
  // matperm follows Eigen's ordering convention: indices()(k) is the ORIGINAL
  // index placed at new position k. toNew must be the other direction -- the new
  // index OF i -- so this inverts, exactly as SupernodalLU::analyzePattern and
  // LeftRightLU::analyzePattern do with the same permutation.
  //
  // Copying it straight across instead (which this did until 2026-08-23) scores
  // the REVERSED elimination order, and reversing an ordering is not a small
  // perturbation of its fill: on testdata/laoss_3 the two differ by 24x
  // (6,242,047 against 257,138, where the solver really produces 520,004 =
  // nnzL+nnzU ~ 2*nnzL). It is the same direction-of-permutation mistake the
  // solvers' own comments warn about, and it hides from everything except fill.
  std::vector<StorageIndex> toNew(n);
  for (StorageIndex i = 0; i < n; ++i) toNew[matperm.indices()(i)] = i;

  std::vector<std::vector<StorageIndex>> adjacency;
  adjacencyForPermutation(n, indexPtr, innerIndices, toNew, adjacency);
  std::vector<StorageIndex> parent;
  computeEliminationTreeOf(n, adjacency, parent);

  // postorder and fold into a second numbering (required so parent[j] > j
  // before computing column structures, exactly as analyzePattern does).
  std::vector<StorageIndex> postorder;
  computePostorderOf(n, parent, postorder);
  std::vector<StorageIndex> relabel(n);
  for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
  std::vector<StorageIndex> toFinal(n);
  for (StorageIndex i = 0; i < n; ++i) toFinal[i] = relabel[toNew[i]];

  std::vector<std::vector<StorageIndex>> adjacency2;
  adjacencyForPermutation(n, indexPtr, innerIndices, toFinal, adjacency2);
  std::vector<StorageIndex> parent2;
  computeEliminationTreeOf(n, adjacency2, parent2);

  std::vector<std::vector<StorageIndex>> columnStructure;
  computeColumnStructuresOf(n, adjacency2, parent2, columnStructure);

  double total = 0.0;
  for (const auto& col : columnStructure) total += static_cast<double>(col.size());
  return total;
}
}  // namespace symbolic
}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_SYMBOLIC_H
