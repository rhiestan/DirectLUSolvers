// The symbolic (values-free) analysis shared by this project's supernodal
// solvers, plus the fill estimate used to RANK candidate fill-reducing
// orderings.
//
// Everything here depends on the sparsity pattern alone, so the whole pipeline
// -- elimination tree, postorder, supernode partition, block structure, update
// lists, scheduling levels -- is the same for any supernodal factorization
// built on a symmetric elimination graph, whatever its numeric core then does
// with the result. These are free functions over plain vectors rather than a
// base class: a solver passes its own members in and keeps them laid out the
// way its numeric phase wants them, and inherits nothing.
//
// This header pulls in nothing beyond Eigen's sparse core, which is what makes
// it includable from anywhere. SupernodalLUAutoOrdering.h cannot serve the same
// role because it pulls in METIS and GKlib, so an ordering functor that only
// needs to score a permutation (PointBlockOrdering) includes this instead.
//
// estimateFillFromPermutation() runs the same pipeline on the graph alone, with
// no amalgamation -- amalgamation adds roughly the same relative overhead to any
// base ordering, so comparing pre-amalgamation fill is enough to RANK
// candidates. The number it returns is not the fill a solver will report.
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef SUPERNODAL_LU_SYMBOLIC_H
#define SUPERNODAL_LU_SYMBOLIC_H

#include <Eigen/SparseCore>

#include <algorithm>
#include <vector>

#include "SupernodalLUSupport.h"

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

// Liu's elimination-tree algorithm with path compression. `adjacency` is the
// symmetric elimination graph in the final (fill-reducing, postordered)
// numbering; only lower neighbours define the tree.
template <typename StorageIndex>
void computeEliminationTree(StorageIndex n, const std::vector<std::vector<StorageIndex>>& adjacency,
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

// Postorder of the elimination tree. Folding this into the numbering makes each
// supernode's columns contiguous; it does not change fill.
template <typename StorageIndex>
void computePostorder(StorageIndex n, const std::vector<StorageIndex>& parent,
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

// Amalgamation and splitting policy read by computeSupernodePartition. The
// defaults are the solvers' own; see their setAmalgamation/setMaxBlockSize
// documentation for what each one trades away.
struct PartitionOptions {
  Index relaxedSize = 4;       // force-merge a supernode narrower than this
  Index maxZeroRows = 4;       // ... or one adding at most this many zero rows
  double fillFraction = 0.3;   // ... or this fraction of the rows already carried
  Index maxBlockSize = 128;    // cap supernode width (0 = unlimited)
};

// Fused symbolic-factorization + supernode-partition pass: per-column symbolic
// fill (Liu's children-merge) interleaved with the supernode-boundary decision,
// so each column's explicit fill list can be FREED the moment its elimination-
// tree parent has consumed it.
//
// The fusion is what bounds peak memory. columnStructure[c] is read in exactly
// two places -- (1) the boundary decision at iteration c+1, comparing it against
// columnStructure[c+1], and (2) the children-merge at iteration parent[c], when
// c's parent folds c's rows into its own. Since parent[c] > c always (postorder),
// (2) never happens before (1), so once iteration parent[c] finishes (2) the list
// has had its last possible use. Only the "open" columns -- produced but not yet
// claimed by their parent -- stay materialized, bounded by the elimination tree's
// live frontier, instead of all n columns' full fill lists at once for the whole
// pass. On large fill-heavy matrices that difference dominates analyze memory.
//
// Produces `supernodes` (firstColumn/lastColumn only -- block structure comes
// from buildRowBlocksAndUpdateSources) and `supernodeOfColumn`, plus one compact
// off-diagonal-row list per finalized supernode, extracted at close time. That
// list is much smaller than a per-column representation: it holds each
// supernode's shared off-diagonal rows ONCE rather than once per member column.
template <typename StorageIndex>
void computeSupernodePartition(StorageIndex n,
                               const std::vector<std::vector<StorageIndex>>& adjacency,
                               const std::vector<StorageIndex>& parent,
                               const PartitionOptions& options,
                               std::vector<Supernode<StorageIndex>>& supernodes,
                               std::vector<StorageIndex>& supernodeOfColumn,
                               std::vector<std::vector<StorageIndex>>& supernodeOffDiagRows) {
  supernodes.clear();
  supernodeOfColumn.assign(n, 0);
  supernodeOffDiagRows.clear();
  if (n == 0) return;

  // Elimination-tree children (parent[] is postorder, so parent[j] > j always;
  // every non-root column is a child of exactly one later column).
  std::vector<std::vector<StorageIndex>> children(n);
  for (StorageIndex j = 0; j < n; ++j)
    if (parent[j] != StorageIndex(-1)) children[parent[j]].push_back(j);

  std::vector<std::vector<StorageIndex>> columnStructure(n);
  // Column j's fill beyond j is the UNION of already-sorted, already-deduped
  // ranges: adjacency[j]'s own tail beyond j (adjacency rows are sorted and
  // deduped by the caller), plus each child's columnStructure tail beyond j
  // (sorted+deduped by this same construction, inductively). A sorted union of
  // sorted inputs is exactly what std::set_union computes, in time linear in the
  // input/output sizes -- so folding the children in one at a time via set_union
  // (ping-ponging between scratch/scratch2) produces the same deduped, sorted
  // result as a mark-and-sweep-then-std::sort approach, but without ever paying
  // an O(m log m) sort of a list that's almost entirely inherited, unchanged,
  // from one child. This is the hot path on large fill-heavy matrices: large
  // near-root supernodes' columns have the biggest m, and a per-column sort would
  // rebuild their near-identical structure from scratch at every member column.
  std::vector<StorageIndex> scratch, scratch2;

  auto rowsBeyond = [](const std::vector<StorageIndex>& structure, StorageIndex col) -> StorageIndex {
    return static_cast<StorageIndex>(structure.end() - std::upper_bound(structure.begin(), structure.end(), col));
  };
  // Extracts the off-diagonal rows (strictly beyond lastColumn) of the supernode
  // that just closed -- its shared row set, per the "last column has the largest
  // structure" invariant -- into a compact per-supernode entry. Done here, at
  // close time, so downstream row-block building never needs the full per-column
  // lists: a wide supernode's early columns list many rows that are still INSIDE
  // the diagonal block, which this discards, keeping the retained data roughly
  // proportional to the real off-diagonal row count rather than
  // (member columns) x (off-diagonal row count).
  auto closeSupernode = [&](StorageIndex lastColumn, const std::vector<StorageIndex>& structure) {
    auto tailBegin = std::upper_bound(structure.begin(), structure.end(), lastColumn);
    supernodeOffDiagRows.emplace_back(tailBegin, structure.end());
  };

  Supernode<StorageIndex> s0;
  s0.firstColumn = 0;
  s0.lastColumn = 0;
  supernodes.push_back(s0);
  StorageIndex currentStart = 0;

  for (StorageIndex j = 0; j < n; ++j) {
    // --- symbolic fill of column j: merge already-sorted tails, no sort ---
    const auto& adjJ = adjacency[j];
    auto adjBeyond = std::upper_bound(adjJ.begin(), adjJ.end(), j);
    scratch.assign(adjBeyond, adjJ.end());  // own new neighbors beyond j (small)
    for (StorageIndex c : children[j]) {
      const std::vector<StorageIndex>& cs = columnStructure[c];
      auto childBeyond = std::upper_bound(cs.begin(), cs.end(), j);
      if (childBeyond == cs.end()) continue;  // nothing from this child survives past j
      // Write into a PRE-SIZED buffer rather than through back_inserter. The
      // union can never exceed the sum of its inputs, so sizing scratch2 to
      // that bound up front lets set_union store through a contiguous iterator
      // -- a plain pointer write per element -- instead of a push_back whose
      // capacity check on every element also defeats any bulk copy of the tail
      // that remains when one input runs out. Same result, and this is the
      // single hottest line in analyzePattern.
      scratch2.resize(scratch.size() + static_cast<std::size_t>(cs.end() - childBeyond));
      const auto unionEnd =
          std::set_union(scratch.begin(), scratch.end(), childBeyond, cs.end(), scratch2.begin());
      scratch2.resize(static_cast<std::size_t>(unionEnd - scratch2.begin()));
      scratch.swap(scratch2);
    }
    columnStructure[j].clear();
    columnStructure[j].reserve(scratch.size() + 1);
    columnStructure[j].push_back(j);  // diagonal, smaller than everything above (all > j)
    columnStructure[j].insert(columnStructure[j].end(), scratch.begin(), scratch.end());

    // --- supernode-boundary decision. A boundary at column j is MANDATORY when
    //     parent[j-1]!=j (j does not continue the elimination-tree path of j-1);
    //     merging across it would violate the "last column has the largest
    //     structure" invariant. When the path does continue, j joins the running
    //     supernode for FREE only if it is a genuine fundamental continuation:
    //     one child AND the same off-diagonal row set as j-1 (deltaRows == 0).
    //     Otherwise the merge costs explicit zeros and has to be AMALGAMATED,
    //     i.e. justified by the cost rules below.
    //
    //     The structure half of that test is load-bearing, not a formality: on a
    //     CHAIN elimination tree (banded matrices -- tridiagonal is the extreme)
    //     every column has exactly one child and continues its parent's path, so
    //     the etree conditions alone hold everywhere and would merge the entire
    //     matrix into one dense supernode. Each column there adds one new row, so
    //     deltaRows == 1 and the merge correctly falls through to the cost rules.
    if (j > 0) {
      const std::vector<StorageIndex>& structPrev = columnStructure[j - 1];
      const std::vector<StorageIndex>& structJ = columnStructure[j];
      bool start;
      if (parent[j - 1] != j) {
        start = true;  // mandatory structural boundary
      } else if (children[j].size() == 1 && rowsBeyond(structJ, j) == rowsBeyond(structPrev, j)) {
        start = false;  // fundamental supernode: no extra fill
      } else {
        // path-continuation branch point: amalgamate if cheap enough.
        const StorageIndex childWidth = j - currentStart;
        const StorageIndex existingRows = rowsBeyond(structPrev, j);
        const StorageIndex deltaRows = rowsBeyond(structJ, j) - existingRows;
        // Absolute rule (governs sparse matrices: few rows, so the fraction term
        // is tiny) OR a RELATIVE rule: accept the merge when the extra zero rows
        // are a small fraction of the rows the supernode already carries.
        const bool merge =
            (static_cast<Index>(childWidth) < options.relaxedSize) ||
            (static_cast<Index>(deltaRows) <= options.maxZeroRows) ||
            (static_cast<double>(deltaRows) <= options.fillFraction * static_cast<double>(existingRows));
        start = !merge;
      }
      // Splitting: force a boundary once the running supernode hits
      // maxBlockSize. This adds no fill (inter-block entries just move to
      // off-diagonal panels) and caps dense-panel width for cache- and
      // task-friendly BLAS.
      if (options.maxBlockSize > 0 && static_cast<Index>(j - currentStart) >= options.maxBlockSize)
        start = true;

      if (start) {
        closeSupernode(static_cast<StorageIndex>(j - 1), structPrev);
        // A closed supernode whose last column is an actual elimination-tree
        // root (no parent) will never be claimed by the children-merge freeing
        // loop below; free it here instead so multi-component matrices don't
        // retain every extra root's full structure for the rest of the pass.
        if (parent[j - 1] == StorageIndex(-1)) std::vector<StorageIndex>().swap(columnStructure[j - 1]);
        Supernode<StorageIndex> s;
        s.firstColumn = j;
        s.lastColumn = j;
        supernodes.push_back(s);
        currentStart = j;
      } else {
        supernodes.back().lastColumn = j;
      }
    }
    supernodeOfColumn[j] = static_cast<StorageIndex>(supernodes.size() - 1);

    // --- free every child's structure: parent[c]==j is always its last
    //     possible use (see the comment above columnStructure's declaration). ---
    for (StorageIndex c : children[j]) std::vector<StorageIndex>().swap(columnStructure[c]);
  }
  closeSupernode(static_cast<StorageIndex>(n - 1), columnStructure[n - 1]);
}

// Turns each supernode's compact off-diagonal-row list into contiguous
// per-facing-supernode RowBlocks, then builds the per-supernode update-source
// lists. Needs supernodeOfColumn for arbitrary later columns, so it can only run
// after computeSupernodePartition has finished ALL supernodes.
//
// Fills in the block fields of `supernodes` (firstRowBlock, rowBlockCount,
// offDiagonalRowCount), which computeSupernodePartition leaves at zero.
template <typename StorageIndex>
void buildRowBlocksAndUpdateSources(
    const std::vector<std::vector<StorageIndex>>& supernodeOffDiagRows,
    const std::vector<StorageIndex>& supernodeOfColumn,
    std::vector<Supernode<StorageIndex>>& supernodes,
    std::vector<RowBlock<StorageIndex>>& rowBlocks,
    std::vector<std::vector<UpdateSource<StorageIndex>>>& updateSources) {
  rowBlocks.clear();
  const StorageIndex supernodeNbr = static_cast<StorageIndex>(supernodes.size());

  // Build off-diagonal row blocks for each supernode from its (already
  // tail-extracted, so every entry is strictly beyond lastColumn) off-diagonal
  // row list. Rows are split at non-contiguities and at facing-supernode
  // boundaries so every block faces exactly one supernode.
  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    Supernode<StorageIndex>& sn = supernodes[s];
    const std::vector<StorageIndex>& structure = supernodeOffDiagRows[s];
    sn.firstRowBlock = static_cast<StorageIndex>(rowBlocks.size());
    sn.rowBlockCount = 0;
    sn.offDiagonalRowCount = 0;

    StorageIndex offset = 0;
    bool inBlock = false;
    RowBlock<StorageIndex> current;
    for (StorageIndex r : structure) {
      const StorageIndex facing = supernodeOfColumn[r];
      if (inBlock && r == current.lastRow + 1 && facing == current.facingSupernode) {
        current.lastRow = r;
      } else {
        if (inBlock) {
          rowBlocks.push_back(current);
          ++sn.rowBlockCount;
          offset += current.height();
        }
        current.firstRow = r;
        current.lastRow = r;
        current.facingSupernode = facing;
        current.panelOffset = offset;
        inBlock = true;
      }
    }
    if (inBlock) {
      rowBlocks.push_back(current);
      ++sn.rowBlockCount;
      offset += current.height();
    }
    sn.offDiagonalRowCount = offset;
  }

  // Build, for each supernode, the list of contributing sources for the
  // left-looking sweep. (The off-diagonal row -> panel position map is not
  // materialized; the solvers derive it from the sorted row blocks.)
  updateSources.assign(supernodeNbr, std::vector<UpdateSource<StorageIndex>>());

  for (StorageIndex s = 0; s < supernodeNbr; ++s) {
    const Supernode<StorageIndex>& sn = supernodes[s];
    StorageIndex previousFacing = StorageIndex(-1);
    for (StorageIndex b = 0; b < sn.rowBlockCount; ++b) {
      const StorageIndex blockIndex = sn.firstRowBlock + b;
      const RowBlock<StorageIndex>& block = rowBlocks[blockIndex];
      // Register one update per (source, facing supernode); a source's rows
      // inside one target's columns are consecutive in the sorted panel, so all
      // blocks facing the same target are consecutive. Record only the first.
      if (block.facingSupernode != previousFacing) {
        UpdateSource<StorageIndex> src;
        src.sourceSupernode = s;
        src.facingRowBlock = blockIndex;  // first block facing this target
        updateSources[block.facingSupernode].push_back(src);
        previousFacing = block.facingSupernode;
      }
    }
  }
}

// Group supernodes into elimination-tree levels: a supernode's level is 1 + the
// max level of its update sources. A bulk-synchronous scheduler runs one level
// at a time; a DAG scheduler uses these only as a diagnostic.
template <typename StorageIndex>
void computeSupernodeLevels(StorageIndex supernodeCount,
                            const std::vector<std::vector<UpdateSource<StorageIndex>>>& updateSources,
                            std::vector<std::vector<StorageIndex>>& levelGroups) {
  std::vector<StorageIndex> level(supernodeCount, 0);
  StorageIndex maxLevel = 0;
  // Update sources of supernode s are tree descendants, so they have smaller
  // ids; processing in increasing id order means level[source] is already final.
  for (StorageIndex s = 0; s < supernodeCount; ++s) {
    StorageIndex lv = 0;
    for (const UpdateSource<StorageIndex>& src : updateSources[s])
      lv = std::max<StorageIndex>(lv, static_cast<StorageIndex>(level[src.sourceSupernode] + 1));
    level[s] = lv;
    maxLevel = std::max(maxLevel, lv);
  }
  levelGroups.assign(supernodeCount == 0 ? 0 : (maxLevel + 1), std::vector<StorageIndex>());
  for (StorageIndex s = 0; s < supernodeCount; ++s) levelGroups[level[s]].push_back(s);
}

// Per-column symbolic fill by Liu's children-merge, retaining every column's
// list for the whole pass. That is what the fill ESTIMATE wants -- it sums them
// all at the end and never builds supernodes. A solver wants
// computeSupernodePartition() instead, which fuses this with the boundary
// decision so a column's list can be freed as soon as its parent has consumed
// it. `parent` must already be in postorder, i.e. parent[j] > j.
template <typename StorageIndex>
void computeColumnStructures(StorageIndex n, const std::vector<std::vector<StorageIndex>>& adjacency,
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
  computeEliminationTree(n, adjacency, parent);

  // postorder and fold into a second numbering (required so parent[j] > j
  // before computing column structures, exactly as analyzePattern does).
  std::vector<StorageIndex> postorder;
  computePostorder(n, parent, postorder);
  std::vector<StorageIndex> relabel(n);
  for (StorageIndex t = 0; t < n; ++t) relabel[postorder[t]] = t;
  std::vector<StorageIndex> toFinal(n);
  for (StorageIndex i = 0; i < n; ++i) toFinal[i] = relabel[toNew[i]];

  std::vector<std::vector<StorageIndex>> adjacency2;
  adjacencyForPermutation(n, indexPtr, innerIndices, toFinal, adjacency2);
  std::vector<StorageIndex> parent2;
  computeEliminationTree(n, adjacency2, parent2);

  std::vector<std::vector<StorageIndex>> columnStructure;
  computeColumnStructures(n, adjacency2, parent2, columnStructure);

  double total = 0.0;
  for (const auto& col : columnStructure) total += static_cast<double>(col.size());
  return total;
}
}  // namespace symbolic
}  // namespace supernodal_lu
}  // namespace Eigen

#endif  // SUPERNODAL_LU_SYMBOLIC_H
