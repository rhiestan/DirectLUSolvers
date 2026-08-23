// PointBlockOrdering -- a fill-reducing ordering for matrices with several
// unknowns per grid point (PDE systems, device Jacobians, coupled multi-physics).
//
// WHY THIS EXISTS
//
// A discretized system with nv unknowns per node has TWO graphs: the scalar
// graph on n = nv*N unknowns, and the NODE graph on N points. AMD and COLAMD
// order the scalar one, which is both bigger and -- more importantly -- free to
// split a node's unknowns apart and eliminate them at unrelated times. The
// unknowns of one node share (almost) the same neighbourhood, so splitting them
// buys nothing and costs a great deal of fill.
//
// This functor orders the NODE graph with AMD instead and then expands that
// order, keeping each node's nv unknowns adjacent. Measured end to end in
// LeftRightLU on testdata/setfos_2 (n = 3048 = 3 variables x 1016 nodes, a 1-D
// semiconductor device Jacobian), stored scalars in the factor:
//
//   Eigen AMD on the 3048-node scalar graph      3,933,570      188 ms
//   COLAMD on the scalar graph                   2,360,714       96 ms
//   AMD on the 1016-node NODE graph (this)       1,651,762       75 ms
//
// WHY IT HELPS, AND WHAT IT IS NOT. The mechanism is more mundane than "node
// unknowns belong together": AMD is a heuristic, and it simply does a better job
// on the 3x smaller node graph than on the scalar one. Ordering the node graph
// lands within a few percent of the best symmetric-pattern ordering known for
// this matrix (1,528,238 structural, from an exact minimum-degree code), where
// Eigen's scalar AMD is 2.6x off it.
//
// It is NOT a way to make a conservative union pattern cheap. setfos_2 ships
// 724,732 stored entries of which only 46,449 are numerically nonzero, and that
// costs 16x in factor flops (4.02e8 against 2.45e7) under EVERY ordering tried,
// node-major and variable-major alike -- measured with the explicit zeros
// preserved, which matters, because a sparse permutation product silently
// prunes them and makes the penalty appear to vanish. If the pattern can be
// tightened, tighten it; no ordering will do it for you.
//
// WHAT IT DOES NOT DO. This orders by point block; it does not STORE by point
// block. On the matrix above only 92.9% of off-diagonal point blocks are 4/9
// occupied, so dense nv x nv storage would be a separate (and unproven) trade.
//
// AUTO-DETECTION. With no nv set, the functor tries each divisor of n up to
// setMaxVariablesPerNode() in both layouts, scores every candidate with the
// same symbolic fill estimate AutoOrdering uses, and keeps the best -- with
// plain AMD on the scalar graph always in the running as candidate zero, so
// this can only pick a node ordering when one actually predicts less fill.
// Set nv explicitly to skip the search.
//
// Usage:
//   #include <PointBlockOrdering.h>
//   Eigen::LeftRightLU<Eigen::SparseMatrix<double>,
//                      Eigen::PointBlockOrdering<int>> solver;
//   solver.compute(A);
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef POINT_BLOCK_ORDERING_H
#define POINT_BLOCK_ORDERING_H

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>

#include "SupernodalLUSymbolic.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace Eigen {
namespace point_block {

/** How the caller's unknowns are laid out. For nv unknowns per node and N
 *  nodes, scalar index i belongs to
 *    VariableMajor: node i % N, variable i / N   (all of variable 0, then all
 *                   of variable 1, ... -- the layout testdata/setfos_2 uses)
 *    NodeMajor:     node i / nv, variable i % nv (a node's unknowns adjacent). */
enum class Layout { VariableMajor, NodeMajor };

/** \brief Fill-reducing ordering that eliminates a node's unknowns together.
 *         See the file header for the measurements motivating it.
 *
 * Returns the same permutation convention as Eigen's AMDOrdering
 * (indices()(k) is the original index placed at new position k), so it drops
 * into SupernodalLU and LeftRightLU as the OrderingType template argument with
 * no further adaptation. */
template <typename StorageIndex>
class PointBlockOrdering {
 public:
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;
  typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;

  PointBlockOrdering() = default;
  PointBlockOrdering(int variablesPerNode, Layout layout)
      : m_variablesPerNode(variablesPerNode), m_layout(layout) {}

  /** Unknowns per node. 0 (the default) auto-detects; any value > 1 skips the
   *  search and uses exactly that blocking with layout(). */
  void setVariablesPerNode(int nv) { m_variablesPerNode = nv; }
  int variablesPerNode() const { return m_variablesPerNode; }

  /** Layout of an explicitly-set blocking. Ignored while auto-detecting, which
   *  tries both. */
  void setLayout(Layout layout) { m_layout = layout; }
  Layout layout() const { return m_layout; }

  /** Largest nv the auto-detection will consider (default 8). Every divisor of
   *  n up to this bound is tried, in both layouts. */
  void setMaxVariablesPerNode(int maxNv) { m_maxVariablesPerNode = maxNv; }
  int maxVariablesPerNode() const { return m_maxVariablesPerNode; }

  /** Above this n, auto-detection is skipped and plain AMD is used: each
   *  candidate costs a full symbolic pass, and on a large matrix that would
   *  multiply analyzePattern()'s cost by the number of candidates. An
   *  explicitly-set nv is always honoured, whatever the size. */
  void setAutoDetectSizeLimit(StorageIndex limit) { m_autoDetectLimit = limit; }
  StorageIndex autoDetectSizeLimit() const { return m_autoDetectLimit; }

  // --- diagnostics from the most recent operator() call ---------------------

  /** Blocking actually used: 1 means the scalar AMD baseline won. */
  int lastVariablesPerNode() const { return m_lastNv; }
  Layout lastLayout() const { return m_lastLayout; }
  /** Predicted nnz(L) of the chosen ordering, and of the scalar AMD baseline it
   *  was compared against. Ranking numbers, not the fill the solver reports
   *  (no amalgamation) -- but their RATIO is the win this functor bought. */
  double lastPredictedNnzL() const { return m_lastNnzL; }
  double lastScalarNnzL() const { return m_lastScalarNnzL; }

  template <typename MatrixType>
  void operator()(const MatrixType& A, PermutationType& matperm) {
    const StorageIndex n = internal::convert_index<StorageIndex>(A.cols());
    m_lastNv = 1;
    m_lastLayout = Layout::VariableMajor;
    m_lastNnzL = std::numeric_limits<double>::quiet_NaN();
    m_lastScalarNnzL = std::numeric_limits<double>::quiet_NaN();
    if (n == 0) {
      matperm.resize(0);
      return;
    }

    // Candidate zero is always plain AMD on the scalar graph, so a node
    // ordering has to EARN its place by predicting less fill.
    AMDOrdering<StorageIndex> amd;
    amd(A, matperm);

    const bool autoDetect = (m_variablesPerNode <= 1);
    if (autoDetect && n > m_autoDetectLimit) return;

    std::vector<Candidate> candidates;
    if (autoDetect) {
      // Cheap pre-filter first. Scoring a candidate costs a full symbolic pass,
      // and trying every divisor in both layouts multiplied analyzePattern() by
      // ~25x on testdata/setfos_2 (36 ms -> 934 ms). Collapsing the pattern onto
      // the node graph is one pass over the nonzeros, and it already separates
      // real blockings from nonsense ones: when nv is right, each node edge
      // carries several scalar entries (4 of 9 on setfos_2), and when it is
      // wrong the entries scatter one per node edge. So collapse everything,
      // keep only the candidates that actually merge, and score those.
      std::vector<ScoredCandidate> viable;
      for (int nv = 2; nv <= m_maxVariablesPerNode; ++nv) {
        if (nv >= n || n % nv != 0) continue;
        for (Layout layout : {Layout::VariableMajor, Layout::NodeMajor}) {
          const double merge = collapseRatio(A, n, nv, layout);
          if (merge >= kMinimumCollapseRatio) viable.push_back({{nv, layout}, merge});
        }
      }
      std::sort(viable.begin(), viable.end(),
                [](const ScoredCandidate& a, const ScoredCandidate& b) { return a.merge > b.merge; });
      if (viable.size() > kMaxScoredCandidates) viable.resize(kMaxScoredCandidates);
      for (const ScoredCandidate& s : viable) candidates.push_back(s.candidate);
    } else {
      if (m_variablesPerNode >= n || n % m_variablesPerNode != 0) return;
      candidates.push_back({m_variablesPerNode, m_layout});
    }
    if (candidates.empty()) return;

    IndexVector indexPtr, innerIndices;
    supernodal_lu::symbolic::buildSymmetrizedGraph<StorageIndex>(A, indexPtr, innerIndices);
    double best = supernodal_lu::symbolic::estimateFillFromPermutation(n, indexPtr, innerIndices, matperm);
    m_lastScalarNnzL = best;
    m_lastNnzL = best;

    PermutationType candidatePerm;
    for (const Candidate& c : candidates) {
      if (!buildNodeOrdering(A, n, c.nv, c.layout, candidatePerm)) continue;
      const double nnzL =
          supernodal_lu::symbolic::estimateFillFromPermutation(n, indexPtr, innerIndices, candidatePerm);
      if (nnzL < best) {
        best = nnzL;
        matperm = candidatePerm;
        m_lastNv = c.nv;
        m_lastLayout = c.layout;
        m_lastNnzL = nnzL;
      }
    }
  }

 private:
  struct Candidate {
    int nv;
    Layout layout;
  };
  struct ScoredCandidate {
    Candidate candidate;
    double merge;
  };

  // Average scalar entries per node-graph edge below which a blocking is not
  // worth the symbolic pass: 1.0 means the collapse merged nothing at all.
  static constexpr double kMinimumCollapseRatio = 1.5;
  static constexpr std::size_t kMaxScoredCandidates = 3;

  // Scalar index of variable v at node `node`, under this layout.
  static StorageIndex scalarOf(StorageIndex node, int v, StorageIndex nodeCount, int nv, Layout layout) {
    return layout == Layout::VariableMajor
               ? static_cast<StorageIndex>(v) * nodeCount + node
               : node * static_cast<StorageIndex>(nv) + static_cast<StorageIndex>(v);
  }

  // Scalar entries per distinct node edge under this (nv, layout) collapse.
  // One pass over the nonzeros plus a marker array, no allocation per column.
  //
  // Iterates NODE by node rather than column by column: the marker array stamps
  // a row with the node column currently being scanned, which only dedups
  // correctly while all nv columns of that node are consecutive. They are in
  // node-major order but NOT in variable-major, where a node's columns sit N
  // apart -- scanning in index order would let an intervening node overwrite the
  // marks and inflate the edge count.
  template <typename MatrixType>
  static double collapseRatio(const MatrixType& A, StorageIndex n, int nv, Layout layout) {
    const StorageIndex nodeCount = n / static_cast<StorageIndex>(nv);
    if (nodeCount < 2) return 0.0;
    std::vector<StorageIndex> markedAt(static_cast<std::size_t>(nodeCount), StorageIndex(-1));
    double entries = 0.0, nodeEdges = 0.0;
    for (StorageIndex node = 0; node < nodeCount; ++node) {
      for (int v = 0; v < nv; ++v) {
        for (typename MatrixType::InnerIterator it(A, scalarOf(node, v, nodeCount, nv, layout)); it; ++it) {
          const StorageIndex nodeRow =
              nodeOf(internal::convert_index<StorageIndex>(it.index()), nodeCount, nv, layout);
          entries += 1.0;
          if (markedAt[static_cast<std::size_t>(nodeRow)] != node) {
            markedAt[static_cast<std::size_t>(nodeRow)] = node;
            nodeEdges += 1.0;
          }
        }
      }
    }
    return nodeEdges > 0.0 ? entries / nodeEdges : 0.0;
  }

  int m_variablesPerNode = 0;  // 0 = auto-detect
  Layout m_layout = Layout::VariableMajor;
  int m_maxVariablesPerNode = 8;
  StorageIndex m_autoDetectLimit = 200000;

  int m_lastNv = 1;
  Layout m_lastLayout = Layout::VariableMajor;
  double m_lastNnzL = std::numeric_limits<double>::quiet_NaN();
  double m_lastScalarNnzL = std::numeric_limits<double>::quiet_NaN();

  static StorageIndex nodeOf(StorageIndex i, StorageIndex nodeCount, int nv, Layout layout) {
    return layout == Layout::VariableMajor ? i % nodeCount : i / static_cast<StorageIndex>(nv);
  }

  // AMD on the node graph, expanded back to a scalar permutation with each
  // node's nv unknowns kept adjacent and in their original variable order.
  // Returns false if the collapse is degenerate (nothing to order).
  template <typename MatrixType>
  static bool buildNodeOrdering(const MatrixType& A, StorageIndex n, int nv, Layout layout,
                                PermutationType& matperm) {
    const StorageIndex nodeCount = n / static_cast<StorageIndex>(nv);
    if (nodeCount < 2) return false;

    // Collapse the scalar pattern onto the node graph. Duplicate node edges are
    // summed away by setFromTriplets; only the pattern matters to AMD.
    typedef SparseMatrix<double, ColMajor, StorageIndex> NodeMatrix;
    std::vector<Triplet<double, StorageIndex>> triplets;
    triplets.reserve(static_cast<std::size_t>(A.nonZeros()));
    for (StorageIndex j = 0; j < n; ++j) {
      const StorageIndex nodeCol = nodeOf(j, nodeCount, nv, layout);
      for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
        const StorageIndex nodeRow =
            nodeOf(internal::convert_index<StorageIndex>(it.index()), nodeCount, nv, layout);
        triplets.emplace_back(nodeRow, nodeCol, 1.0);
      }
    }
    NodeMatrix nodeGraph(nodeCount, nodeCount);
    nodeGraph.setFromTriplets(triplets.begin(), triplets.end());
    nodeGraph.makeCompressed();

    PermutationType nodePerm;
    AMDOrdering<StorageIndex> amd;
    amd(nodeGraph, nodePerm);
    if (nodePerm.size() != nodeCount) return false;

    // nodePerm.indices()(k) is the ORIGINAL node placed at new position k, and
    // this functor returns the same convention, so the expansion writes
    // matperm.indices()(newScalarPosition) = originalScalarIndex.
    matperm.resize(n);
    StorageIndex pos = 0;
    for (StorageIndex k = 0; k < nodeCount; ++k) {
      const StorageIndex node = nodePerm.indices()(k);
      for (int v = 0; v < nv; ++v) matperm.indices()(pos++) = scalarOf(node, v, nodeCount, nv, layout);
    }
    return pos == n;
  }
};

}  // namespace point_block

template <typename StorageIndex>
using PointBlockOrdering = point_block::PointBlockOrdering<StorageIndex>;

}  // namespace Eigen

#endif  // POINT_BLOCK_ORDERING_H
