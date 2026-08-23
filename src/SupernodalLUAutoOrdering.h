// Automatic ordering selection for SupernodalLU: tries Eigen's AMD ordering
// plus several METIS_NodeND nested-dissection candidates (different seeds),
// predicts the symbolic fill (nnz of the factor) each would produce, and
// returns whichever candidate has the smallest predicted fill.
//
// Why: measurements recorded in supernodallu-project memory (and reproducible
// via compare_testdata) show METIS nested dissection is NOT uniformly better
// than AMD here -- it wins on some matrices (YaleB_10NN, -5% fill) and loses
// badly on others (bayer05, +200% fill; rdb2048, +19%) because these test
// matrices are small/irregular rather than the large, well-separated PDE
// meshes nested dissection is built for. Rather than hand-picking an ordering
// per matrix, this functor tries several candidates and keeps the one with
// the least predicted fill, automating the choice at a bounded extra cost
// (a few cheap, VALUES-FREE symbolic passes during analyzePattern(), no
// numeric factorization). Also fixes METIS's seed explicitly for
// reproducibility -- Eigen's plain MetisOrdering passes options=NULL, which
// leaves METIS's internal seed at its non-deterministic default.
//
// Kept SEPARATE from SupernodalLU.h (like SupernodalLUMetis.h): including
// this header pulls in METIS + GKlib.
//
// Usage:
//   #include <SupernodalLUAutoOrdering.h>   // DirectLUSolvers/src on include path
//   Eigen::SupernodalLUAuto<Eigen::SparseMatrix<double>> solver;
//   solver.compute(A);
//   x = solver.solve(b);
// and link against metis + GKlib (e.g. -lmetis -lGKlib).

#ifndef SUPERNODAL_LU_AUTO_ORDERING_H
#define SUPERNODAL_LU_AUTO_ORDERING_H

// Eigen/MetisSupport uses std::cerr on a METIS error but does not include
// <iostream> itself; pull it in here so this header is self-contained.
#include <iostream>

#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>

#include "SupernodalLU.h"
#include "SupernodalLUSymbolic.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace Eigen {
namespace supernodal_lu {

// Fill-reducing ordering that automatically chooses between AMD and several
// METIS nested-dissection restarts by predicting each candidate's symbolic
// fill. See file header for rationale.
template <typename StorageIndex>
class AutoOrdering {
 public:
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;
  typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;

  enum class Choice { AMD, Metis };

  // Diagnostics from the most recent operator() call (mainly for testing /
  // benchmarking the ordering choice in isolation).
  Choice lastChoice() const { return m_lastChoice; }
  int lastMetisSeed() const { return m_lastSeed; }  // valid iff lastChoice()==Metis
  double lastPredictedNnzL() const { return m_lastNnzL; }  // NaN if not estimated (huge-n path)

  template <typename MatrixType>
  void operator()(const MatrixType& A, PermutationType& matperm) {
    const StorageIndex n = internal::convert_index<StorageIndex>(A.cols());
    m_lastChoice = Choice::AMD;
    m_lastSeed = -1;
    m_lastNnzL = std::numeric_limits<double>::quiet_NaN();
    if (n == 0) {
      matperm.resize(0);
      return;
    }

    // Above this size, skip the multi-candidate comparison: nested dissection
    // is the established better choice for large, well-separated problems
    // (see pastix_algorithms.md / SupernodalLUMetis.h), and running the full
    // symbolic fill estimate for several candidates here would roughly double
    // the real analyzePattern() cost right when that cost is largest. Just
    // delegate straight to METIS, single pass, no estimate.
    static constexpr StorageIndex kFullSelectionLimit = 20000;
    if (n > kFullSelectionLimit) {
      IndexVector indexPtr, innerIndices;
      symbolic::buildSymmetrizedGraph<StorageIndex>(A, indexPtr, innerIndices);
      if (runMetis(n, indexPtr, innerIndices, /*seed=*/0, matperm)) {
        m_lastChoice = Choice::Metis;
        m_lastSeed = 0;
        return;
      }
      // METIS failed outright (should not happen in practice): fall back.
      AMDOrdering<StorageIndex> amd;
      amd(A, matperm);
      return;
    }

    // Symmetrized adjacency (A+A^T pattern, no diagonal), shared by every
    // candidate's fill estimate and by every METIS call.
    IndexVector indexPtr, innerIndices;
    symbolic::buildSymmetrizedGraph<StorageIndex>(A, indexPtr, innerIndices);

    // Candidate 0: Eigen's AMD (matches SupernodalLU's own default ordering,
    // called the same way -- directly on A, which AMDOrdering symmetrizes
    // internally).
    PermutationType best;
    AMDOrdering<StorageIndex> amd;
    amd(A, best);
    double bestNnzL = symbolic::estimateFillFromPermutation(n, indexPtr, innerIndices, best);
    m_lastNnzL = bestNnzL;

    // METIS_NodeND restarts with fixed, distinct seeds (reproducible, unlike
    // Eigen's MetisOrdering which passes options=NULL). Fewer seeds on larger
    // graphs bounds the extra symbolic-analysis cost.
    const int numSeeds = seedsForSize(n);
    for (int seed = 0; seed < numSeeds; ++seed) {
      PermutationType candidate;
      if (!runMetis(n, indexPtr, innerIndices, seed, candidate)) continue;
      const double nnzL = symbolic::estimateFillFromPermutation(n, indexPtr, innerIndices, candidate);
      if (nnzL < bestNnzL) {
        bestNnzL = nnzL;
        best = candidate;
        m_lastChoice = Choice::Metis;
        m_lastSeed = seed;
        m_lastNnzL = nnzL;
      }
    }
    matperm = best;
  }

 private:
  Choice m_lastChoice = Choice::AMD;
  int m_lastSeed = -1;
  double m_lastNnzL = std::numeric_limits<double>::quiet_NaN();

  static int seedsForSize(StorageIndex n) {
    if (n <= 5000) return 3;
    if (n <= 10000) return 2;
    return 1;  // up to kFullSelectionLimit
  }


  // Direct METIS_NodeND call with a fixed seed (everything else at METIS's
  // own defaults). Mirrors MetisOrdering::operator()'s perm/iperm handling:
  // row (column) i of A is row (column) matperm(i) of the permuted matrix.
  static bool runMetis(StorageIndex n, const IndexVector& indexPtr, const IndexVector& innerIndices, int seed,
                       PermutationType& matperm) {
    IndexVector perm(n), iperm(n);
    IndexVector xadj = indexPtr;      // METIS_NodeND wants non-const pointers
    IndexVector adjncy = innerIndices;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_SEED] = seed;
    idx_t m = n;
    const int err = METIS_NodeND(&m, xadj.data(), adjncy.data(), NULL, options, perm.data(), iperm.data());
    if (err != METIS_OK) return false;
    matperm.resize(n);
    for (StorageIndex j = 0; j < n; ++j) matperm.indices()(iperm(j)) = j;
    return true;
  }

};

}  // namespace supernodal_lu

template <typename StorageIndex>
using AutoOrdering = supernodal_lu::AutoOrdering<StorageIndex>;

/** \brief SupernodalLU that automatically chooses between AMD and several
 *         METIS nested-dissection restarts by predicted fill (see
 *         supernodal_lu::AutoOrdering / this file's header comment). Usually
 *         the best default when METIS is available and matrix provenance
 *         (mesh-like vs. irregular) isn't known ahead of time.
 *
 * \tparam MatrixType_ a column-major Eigen::SparseMatrix.
 * \tparam Executor_   optional parallel-execution backend, matching
 *                     SupernodalLU (default SerialExecutor).
 */
template <typename MatrixType_, typename Executor_ = supernodal_lu::SerialExecutor>
using SupernodalLUAuto =
    SupernodalLU<MatrixType_, AutoOrdering<typename MatrixType_::StorageIndex>, Executor_>;

}  // namespace Eigen

#endif  // SUPERNODAL_LU_AUTO_ORDERING_H
