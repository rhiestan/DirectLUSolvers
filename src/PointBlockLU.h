// PointBlockLU -- a direct sparse LU solver for systems with several unknowns
// per grid point, factored repeatedly against a fixed pattern.
//
// WHAT IT IS FOR, AND WHY IT IS NOT SUPERNODAL
//
// SupernodalLU and LeftRightLU are supernodal BLAS-3 solvers that eliminate the
// SYMMETRIZED pattern (A + A^T). This one keeps the UNSYMMETRIC pattern and
// factors it column by column -- left-looking Gilbert-Peierls with partial
// pivoting, scalar kernels, no supernodes and no BLAS-3 -- and it records the
// pattern and pivot sequence of the first factorization so that every later one
// on the same pattern is a pure numeric replay.
//
// USE IT WHEN THE FACTOR STAYS SPARSE. That is the whole envelope, and it is
// narrower than "small matrices". Symmetrizing an unsymmetric pattern can cost
// an enormous amount of fill, and avoiding that is what this solver buys; but a
// scalar column algorithm runs at roughly a fifth of the throughput of a
// supernodal one on dense panels, so once the factor densifies the fill
// advantage is spent and the sibling solvers win. Measured against LeftRightLU
// on this project's testdata, factor(+solve) with the same COLAMD ordering,
// single-threaded (PointBlockLU timed on its REPLAY, which is the call a Newton
// loop makes):
//
//   matrix        n     PointBlockLU fill / time   LeftRightLU fill / time   ratio
//   setfos      1015        4,080 /   0.03 ms      116,786 /  2.06 ms       69x
//   bayer05     3268       77,462 /   1.25 ms      456,036 / 13.41 ms       10.7x
//   gemat11     4929       79,614 /   1.52 ms      131,156 /  3.32 ms        2.2x
//   tomography   500       46,540 /   2.39 ms      180,154 /  4.52 ms        1.9x
//   sherman1    1000       32,916 /   0.84 ms       40,884 /  0.84 ms        1.0x
//   laoss_3     4180      731,852 /  44.0 ms     1,210,476 / 24.9 ms         0.57x
//   YaleB_10NN  2414    1,232,024 / 258.1 ms     1,638,482 / 80.2 ms         0.31x
//   setfos_2    3048    1,935,546 / 452.8 ms     2,360,714 / 100.7 ms        0.22x
//
// The crossover sits near 100k stored scalars in the factor. Below it this
// solver wins by a lot; above it, use LeftRightLU.
//
// STRUCTURALLY SINGULAR INPUT IS DECLINED, not patched. An unsymmetric LU needs
// a pivot in every column, and testdata/bcsstm13 has 762 numerically empty
// columns out of 2003, so factorize() reports NumericalIssue and says which
// column -- exactly as Eigen::SparseLU does on the same matrix. The supernodal
// siblings appear to succeed there only because symmetrizing the pattern fills
// those columns in from the transpose; their answer on that matrix carries a
// relative error of 0.62 either way.
//
// ORDERING. The default is COLAMD, and the table above uses it. Note that
// PointBlockOrdering -- despite the shared name, which refers to the target
// matrix class rather than to the ordering -- is a poor default HERE: it ranks
// its candidates by SYMMETRIC-pattern fill, which is the right objective for the
// supernodal siblings and the wrong one for an unsymmetric-pattern LU. Measured
// on gemat11 it produces 6,742,455 stored scalars against COLAMD's 79,479, and
// 5.3 s against 2.9 ms. Use it with LeftRightLU, not with this solver.
//
// SINGLE-THREADED, DELIBERATELY -- AND MEASURED, NOT ASSUMED. Unlike its two
// siblings this solver takes no Executor template parameter. It is not an
// oversight and not a to-do: a parallel version was built, measured, and
// removed.
//
// The REPLAY is the half that could be scheduled at all. By then the pattern and
// the pivots are fixed, so the column dependency DAG is known (column k needs
// every column j < k appearing in U(:,k), and writes only its own L and U
// entries). The first factorization cannot be scheduled even in principle:
// partial pivoting means the choice made in column k determines the structure of
// every later column. So the implementation was a DAG scheduler over the replay,
// one fork-join dispatch for the whole factorization with per-lane scratch rows
// and per-worker deques with stealing -- and it was bit-identical to the serial
// replay, because each column applies its updates in the same order whichever
// lane runs it. It still did not pay. Replay microseconds:
//
//   matrix      ideal   1 lane      2      4      8     16
//   setfos      1.25x       24     70    163    219    459
//   bayer05     3.32x     1129   1069   1015    988   1294
//   gemat11     1.78x     1380   1336   1263   1241   1640
//   setfos_2    1.00x   437610 401312 471920 491068 502278
//
// The best result anywhere was 1.14x (bayer05 at 8 lanes); past 8 lanes every
// case lost, and setfos degraded 19x. Two structural reasons, either of which
// would be enough:
//
//   1. THE DAG HAS NO WIDTH. "ideal" above is total work over critical path --
//      the ceiling on ANY schedule, before a single thread is created. It is
//      1.00x on setfos_2, 1.02x on YaleB_10NN, 1.01x on tomography, 1.25x on
//      setfos, and reaches only 3.3x on bayer05. A 2-D Laplacian control scores
//      just 1.4x, so this is not a quirk of the target matrices: it is the same
//      fact doc/Parallelism.md records for the supernodal solvers, where level/DAG
//      parallelism alone never exceeded 1.79x and all the real scaling came from
//      chunking INSIDE dense panels. This solver has no dense panels to chunk --
//      having none is the point of it.
//   2. THE TASKS ARE TOO SMALL TO SCHEDULE. A column of bayer05's replay costs
//      ~350 ns, while its plan has ~37k DAG edges each needing an atomic
//      decrement to release a consumer. The bookkeeping costs what the
//      arithmetic costs.
//
// If you need threads on a matrix in this class, the useful lever is not this
// solver: it is LeftRightLU, whose supernodal panels are coarse enough to
// schedule (see doc/Parallelism.md's parallel scaling section). Reach for it when the
// factor is dense enough that the crossover above has been passed anyway.
//
// REFACTORIZATION IS THE POINT. analyzePattern() chooses the ordering; the
// FIRST factorize() does the symbolic search (a depth-first reachability pass
// per column) together with the numeric work and records the pattern of L and U
// and the pivot sequence; every later factorize() on the same pattern replays
// that record with no search, no allocation and no pivot choice. That is the
// call a Newton loop actually makes.
//
// Usage:
//   #include <PointBlockLU.h>
//   Eigen::PointBlockLU<Eigen::SparseMatrix<double>> solver;
//   solver.analyzePattern(A);          // once
//   for (each Newton step) {
//     solver.factorize(A);             // replays the recorded plan
//     x = solver.solve(b);
//   }
//
// This Source Code Form is licensed under the Mozilla Public License v.2.0,
// matching the surrounding Eigen code it integrates with.

#ifndef POINT_BLOCK_LU_H
#define POINT_BLOCK_LU_H

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>

#include <algorithm>
#include <string>
#include <vector>

namespace Eigen {
namespace point_block {

// Eigen's ordering functors do not agree on which direction of the permutation
// they return, and nothing in the numbers distinguishes them -- getting it
// backwards leaves residuals at machine precision and only shows up as fill
// (measured here: lap2d_30^2 under COLAMD, 126,180 stored scalars the wrong way
// round against 22,176 the right way). So it is a compile-time trait, exactly as
// left_right_lu::OrderingConvention is for LeftRightLU.
//
//   returnsInverse == true : indices()(k) is the ORIGINAL index placed at new
//                            position k -- AMDOrdering, MetisOrdering,
//                            PointBlockOrdering. Used as the elimination order
//                            directly.
//   returnsInverse == false: indices()(i) is the NEW position of original i --
//                            COLAMDOrdering. Inverted before use.
//
// Specialize it for a custom functor that returns the direct map.
template <typename OrderingType>
struct OrderingConvention {
  static constexpr bool returnsInverse = true;
};

template <typename StorageIndex>
struct OrderingConvention<COLAMDOrdering<StorageIndex>> {
  static constexpr bool returnsInverse = false;
};

}  // namespace point_block

/** \class PointBlockLU
 * \brief Unsymmetric-pattern sparse LU with partial pivoting, built for small
 *        systems refactorized many times against one pattern.
 *
 * \tparam MatrixType_   a column-major Eigen::SparseMatrix.
 * \tparam OrderingType_ column preordering functor. The default is COLAMD,
 *                       which is the ordering designed for exactly this
 *                       factorization (it bounds the fill of LU with partial
 *                       pivoting); AMDOrdering, NaturalOrdering and
 *                       PointBlockOrdering also work.
 */
template <typename MatrixType_,
          typename OrderingType_ = COLAMDOrdering<typename MatrixType_::StorageIndex>>
class PointBlockLU : public SparseSolverBase<PointBlockLU<MatrixType_, OrderingType_>> {
 protected:
  typedef SparseSolverBase<PointBlockLU<MatrixType_, OrderingType_>> Base;
  using Base::m_isInitialized;

 public:
  typedef MatrixType_ MatrixType;
  typedef OrderingType_ OrderingType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  using Base::_solve_impl;

  PointBlockLU() { init(); }
  explicit PointBlockLU(const MatrixType& matrix) {
    init();
    compute(matrix);
  }

  // --- main driver ----------------------------------------------------------

  /** Choose the column preordering. Cheap: no symbolic factorization happens
   *  here, because with partial pivoting the pattern of L and U is not known
   *  until the numeric values are seen. The first factorize() does that work. */
  void analyzePattern(const MatrixType& matrix);

  /** Numeric factorization. The first call after analyzePattern() searches for
   *  the pattern and the pivots; later calls replay them (see refactorizations()
   *  and forceFullFactorization()). */
  void factorize(const MatrixType& matrix);

  void compute(const MatrixType& matrix) {
    analyzePattern(matrix);
    factorize(matrix);
  }

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

  // --- queries --------------------------------------------------------------

  inline Index rows() const { return m_size; }
  inline Index cols() const { return m_size; }

  /** Status of the last operation. */
  ComputationInfo info() const { return m_info; }
  bool isFactorized() const { return m_factorized; }
  const std::string& lastErrorMessage() const { return m_lastError; }

  /** Stored scalars in L (including its unit diagonal) and in U. */
  Index nnzL() const { return static_cast<Index>(m_lRow.size()); }
  Index nnzU() const { return static_cast<Index>(m_uRow.size()); }

  /** How many times the recorded plan has been replayed since the last full
   *  factorization. 0 means the next factorize() will still be a full one. */
  Index refactorizations() const { return m_refactorizations; }

  /** Column preordering, in Eigen's convention: indices()(k) is the original
   *  column eliminated k-th. */
  const PermutationType& colsPermutation() const { return m_colPermutation; }
  /** Row permutation chosen by partial pivoting: indices()(i) is the position at
   *  which original row i became pivotal. Valid after the first factorize(). */
  const PermutationType& rowsPermutation() const { return m_rowPermutation; }

  /** Access the ordering functor, e.g. to declare the blocking explicitly
   *  (`solver.orderingFunctor().setVariablesPerNode(3)`) or to read back what
   *  auto-detection chose. Must be configured BEFORE analyzePattern(). */
  OrderingType& orderingFunctor() { return m_orderingFunctor; }
  const OrderingType& orderingFunctor() const { return m_orderingFunctor; }

  // --- options --------------------------------------------------------------

  /** Partial-pivoting threshold in (0, 1]. A candidate diagonal entry is kept
   *  as the pivot when |a_kk| >= threshold * max|a_ik| over the column, so 1.0
   *  is strict partial pivoting (most stable, most fill) and a small value
   *  prefers the diagonal (less fill, and a pivot sequence that survives
   *  refactorization better). Default 1.0. */
  void setPivotThreshold(const RealScalar& threshold) {
    m_pivotThreshold = numext::mini(RealScalar(1), numext::maxi(RealScalar(0), threshold));
  }
  RealScalar pivotThreshold() const { return m_pivotThreshold; }

  /** Row/column equilibration (Ruiz). On by default and worth keeping on for
   *  this matrix class: testdata/setfos_2 spans 4e48 in magnitude, where an
   *  unscaled pivot comparison is meaningless. Folded transparently into
   *  solve(). */
  void setEquilibration(bool on) { m_equilibrate = on; }
  bool equilibration() const { return m_equilibrate; }

  /** Force every factorize() to redo the symbolic search and re-choose pivots
   *  instead of replaying. Off by default; useful when the VALUES change enough
   *  between calls that the recorded pivot sequence is no longer appropriate. */
  void setForceFullFactorization(bool on) { m_forceFull = on; }
  bool forceFullFactorization() const { return m_forceFull; }

  /** A replay is rejected, and a full factorization redone, when a pivot's
   *  magnitude falls below this fraction of the magnitude it had when the plan
   *  was recorded. Default 1e-8; 0 disables the check. This is what keeps
   *  replaying safe when the caller's values drift. */
  void setMinPivotRatio(const RealScalar& ratio) { m_minPivotRatio = ratio; }
  RealScalar minPivotRatio() const { return m_minPivotRatio; }

  /** log|det(A)| as a sum of logs (stays finite where determinant() overflows),
   *  paired with determinantSign(). */
  Scalar logAbsDeterminant() const;
  Scalar determinantSign() const;
  Scalar determinant() const;

 private:
  void init() {
    m_size = 0;
    m_info = Success;
    m_analyzed = false;
    m_factorized = false;
    m_planRecorded = false;
    m_refactorizations = 0;
    m_pivotThreshold = RealScalar(1);
    m_equilibrate = true;
    m_forceFull = false;
    m_minPivotRatio = RealScalar(1e-8);
    m_isInitialized = false;
  }

  void computeScaling(const MatrixType& matrix);
  bool fullFactorize(const MatrixType& matrix);
  bool replayFactorize(const MatrixType& matrix);
  void replayColumn(const MatrixType& matrix, StorageIndex k, Scalar* work, bool& reject);
  void sortFactorColumns();

  // Depth-first reachability over the graph of the L columns built so far,
  // starting from the pattern of one column of A. Returns `top`: the reachable
  // nodes are m_stackXi[top .. n-1], in topological order.
  StorageIndex reach(const MatrixType& matrix, StorageIndex column);
  void dfs(StorageIndex start, StorageIndex& top);

  OrderingType m_orderingFunctor;

  StorageIndex m_size;
  ComputationInfo m_info;
  std::string m_lastError;
  bool m_analyzed, m_factorized, m_planRecorded, m_equilibrate, m_forceFull;
  Index m_refactorizations;
  RealScalar m_pivotThreshold, m_minPivotRatio;

  PermutationType m_colPermutation;  // indices()(k) = original column eliminated k-th
  PermutationType m_rowPermutation;  // indices()(i) = pivotal position of original row i
  std::vector<StorageIndex> m_colOf;   // m_colOf[k] = original column eliminated k-th
  std::vector<StorageIndex> m_pinv;    // pinv[originalRow] = pivotal position, -1 while unused
  std::vector<StorageIndex> m_rowOf;   // m_rowOf[k] = original row that became pivotal at k

  // L and U in compressed columns, row indices in PIVOTAL numbering.
  // L is unit lower triangular with its diagonal stored explicitly first.
  std::vector<StorageIndex> m_lPtr, m_lRow, m_uPtr, m_uRow;
  std::vector<Scalar> m_lVal, m_uVal;
  std::vector<Scalar> m_pivotRecorded;  // |pivot| at the time the plan was recorded

  std::vector<RealScalar> m_rowScale, m_colScale;

  // scratch, kept across calls so a replay allocates nothing
  std::vector<Scalar> m_work;
  std::vector<StorageIndex> m_stackXi, m_stackPos, m_marker;

};

// ---------------------------------------------------------------------------
//  analyzePattern
// ---------------------------------------------------------------------------

template <typename MatrixType, typename OrderingType>
void PointBlockLU<MatrixType, OrderingType>::analyzePattern(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "PointBlockLU: matrix must be square");
  m_size = internal::convert_index<StorageIndex>(matrix.cols());
  m_info = Success;
  m_lastError.clear();
  m_analyzed = false;
  m_factorized = false;
  m_planRecorded = false;
  m_refactorizations = 0;
  m_isInitialized = false;

  const StorageIndex n = m_size;
  m_colOf.resize(static_cast<std::size_t>(n));
  if (n == 0) {
    m_analyzed = true;
    m_isInitialized = true;
    return;
  }

  PermutationType perm;
  m_orderingFunctor(matrix, perm);
  if (perm.size() == 0) {  // NaturalOrdering reports the identity as empty
    for (StorageIndex k = 0; k < n; ++k) m_colOf[k] = k;
  } else if (point_block::OrderingConvention<OrderingType>::returnsInverse) {
    // indices()(k) is the original column placed at new position k, which is
    // exactly "the column eliminated k-th" -- used as it comes.
    for (StorageIndex k = 0; k < n; ++k) m_colOf[k] = perm.indices()(k);
  } else {
    // indices()(i) is the new position of original column i, so invert.
    for (StorageIndex i = 0; i < n; ++i) m_colOf[static_cast<std::size_t>(perm.indices()(i))] = i;
  }
  m_colPermutation.resize(n);
  for (StorageIndex k = 0; k < n; ++k) m_colPermutation.indices()(k) = m_colOf[k];

  m_work.assign(static_cast<std::size_t>(n), Scalar(0));
  m_stackXi.resize(static_cast<std::size_t>(n));
  m_stackPos.resize(static_cast<std::size_t>(n));
  m_marker.assign(static_cast<std::size_t>(n), StorageIndex(-1));
  m_analyzed = true;
  m_isInitialized = true;
}

// ---------------------------------------------------------------------------
//  scaling
// ---------------------------------------------------------------------------

template <typename MatrixType, typename OrderingType>
void PointBlockLU<MatrixType, OrderingType>::computeScaling(const MatrixType& matrix) {
  const StorageIndex n = m_size;
  m_rowScale.assign(static_cast<std::size_t>(n), RealScalar(1));
  m_colScale.assign(static_cast<std::size_t>(n), RealScalar(1));
  if (!m_equilibrate) return;

  // Ruiz equilibration on |A|: iterate row and column scalings towards a matrix
  // whose rows and columns all have infinity norm 1, so that the pivot
  // comparison in factorize() is meaningful (testdata/setfos_2 spans 4e48).
  //
  // ITERATE TO CONVERGENCE, DO NOT RUN A FIXED COUNT. This runs on every
  // factorize(), replays included, and the sweep is O(nnz) while the
  // factorization it is preparing can be far cheaper: on testdata/setfos (a
  // tridiagonal matrix needing 2050 flops) a fixed eight sweeps WAS 80% of the
  // whole replay, 97.7 us against 19.1 us with scaling off. A well-scaled matrix
  // converges in one or two sweeps and should pay for one or two.
  const RealScalar tolerance = RealScalar(0.1);  // max deviation of any row/col norm from 1
  const int maxSweeps = 20;
  std::vector<RealScalar> rowMax(static_cast<std::size_t>(n)), colMax(static_cast<std::size_t>(n));
  for (int sweep = 0; sweep < maxSweeps; ++sweep) {
    std::fill(rowMax.begin(), rowMax.end(), RealScalar(0));
    std::fill(colMax.begin(), colMax.end(), RealScalar(0));
    for (StorageIndex j = 0; j < n; ++j) {
      for (typename MatrixType::InnerIterator it(matrix, j); it; ++it) {
        const StorageIndex i = internal::convert_index<StorageIndex>(it.index());
        const RealScalar v = numext::abs(it.value()) * m_rowScale[i] * m_colScale[j];
        rowMax[i] = numext::maxi(rowMax[i], v);
        colMax[j] = numext::maxi(colMax[j], v);
      }
    }
    RealScalar deviation = RealScalar(0);
    for (StorageIndex i = 0; i < n; ++i)
      if (rowMax[i] > RealScalar(0))
        deviation = numext::maxi(deviation, numext::abs(RealScalar(1) - rowMax[i]));
    for (StorageIndex j = 0; j < n; ++j)
      if (colMax[j] > RealScalar(0))
        deviation = numext::maxi(deviation, numext::abs(RealScalar(1) - colMax[j]));
    if (deviation <= tolerance) break;

    for (StorageIndex i = 0; i < n; ++i)
      if (rowMax[i] > RealScalar(0)) m_rowScale[i] /= numext::sqrt(rowMax[i]);
    for (StorageIndex j = 0; j < n; ++j)
      if (colMax[j] > RealScalar(0)) m_colScale[j] /= numext::sqrt(colMax[j]);
  }
}

// ---------------------------------------------------------------------------
//  depth-first reachability (the symbolic half of Gilbert-Peierls)
// ---------------------------------------------------------------------------

// Non-recursive DFS from `start` over the graph whose node set is the original
// row indices and whose edges out of node i are the off-diagonal rows of the
// already-built L column m_pinv[i] (nodes with m_pinv[i] < 0 are sinks: their
// column does not exist yet). Pushes finished nodes onto m_stackXi from the top
// down, which leaves m_stackXi[top .. n-1] in topological order.
template <typename MatrixType, typename OrderingType>
void PointBlockLU<MatrixType, OrderingType>::dfs(StorageIndex start, StorageIndex& top) {
  const StorageIndex n = m_size;
  StorageIndex head = 0;
  m_stackXi[0] = start;  // the DFS stack grows from index 0 upwards
  while (head >= 0) {
    const StorageIndex node = m_stackXi[static_cast<std::size_t>(head)];
    const StorageIndex col = m_pinv[static_cast<std::size_t>(node)];
    if (m_marker[static_cast<std::size_t>(node)] != StorageIndex(-2)) {
      m_marker[static_cast<std::size_t>(node)] = StorageIndex(-2);  // mark visited
      // resume position: first off-diagonal entry of L(:, col), if col exists
      m_stackPos[static_cast<std::size_t>(head)] =
          col < 0 ? StorageIndex(0) : static_cast<StorageIndex>(m_lPtr[static_cast<std::size_t>(col)] + 1);
    }
    bool descended = false;
    if (col >= 0) {
      const StorageIndex end = static_cast<StorageIndex>(m_lPtr[static_cast<std::size_t>(col) + 1]);
      StorageIndex p = m_stackPos[static_cast<std::size_t>(head)];
      for (; p < end; ++p) {
        const StorageIndex child = m_lRow[static_cast<std::size_t>(p)];
        if (m_marker[static_cast<std::size_t>(child)] == StorageIndex(-2)) continue;
        m_stackPos[static_cast<std::size_t>(head)] = p + 1;
        m_stackXi[static_cast<std::size_t>(++head)] = child;
        descended = true;
        break;
      }
      if (!descended) m_stackPos[static_cast<std::size_t>(head)] = end;
    }
    if (!descended) {
      --head;
      m_stackXi[static_cast<std::size_t>(--top)] = node;  // node is finished
    }
  }
  (void)n;
}

template <typename MatrixType, typename OrderingType>
typename MatrixType::StorageIndex PointBlockLU<MatrixType, OrderingType>::reach(const MatrixType& matrix,
                                                                                StorageIndex column) {
  const StorageIndex n = m_size;
  StorageIndex top = n;
  for (typename MatrixType::InnerIterator it(matrix, column); it; ++it) {
    const StorageIndex i = internal::convert_index<StorageIndex>(it.index());
    if (m_marker[static_cast<std::size_t>(i)] != StorageIndex(-2)) dfs(i, top);
  }
  for (StorageIndex p = top; p < n; ++p)
    m_marker[static_cast<std::size_t>(m_stackXi[static_cast<std::size_t>(p)])] = StorageIndex(-1);
  return top;
}

// ---------------------------------------------------------------------------
//  factorize
// ---------------------------------------------------------------------------

template <typename MatrixType, typename OrderingType>
void PointBlockLU<MatrixType, OrderingType>::factorize(const MatrixType& matrix) {
  eigen_assert(m_analyzed && "PointBlockLU: factorize() called before analyzePattern()");
  eigen_assert(matrix.rows() == m_size && matrix.cols() == m_size &&
               "PointBlockLU: factorize() matrix size differs from analyzePattern()");
  m_info = Success;
  m_lastError.clear();
  m_factorized = false;
  if (m_size == 0) {
    m_factorized = true;
    return;
  }

  computeScaling(matrix);

  // Replay the recorded plan when there is one. A rejected replay (a pivot that
  // collapsed relative to the value it had when the plan was recorded) falls
  // back to a full factorization rather than returning a bad factor.
  if (m_planRecorded && !m_forceFull) {
    if (replayFactorize(matrix)) {
      ++m_refactorizations;
      m_factorized = true;
      return;
    }
    m_planRecorded = false;  // the recorded pivots no longer fit the values
  }

  m_refactorizations = 0;
  if (!fullFactorize(matrix)) return;  // m_info / m_lastError already set
  sortFactorColumns();
  m_planRecorded = true;
  m_factorized = true;
}

template <typename MatrixType, typename OrderingType>
bool PointBlockLU<MatrixType, OrderingType>::fullFactorize(const MatrixType& matrix) {
  const StorageIndex n = m_size;
  const std::size_t un = static_cast<std::size_t>(n);

  m_pinv.assign(un, StorageIndex(-1));
  m_rowOf.assign(un, StorageIndex(-1));
  m_lPtr.assign(un + 1, StorageIndex(0));
  m_uPtr.assign(un + 1, StorageIndex(0));
  m_lRow.clear();
  m_lVal.clear();
  m_uRow.clear();
  m_uVal.clear();
  const std::size_t guess = static_cast<std::size_t>(matrix.nonZeros()) * 2 + un;
  m_lRow.reserve(guess);
  m_lVal.reserve(guess);
  m_uRow.reserve(guess);
  m_uVal.reserve(guess);
  m_pivotRecorded.assign(un, Scalar(0));
  std::fill(m_marker.begin(), m_marker.end(), StorageIndex(-1));
  std::fill(m_work.begin(), m_work.end(), Scalar(0));

  for (StorageIndex k = 0; k < n; ++k) {
    m_lPtr[static_cast<std::size_t>(k)] = static_cast<StorageIndex>(m_lRow.size());
    m_uPtr[static_cast<std::size_t>(k)] = static_cast<StorageIndex>(m_uRow.size());
    const StorageIndex col = m_colOf[static_cast<std::size_t>(k)];

    // --- symbolic: which rows can this column touch, in topological order ----
    const StorageIndex top = reach(matrix, col);

    // --- numeric: x = L \ A(:, col), over that pattern only -----------------
    for (StorageIndex p = top; p < n; ++p)
      m_work[static_cast<std::size_t>(m_stackXi[static_cast<std::size_t>(p)])] = Scalar(0);
    for (typename MatrixType::InnerIterator it(matrix, col); it; ++it) {
      const StorageIndex i = internal::convert_index<StorageIndex>(it.index());
      m_work[static_cast<std::size_t>(i)] =
          it.value() * m_rowScale[static_cast<std::size_t>(i)] * m_colScale[static_cast<std::size_t>(col)];
    }
    for (StorageIndex px = top; px < n; ++px) {
      const StorageIndex j = m_stackXi[static_cast<std::size_t>(px)];  // original row
      const StorageIndex jj = m_pinv[static_cast<std::size_t>(j)];     // its column of L, or -1
      if (jj < 0) continue;
      const Scalar xj = m_work[static_cast<std::size_t>(j)];
      if (xj == Scalar(0)) continue;
      const StorageIndex end = m_lPtr[static_cast<std::size_t>(jj) + 1];
      for (StorageIndex p = m_lPtr[static_cast<std::size_t>(jj)] + 1; p < end; ++p)
        m_work[static_cast<std::size_t>(m_lRow[static_cast<std::size_t>(p)])] -=
            m_lVal[static_cast<std::size_t>(p)] * xj;
    }

    // --- pivot: largest magnitude among the rows not yet pivotal ------------
    StorageIndex ipiv = StorageIndex(-1);
    RealScalar best = RealScalar(-1);
    for (StorageIndex p = top; p < n; ++p) {
      const StorageIndex i = m_stackXi[static_cast<std::size_t>(p)];
      if (m_pinv[static_cast<std::size_t>(i)] >= 0) {  // already pivotal: this is a U entry
        m_uRow.push_back(m_pinv[static_cast<std::size_t>(i)]);
        m_uVal.push_back(m_work[static_cast<std::size_t>(i)]);
      } else {
        const RealScalar t = numext::abs(m_work[static_cast<std::size_t>(i)]);
        if (t > best) {
          best = t;
          ipiv = i;
        }
      }
    }
    if (ipiv < 0 || !(best > RealScalar(0))) {
      m_info = NumericalIssue;
      m_lastError = "PointBlockLU: no acceptable pivot in column " + std::to_string(k) +
                    " (matrix is structurally or numerically singular).";
      return false;
    }
    // Prefer the structural diagonal when it is within the threshold of the
    // best candidate: it keeps the pivot sequence closer to the caller's own
    // numbering, which is what makes a recorded plan survive value changes.
    if (m_pinv[static_cast<std::size_t>(col)] < 0 &&
        numext::abs(m_work[static_cast<std::size_t>(col)]) >= best * m_pivotThreshold)
      ipiv = col;

    const Scalar pivot = m_work[static_cast<std::size_t>(ipiv)];
    m_uRow.push_back(k);  // U's diagonal goes last in the column
    m_uVal.push_back(pivot);
    m_pivotRecorded[static_cast<std::size_t>(k)] = pivot;
    m_pinv[static_cast<std::size_t>(ipiv)] = k;
    m_rowOf[static_cast<std::size_t>(k)] = ipiv;

    m_lRow.push_back(ipiv);  // L's unit diagonal, stored explicitly
    m_lVal.push_back(Scalar(1));
    for (StorageIndex p = top; p < n; ++p) {
      const StorageIndex i = m_stackXi[static_cast<std::size_t>(p)];
      if (m_pinv[static_cast<std::size_t>(i)] < 0) {
        m_lRow.push_back(i);
        m_lVal.push_back(m_work[static_cast<std::size_t>(i)] / pivot);
      }
      m_work[static_cast<std::size_t>(i)] = Scalar(0);
    }
  }
  m_lPtr[un] = static_cast<StorageIndex>(m_lRow.size());
  m_uPtr[un] = static_cast<StorageIndex>(m_uRow.size());

  // L was built with ORIGINAL row indices (the DFS walks that numbering);
  // relabel it into pivotal numbering now that every row has a pivot position.
  for (std::size_t p = 0; p < m_lRow.size(); ++p)
    m_lRow[p] = m_pinv[static_cast<std::size_t>(m_lRow[p])];

  m_rowPermutation.resize(n);
  for (StorageIndex i = 0; i < n; ++i)
    m_rowPermutation.indices()(i) = m_pinv[static_cast<std::size_t>(i)];
  return true;
}

// Sort each column of L and U by row index. Both the replay and the triangular
// solves rely on the resulting canonical form: ascending order puts L's unit
// diagonal (row k in column k, its smallest) first and U's diagonal (row k, its
// largest) last, and it is a valid topological order for the replay -- column k
// only ever depends on columns j < k.
template <typename MatrixType, typename OrderingType>
void PointBlockLU<MatrixType, OrderingType>::sortFactorColumns() {
  const StorageIndex n = m_size;
  std::vector<StorageIndex> order;
  std::vector<StorageIndex> rowBuf;
  std::vector<Scalar> valBuf;
  for (int which = 0; which < 2; ++which) {
    std::vector<StorageIndex>& ptr = which == 0 ? m_lPtr : m_uPtr;
    std::vector<StorageIndex>& row = which == 0 ? m_lRow : m_uRow;
    std::vector<Scalar>& val = which == 0 ? m_lVal : m_uVal;
    for (StorageIndex k = 0; k < n; ++k) {
      const StorageIndex begin = ptr[static_cast<std::size_t>(k)];
      const StorageIndex end = ptr[static_cast<std::size_t>(k) + 1];
      const StorageIndex len = end - begin;
      if (len < 2) continue;
      order.resize(static_cast<std::size_t>(len));
      for (StorageIndex t = 0; t < len; ++t) order[static_cast<std::size_t>(t)] = t;
      std::sort(order.begin(), order.end(), [&](StorageIndex a, StorageIndex b) {
        return row[static_cast<std::size_t>(begin + a)] < row[static_cast<std::size_t>(begin + b)];
      });
      rowBuf.resize(static_cast<std::size_t>(len));
      valBuf.resize(static_cast<std::size_t>(len));
      for (StorageIndex t = 0; t < len; ++t) {
        rowBuf[static_cast<std::size_t>(t)] = row[static_cast<std::size_t>(begin + order[static_cast<std::size_t>(t)])];
        valBuf[static_cast<std::size_t>(t)] = val[static_cast<std::size_t>(begin + order[static_cast<std::size_t>(t)])];
      }
      for (StorageIndex t = 0; t < len; ++t) {
        row[static_cast<std::size_t>(begin + t)] = rowBuf[static_cast<std::size_t>(t)];
        val[static_cast<std::size_t>(begin + t)] = valBuf[static_cast<std::size_t>(t)];
      }
    }
  }
}

// One column of the replay, against a caller-supplied scratch row. Every entry
// this touches is zeroed on ENTRY rather than on exit, so a lane may reuse one
// scratch row across columns without clearing it in between.
template <typename MatrixType, typename OrderingType>
void PointBlockLU<MatrixType, OrderingType>::replayColumn(const MatrixType& matrix,
                                                                    StorageIndex k, Scalar* work,
                                                                    bool& reject) {
  const StorageIndex uBegin = m_uPtr[static_cast<std::size_t>(k)];
  const StorageIndex uEnd = m_uPtr[static_cast<std::size_t>(k) + 1];
  const StorageIndex lBegin = m_lPtr[static_cast<std::size_t>(k)];
  const StorageIndex lEnd = m_lPtr[static_cast<std::size_t>(k) + 1];

  for (StorageIndex p = uBegin; p < uEnd; ++p) work[m_uRow[static_cast<std::size_t>(p)]] = Scalar(0);
  for (StorageIndex p = lBegin; p < lEnd; ++p) work[m_lRow[static_cast<std::size_t>(p)]] = Scalar(0);

  const StorageIndex col = m_colOf[static_cast<std::size_t>(k)];
  for (typename MatrixType::InnerIterator it(matrix, col); it; ++it) {
    const StorageIndex i = internal::convert_index<StorageIndex>(it.index());
    work[m_pinv[static_cast<std::size_t>(i)]] =
        it.value() * m_rowScale[static_cast<std::size_t>(i)] * m_colScale[static_cast<std::size_t>(col)];
  }

  for (StorageIndex p = uBegin; p + 1 < uEnd; ++p) {
    const StorageIndex j = m_uRow[static_cast<std::size_t>(p)];
    const Scalar xj = work[j];
    m_uVal[static_cast<std::size_t>(p)] = xj;
    if (xj == Scalar(0)) continue;
    const StorageIndex end = m_lPtr[static_cast<std::size_t>(j) + 1];
    for (StorageIndex q = m_lPtr[static_cast<std::size_t>(j)] + 1; q < end; ++q)
      work[m_lRow[static_cast<std::size_t>(q)]] -= m_lVal[static_cast<std::size_t>(q)] * xj;
  }

  const Scalar pivot = work[k];
  const RealScalar recorded = numext::abs(m_pivotRecorded[static_cast<std::size_t>(k)]);
  if (pivot == Scalar(0) || !(numext::abs(pivot) == numext::abs(pivot))) {
    reject = true;
    return;
  }
  if (m_minPivotRatio > RealScalar(0) && recorded > RealScalar(0) &&
      numext::abs(pivot) < m_minPivotRatio * recorded) {
    reject = true;
    return;
  }
  m_uVal[static_cast<std::size_t>(uEnd - 1)] = pivot;
  m_lVal[static_cast<std::size_t>(lBegin)] = Scalar(1);
  for (StorageIndex p = lBegin + 1; p < lEnd; ++p)
    m_lVal[static_cast<std::size_t>(p)] = work[m_lRow[static_cast<std::size_t>(p)]] / pivot;
}

// Replay: the pattern of L and U and the pivot rows are already known, so this
// is the same column sweep with the symbolic half deleted -- no reachability
// search, no pivot comparison, no allocation. This is the call a Newton loop
// makes, and it is why the solver exists.
template <typename MatrixType, typename OrderingType>
bool PointBlockLU<MatrixType, OrderingType>::replayFactorize(const MatrixType& matrix) {
  const StorageIndex n = m_size;
  for (StorageIndex k = 0; k < n; ++k) {
    bool reject = false;
    replayColumn(matrix, k, m_work.data(), reject);
    if (reject) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
//  solve
// ---------------------------------------------------------------------------

template <typename MatrixType, typename OrderingType>
template <typename Rhs, typename Dest>
void PointBlockLU<MatrixType, OrderingType>::_solve_impl(const MatrixBase<Rhs>& b,
                                                         MatrixBase<Dest>& x) const {
  eigen_assert(m_factorized && "PointBlockLU: solve() called before a successful factorize()");
  const StorageIndex n = m_size;
  x.derived().resize(n, b.cols());
  if (n == 0) return;

  // L U y = Pr Dr b, then z = Pc y, then x = Dc z.
  std::vector<Scalar> w(static_cast<std::size_t>(n));
  for (Index rhs = 0; rhs < b.cols(); ++rhs) {
    for (StorageIndex i = 0; i < n; ++i)
      w[static_cast<std::size_t>(m_pinv[static_cast<std::size_t>(i)])] =
          b.derived().coeff(i, rhs) * m_rowScale[static_cast<std::size_t>(i)];

    for (StorageIndex k = 0; k < n; ++k) {  // unit lower triangular, diagonal first
      const Scalar xk = w[static_cast<std::size_t>(k)];
      if (xk == Scalar(0)) continue;
      const StorageIndex end = m_lPtr[static_cast<std::size_t>(k) + 1];
      for (StorageIndex p = m_lPtr[static_cast<std::size_t>(k)] + 1; p < end; ++p)
        w[static_cast<std::size_t>(m_lRow[static_cast<std::size_t>(p)])] -=
            m_lVal[static_cast<std::size_t>(p)] * xk;
    }
    for (StorageIndex k = n - 1; k >= 0; --k) {  // upper triangular, diagonal last
      const StorageIndex begin = m_uPtr[static_cast<std::size_t>(k)];
      const StorageIndex end = m_uPtr[static_cast<std::size_t>(k) + 1];
      w[static_cast<std::size_t>(k)] /= m_uVal[static_cast<std::size_t>(end - 1)];
      const Scalar xk = w[static_cast<std::size_t>(k)];
      if (xk != Scalar(0))
        for (StorageIndex p = begin; p + 1 < end; ++p)
          w[static_cast<std::size_t>(m_uRow[static_cast<std::size_t>(p)])] -=
              m_uVal[static_cast<std::size_t>(p)] * xk;
      if (k == 0) break;  // StorageIndex may be unsigned
    }
    for (StorageIndex k = 0; k < n; ++k) {
      const StorageIndex j = m_colOf[static_cast<std::size_t>(k)];
      x.derived().coeffRef(j, rhs) = w[static_cast<std::size_t>(k)] * m_colScale[static_cast<std::size_t>(j)];
    }
  }
}

// ---------------------------------------------------------------------------
//  determinant
// ---------------------------------------------------------------------------

namespace point_block {
// Parity of a permutation given as newIndexOf[i], by cycle decomposition.
template <typename StorageIndex>
inline int permutationParity(const std::vector<StorageIndex>& perm) {
  std::vector<char> seen(perm.size(), 0);
  int sign = 1;
  for (std::size_t start = 0; start < perm.size(); ++start) {
    if (seen[start]) continue;
    std::size_t len = 0, cur = start;
    while (!seen[cur]) {
      seen[cur] = 1;
      cur = static_cast<std::size_t>(perm[cur]);
      ++len;
    }
    if (len % 2 == 0) sign = -sign;
  }
  return sign;
}
}  // namespace point_block

template <typename MatrixType, typename OrderingType>
typename MatrixType::Scalar PointBlockLU<MatrixType, OrderingType>::logAbsDeterminant() const {
  eigen_assert(m_factorized && "PointBlockLU: logAbsDeterminant() before factorize()");
  RealScalar acc = RealScalar(0);
  for (StorageIndex k = 0; k < m_size; ++k)
    acc += numext::log(numext::abs(m_uVal[static_cast<std::size_t>(m_uPtr[static_cast<std::size_t>(k) + 1] - 1)]));
  // divide out the (positive) equilibration: log|det A| = log|det A~| - sum log Dr - sum log Dc
  for (StorageIndex i = 0; i < m_size; ++i)
    acc -= numext::log(m_rowScale[static_cast<std::size_t>(i)]) + numext::log(m_colScale[static_cast<std::size_t>(i)]);
  return Scalar(acc);
}

template <typename MatrixType, typename OrderingType>
typename MatrixType::Scalar PointBlockLU<MatrixType, OrderingType>::determinantSign() const {
  eigen_assert(m_factorized && "PointBlockLU: determinantSign() before factorize()");
  int sign = point_block::permutationParity(m_pinv) * point_block::permutationParity(m_colOf);
  for (StorageIndex k = 0; k < m_size; ++k) {
    const Scalar d = m_uVal[static_cast<std::size_t>(m_uPtr[static_cast<std::size_t>(k) + 1] - 1)];
    if (numext::real(d) < RealScalar(0)) sign = -sign;
  }
  return Scalar(sign);
}

template <typename MatrixType, typename OrderingType>
typename MatrixType::Scalar PointBlockLU<MatrixType, OrderingType>::determinant() const {
  return determinantSign() * numext::exp(logAbsDeterminant());
}

}  // namespace Eigen

#endif  // POINT_BLOCK_LU_H
