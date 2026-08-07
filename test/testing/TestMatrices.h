// Test matrix generators and sparsity-pattern helpers shared by the
// DirectLUSolvers test suites.
//
// These were duplicated (verbatim, or near enough) across
// test_supernodal_lu.cpp, test_leftright_lu.cpp and test_parallel_lu.cpp. All
// generators are deterministic given their seed, so baselines computed from
// them are reproducible.

#ifndef DIRECTLUSOLVERS_TEST_TESTING_TESTMATRICES_H
#define DIRECTLUSOLVERS_TEST_TESTING_TESTMATRICES_H

#include <Eigen/SparseCore>

#include <algorithm>
#include <random>
#include <vector>

namespace lu_testing {

// A matrix with a SYMMETRIC pattern and general (unsymmetric) values, made
// diagonally dominant so plain unpivoted LU is stable.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> randomSymmetricPatternAs(int n, double offDiagProb, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  std::vector<Eigen::Triplet<Scalar>> triplets;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (prob(rng) < offDiagProb) {
        triplets.emplace_back(i, j, Scalar(uni(rng)));  // pattern symmetric, values differ
        triplets.emplace_back(j, i, Scalar(uni(rng)));
      }
  for (int i = 0; i < n; ++i) triplets.emplace_back(i, i, Scalar(n + uni(rng)));  // strong diagonal

  Eigen::SparseMatrix<Scalar> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

inline Eigen::SparseMatrix<double> randomSymmetricPattern(int n, double offDiagProb, unsigned seed) {
  return randomSymmetricPatternAs<double>(n, offDiagProb, seed);
}

// 5-point 2D Laplacian on a gx-by-gy grid (symmetric pattern, well-separated --
// the friendly case for nested dissection and tree parallelism).
template <typename Scalar>
Eigen::SparseMatrix<Scalar> laplacian2dAs(int gx, int gy) {
  const int n = gx * gy;
  std::vector<Eigen::Triplet<Scalar>> t;
  auto id = [gx](int x, int y) { return y * gx + x; };
  for (int y = 0; y < gy; ++y)
    for (int x = 0; x < gx; ++x) {
      const int i = id(x, y);
      t.emplace_back(i, i, Scalar(4));
      if (x > 0) t.emplace_back(i, id(x - 1, y), Scalar(-1));
      if (x + 1 < gx) t.emplace_back(i, id(x + 1, y), Scalar(-1));
      if (y > 0) t.emplace_back(i, id(x, y - 1), Scalar(-1));
      if (y + 1 < gy) t.emplace_back(i, id(x, y + 1), Scalar(-1));
    }
  Eigen::SparseMatrix<Scalar> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

inline Eigen::SparseMatrix<double> laplacian2d(int gx, int gy) { return laplacian2dAs<double>(gx, gy); }

// 7-point 3D Laplacian. Fills far more than the 2D case for the same n, which
// is what makes it the useful stress case for fill prediction and for the
// separator-quality behaviour these solvers are sensitive to.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> laplacian3dAs(int gx, int gy, int gz) {
  const int n = gx * gy * gz;
  std::vector<Eigen::Triplet<Scalar>> t;
  auto id = [gx, gy](int x, int y, int z) { return (z * gy + y) * gx + x; };
  for (int z = 0; z < gz; ++z)
    for (int y = 0; y < gy; ++y)
      for (int x = 0; x < gx; ++x) {
        const int i = id(x, y, z);
        t.emplace_back(i, i, Scalar(6));
        if (x > 0) t.emplace_back(i, id(x - 1, y, z), Scalar(-1));
        if (x + 1 < gx) t.emplace_back(i, id(x + 1, y, z), Scalar(-1));
        if (y > 0) t.emplace_back(i, id(x, y - 1, z), Scalar(-1));
        if (y + 1 < gy) t.emplace_back(i, id(x, y + 1, z), Scalar(-1));
        if (z > 0) t.emplace_back(i, id(x, y, z - 1), Scalar(-1));
        if (z + 1 < gz) t.emplace_back(i, id(x, y, z + 1), Scalar(-1));
      }
  Eigen::SparseMatrix<Scalar> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

inline Eigen::SparseMatrix<double> laplacian3d(int gx, int gy, int gz) {
  return laplacian3dAs<double>(gx, gy, gz);
}

// A symmetric-pattern matrix with a numerically WEAK diagonal but strong
// off-diagonal entries: exactly the case where in-block pivoting matters. The
// diagonal is small; large entries sit off-diagonal (still pattern-symmetric).
template <typename Scalar>
Eigen::SparseMatrix<Scalar> weakDiagonalAs(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::vector<Eigen::Triplet<Scalar>> t;
  for (int i = 0; i + 1 < n; ++i) {
    // strong off-diagonal couplings
    t.emplace_back(i, i + 1, Scalar(2.0 + uni(rng)));
    t.emplace_back(i + 1, i, Scalar(2.0 + uni(rng)));
  }
  for (int i = 0; i < n; i += 3) {  // a few longer-range couplings for real fill
    const int j = (i + 5) % n;
    t.emplace_back(std::min(i, j), std::max(i, j), Scalar(1.5 + uni(rng)));
    t.emplace_back(std::max(i, j), std::min(i, j), Scalar(1.5 + uni(rng)));
  }
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, Scalar(1e-3 * uni(rng)));  // weak diagonal
  Eigen::SparseMatrix<Scalar> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

inline Eigen::SparseMatrix<double> weakDiagonal(int n, unsigned seed) {
  return weakDiagonalAs<double>(n, seed);
}

// ---------------------------------------------------------------------------
//  Pattern helpers
// ---------------------------------------------------------------------------

// True when A(i,j) != 0 <=> A(j,i) != 0 structurally. Compares the column
// index sets of A and A^T; a genuine structural zero counts as present.
template <typename Scalar>
bool patternIsSymmetric(const Eigen::SparseMatrix<Scalar>& A) {
  if (A.rows() != A.cols()) return false;
  Eigen::SparseMatrix<Scalar> AT = A.transpose();
  AT.makeCompressed();
  if (AT.nonZeros() != A.nonZeros()) return false;
  for (Eigen::Index col = 0; col <= A.outerSize(); ++col)
    if (A.outerIndexPtr()[col] != AT.outerIndexPtr()[col]) return false;
  for (Eigen::Index k = 0; k < A.nonZeros(); ++k)
    if (A.innerIndexPtr()[k] != AT.innerIndexPtr()[k]) return false;
  return true;
}

// A copy of A whose pattern is symmetric, by inserting an explicit structural
// zero at (j,i) wherever (i,j) is present but (j,i) is not. This does not change
// the linear system -- a genuine zero contributes nothing to the sum -- it only
// gives the solvers the symmetric pattern they require.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> symmetrizePattern(const Eigen::SparseMatrix<Scalar>& A) {
  std::vector<Eigen::Triplet<Scalar>> t;
  t.reserve(static_cast<std::size_t>(A.nonZeros()) * 2);
  for (Eigen::Index col = 0; col < A.outerSize(); ++col)
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, col); it; ++it) {
      t.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
      t.emplace_back(static_cast<int>(it.col()), static_cast<int>(it.row()), Scalar(0));
    }
  Eigen::SparseMatrix<Scalar> S(A.rows(), A.cols());
  S.setFromTriplets(t.begin(), t.end());  // duplicates sum -> the real value survives
  S.makeCompressed();
  return S;
}

// symmetrizePattern(A) only when needed, so an already-symmetric pattern is not
// needlessly copied.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> ensureSymmetricPattern(const Eigen::SparseMatrix<Scalar>& A) {
  return patternIsSymmetric(A) ? A : symmetrizePattern(A);
}

}  // namespace lu_testing

#endif  // DIRECTLUSOLVERS_TEST_TESTING_TESTMATRICES_H
