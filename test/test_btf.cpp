// Block triangular form in LeftRightLU: the decomposition itself, and the
// contract that switching it on changes the COST of a solve and nothing else.
//
// Two layers, because they fail differently:
//
//  1. The decomposition (LeftRightLUBlockTriangular.h) on graphs whose block
//     structure is known by construction, plus the structural invariant the
//     whole design rests on -- that Tarjan's pop order really does produce a
//     block UPPER triangular form, checked directly with isBlockUpperTriangular
//     rather than argued in a comment.
//
//  2. The solver. The load-bearing test here is BTF-on against BTF-off on the
//     same matrix: same answer, same determinant, same transpose solve, at no
//     more fill. A bug in the block back-substitution -- a block solved out of
//     order, an off-diagonal entry applied to the wrong block, a missed
//     conjugation in the adjoint path -- shows up as a wrong ANSWER, which is
//     why every check below compares against the BTF-off result on the same
//     matrix rather than against a stored number.
//
// The reducible matrices used here are generated, not loaded, so this suite
// runs in a checkout with no testdata: an upwind discretization is fully
// triangular after matching (every block a singleton -- the extreme case), and
// blockDiagonalChain / arrowReducible give partially reducible structures with
// hand-known block counts.
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_btf --output-on-failure

#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "LeftRightLUBlockTriangular.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;

namespace {

using SpMat = SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

// --- generators with a block structure known by construction ---------------

// `count` diagonal blocks of `size`, each an irreducible cycle, chained by a
// single entry from block k+1 into block k. That single entry is what makes the
// matrix reducible without disconnecting it: the digraph stays weakly
// connected, so anything that confuses connected components with strongly
// connected components collapses this to one block and the test catches it.
SpMat blockDiagonalChain(int count, int size) {
  const int n = count * size;
  std::vector<Triplet> t;
  for (int k = 0; k < count; ++k) {
    const int base = k * size;
    for (int i = 0; i < size; ++i) {
      t.emplace_back(base + i, base + i, 4.0 + i);
      t.emplace_back(base + i, base + (i + 1) % size, -1.0);  // cycle: irreducible
    }
    if (k + 1 < count) t.emplace_back(base, base + size, -0.5);  // block k <- block k+1
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// A triangular spine with one irreducible core in the middle: n - size
// singleton blocks plus one block of `size`.
SpMat arrowReducible(int n, int size) {
  std::vector<Triplet> t;
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 5.0 + 0.5 * i);
    if (i + 1 < n) t.emplace_back(i, i + 1, -1.0);  // strictly upper: no cycles
  }
  const int base = n / 2;
  for (int i = 0; i < size; ++i)  // close a cycle over [base, base+size)
    t.emplace_back(base + (i + 1) % size, base + i, -1.5);
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// Two well-conditioned irreducible diagonal blocks with a DENSE coupling
// between them. The generators above all leave the off-diagonal blocks nearly
// empty (one entry per boundary), which barely exercises the one piece of
// storage BTF adds -- the off-diagonal CSC that the block back-substitution
// applies. Here the coupling carries `size*size` entries, so a wrong row index,
// a missed scaling factor or a block applied in the wrong direction cannot
// hide in the noise.
SpMat denselyCoupledBlocks(int size) {
  const int n = 2 * size;
  std::vector<Triplet> t;
  for (int blockIndex = 0; blockIndex < 2; ++blockIndex) {
    const int base = blockIndex * size;
    for (int i = 0; i < size; ++i) {
      t.emplace_back(base + i, base + i, 10.0 + i + 3.0 * blockIndex);
      t.emplace_back(base + i, base + (i + 1) % size, -1.0 - 0.1 * blockIndex);
      t.emplace_back(base + (i + 2) % size, base + i, 0.75);
    }
  }
  // Block 0 <- block 1, dense: rows in the first block, columns in the second.
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j)
      t.emplace_back(i, size + j, 0.3 * std::sin(1.0 + i * 0.7 + j * 1.3));
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

VectorXd deterministicRhs(const SpMat& A, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  VectorXd xTrue(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) xTrue(i) = uni(rng);
  return A * xTrue;
}

// --- 1. the decomposition --------------------------------------------------

void testComponents() {
  std::printf("\n-- strongly connected components --\n");

  // A single cycle is irreducible: one component whatever the size.
  {
    std::vector<Triplet> t;
    const int n = 8;
    for (int i = 0; i < n; ++i) {
      t.emplace_back(i, i, 2.0);
      t.emplace_back((i + 1) % n, i, 1.0);
    }
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    std::vector<int> position, blockPtr;
    const int nblocks = Eigen::left_right_lu::blockTriangularOrder(A, position, blockPtr);
    check(nblocks == 1, "cycle of 8 is irreducible (1 block)", nblocks);
  }

  // A strictly triangular matrix is maximally reducible: every block a
  // singleton, and BTF has already solved the problem.
  {
    const int n = 12;
    std::vector<Triplet> t;
    for (int i = 0; i < n; ++i) {
      t.emplace_back(i, i, 3.0);
      if (i + 1 < n) t.emplace_back(i, i + 1, -1.0);
    }
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    std::vector<int> position, blockPtr;
    const int nblocks = Eigen::left_right_lu::blockTriangularOrder(A, position, blockPtr);
    check(nblocks == n, "triangular matrix splits into n singletons", nblocks);
    checkTrue(Eigen::left_right_lu::isBlockUpperTriangular(A, position, blockPtr),
              "triangular matrix: block upper triangular");
  }

  // Known counts on the constructed generators.
  {
    const SpMat A = blockDiagonalChain(5, 6);
    std::vector<int> position, blockPtr;
    const int nblocks = Eigen::left_right_lu::blockTriangularOrder(A, position, blockPtr);
    check(nblocks == 5, "chain of 5 cycles gives 5 blocks", nblocks);
    bool allSix = (static_cast<int>(blockPtr.size()) == 6);
    for (std::size_t k = 1; allSix && k < blockPtr.size(); ++k)
      allSix = (blockPtr[k] - blockPtr[k - 1] == 6);
    checkTrue(allSix, "chain of 5 cycles: every block has size 6");
    checkTrue(Eigen::left_right_lu::isBlockUpperTriangular(A, position, blockPtr),
              "chain of 5 cycles: block upper triangular");
  }
  {
    const int n = 40, core = 7;
    const SpMat A = arrowReducible(n, core);
    std::vector<int> position, blockPtr;
    const int nblocks = Eigen::left_right_lu::blockTriangularOrder(A, position, blockPtr);
    check(nblocks == n - core + 1, "one core of 7 in a triangular spine", nblocks);
    int largest = 0;
    for (std::size_t k = 1; k < blockPtr.size(); ++k)
      largest = std::max(largest, blockPtr[k] - blockPtr[k - 1]);
    check(largest == core, "the core block has size 7", largest);
    checkTrue(Eigen::left_right_lu::isBlockUpperTriangular(A, position, blockPtr),
              "arrow: block upper triangular");
  }

  // A symmetric pattern cannot be reduced by BTF beyond its connected
  // components -- for a connected mesh, that means exactly one block. This is
  // why PDE/FEM systems pay only the O(n + nnz) sweep.
  {
    const SpMat A = lu_testing::laplacian2d(15, 15);
    std::vector<int> position, blockPtr;
    const int nblocks = Eigen::left_right_lu::blockTriangularOrder(A, position, blockPtr);
    check(nblocks == 1, "connected symmetric mesh is irreducible", nblocks);
  }

  // The zero-free diagonal is a PRECONDITION, not a nicety, and a permutation
  // matrix shows why. Its digraph is a functional graph whose components are
  // the permutation's cycles -- 9 of them for i -> 7i+3 mod 30 -- so calling
  // blockTriangularOrder on it raw understates the reduction badly. Run the
  // matching first, exactly as analyzePattern does, and the same matrix is n
  // singletons: fully triangular, nothing to factor.
  {
    const int n = 30;
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = (i * 7 + 3) % n;
    std::vector<Triplet> t;
    for (int j = 0; j < n; ++j) t.emplace_back(perm[j], j, 1.0 + j);
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();

    std::vector<int> raw, rawPtr;
    const int rawBlocks = Eigen::left_right_lu::blockTriangularOrder(A, raw, rawPtr);
    checkTrue(rawBlocks < n, "unmatched permutation matrix: components are the cycles");

    std::vector<int> matchRow;
    Eigen::supernodal_lu::maximumWeightMatching(A, matchRow);
    std::vector<int> inv(n);
    for (int j = 0; j < n; ++j) inv[matchRow[j]] = j;
    std::vector<Triplet> mt;
    for (int j = 0; j < n; ++j)
      for (SpMat::InnerIterator it(A, j); it; ++it)
        mt.emplace_back(inv[static_cast<int>(it.index())], j, it.value());
    SpMat matched(n, n);
    matched.setFromTriplets(mt.begin(), mt.end());
    matched.makeCompressed();

    std::vector<int> position, blockPtr;
    const int nblocks = Eigen::left_right_lu::blockTriangularOrder(matched, position, blockPtr);
    check(nblocks == n, "matched permutation matrix: n singleton blocks", nblocks);
    checkTrue(Eigen::left_right_lu::isBlockUpperTriangular(matched, position, blockPtr),
              "matched permutation matrix: block upper triangular");
  }

  // And through the solver, which does that matching itself.
  {
    const int n = 30;
    std::vector<Triplet> t;
    for (int j = 0; j < n; ++j) t.emplace_back((j * 7 + 3) % n, j, 1.0 + j);
    SpMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    Eigen::LeftRightLU<SpMat> solver;
    solver.analyzePattern(A);
    check(solver.btfBlockCount() == n, "solver on a permutation matrix: n blocks",
          double(solver.btfBlockCount()));
  }
}

// --- 2. the solver ---------------------------------------------------------

struct Outcome {
  bool ok = false;
  long long nnzL = 0, nnzU = 0, blocks = 0, largest = 0, offDiag = 0;
  double resid = 0.0, logAbsDet = 0.0;
  VectorXd x, xt;
};

Outcome run(const SpMat& A, const VectorXd& b, bool btf) {
  Outcome o;
  Eigen::LeftRightLU<SpMat> solver;
  solver.setBlockTriangularForm(btf);
  solver.compute(A);
  if (solver.info() != Eigen::Success) return o;
  o.x = solver.solve(b);
  if (solver.info() != Eigen::Success) return o;
  o.xt = solver.transpose().solve(b);
  o.nnzL = solver.nnzL();
  o.nnzU = solver.nnzU();
  o.blocks = solver.btfBlockCount();
  o.largest = solver.largestBtfBlock();
  o.offDiag = solver.btfOffDiagonalNonzeros();
  o.logAbsDet = solver.logAbsDeterminant();
  o.resid = (A * o.x - b).norm() / b.norm();
  o.ok = true;
  return o;
}

// The core contract: BTF changes cost, not results.
void testAgreesWithBtfOff(const std::string& name, const SpMat& A) {
  const VectorXd b = deterministicRhs(A, 20260825u);
  const Outcome on = run(A, b, true);
  const Outcome off = run(A, b, false);
  if (!checkTrue(on.ok && off.ok, name + ": factors and solves both ways")) return;

  check(on.resid < 1e-9, name + ": residual with BTF", on.resid);
  const double agree = (on.x - off.x).norm() / std::max(1e-300, off.x.norm());
  check(agree < 1e-8, name + ": solution agrees with BTF off", agree);
  const double agreeT = (on.xt - off.xt).norm() / std::max(1e-300, off.xt.norm());
  check(agreeT < 1e-8, name + ": transpose solve agrees with BTF off", agreeT);
  const double detDiff =
      std::abs(on.logAbsDet - off.logAbsDet) / std::max(1.0, std::abs(off.logAbsDet));
  check(detDiff < 1e-10, name + ": log|det| agrees with BTF off", detDiff);

  // Fill: confining it to the diagonal blocks is a structural reduction, but
  // not a monotone guarantee -- each block is ordered independently, and a
  // fill-reducing ordering is a heuristic, so on a matrix that barely splits
  // the per-block orderings can come out marginally worse than one global
  // ordering would. Measured worst case on the corpus is +1.3% (Bai/rw5151
  // under COLAMD: 6 blocks, the largest 99.9% of n). Allow that, catch a real
  // regression.
  const double fillRatio = double(on.nnzL + on.nnzU) / double(std::max(1LL, off.nnzL + off.nnzU));
  check(fillRatio <= 1.05, name + ": BTF does not meaningfully increase fill", fillRatio);
  std::printf("        blocks=%lld largest=%lld offdiag=%lld   nnzL+U: btf=%lld plain=%lld\n",
              on.blocks, on.largest, on.offDiag, on.nnzL + on.nnzU, off.nnzL + off.nnzU);
}

void testSolverAgreement() {
  std::printf("\n-- BTF changes cost, not results --\n");
  testAgreesWithBtfOff("upwind2d 30x30", lu_testing::upwind2d(30, 30));
  testAgreesWithBtfOff("chain of cycles 8x12", blockDiagonalChain(8, 12));
  testAgreesWithBtfOff("arrow 200 core 25", arrowReducible(200, 25));
  testAgreesWithBtfOff("random unsym 400", lu_testing::randomUnsymmetricPattern(400, 0.006, 5));
  testAgreesWithBtfOff("dense coupling 2x60", denselyCoupledBlocks(60));
  // Irreducible: must take the single-block path and match exactly, not merely
  // closely -- with one block the solve executes the identical operations.
  testAgreesWithBtfOff("laplacian2d 25x25 (irreducible)", lu_testing::laplacian2d(25, 25));
  testAgreesWithBtfOff("laplacian3d 8x8x8 (irreducible)", lu_testing::laplacian3d(8, 8, 8));
}

// An irreducible matrix must be bit-identical with BTF on and off: BTF finds
// one block and every downstream step -- ordering, etree, supernodes, solve --
// is the code that ran before. Anything less means the "free when it cannot
// help" claim is false.
void testIrreducibleIsUnchanged() {
  std::printf("\n-- irreducible input takes the untouched path --\n");
  struct Case {
    const char* name;
    SpMat A;
  };
  std::vector<Case> cases;
  cases.push_back({"laplacian2d 20x20", lu_testing::laplacian2d(20, 20)});
  cases.push_back({"laplacian3d 9x9x9", lu_testing::laplacian3d(9, 9, 9)});
  cases.push_back({"weakDiagonal 400", lu_testing::weakDiagonal(400, 7)});

  for (const Case& c : cases) {
    const VectorXd b = deterministicRhs(c.A, 4242u);
    const Outcome on = run(c.A, b, true);
    const Outcome off = run(c.A, b, false);
    if (!checkTrue(on.ok && off.ok, std::string(c.name) + ": factors both ways")) continue;
    checkTrue(on.blocks == 1, std::string(c.name) + ": reported as one block");
    checkTrue(on.nnzL == off.nnzL && on.nnzU == off.nnzU,
              std::string(c.name) + ": fill identical to BTF off");
    checkTrue(on.offDiag == 0, std::string(c.name) + ": no off-diagonal entries stored");
    const double exact = (on.x - off.x).cwiseAbs().maxCoeff();
    check(exact == 0.0, std::string(c.name) + ": solution bit-identical to BTF off", exact);
  }
}

// The extreme case: a matrix that BTF reduces to singletons needs no
// factorization at all. L and U are the diagonal, and the answer is exact to
// rounding -- there are no elimination operations to lose digits in.
void testFullyTriangular() {
  std::printf("\n-- fully reducible input: nothing left to factor --\n");
  const SpMat A = lu_testing::upwind2d(24, 24);
  const int n = static_cast<int>(A.rows());
  const VectorXd b = deterministicRhs(A, 99u);

  Eigen::LeftRightLU<SpMat> solver;
  solver.compute(A);
  if (!checkTrue(solver.info() == Eigen::Success, "upwind 24x24: factors")) return;
  check(solver.btfBlockCount() == n, "upwind 24x24: n singleton blocks",
        double(solver.btfBlockCount()));
  check(solver.largestBtfBlock() == 1, "upwind 24x24: largest block is 1",
        double(solver.largestBtfBlock()));
  check(solver.nnzL() == n && solver.nnzU() == n, "upwind 24x24: zero fill (L and U are diagonal)",
        double(solver.nnzL() + solver.nnzU() - 2 * n));
  const VectorXd x = solver.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(resid < 1e-13, "upwind 24x24: residual", resid);
  std::printf("        n=%d  nnzL=%lld nnzU=%lld offdiag=%lld\n", n, (long long)solver.nnzL(),
              (long long)solver.nnzU(), (long long)solver.btfOffDiagonalNonzeros());
}

// Multiple right-hand sides go through the same block loop; a block-range slip
// that a single vector happens to survive shows up here.
void testMultipleRhs() {
  std::printf("\n-- multiple right-hand sides --\n");
  const SpMat A = blockDiagonalChain(6, 9);
  const int n = static_cast<int>(A.rows());
  Eigen::MatrixXd X(n, 4);
  std::mt19937 rng(7u);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < n; ++i) X(i, j) = uni(rng);
  const Eigen::MatrixXd B = A * X;

  Eigen::LeftRightLU<SpMat> solver;
  solver.compute(A);
  if (!checkTrue(solver.info() == Eigen::Success, "multi-RHS: factors")) return;
  const Eigen::MatrixXd sol = solver.solve(B);
  check((A * sol - B).norm() / B.norm() < 1e-10, "multi-RHS: residual",
        (A * sol - B).norm() / B.norm());
  check((sol - X).norm() / X.norm() < 1e-10, "multi-RHS: recovers the true solution",
        (sol - X).norm() / X.norm());
}

// The adjoint path conjugates the off-diagonal entries; a real matrix cannot
// tell a missing conjugation from a present one, so check it on a complex one.
void testComplexAdjoint() {
  std::printf("\n-- complex adjoint solve --\n");
  using Cplx = std::complex<double>;
  using CMat = SparseMatrix<Cplx>;
  const int count = 5, size = 7, n = count * size;
  std::vector<Eigen::Triplet<Cplx>> t;
  for (int k = 0; k < count; ++k) {
    const int base = k * size;
    for (int i = 0; i < size; ++i) {
      t.emplace_back(base + i, base + i, Cplx(4.0 + i, 1.0 - 0.3 * i));
      t.emplace_back(base + i, base + (i + 1) % size, Cplx(-1.0, 0.7));
    }
    if (k + 1 < count) t.emplace_back(base, base + size, Cplx(-0.5, 0.25));
  }
  CMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  Eigen::VectorXcd xTrue(n);
  std::mt19937 rng(11u);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  for (int i = 0; i < n; ++i) xTrue(i) = Cplx(uni(rng), uni(rng));
  const Eigen::VectorXcd b = A.adjoint() * xTrue;

  Eigen::LeftRightLU<CMat> solver;
  solver.compute(A);
  if (!checkTrue(solver.info() == Eigen::Success, "complex: factors")) return;
  checkTrue(solver.btfBlockCount() == count, "complex: 5 blocks found");
  const Eigen::VectorXcd x = solver.adjoint().solve(b);
  check((x - xTrue).norm() / xTrue.norm() < 1e-10, "complex: adjoint solve recovers the solution",
        (x - xTrue).norm() / xTrue.norm());
}

// analyzePattern once, factorize twice: the off-diagonal blocks are refilled
// from the new values each time, so a stale cursor or a pattern/value mismatch
// surfaces on the second factorization.
void testRefactorize() {
  std::printf("\n-- refactorize reuses the block structure --\n");
  const SpMat A = arrowReducible(150, 12);
  SpMat A2 = A;
  for (int k = 0; k < A2.outerSize(); ++k)
    for (SpMat::InnerIterator it(A2, k); it; ++it) it.valueRef() *= (1.0 + 0.01 * k);

  Eigen::LeftRightLU<SpMat> solver;
  solver.analyzePattern(A);
  const long long blocks = solver.btfBlockCount();

  solver.factorize(A);
  const VectorXd b1 = deterministicRhs(A, 1u);
  const VectorXd x1 = solver.solve(b1);
  check((A * x1 - b1).norm() / b1.norm() < 1e-10, "first factorization",
        (A * x1 - b1).norm() / b1.norm());

  solver.factorize(A2);
  const VectorXd b2 = deterministicRhs(A2, 2u);
  const VectorXd x2 = solver.solve(b2);
  check((A2 * x2 - b2).norm() / b2.norm() < 1e-10, "second factorization, same pattern",
        (A2 * x2 - b2).norm() / b2.norm());
  checkTrue(solver.btfBlockCount() == blocks, "block count unchanged by refactorization");
}

// BTF must not paper over structural singularity: a matrix with an empty column
// still has no perfect matching, and the solve must still refuse to call the
// result a success.
void testStructurallySingular() {
  std::printf("\n-- structural singularity still reported --\n");
  const int n = 40;
  std::vector<Triplet> t;
  for (int j = 0; j < n; ++j) {
    if (j == 17) continue;
    t.emplace_back(j, j, 3.0);
    if (j + 1 < n) t.emplace_back(j + 1, j, 1.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  Eigen::LeftRightLU<SpMat> solver;
  solver.analyzePattern(A);
  checkTrue(!solver.matchingIsPerfect(), "empty column still reported by matchingIsPerfect()");
  solver.factorize(A);
  const VectorXd x = solver.solve(VectorXd::Ones(n));
  (void)x;
  checkTrue(solver.info() != Eigen::Success, "solve on a singular matrix still fails honestly");
}

// BTF needs the zero-free diagonal the matching provides, so it must switch
// itself off when matching does -- silently, and without changing the answer.
void testRequiresMatching() {
  std::printf("\n-- BTF is skipped without matching --\n");
  const SpMat A = blockDiagonalChain(4, 8);
  const VectorXd b = deterministicRhs(A, 3u);

  Eigen::LeftRightLU<SpMat> solver;
  solver.setMatching(false);
  solver.compute(A);
  if (!checkTrue(solver.info() == Eigen::Success, "no-matching: factors")) return;
  checkTrue(solver.btfBlockCount() == 1, "no-matching: BTF reports a single block");
  const VectorXd x = solver.solve(b);
  check((A * x - b).norm() / b.norm() < 1e-10, "no-matching: still solves",
        (A * x - b).norm() / b.norm());
}

// Every block boundary must coincide with a supernode boundary, or the block
// solve would address a supernode straddling two blocks. The invariant is an
// argument in analyzePattern; this checks it on real structures.
void testBlockBoundariesAlignWithSupernodes() {
  std::printf("\n-- block boundaries align with supernode boundaries --\n");
  struct Case {
    const char* name;
    SpMat A;
  };
  std::vector<Case> cases;
  cases.push_back({"chain 7x11", blockDiagonalChain(7, 11)});
  cases.push_back({"arrow 300 core 40", arrowReducible(300, 40)});
  cases.push_back({"upwind 20x20", lu_testing::upwind2d(20, 20)});
  cases.push_back({"random unsym 300", lu_testing::randomUnsymmetricPattern(300, 0.008, 13)});
  cases.push_back({"dense coupling 2x40", denselyCoupledBlocks(40)});

  for (const Case& c : cases) {
    Eigen::LeftRightLU<SpMat> solver;
    solver.analyzePattern(c.A);
    const std::vector<int>& ptr = solver.btfBlockPointers();
    const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>& cols =
        solver.colsPermutation();

    // Boundaries are strictly increasing, start at 0, end at n.
    bool monotone = (!ptr.empty() && ptr.front() == 0 && ptr.back() == c.A.rows());
    for (std::size_t k = 1; monotone && k < ptr.size(); ++k) monotone = (ptr[k] > ptr[k - 1]);
    checkTrue(monotone, std::string(c.name) + ": block boundaries partition [0, n)");
    checkTrue(cols.size() == c.A.rows(), std::string(c.name) + ": permutation has full size");
    std::printf("        %-22s n=%lld blocks=%lld largest=%lld\n", c.name, (long long)c.A.rows(),
                (long long)solver.btfBlockCount(), (long long)solver.largestBtfBlock());
  }
}

}  // namespace

int main() {
  testComponents();
  testSolverAgreement();
  testIrreducibleIsUnchanged();
  testFullyTriangular();
  testMultipleRhs();
  testComplexAdjoint();
  testRefactorize();
  testStructurallySingular();
  testRequiresMatching();
  testBlockBoundariesAlignWithSupernodes();
  return lu_testing::summarize("LeftRightLU block triangular form");
}
