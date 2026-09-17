// Correctness tests for Eigen::LeftRightLU (PARDISO-style left-right-looking LU
// with a barrier-free dynamic scheduler and in-block complete pivoting).
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_leftright_lu --output-on-failure
//
// Covers: direct solves, multiple RHS, factor accessors, transpose/adjoint,
// equilibration, complete vs partial vs no in-block pivoting, log-determinant,
// honest failure reporting, parallel(dynamic-scheduler)-vs-serial agreement, and
// the input-validation contract: re-analysis invalidates old factors, non-finite
// input and pattern/size mismatches are declined, a solve without factors is
// refused, and determinant() survives intermediate overflow.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using lu_testing::check;
using lu_testing::checkTrue;
using lu_testing::laplacian2d;
using lu_testing::randomSymmetricPattern;
using lu_testing::randomUnsymmetricPattern;
using lu_testing::upwind2d;
using lu_testing::weakDiagonal;
namespace lr = Eigen::left_right_lu;

namespace {

template <typename Solver>
double solveResidual(Solver& solver, const SparseMatrix<double>& A, const VectorXd& b) {
  const VectorXd x = solver.solve(b);
  return (A * x - b).norm() / b.norm();
}

double solveAndMeasure(const SparseMatrix<double>& A, const char* name) {
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    std::printf("  [FAIL] %-46s factorization failed: %s\n", name, solver.lastErrorMessage().c_str());
    ++lu_testing::failureCount();
    return 1.0;
  }
  VectorXd x = solver.solve(b);
  const double err = (x - xTrue).norm() / xTrue.norm();
  const double resid = (A * x - b).norm() / b.norm();
  const double worst = std::max(err, resid);
  check(worst < 1e-8, name, worst);

  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  if (ref.info() == Eigen::Success) {
    const double d1 = std::abs(solver.determinant());
    const double d2 = std::abs(ref.determinant());
    const double drel = std::abs(d1 - d2) / std::max(1.0, std::abs(d2));
    std::printf("        det(ours)=%.6e det(SparseLU)=%.6e relDiff=%.2e  nnzL=%lld snodes=%lld\n",
                d1, d2, drel, (long long)solver.nnzL(), (long long)solver.supernodeCount());
  }
  return worst;
}

void testMultipleRhs() {
  SparseMatrix<double> A = laplacian2d(8, 8);
  const int n = static_cast<int>(A.rows());
  MatrixXd X = MatrixXd::Random(n, 4);
  MatrixXd B = A * X;
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  MatrixXd Xs = solver.solve(B);
  check((A * Xs - B).norm() / B.norm() < 1e-8, "multiple RHS (4 cols)", (A * Xs - B).norm() / B.norm());
}

void testFactorAccessors() {
  SparseMatrix<double> A = laplacian2d(12, 10);
  const int n = static_cast<int>(A.rows());
  VectorXd b = A * VectorXd::Random(n);
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.setMaxIterativeRefinements(0);
  solver.setEquilibration(false);
  solver.compute(A);

  VectorXd y = solver.rowsPermutation() * b;
  solver.matrixL().solveInPlace(y);
  solver.matrixU().solveInPlace(y);
  VectorXd xManual = solver.colsPermutation().transpose() * y;
  const double resid = (A * xManual - b).norm() / b.norm();
  check(resid < 1e-10, "matrixL()/matrixU() reproduce the solve", resid);
}

void testTransposeSolve() {
  SparseMatrix<double> A = randomSymmetricPattern(150, 0.05, 99);
  const int n = static_cast<int>(A.rows());
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);

  VectorXd bT = A.transpose() * VectorXd::Random(n);
  VectorXd xT = solver.transpose().solve(bT);
  check((A.transpose() * xT - bT).norm() / bT.norm() < 1e-8, "transpose().solve(): A^T x = b",
        (A.transpose() * xT - bT).norm() / bT.norm());

  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  VectorXd xRef = ref.transpose().solve(bT);
  check((xT - xRef).norm() / xRef.norm() < 1e-8, "transpose().solve() matches Eigen::SparseLU",
        (xT - xRef).norm() / xRef.norm());

  VectorXd bA = A.adjoint() * VectorXd::Random(n);
  VectorXd xA = solver.adjoint().solve(bA);
  check((A.adjoint() * xA - bA).norm() / bA.norm() < 1e-8, "adjoint().solve(): A^H x = b",
        (A.adjoint() * xA - bA).norm() / bA.norm());
}

// In-block pivoting: a weak-diagonal matrix. Complete pivoting (default) should
// solve accurately; compare the three modes. All go through matching + static
// pivoting + refinement, so all should be usable, but this exercises the row+col
// interchange path and its solve folding directly.
void testCompletePivoting() {
  SparseMatrix<double> A = weakDiagonal(300, 7);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  auto run = [&](lr::Pivoting mode, const char* name) {
    Eigen::LeftRightLU<SparseMatrix<double>> s;
    s.setPivoting(mode);
    s.compute(A);
    const double resid = solveResidual(s, A, b);
    std::printf("        %-10s resid=%.2e bumped=%lld\n", name, resid, (long long)s.replacedPivots());
    return resid;
  };
  const double complete = run(lr::Pivoting::Complete, "complete");
  const double partial = run(lr::Pivoting::Partial, "partial");
  run(lr::Pivoting::None, "none");

  check(complete < 1e-8, "complete pivoting solves weak-diagonal system", complete);
  check(partial < 1e-8, "partial pivoting solves weak-diagonal system", partial);

  // determinant sign/magnitude with column swaps must still match Eigen::SparseLU.
  Eigen::LeftRightLU<SparseMatrix<double>> s;
  s.compute(A);
  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  const double drel = std::abs(std::abs(s.determinant()) - std::abs(ref.determinant())) /
                      std::max(1.0, std::abs(ref.determinant()));
  check(drel < 1e-6, "determinant matches SparseLU under complete pivoting", drel);
}

// Force the column-swap path: with matching OFF, the raw weak diagonal makes
// in-block complete pivoting actually interchange rows AND columns. This is the
// hardest code to get right (the per-supernode column permutation Q_s folded
// through the forward/backward and transpose solves and the determinant sign).
void testCompletePivotingColumnSwaps() {
  SparseMatrix<double> A = weakDiagonal(200, 3);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.setMatching(false);                    // no diagonal help -> pivoting must work
  solver.setPivoting(lr::Pivoting::Complete);
  solver.compute(A);
  check(solver.info() == Eigen::Success, "complete pivoting factorizes (matching off)",
        solver.info() == Eigen::Success ? 0.0 : 1.0);

  const VectorXd x = solver.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(resid < 1e-8, "column-swap path: forward solve correct", resid);

  VectorXd bT = A.transpose() * xTrue;
  const VectorXd xT = solver.transpose().solve(bT);
  const double residT = (A.transpose() * xT - bT).norm() / bT.norm();
  check(residT < 1e-8, "column-swap path: transpose solve correct", residT);

  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  const double drel = std::abs(std::abs(solver.determinant()) - std::abs(ref.determinant())) /
                      std::max(1.0, std::abs(ref.determinant()));
  const bool signOk = (solver.determinant() > 0) == (ref.determinant() > 0);
  check(drel < 1e-6 && signOk, "column-swap path: determinant magnitude & sign", drel);
  std::printf("        det(ours)=%+.6e det(SparseLU)=%+.6e\n", solver.determinant(), ref.determinant());
}

void testLogDeterminant() {
  SparseMatrix<double> A = laplacian2d(15, 15);  // SPD -> positive determinant
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  const double logAbs = solver.logAbsDeterminant();
  const double fromDet = std::log(std::abs(solver.determinant()));
  const double rel = std::abs(logAbs - fromDet) / std::max(1.0, std::abs(fromDet));
  check(rel < 1e-10, "logAbsDeterminant() consistent with log|determinant()|", rel);
  check(solver.determinantSign() > 0.0, "determinantSign() positive for SPD matrix",
        solver.determinantSign() > 0 ? 0.0 : 1.0);
  std::printf("        log|det| = %.6e  sign = %+.0f\n", logAbs, solver.determinantSign());
}

void testEquilibration() {
  SparseMatrix<double> A = randomSymmetricPattern(150, 0.05, 31);
  const int n = static_cast<int>(A.rows());
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> expo(-6.0, 6.0);
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      it.valueRef() *= std::pow(10.0, expo(rng));  // wreck the scaling
  VectorXd b = A * VectorXd::Random(n);
  Eigen::LeftRightLU<SparseMatrix<double>> on;
  on.compute(A);
  const double residOn = solveResidual(on, A, b);
  check(residOn < 1e-8, "equilibration solves badly-scaled system", residOn);
}

void testHonestFailure() {
  SparseMatrix<double> A = randomSymmetricPattern(80, 0.06, 23);
  const int dead = 40;
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      if (it.row() == dead || it.col() == dead) it.valueRef() = 0.0;
  A.prune(0.0);

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  const bool factored = solver.isFactorized();
  VectorXd b = VectorXd::Random(80);
  const VectorXd x = solver.solve(b);
  const double resid = (A * x - b).norm() / b.norm();
  check(factored && solver.info() == Eigen::NumericalIssue,
        "honest check: singular solve reports NumericalIssue", resid);

  VectorXd xr = VectorXd::Random(80);
  xr(dead) = 0.0;
  VectorXd bIn = A * xr;
  const VectorXd x2 = solver.solve(bIn);
  const double resid2 = (A * x2 - bIn).norm() / std::max(1e-300, bIn.norm());
  check(solver.info() == Eigen::Success && resid2 < 1e-8,
        "honest check: status recovers on a consistent RHS", resid2);
}

// Fail-fast fill guard: predictedFactorNonzeros() is available after analyze and
// exactly matches the realized fill; setMaxFactorNonzeros() below the real fill
// makes factorize() abort cleanly (NumericalIssue, nothing allocated) instead of
// attempting the allocation; above it, factorization proceeds normally.
void testFillGuard() {
  SparseMatrix<double> A = laplacian2d(20, 20);
  const int n = static_cast<int>(A.rows());

  Eigen::LeftRightLU<SparseMatrix<double>> probe;
  probe.analyzePattern(A);
  const long long predicted = probe.predictedFactorNonzeros();
  check(predicted > 0, "predictedFactorNonzeros() > 0 after analyze", predicted > 0 ? 0.0 : 1.0);

  Eigen::LeftRightLU<SparseMatrix<double>> guarded;
  guarded.setMaxFactorNonzeros(1000);  // far below the true fill -> must trip
  guarded.compute(A);
  check(guarded.info() == Eigen::NumericalIssue && !guarded.isFactorized(),
        "fill guard aborts factorize below limit", guarded.isFactorized() ? 1.0 : 0.0);

  Eigen::LeftRightLU<SparseMatrix<double>> ok;
  ok.setMaxFactorNonzeros(predicted + 1);  // generous -> normal factorization
  ok.compute(A);
  const bool factored = ok.info() == Eigen::Success && ok.isFactorized();
  const long long realized = static_cast<long long>(ok.nnzL()) + ok.nnzU() - n;
  check(factored && realized == predicted, "guard passes; prediction == realized fill (nnzL+U-n)",
        factored && realized == predicted ? 0.0 : 1.0);
  VectorXd b = A * VectorXd::Random(n);
  const double resid = (A * ok.solve(b) - b).norm() / b.norm();
  check(resid < 1e-8, "guarded (passing) solve is correct", resid);
  std::printf("        predicted=%lld  realized(nnzL+U-n)=%lld\n", predicted, realized);
}

// Parallel dynamic scheduler vs serial: same matrix, StdThreadExecutor vs the
// serial default must agree to solver accuracy (bit-identity is NOT a goal). Also
// checks the scheduler completes without deadlock and reports no error.
void testParallelVsSerial() {
  SparseMatrix<double> A = laplacian2d(60, 60);  // 3600 unknowns, wide tree
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> serial;
  serial.compute(A);
  const VectorXd xs = serial.solve(b);
  const double residS = (A * xs - b).norm() / b.norm();

  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                     Eigen::supernodal_lu::StdThreadExecutor>
      parallel;
  parallel.compute(A);
  if (parallel.info() != Eigen::Success) {
    std::printf("  [FAIL] %-46s parallel factorization failed: %s\n", "parallel scheduler",
                parallel.lastErrorMessage().c_str());
    ++lu_testing::failureCount();
    return;
  }
  const VectorXd xp = parallel.solve(b);
  const double residP = (A * xp - b).norm() / b.norm();
  const double agree = (xp - xs).norm() / xs.norm();

  check(residP < 1e-8, "parallel dynamic scheduler solves accurately", residP);
  check(agree < 1e-8, "parallel solution agrees with serial", agree);
  std::printf("        threads=%d serial resid=%.2e parallel resid=%.2e agree=%.2e\n",
              parallel.executor().concurrency(), residS, residP, agree);

  // stress: run the parallel factorization several times to shake out any
  // scheduler race/deadlock (different matrices, repeated compute()).
  for (int rep = 0; rep < 5; ++rep) {
    SparseMatrix<double> M = randomSymmetricPattern(400, 0.02, 100 + rep);
    Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                       Eigen::supernodal_lu::StdThreadExecutor>
        p;
    p.compute(M);
    VectorXd bb = M * VectorXd::Random(400);
    const double r = solveResidual(p, M, bb);
    if (!(p.info() == Eigen::Success && r < 1e-8)) {
      check(false, "parallel scheduler stress repetition", r);
      return;
    }
  }
  check(true, "parallel scheduler stress (5 repetitions, no deadlock)", 0.0);
}

// ---------------------------------------------------------------------------
//  Unsymmetric nonzero patterns
// ---------------------------------------------------------------------------

// The solver takes a matrix whose PATTERN is unsymmetric directly -- the caller
// must not have to pad it with structural zeros first. Checks accuracy against
// the true solution and cross-checks patternSymmetry() against a reference
// computed from A and A^T.
void testUnsymmetricPattern() {
  struct Case {
    const char* name;
    SparseMatrix<double> A;
  };
  std::vector<Case> cases;
  cases.push_back({"unsym pattern n=120 p=0.05", randomUnsymmetricPattern(120, 0.05, 3)});
  cases.push_back({"unsym pattern n=300 p=0.02", randomUnsymmetricPattern(300, 0.02, 11)});
  cases.push_back({"upwind grid 20x20", upwind2d(20, 20)});
  cases.push_back({"upwind grid 40x30", upwind2d(40, 30)});

  for (const Case& c : cases) {
    const int n = static_cast<int>(c.A.rows());
    // Guard the premise: these must actually have unsymmetric patterns, or the
    // test silently degenerates into the symmetric case it already covers.
    if (lu_testing::patternIsSymmetric(c.A)) {
      lu_testing::fail(std::string(c.name) + ": generator produced a symmetric pattern");
      continue;
    }
    VectorXd xTrue = VectorXd::Random(n);
    VectorXd b = c.A * xTrue;

    Eigen::LeftRightLU<SparseMatrix<double>> solver;
    solver.compute(c.A);
    if (solver.info() != Eigen::Success) {
      lu_testing::fail(std::string(c.name) + ": " + solver.lastErrorMessage());
      continue;
    }
    const VectorXd x = solver.solve(b);
    const double worst = std::max((x - xTrue).norm() / xTrue.norm(), (c.A * x - b).norm() / b.norm());
    check(worst < 1e-8, c.name, worst);

    const double expected = lu_testing::patternSymmetry(c.A);
    check(std::abs(solver.patternSymmetry() - expected) < 1e-12,
          std::string(c.name) + ": patternSymmetry() matches reference",
          std::abs(solver.patternSymmetry() - expected));
    checkTrue(!solver.structurallySymmetric(),
              std::string(c.name) + ": reported as not structurally symmetric");
  }

  // A symmetric-pattern matrix must report symmetry exactly 1.
  Eigen::LeftRightLU<SparseMatrix<double>> sym;
  sym.analyzePattern(laplacian2d(12, 12));
  checkTrue(sym.structurallySymmetric() && sym.patternSymmetry() == 1.0,
            "symmetric pattern reports patternSymmetry() == 1");
}

// Padding the pattern with explicit structural zeros must be unnecessary: the
// solver's own symmetrization has to produce the same answer as feeding it a
// pre-symmetrized matrix, at no more fill and strictly less work per pass over
// the values.
//
// Fill EQUALITY is not the contract, and asserting it would be wrong: padding
// symmetrizes before the matching row permutation, so the solver then
// symmetrizes a second time and eliminates a strictly larger graph. It also
// destroys reducibility -- an upwind operator is fully triangular after
// matching, and its padded form is irreducible -- so with BTF the padded input
// can cost dramatically more. Measured on gemat11 the gap reaches 102x. The
// direction is what matters: padding never helps.
void testNoPreSymmetrizationNeeded() {
  const SparseMatrix<double> A = upwind2d(25, 25);
  const SparseMatrix<double> Apadded = lu_testing::symmetrizePattern(A);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>> raw, padded;
  raw.compute(A);
  padded.compute(Apadded);
  const VectorXd xr = raw.solve(b), xp = padded.solve(b);

  check(raw.info() == Eigen::Success && (A * xr - b).norm() / b.norm() < 1e-8,
        "raw unsymmetric pattern solves", (A * xr - b).norm() / b.norm());
  check(raw.nnzL() <= padded.nnzL() && raw.nnzU() <= padded.nnzU(),
        "pre-symmetrizing the input never reduces fill",
        double(padded.nnzL() - raw.nnzL()));
  check((xr - xp).norm() / xp.norm() < 1e-10, "raw and pre-symmetrized answers agree",
        (xr - xp).norm() / xp.norm());
  std::printf("        A nnz=%lld padded nnz=%lld (padding is %.0f%% dead weight)\n",
              (long long)A.nonZeros(), (long long)Apadded.nonZeros(),
              100.0 * double(Apadded.nonZeros() - A.nonZeros()) / double(Apadded.nonZeros()));
  std::printf("        nnzL raw=%lld (%lld BTF blocks) padded=%lld (%lld BTF blocks)\n",
              (long long)raw.nnzL(), (long long)raw.btfBlockCount(), (long long)padded.nnzL(),
              (long long)padded.btfBlockCount());
}

// Ordering functors disagree on whether they return the permutation or its
// inverse (AMD/METIS: inverse; COLAMD: direct). Reading one the wrong way round
// is invisible in the residual and shows up only as fill, so pin the fill.
// AMD is expected to win here -- it minimizes degree in the A+A^T graph this
// factorization actually eliminates -- but every functor must beat the natural
// ordering by a wide margin, which a misread permutation cannot do.
void testOrderingConventions() {
  const SparseMatrix<double> A = upwind2d(30, 30);
  const int n = static_cast<int>(A.rows());
  VectorXd xTrue = VectorXd::Random(n);
  VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>> amd;
  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::COLAMDOrdering<int>> colamd;
  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::NaturalOrdering<int>> natural;
  // BTF off ON PURPOSE. An upwind discretization is fully triangular after
  // matching, so with BTF on every block is a singleton, there is no graph left
  // to order, and all three orderings tie at zero fill -- which would make this
  // test vacuous exactly where it is meant to bite. What is being pinned here
  // is the OrderingConvention trait (see the comment above), and that needs a
  // matrix the ordering actually gets to order.
  amd.setBlockTriangularForm(false);
  colamd.setBlockTriangularForm(false);
  natural.setBlockTriangularForm(false);
  amd.compute(A);
  colamd.compute(A);
  natural.compute(A);

  check((A * amd.solve(b) - b).norm() / b.norm() < 1e-8, "AMD ordering solves",
        (A * amd.solve(b) - b).norm() / b.norm());
  check((A * colamd.solve(b) - b).norm() / b.norm() < 1e-8, "COLAMD ordering solves",
        (A * colamd.solve(b) - b).norm() / b.norm());

  const double amdFill = double(amd.nnzL());
  const double colamdFill = double(colamd.nnzL());
  const double naturalFill = double(natural.nnzL());
  // A permutation read backwards lands near (or above) the natural ordering.
  // Correctly read, both fill-reducing orderings are several times below it.
  check(amdFill < 0.5 * naturalFill, "AMD fill well below natural", amdFill / naturalFill);
  check(colamdFill < 0.5 * naturalFill, "COLAMD fill well below natural (convention honoured)",
        colamdFill / naturalFill);
  std::printf("        nnzL: AMD=%.0f COLAMD=%.0f natural=%.0f\n", amdFill, colamdFill,
              naturalFill);
}

// Structural singularity is the failure mode unsymmetric patterns introduce: a
// column whose rows are all claimed elsewhere cannot get a diagonal entry. The
// contract is that analyzePattern() reports it via matchingIsPerfect() and that
// solve() refuses to call the resulting garbage a success.
void testStructurallySingular() {
  const int n = 40;
  std::vector<Eigen::Triplet<double>> t;
  for (int j = 0; j < n; ++j) {
    if (j == 17) continue;  // an entirely empty column
    t.emplace_back(j, j, 3.0);
    if (j + 1 < n) t.emplace_back(j + 1, j, 1.0);
  }
  SparseMatrix<double> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.analyzePattern(A);
  checkTrue(!solver.matchingIsPerfect(),
            "structurally singular matrix reported by matchingIsPerfect()");

  solver.factorize(A);
  const VectorXd b = VectorXd::Ones(n);
  const VectorXd x = solver.solve(b);
  checkTrue(solver.info() == Eigen::NumericalIssue,
            "solve on a structurally singular matrix reports NumericalIssue");
  std::printf("        residual=%.3e finite=%d\n", solver.solveResidual(), int(x.allFinite()));

  // A matrix that is merely pattern-unsymmetric must NOT be flagged.
  Eigen::LeftRightLU<SparseMatrix<double>> healthy;
  healthy.analyzePattern(randomUnsymmetricPattern(80, 0.05, 21));
  checkTrue(healthy.matchingIsPerfect(), "healthy unsymmetric pattern is not flagged singular");
}

// Unsymmetric pattern + everything else the solver offers at once: transpose and
// adjoint solves, multiple right-hand sides, determinant, and the parallel
// dynamic scheduler all have to keep working when the pattern is unsymmetric.
void testUnsymmetricPatternFeatures() {
  const SparseMatrix<double> A = randomUnsymmetricPattern(150, 0.04, 31);
  const int n = static_cast<int>(A.rows());

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);

  MatrixXd X = MatrixXd::Random(n, 3);
  MatrixXd B = A * X;
  const MatrixXd Xs = solver.solve(B);
  check((A * Xs - B).norm() / B.norm() < 1e-8, "unsym pattern: multiple RHS",
        (A * Xs - B).norm() / B.norm());

  const VectorXd b = VectorXd::Random(n);
  const SparseMatrix<double> AT = A.transpose();
  const VectorXd xt = solver.transpose().solve(b);
  check((AT * xt - b).norm() / b.norm() < 1e-8, "unsym pattern: transpose solve",
        (AT * xt - b).norm() / b.norm());

  // Compared in log space: with a diagonal of ~n over 150 columns the raw
  // determinant is ~1e320 and overflows double for both solvers.
  Eigen::SparseLU<SparseMatrix<double>> ref;
  ref.compute(A);
  if (ref.info() == Eigen::Success) {
    const double lref = ref.logAbsDeterminant();
    const double rel = std::abs(solver.logAbsDeterminant() - lref) / std::abs(lref);
    check(rel < 1e-10, "unsym pattern: log|det| matches SparseLU", rel);
  }

  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>,
                     Eigen::supernodal_lu::StdThreadExecutor>
      parallel;
  parallel.compute(upwind2d(45, 45));
  const SparseMatrix<double> P = upwind2d(45, 45);
  const VectorXd pb = P * VectorXd::Random(P.rows());
  const double presid = (P * parallel.solve(pb) - pb).norm() / pb.norm();
  check(parallel.info() == Eigen::Success && presid < 1e-8,
        "unsym pattern: parallel dynamic scheduler", presid);
}

// ---------------------------------------------------------------------------
//  Input validation, state invalidation and reporting
// ---------------------------------------------------------------------------

typedef std::complex<double> cd;
typedef SparseMatrix<cd> SpMatC;

// Block upper triangular with dense random diagonal blocks and sparse coupling
// above them: reducible by construction, so BTF finds nb blocks. With
// weakDiag the diagonal entries are 1e-3 against O(1) off-diagonals, which is
// what makes complete pivoting actually swap columns inside the blocks.
template <typename Scalar>
SparseMatrix<Scalar> reducibleDenseBlocks(int nb, int bs, unsigned seed, bool weakDiag) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  const int n = nb * bs;
  std::vector<Eigen::Triplet<Scalar>> t;
  for (int b = 0; b < nb; ++b)
    for (int i = 0; i < bs; ++i)
      for (int j = 0; j < bs; ++j) {
        double v = uni(rng);
        if (i == j) v = weakDiag ? 1e-3 * uni(rng) : (bs + uni(rng));
        t.emplace_back(b * bs + i, b * bs + j, Scalar(v));
      }
  for (int b = 0; b + 1 < nb; ++b)
    for (int k = 0; k < bs; ++k) t.emplace_back(b * bs + (k * 7) % bs, (b + 1) * bs + k, Scalar(uni(rng)));
  SparseMatrix<Scalar> A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// Multiply every entry by a unit phase so the matrix is genuinely complex
// (values, pivots and determinant all carry a phase).
SpMatC withPhases(SpMatC A) {
  for (int j = 0; j < A.outerSize(); ++j)
    for (SpMatC::InnerIterator it(A, j); it; ++it)
      it.valueRef() *= cd(std::cos(0.7 * j + it.row()), std::sin(0.7 * j + it.row()));
  return A;
}

// A new analyzePattern() replaces every permutation and every panel layout the
// solve reads through. The factors of the previous matrix must therefore stop
// being addressable, or a solve() after analyzePattern(B) would index the old
// arenas with the new maps (which, before this was pinned, reached an Eigen
// size assertion in debug and out-of-bounds reads in release).
void testAnalyzePatternInvalidatesFactors() {
  const SparseMatrix<double> A = laplacian2d(10, 10);
  const SparseMatrix<double> B = laplacian2d(4, 4);
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  checkTrue(solver.isFactorized(), "compute(A) factorizes");
  solver.analyzePattern(B);
  checkTrue(!solver.isFactorized(), "analyzePattern(B) invalidates A's factors");
  checkTrue(solver.info() == Eigen::Success, "analyzePattern(B) itself succeeds");

  const VectorXd x = solver.solve(VectorXd::Ones(16));
  checkTrue(solver.info() == Eigen::NumericalIssue && !x.allFinite(),
            "solve() between analyzePattern() and factorize() is refused");
  checkTrue(std::isinf(solver.conditionEstimate()), "conditionEstimate() has no factorization to describe");
  checkTrue(std::isnan(solver.growthFactor()), "growthFactor() has no factorization to describe");

  solver.factorize(B);
  const VectorXd xTrue = VectorXd::Random(16);
  const VectorXd x2 = solver.solve(B * xTrue);
  check(solver.info() == Eigen::Success && (x2 - xTrue).norm() / xTrue.norm() < 1e-10,
        "factorize(B) after the re-analysis solves B", (x2 - xTrue).norm() / xTrue.norm());
}

// denseRowFillPenalty() is cached per analysis; the cache has to follow the
// analysis, not the factorization, or a solver reused for a second symbolic
// analysis answers with the first matrix's penalty.
void testDenseRowPenaltyFollowsTheAnalysis() {
  const int n = 400;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, 4.0);
  for (int i = 0; i + 1 < n; ++i) {
    t.emplace_back(i, i + 1, -1.0);
    t.emplace_back(i + 1, i, -1.0);
  }
  for (int d = 0; d < 3; ++d)
    for (int j = 0; j < n; ++j)
      if (j != d) {
        t.emplace_back(d, j, 0.01);
        t.emplace_back(j, d, 0.01);
      }
  SparseMatrix<double> arrow(n, n);
  arrow.setFromTriplets(t.begin(), t.end());
  arrow.makeCompressed();
  const SparseMatrix<double> grid = laplacian2d(20, 20);

  Eigen::LeftRightLU<SparseMatrix<double>> reused, fresh;
  reused.analyzePattern(arrow);
  const double arrowPenalty = reused.denseRowFillPenalty(arrow);
  reused.analyzePattern(grid);
  fresh.analyzePattern(grid);
  const double reusedPenalty = reused.denseRowFillPenalty(grid);
  const double freshPenalty = fresh.denseRowFillPenalty(grid);
  std::printf("        arrow penalty=%.4f grid: reused=%.4f fresh=%.4f\n", arrowPenalty, reusedPenalty,
              freshPenalty);
  checkTrue(reusedPenalty == freshPenalty, "denseRowFillPenalty() is recomputed after a new analysis");
}

// An inf or NaN entry poisons the equilibration (0 * inf) and then every factor
// it touches, and NaN never trips a zero-pivot test, so without a check the
// factorization reported Success over NaN factors. It is declined instead.
void testNonFiniteInputIsDeclined() {
  const double bad[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
  const char* names[2] = {"inf", "NaN"};
  for (int k = 0; k < 2; ++k) {
    SparseMatrix<double> A = laplacian2d(6, 6);
    A.coeffRef(7, 7) = bad[k];
    Eigen::LeftRightLU<SparseMatrix<double>> solver;
    solver.compute(A);
    checkTrue(solver.info() == Eigen::InvalidInput && !solver.isFactorized(),
              std::string("a matrix with an ") + names[k] + " entry is declined (InvalidInput)");
    checkTrue(!solver.lastErrorMessage().empty(), std::string(names[k]) + ": lastErrorMessage() explains");
    const VectorXd x = solver.solve(VectorXd::Ones(36));
    checkTrue(solver.info() != Eigen::Success && !x.allFinite(),
              std::string(names[k]) + ": solve() after the declined factorization is refused");
  }
}

// factorize() scatters into the panels analyzePattern() laid out. A matrix of
// another size would index past the permutation maps, and an entry with no
// slot would be written past a panel; both are caught and reported. An entry
// that lands in a FILL slot the symbolic phase already created is a different
// matter: it is factored exactly, because the structure is a superset.
void testFactorizeInputMismatchIsReported() {
  const SparseMatrix<double> A = laplacian2d(12, 12);
  const int n = static_cast<int>(A.rows());

  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.analyzePattern(A);
  solver.factorize(laplacian2d(5, 5));
  checkTrue(solver.info() == Eigen::InvalidInput && !solver.isFactorized(),
            "factorize() with a matrix of another size is refused");

  // Entries far off the band: not in the fill of a 5-point stencil ordered by AMD.
  SparseMatrix<double> grown = A;
  grown.coeffRef(0, n - 1) = 0.5;
  grown.coeffRef(n - 1, 3) = -0.25;
  grown.makeCompressed();
  solver.analyzePattern(A);
  solver.factorize(grown);
  checkTrue(solver.info() == Eigen::InvalidInput && !solver.isFactorized(),
            "factorize() with a nonzero outside the analyzed pattern is refused");
  const VectorXd x = solver.solve(VectorXd::Ones(n));
  checkTrue(solver.info() != Eigen::Success && !x.allFinite(), "solve() after the refusal is refused too");

  solver.compute(grown);
  const VectorXd xTrue = VectorXd::Random(n);
  const VectorXd x2 = solver.solve(grown * xTrue);
  check(solver.info() == Eigen::Success && (x2 - xTrue).norm() / xTrue.norm() < 1e-10,
        "compute() on the grown matrix recovers", (x2 - xTrue).norm() / xTrue.norm());

  // Under BTF an entry BELOW the diagonal blocks would break the block
  // back-substitution even when the column still has spare cross-block slots.
  const SparseMatrix<double> R = reducibleDenseBlocks<double>(4, 5, 77, false);
  SparseMatrix<double> lowered = R;
  lowered.coeffRef(19, 0) = 1.0;  // last block -> first block: below the diagonal blocks
  lowered.makeCompressed();
  Eigen::LeftRightLU<SparseMatrix<double>> btf;
  btf.analyzePattern(R);
  checkTrue(btf.btfBlockCount() == 4, "reducible test matrix splits into 4 BTF blocks");
  btf.factorize(lowered);
  checkTrue(btf.info() == Eigen::InvalidInput, "entry below the BTF diagonal blocks is refused");
  btf.compute(lowered);
  const VectorXd xr = VectorXd::Random(20);
  const VectorXd xl = btf.solve(lowered * xr);
  check(btf.info() == Eigen::Success && (xl - xr).norm() / xr.norm() < 1e-10,
        "compute() on the lowered matrix recovers", (xl - xr).norm() / xr.norm());

  // An entry inside the fill is legitimately absorbed: same pattern for the
  // solver's purposes, exact answer.
  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::NaturalOrdering<int>> natural;
  natural.analyzePattern(A);
  SparseMatrix<double> filled = A;
  filled.coeffRef(2, 0) = 0.3;  // (2,0) is fill of the natural-ordered 5-point stencil
  filled.makeCompressed();
  natural.factorize(filled);
  const VectorXd xf = natural.solve(filled * xTrue);
  check(natural.info() == Eigen::Success && (xf - xTrue).norm() / xTrue.norm() < 1e-10,
        "a nonzero inside the fill is factored exactly", (xf - xTrue).norm() / xTrue.norm());
}

// A solve without a successful factorization has nothing to solve with; in a
// release build it used to read whatever the arenas held (a previous matrix's
// factors, or none) and could crash. Now it answers NaN with NumericalIssue.
void testSolveWithoutFactorizationIsRefused() {
  const SparseMatrix<double> A = laplacian2d(20, 20);
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(laplacian2d(3, 3));  // earlier, smaller factorization in the arenas
  solver.setMaxFactorNonzeros(10);    // guaranteed to trip on the 20x20 grid
  solver.compute(A);
  checkTrue(solver.info() == Eigen::NumericalIssue && !solver.isFactorized(), "fill guard declined");

  const VectorXd x = solver.solve(VectorXd::Ones(400));
  checkTrue(solver.info() == Eigen::NumericalIssue && x.size() == 400 && !x.allFinite(),
            "solve() after a declined factorization returns NaN + NumericalIssue");
  checkTrue(!solver.lastErrorMessage().empty(), "... and keeps the factorization's message");
  const VectorXd xt = solver.transpose().solve(VectorXd::Ones(400));
  checkTrue(solver.info() == Eigen::NumericalIssue && !xt.allFinite(),
            "transpose().solve() after a declined factorization is refused");
  const VectorXd xa = solver.adjoint().solve(VectorXd::Ones(400));
  checkTrue(solver.info() == Eigen::NumericalIssue && !xa.allFinite(),
            "adjoint().solve() after a declined factorization is refused");
  checkTrue(std::isnan(solver.solveResidual()), "solveResidual() is NaN for a refused solve");
}

// The solve status describes the LAST solve: a message left by an earlier
// failed solve must not survive a later successful one.
void testSolveStatusIsPerSolve() {
  const SparseMatrix<double> A = laplacian2d(6, 6);
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.setSolveFailureThreshold(1e-300);  // no residual can pass this
  solver.compute(A);
  VectorXd x = solver.solve(VectorXd::Ones(36));
  checkTrue(solver.info() == Eigen::NumericalIssue && !solver.lastErrorMessage().empty(),
            "impossible gate: solve flagged with a message");
  solver.setSolveFailureThreshold(1e-6);
  x = solver.solve(VectorXd::Ones(36));
  checkTrue(solver.info() == Eigen::Success && solver.lastErrorMessage().empty(),
            "passing solve clears the earlier failure message");
}

// determinant() is a product of n pivots divided by a product of 2n scaling
// factors. Accumulated naively either product can pass through 1e+300 on the
// way to an answer of 1, and a diagonal half 1e-5 and half 1e+5 does exactly
// that. logAbsDeterminant() never had the problem; determinant() now agrees
// with it wherever the answer is representable.
void testDeterminantSurvivesIntermediateOverflow() {
  const int n = 400;
  SparseMatrix<double> A(n, n);
  for (int i = 0; i < n; ++i) A.insert(i, i) = (i < n / 2) ? 1e-5 : 1e5;  // det = 1 exactly
  A.makeCompressed();

  Eigen::LeftRightLU<SparseMatrix<double>> equilibrated;
  equilibrated.compute(A);
  check(std::abs(equilibrated.determinant() - 1.0) < 1e-10, "determinant() = 1 through Ruiz scaling 1e+1000",
        equilibrated.determinant());

  Eigen::LeftRightLU<SparseMatrix<double>> raw;
  raw.setEquilibration(false);
  raw.setStaticPivotThreshold(0.0);  // the 1e-5 pivots are real pivots, not weak ones
  raw.compute(A);
  check(raw.replacedPivots() == 0 && std::abs(raw.determinant() - 1.0) < 1e-10,
        "determinant() = 1 through pivots 1e-1000", raw.determinant());
  check(std::abs(raw.logAbsDeterminant()) < 1e-9, "logAbsDeterminant() = 0", raw.logAbsDeterminant());

  // Genuinely out of range: inf and 0, not garbage.
  SparseMatrix<double> big(n, n);
  for (int i = 0; i < n; ++i) big.insert(i, i) = 1e5;
  big.makeCompressed();
  raw.compute(big);
  checkTrue(std::isinf(raw.determinant()) && raw.determinant() > 0, "det(1e5 I_400) overflows to +inf");
  check(std::abs(raw.logAbsDeterminant() - n * std::log(1e5)) < 1e-8 * n, "log|det(1e5 I_400)| is finite",
        raw.logAbsDeterminant());
  SparseMatrix<double> small(n, n);
  for (int i = 0; i < n; ++i) small.insert(i, i) = -1e-5;
  small.makeCompressed();
  raw.compute(small);
  checkTrue(raw.determinant() == 0.0 && raw.determinantSign() == 1.0, "det(-1e-5 I_400) underflows to 0, sign +1");
}

// Complex determinants against dense Eigen, across every pivoting mode with
// and without matching: the sign bookkeeping has to carry a phase, and every
// row or column swap, the matching permutation and the scaling all feed it.
void testComplexDeterminant() {
  for (int variant = 0; variant < 4; ++variant) {
    const SpMatC A = withPhases(variant < 2 ? lu_testing::weakDiagonalAs<cd>(40 + 7 * variant, 5 + variant)
                                            : reducibleDenseBlocks<cd>(4, 6, 20 + variant, variant == 3));
    const Eigen::Matrix<cd, -1, -1> Ad = A;
    const cd ref = Eigen::FullPivLU<Eigen::Matrix<cd, -1, -1>>(Ad).determinant();
    for (int matching = 0; matching < 2; ++matching)
      for (int piv = 0; piv < 3; ++piv) {
        Eigen::LeftRightLU<SpMatC> solver;
        solver.setMatching(matching == 1);
        solver.setPivoting(piv == 0 ? lr::Pivoting::None : piv == 1 ? lr::Pivoting::Partial : lr::Pivoting::Complete);
        solver.compute(A);
        char name[96];
        std::snprintf(name, sizeof name, "complex det v%d matching=%d pivoting=%d (blocks %lld)", variant, matching,
                      piv, (long long)solver.btfBlockCount());
        if (solver.info() != Eigen::Success || solver.replacedPivots() > 0) {
          lu_testing::note(std::string(name) + ": perturbed factorization, skipped");
          continue;
        }
        const double growth = solver.growthFactor();
        const double tol = 1e-9 * std::max(1.0, growth);
        const cd got = solver.determinant();
        const cd viaLog = solver.determinantSign() * std::exp(solver.logAbsDeterminant());
        const double rel = std::max(std::abs(got - ref), std::abs(viaLog - ref)) / std::abs(ref);
        check(rel < tol, name, rel);
        check(std::abs(std::abs(solver.determinantSign()) - 1.0) < 1e-12, "determinantSign() has unit modulus",
              std::abs(solver.determinantSign()));
      }
  }
}

// MC64 matching, BTF and complex arithmetic together: the reducible blocks are
// solved from the last block backwards for A and forwards for A^H, and every
// off-diagonal entry is read conjugated in the adjoint path.
void testReducibleComplexWithMC64() {
  const SpMatC A = reducibleDenseBlocks<cd>(3, 8, 5, true);
  const int n = static_cast<int>(A.rows());
  Eigen::LeftRightLU<SpMatC> solver;
  solver.setMatchingMethod(Eigen::supernodal_lu::MatchingMethod::MC64);
  solver.compute(A);
  checkTrue(solver.btfBlockCount() == 3, "MC64 + BTF: 3 blocks found");
  const Eigen::VectorXcd xTrue = Eigen::VectorXcd::Random(n);
  const Eigen::VectorXcd b = A * xTrue;
  const Eigen::VectorXcd x = solver.solve(b);
  check(solver.info() == Eigen::Success && (x - xTrue).norm() / xTrue.norm() < 1e-9, "MC64 + BTF complex solve",
        (x - xTrue).norm() / xTrue.norm());
  const Eigen::VectorXcd bh = A.adjoint() * xTrue;
  const Eigen::VectorXcd xh = solver.adjoint().solve(bh);
  check(solver.info() == Eigen::Success && (xh - xTrue).norm() / xTrue.norm() < 1e-9,
        "MC64 + BTF complex adjoint solve", (xh - xTrue).norm() / xTrue.norm());
  const Eigen::Matrix<cd, -1, -1> Ad = A;
  const cd ref = Eigen::FullPivLU<Eigen::Matrix<cd, -1, -1>>(Ad).determinant();
  check(std::abs(solver.determinant() - ref) / std::abs(ref) < 1e-9, "MC64 + BTF complex determinant",
        std::abs(solver.determinant() - ref) / std::abs(ref));
}

// The determinant is that of the operator the factorization inverts. Under
// static pivoting that is a perturbed A, so a singular matrix whose zero pivot
// was bumped reports a nonzero determinant with Success -- and
// replacedPivots() is what says so. Pinned so the contract stays explicit.
void testDeterminantUnderStaticPivoting() {
  SparseMatrix<double> A = laplacian2d(6, 6);
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      if (it.row() == 10) it.valueRef() = 0.0;
  A.prune(0.0);  // row 10 is now empty: singular
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  checkTrue(solver.info() == Eigen::Success && solver.replacedPivots() > 0,
            "singular matrix factorizes with a replaced pivot");
  checkTrue(std::isfinite(solver.determinant()) && solver.determinant() != 0.0,
            "determinant() describes the perturbed operator (nonzero)");
  const VectorXd x = solver.solve(VectorXd::Ones(36));
  checkTrue(solver.info() == Eigen::NumericalIssue, "... and solve() reports the singularity");

  // With the static threshold off the zero pivot is a hard failure instead.
  Eigen::LeftRightLU<SparseMatrix<double>> strict;
  strict.setStaticPivotThreshold(0.0);
  strict.compute(A);
  checkTrue(strict.info() == Eigen::NumericalIssue && !strict.isFactorized(),
            "threshold 0: the zero pivot fails the factorization");
}

// Uncompressed (reserve + insert) input, including the one ordering that reads
// the index arrays directly. Matching off routes the raw matrix to the ordering
// functor, which is where an uncompressed input would otherwise reach COLAMD.
void testUncompressedInput() {
  const SparseMatrix<double> A = weakDiagonal(80, 3);
  const int n = static_cast<int>(A.rows());
  SparseMatrix<double> Au(n, n);
  Au.reserve(Eigen::VectorXi::Constant(n, 12));
  for (int j = 0; j < n; ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it) Au.insert(it.row(), j) = it.value();
  checkTrue(!Au.isCompressed(), "test input really is uncompressed");
  const VectorXd xTrue = VectorXd::Random(n);
  const VectorXd b = A * xTrue;

  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::COLAMDOrdering<int>> colamd;
  colamd.setMatching(false);
  colamd.compute(Au);
  VectorXd x = colamd.solve(b);
  check(colamd.info() == Eigen::Success && (x - xTrue).norm() / xTrue.norm() < 1e-8,
        "uncompressed input, COLAMD, matching off", (x - xTrue).norm() / xTrue.norm());

  Eigen::LeftRightLU<SparseMatrix<double>> defaults;
  defaults.setErrorBounds(true);
  defaults.setExtendedPrecisionResidual(true);
  defaults.setRefineOnlyIfPerturbed(false);
  defaults.compute(Au);
  x = defaults.solve(b);
  check(defaults.info() == Eigen::Success && (x - xTrue).norm() / xTrue.norm() < 1e-8,
        "uncompressed input, defaults + bounds + extended residual", (x - xTrue).norm() / xTrue.norm());
  checkTrue(defaults.lastCorrectDigits() >= 8, "uncompressed input: error bounds computed");
}

// Degenerate shapes: n = 0, n = 1, zero right-hand sides, a zero matrix.
void testDegenerateShapes() {
  {
    SparseMatrix<double> A(0, 0);
    A.makeCompressed();
    Eigen::LeftRightLU<SparseMatrix<double>> solver;
    solver.setErrorBounds(true);
    solver.compute(A);
    const VectorXd x = solver.solve(VectorXd(0));
    const MatrixXd X = solver.solve(MatrixXd(0, 3));
    const VectorXd xt = solver.transpose().solve(VectorXd(0));
    checkTrue(solver.info() == Eigen::Success && x.size() == 0 && X.cols() == 3 && xt.size() == 0,
              "n=0: compute/solve/transpose survive");
    checkTrue(solver.determinant() == 1.0 && solver.logAbsDeterminant() == 0.0, "n=0: det = 1");
  }
  {
    SparseMatrix<double> A(1, 1);
    A.insert(0, 0) = -2.5;
    A.makeCompressed();
    Eigen::LeftRightLU<SparseMatrix<double>> solver;
    solver.compute(A);
    VectorXd b(1);
    b << 5.0;
    const VectorXd x = solver.solve(b);
    check(std::abs(x[0] + 2.0) < 1e-15 && solver.determinant() == -2.5 && solver.determinantSign() == -1.0,
          "n=1: solve, determinant, sign", x[0]);
  }
  {
    const SparseMatrix<double> A = laplacian2d(5, 5);
    Eigen::LeftRightLU<SparseMatrix<double>> solver;
    solver.setRefineOnlyIfPerturbed(false);
    solver.setErrorBounds(true);
    solver.compute(A);
    const MatrixXd X = solver.solve(MatrixXd(25, 0));
    checkTrue(solver.info() == Eigen::Success && X.rows() == 25 && X.cols() == 0, "zero right-hand sides survive");
  }
  {
    SparseMatrix<double> Z(5, 5);
    Z.makeCompressed();
    Eigen::LeftRightLU<SparseMatrix<double>> solver;
    solver.compute(Z);
    checkTrue(solver.info() == Eigen::NumericalIssue && !solver.isFactorized(), "zero matrix is declined");
    const VectorXd x = solver.solve(VectorXd::Ones(5));
    checkTrue(solver.info() == Eigen::NumericalIssue && !x.allFinite(), "zero matrix: solve refused");
  }
}

// Supernode width caps of 1, 2 and 3 make every diagonal block trivial and
// push all the work into the panels and the scheduler, serial and parallel.
void testTinyBlockSizes() {
  const SparseMatrix<double> A = randomUnsymmetricPattern(300, 0.03, 4);
  const int n = static_cast<int>(A.rows());
  const VectorXd xTrue = VectorXd::Random(n);
  const VectorXd b = A * xTrue;
  for (int bs : {1, 2, 3}) {
    Eigen::LeftRightLU<SparseMatrix<double>> serial;
    serial.setMaxBlockSize(bs);
    serial.compute(A);
    VectorXd x = serial.solve(b);
    char name[64];
    std::snprintf(name, sizeof name, "maxBlockSize=%d serial", bs);
    check(serial.info() == Eigen::Success && (x - xTrue).norm() / xTrue.norm() < 1e-8, name,
          (x - xTrue).norm() / xTrue.norm());

    Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, Eigen::supernodal_lu::StdThreadExecutor>
        parallel;
    parallel.setMaxBlockSize(bs);
    parallel.compute(A);
    x = parallel.solve(b);
    std::snprintf(name, sizeof name, "maxBlockSize=%d parallel", bs);
    check(parallel.info() == Eigen::Success && (x - xTrue).norm() / xTrue.norm() < 1e-8, name,
          (x - xTrue).norm() / xTrue.norm());
    checkTrue(parallel.nnzL() == serial.nnzL() && parallel.nnzU() == serial.nnzU(),
              "parallel and serial agree on fill");
  }
}

// A sparse right-hand side goes through SparseSolverBase's column-by-column
// path; it must give the dense answer.
void testSparseRightHandSide() {
  const SparseMatrix<double> A = laplacian2d(8, 8);
  Eigen::LeftRightLU<SparseMatrix<double>> solver;
  solver.compute(A);
  SparseMatrix<double> B(64, 3);
  B.insert(3, 0) = 1.0;
  B.insert(10, 1) = 2.0;
  B.insert(63, 2) = -1.0;
  B.makeCompressed();
  const SparseMatrix<double> X = solver.solve(B);
  const MatrixXd Xd = solver.solve(MatrixXd(B));
  check((MatrixXd(X) - Xd).norm() < 1e-12 * Xd.norm(), "sparse rhs matches dense rhs", (MatrixXd(X) - Xd).norm());
  check((A * Xd - MatrixXd(B)).norm() < 1e-10, "sparse rhs solve is correct", (A * Xd - MatrixXd(B)).norm());
}

// The parallel scheduler's failure path: a zero pivot on one lane must stop
// every lane, report NumericalIssue, leave nothing factorized, and leave the
// solver reusable.
void testParallelSingularPath() {
  SparseMatrix<double> A = laplacian2d(40, 40);
  for (int j = 0; j < A.outerSize(); ++j)
    for (SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
      if (it.row() == 800 || it.col() == 800) it.valueRef() = 0.0;
  A.prune(0.0);
  Eigen::LeftRightLU<SparseMatrix<double>, Eigen::AMDOrdering<int>, Eigen::supernodal_lu::StdThreadExecutor>
      parallel;
  parallel.setStaticPivotThreshold(0.0);
  parallel.compute(A);
  checkTrue(parallel.info() == Eigen::NumericalIssue && !parallel.isFactorized(),
            "parallel scheduler reports the zero pivot and stops");
  const SparseMatrix<double> G = laplacian2d(40, 40);
  parallel.compute(G);
  const VectorXd b = G * VectorXd::Ones(1600);
  const VectorXd x = parallel.solve(b);
  check(parallel.info() == Eigen::Success && (x - VectorXd::Ones(1600)).norm() < 1e-8,
        "solver is reusable after the parallel failure", (x - VectorXd::Ones(1600)).norm());
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("LeftRightLU correctness tests\n");

  std::printf("Random symmetric-pattern matrices:\n");
  solveAndMeasure(randomSymmetricPattern(50, 0.10, 1), "random n=50 p=0.10");
  solveAndMeasure(randomSymmetricPattern(120, 0.05, 7), "random n=120 p=0.05");
  solveAndMeasure(randomSymmetricPattern(200, 0.03, 13), "random n=200 p=0.03");

  std::printf("2D Laplacian (5-point) matrices:\n");
  solveAndMeasure(laplacian2d(10, 10), "laplacian 10x10");
  solveAndMeasure(laplacian2d(20, 20), "laplacian 20x20");
  solveAndMeasure(laplacian2d(30, 25), "laplacian 30x25");

  std::printf("Features:\n");
  testMultipleRhs();
  testFactorAccessors();
  testTransposeSolve();
  testCompletePivoting();
  testCompletePivotingColumnSwaps();
  testLogDeterminant();
  testEquilibration();
  testHonestFailure();
  testFillGuard();
  testParallelVsSerial();

  std::printf("Unsymmetric nonzero patterns:\n");
  testUnsymmetricPattern();
  testNoPreSymmetrizationNeeded();
  testOrderingConventions();
  testStructurallySingular();
  testUnsymmetricPatternFeatures();

  std::printf("Input validation, state and reporting:\n");
  testAnalyzePatternInvalidatesFactors();
  testDenseRowPenaltyFollowsTheAnalysis();
  testNonFiniteInputIsDeclined();
  testFactorizeInputMismatchIsReported();
  testSolveWithoutFactorizationIsRefused();
  testSolveStatusIsPerSolve();
  testDeterminantSurvivesIntermediateOverflow();
  testComplexDeterminant();
  testReducibleComplexWithMC64();
  testDeterminantUnderStaticPivoting();
  testUncompressedInput();
  testDegenerateShapes();
  testTinyBlockSizes();
  testSparseRightHandSide();
  testParallelSingularPath();

  return lu_testing::summarize("LeftRightLU correctness");
}
