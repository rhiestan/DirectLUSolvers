// Correctness tests for Eigen::LeftRightLU (PARDISO-style left-right-looking LU
// with a barrier-free dynamic scheduler and in-block complete pivoting).
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_leftright_lu --output-on-failure
//
// Covers: direct solves, multiple RHS, factor accessors, transpose/adjoint,
// equilibration, complete vs partial vs no in-block pivoting, log-determinant,
// honest failure reporting, and parallel(dynamic-scheduler)-vs-serial agreement.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
// solver's own symmetrization has to produce the same factor structure and the
// same answer as feeding it a pre-symmetrized matrix, while doing strictly less
// work per pass over the values.
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
  check(raw.nnzL() == padded.nnzL() && raw.nnzU() == padded.nnzU(),
        "raw and pre-symmetrized inputs give identical fill",
        double(raw.nnzL() - padded.nnzL()));
  check((xr - xp).norm() / xp.norm() < 1e-10, "raw and pre-symmetrized answers agree",
        (xr - xp).norm() / xp.norm());
  std::printf("        A nnz=%lld padded nnz=%lld (padding is %.0f%% dead weight); nnzL=%lld\n",
              (long long)A.nonZeros(), (long long)Apadded.nonZeros(),
              100.0 * double(Apadded.nonZeros() - A.nonZeros()) / double(Apadded.nonZeros()),
              (long long)raw.nnzL());
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

  return lu_testing::summarize("LeftRightLU correctness");
}
