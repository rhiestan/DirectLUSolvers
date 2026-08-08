// MC64 maximum-product matching: optimality, duals, and integration.
//
//   ctest --test-dir build -R test_mc64 --output-on-failure
//
// WHY THIS EXISTS
//
// A subtly-wrong linear assignment solver does not crash -- it returns a valid
// permutation that is merely not optimal, the factorization proceeds, and the
// only symptom is a worse answer on some matrices. That is exactly the failure
// mode of the transversal this replaces, so "it produced a permutation and the
// tests passed" is not evidence of anything.
//
// The strong oracle is BRUTE FORCE: for small n the optimal assignment can be
// found exhaustively, so optimality is checked outright rather than inferred.
// The second oracle is the dual scaling, which is self-checking: if the duals
// are right then |Dr_i * a_ij * Dc_j| <= 1 everywhere with equality exactly on
// the matched entries, and that can be verified without any reference
// implementation.

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "LeftRightLU.h"
#include "SupernodalLU.h"
#include "SupernodalLUMC64.h"
#include "testing/Check.h"
#include "testing/TestMatrices.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using lu_testing::check;
using lu_testing::checkTrue;

namespace slu = Eigen::supernodal_lu;

namespace {

typedef SparseMatrix<double> SpMat;

// Optimal log-product over all perfect matchings, by exhaustive search.
double bruteForceLogProduct(const MatrixXd& D, bool& feasible) {
  const int n = static_cast<int>(D.rows());
  std::vector<int> p(static_cast<std::size_t>(n));
  std::iota(p.begin(), p.end(), 0);
  double best = -1e300;
  feasible = false;
  do {
    double total = 0.0;
    bool ok = true;
    for (int j = 0; j < n; ++j) {
      const double a = std::abs(D(p[static_cast<std::size_t>(j)], j));
      if (a <= 0.0) { ok = false; break; }
      total += std::log(a);
    }
    if (ok) { feasible = true; best = std::max(best, total); }
  } while (std::next_permutation(p.begin(), p.end()));
  return best;
}

// Exhaustive optimality + the dual scaling property, over many random matrices
// with magnitudes spanning ~12 orders so the assignment is genuinely contested.
void testOptimalityAgainstBruteForce() {
  std::mt19937 rng(12345);
  int suboptimal = 0, invalidPerm = 0, feasibilityMismatch = 0, scalingViolation = 0;
  int perfectCases = 0;
  double worstScaledMagnitude = 0.0, worstDiagonalError = 0.0;
  const int kTrials = 3000;

  for (int trial = 0; trial < kTrials; ++trial) {
    const int n = 1 + static_cast<int>(rng() % 7);
    const double density = 0.25 + 0.75 * static_cast<double>(rng() % 100) / 100.0;
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> expo(-6.0, 6.0);

    MatrixXd D = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (prob(rng) < density) D(i, j) = uni(rng) * std::pow(10.0, expo(rng));

    SpMat A = D.sparseView();
    A.makeCompressed();

    std::vector<int> match;
    std::vector<double> rowScale, colScale;
    const bool perfect = slu::mc64Matching(A, match, rowScale, colScale);

    // Always a valid permutation, even when structurally singular.
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    bool valid = static_cast<int>(match.size()) == n;
    for (int j = 0; valid && j < n; ++j) {
      if (match[static_cast<std::size_t>(j)] < 0 || match[static_cast<std::size_t>(j)] >= n ||
          seen[static_cast<std::size_t>(match[static_cast<std::size_t>(j)])])
        valid = false;
      else
        seen[static_cast<std::size_t>(match[static_cast<std::size_t>(j)])] = 1;
    }
    if (!valid) { ++invalidPerm; continue; }

    bool feasible = false;
    const double best = bruteForceLogProduct(D, feasible);
    if (feasible != perfect) { ++feasibilityMismatch; continue; }
    if (!perfect) continue;
    ++perfectCases;

    double got = 0.0;
    for (int j = 0; j < n; ++j)
      got += std::log(std::abs(D(match[static_cast<std::size_t>(j)], j)));
    if (got < best - 1e-9 * std::max(1.0, std::abs(best))) ++suboptimal;

    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        if (D(i, j) != 0.0) {
          const double m = std::abs(rowScale[static_cast<std::size_t>(i)] * D(i, j) *
                                    colScale[static_cast<std::size_t>(j)]);
          worstScaledMagnitude = std::max(worstScaledMagnitude, m);
          if (m > 1.0 + 1e-8) ++scalingViolation;
        }
    for (int j = 0; j < n; ++j) {
      const int i = match[static_cast<std::size_t>(j)];
      const double m = std::abs(rowScale[static_cast<std::size_t>(i)] * D(i, j) *
                                colScale[static_cast<std::size_t>(j)]);
      worstDiagonalError = std::max(worstDiagonalError, std::abs(m - 1.0));
    }
  }

  std::printf("        %d trials, %d with a perfect matching\n", kTrials, perfectCases);
  checkTrue(invalidPerm == 0, "MC64 always returns a valid permutation");
  checkTrue(feasibilityMismatch == 0, "MC64 agrees with brute force on feasibility");
  checkTrue(suboptimal == 0, "MC64 attains the brute-force optimal product");
  checkTrue(scalingViolation == 0, "dual scaling: no entry exceeds magnitude 1");
  check(std::abs(worstScaledMagnitude - 1.0) < 1e-8,
        "dual scaling: the largest scaled entry is exactly 1", worstScaledMagnitude);
  check(worstDiagonalError < 1e-9, "dual scaling: matched diagonal has magnitude 1",
        worstDiagonalError);
}

// The property the transversal cannot promise: MC64 maximizes the product over
// ALL permutations, so in particular it can never choose a worse diagonal than
// the identity. That is precisely the guarantee whose absence made the
// transversal break otherwise-solvable matrices.
void testNeverWorseThanIdentity() {
  std::mt19937 rng(777);
  int worseThanIdentity = 0, cases = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const int n = 30 + static_cast<int>(rng() % 40);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> expo(-4.0, 4.0);
    MatrixXd D = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (i == j || prob(rng) < 0.12) D(i, j) = uni(rng) * std::pow(10.0, expo(rng));
    for (int i = 0; i < n; ++i)
      if (D(i, i) == 0.0) D(i, i) = 1.0;  // keep the identity feasible

    SpMat A = D.sparseView();
    A.makeCompressed();
    std::vector<int> match;
    std::vector<double> rs, cs;
    if (!slu::mc64Matching(A, match, rs, cs)) continue;
    ++cases;

    double identity = 0.0, matched = 0.0;
    for (int j = 0; j < n; ++j) {
      identity += std::log(std::abs(D(j, j)));
      matched += std::log(std::abs(D(match[static_cast<std::size_t>(j)], j)));
    }
    if (matched < identity - 1e-9 * std::max(1.0, std::abs(identity))) ++worseThanIdentity;
  }
  std::printf("        %d feasible cases\n", cases);
  checkTrue(worseThanIdentity == 0, "MC64 is never worse than the un-permuted diagonal");
}

// End to end through both solvers, on a matrix whose diagonal is far too weak
// for static pivoting to cope with unaided.
void testIntegration() {
  const SpMat A = lu_testing::weakDiagonal(400, 5);
  const Eigen::Index n = A.rows();
  VectorXd xTrue(n);
  for (Eigen::Index i = 0; i < n; ++i) xTrue(i) = 1.0 + 0.25 * static_cast<double>(i % 5);
  const VectorXd b = A * xTrue;

  {
    Eigen::SupernodalLU<SpMat> s;
    s.setMatchingMethod(slu::MatchingMethod::MC64);
    s.compute(A);
    checkTrue(s.info() == Eigen::Success, "SupernodalLU + MC64: factorizes");
    const double resid = (A * VectorXd(s.solve(b)) - b).norm() / b.norm();
    check(resid < 1e-8, "SupernodalLU + MC64: solves a weak-diagonal system", resid);
    checkTrue(s.matchingMethod() == slu::MatchingMethod::MC64, "matchingMethod() round-trips");
  }
  {
    Eigen::LeftRightLU<SpMat> t;
    t.setMatchingMethod(slu::MatchingMethod::MC64);
    t.compute(A);
    checkTrue(t.info() == Eigen::Success, "LeftRightLU + MC64: factorizes");
    const double resid = (A * VectorXd(t.solve(b)) - b).norm() / b.norm();
    check(resid < 1e-8, "LeftRightLU + MC64: solves a weak-diagonal system", resid);
  }

  // setMatching(bool) must keep working as the alias it now is.
  Eigen::SupernodalLU<SpMat> legacy;
  legacy.setMatching(false);
  checkTrue(legacy.matchingMethod() == slu::MatchingMethod::None,
            "setMatching(false) maps to MatchingMethod::None");
  legacy.setMatching(true);
  checkTrue(legacy.matchingMethod() == slu::MatchingMethod::Transversal,
            "setMatching(true) maps to MatchingMethod::Transversal");
}

// A structurally singular matrix must still yield a usable permutation and be
// reported as imperfect, not silently accepted.
void testStructurallySingular() {
  const int n = 40;
  SpMat A(n, n);
  for (int i = 0; i < n; ++i) {
    if (i == 17) continue;  // leave column 17 (and row 17) empty
    A.insert(i, i) = 2.0 + 0.1 * i;
  }
  A.makeCompressed();

  std::vector<int> match;
  std::vector<double> rs, cs;
  const bool perfect = slu::mc64Matching(A, match, rs, cs);
  checkTrue(!perfect, "structurally singular matrix reports an imperfect matching");

  std::vector<int> seen(static_cast<std::size_t>(n), 0);
  bool valid = true;
  for (int j = 0; j < n; ++j) {
    if (match[static_cast<std::size_t>(j)] < 0 || match[static_cast<std::size_t>(j)] >= n ||
        seen[static_cast<std::size_t>(match[static_cast<std::size_t>(j)])])
      valid = false;
    else
      seen[static_cast<std::size_t>(match[static_cast<std::size_t>(j)])] = 1;
  }
  checkTrue(valid, "structurally singular matrix still yields a valid permutation");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("MC64 maximum-product matching\n");

  std::printf("Optimality (brute-force oracle):\n");
  testOptimalityAgainstBruteForce();
  std::printf("Never worse than the identity:\n");
  testNeverWorseThanIdentity();
  std::printf("Degenerate input:\n");
  testStructurallySingular();
  std::printf("Integration:\n");
  testIntegration();

  return lu_testing::summarize("MC64");
}
