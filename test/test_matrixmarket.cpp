// Unit tests for the shared MatrixMarket reader (testing/MatrixMarket.h).
//
// Build + run via CTest (from the DirectLUSolvers directory):
//   cmake -S . -B build -G Ninja && cmake --build build
//   ctest --test-dir build -R test_matrixmarket --output-on-failure
//
// The reader is the one piece of test infrastructure whose bugs are silent: a
// misparsed matrix still factors, still solves, and still reports a small
// residual -- for the wrong system. So it gets its own tests, written against
// small files with hand-checkable contents.
//
// The skew-symmetric case is a REGRESSION test. The previous readers detected
// symmetry with `banner.find("symmetric")`, which also matches
// "skew-symmetric", so a skew file was mirrored as A(j,i) = +v instead of -v.

#include <Eigen/SparseCore>

#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "testing/Check.h"
#include "testing/MatrixMarket.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

using Eigen::SparseMatrix;
using lu_testing::check;
using lu_testing::checkTrue;

namespace {

namespace fs = std::filesystem;

// A scratch directory removed at process exit.
class ScratchDir {
 public:
  ScratchDir() {
    m_path = fs::temp_directory_path() / "dlu_mm_test";
    fs::remove_all(m_path);
    fs::create_directories(m_path);
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(m_path, ec);
  }
  std::string write(const std::string& name, const std::string& contents) const {
    const fs::path p = m_path / name;
    std::ofstream out(p);
    out << contents;
    out.close();
    return p.string();
  }

 private:
  fs::path m_path;
};

// Dense view, so expectations can be written out entry by entry.
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense(const Eigen::SparseMatrix<Scalar>& A) {
  return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(A);
}

void testGeneralCoordinate(const ScratchDir& dir) {
  const std::string path = dir.write("general.mtx",
                                     "%%MatrixMarket matrix coordinate real general\n"
                                     "% a comment line\n"
                                     "3 3 4\n"
                                     "1 1 1.0\n"
                                     "1 3 2.0\n"
                                     "3 1 3.0\n"
                                     "2 2 4.0\n");
  lu_testing::MatrixMarketHeader h;
  const SparseMatrix<double> A = lu_testing::loadMatrixMarket(path, &h);
  Eigen::MatrixXd expect(3, 3);
  expect << 1, 0, 2,
            0, 4, 0,
            3, 0, 0;
  const double diff = (dense(A) - expect).norm();
  check(diff == 0.0, "general coordinate: values", diff);
  checkTrue(h.symmetry == lu_testing::MMSymmetry::General && h.rows == 3 && h.cols == 3,
            "general coordinate: header");
}

void testSymmetric(const ScratchDir& dir) {
  const std::string path = dir.write("sym.mtx",
                                     "%%MatrixMarket matrix coordinate real symmetric\n"
                                     "3 3 3\n"
                                     "1 1 5.0\n"
                                     "2 1 7.0\n"
                                     "3 2 9.0\n");
  const SparseMatrix<double> A = lu_testing::loadMatrixMarket(path);
  Eigen::MatrixXd expect(3, 3);
  expect << 5, 7, 0,
            7, 0, 9,
            0, 9, 0;
  const double diff = (dense(A) - expect).norm();
  check(diff == 0.0, "symmetric: off-diagonals mirrored as +v", diff);
}

// REGRESSION: "skew-symmetric" contains the substring "symmetric". The old
// readers therefore mirrored with +v, producing a symmetric matrix where the
// file describes an antisymmetric one -- silently the wrong operator.
void testSkewSymmetric(const ScratchDir& dir) {
  const std::string path = dir.write("skew.mtx",
                                     "%%MatrixMarket matrix coordinate real skew-symmetric\n"
                                     "3 3 2\n"
                                     "2 1 7.0\n"
                                     "3 2 9.0\n");
  const SparseMatrix<double> A = lu_testing::loadMatrixMarket(path);
  Eigen::MatrixXd expect(3, 3);
  expect <<  0, -7,  0,
             7,  0, -9,
             0,  9,  0;
  const double diff = (dense(A) - expect).norm();
  check(diff == 0.0, "skew-symmetric: mirrored as -v (regression)", diff);

  // Guard the property directly too: A^T == -A.
  const Eigen::MatrixXd D = dense(A);
  const double antisym = (D.transpose() + D).norm();
  check(antisym == 0.0, "skew-symmetric: A^T == -A", antisym);
}

void testHermitian(const ScratchDir& dir) {
  const std::string path = dir.write("herm.mtx",
                                     "%%MatrixMarket matrix coordinate complex hermitian\n"
                                     "2 2 2\n"
                                     "1 1 4.0 0.0\n"
                                     "2 1 1.0 2.0\n");
  typedef std::complex<double> C;
  const SparseMatrix<C> A = lu_testing::loadMatrixMarketAs<C>(path);
  Eigen::MatrixXcd expect(2, 2);
  expect << C(4, 0), C(1, -2),
            C(1, 2), C(0, 0);
  const double diff = (dense(A) - expect).norm();
  check(diff == 0.0, "hermitian: mirrored as conj(v)", diff);

  // A reader that matches no symmetry keyword for "hermitian" reads only the
  // lower triangle -- assert the upper triangle really is populated.
  checkTrue(A.nonZeros() == 3, "hermitian: upper triangle present");
}

void testComplexIntoRealIsRejected(const ScratchDir& dir) {
  const std::string path = dir.write("cplx.mtx",
                                     "%%MatrixMarket matrix coordinate complex general\n"
                                     "2 2 1\n"
                                     "1 1 3.0 4.0\n");
  bool threw = false;
  try {
    (void)lu_testing::loadMatrixMarket(path);  // double Scalar
  } catch (const std::exception&) {
    threw = true;
  }
  checkTrue(threw, "complex file into a real Scalar is rejected");

  // ... and reads correctly into a complex Scalar.
  const SparseMatrix<std::complex<double>> A = lu_testing::loadMatrixMarketAs<std::complex<double>>(path);
  checkTrue(A.coeff(0, 0) == std::complex<double>(3, 4), "complex file into a complex Scalar");
}

void testPatternField(const ScratchDir& dir) {
  const std::string path = dir.write("pat.mtx",
                                     "%%MatrixMarket matrix coordinate pattern symmetric\n"
                                     "3 3 2\n"
                                     "2 1\n"
                                     "3 3\n");
  const SparseMatrix<double> A = lu_testing::loadMatrixMarket(path);
  Eigen::MatrixXd expect(3, 3);
  expect << 0, 1, 0,
            1, 0, 0,
            0, 0, 1;
  const double diff = (dense(A) - expect).norm();
  check(diff == 0.0, "pattern field: implied value 1", diff);
}

void testArrayFormat(const ScratchDir& dir) {
  // Array format is column-major. This is what testdata/*/spmatrix_b.mtx uses;
  // a coordinate-only reader would misread such a file.
  const std::string path = dir.write("arr.mtx",
                                     "%%MatrixMarket matrix array real general\n"
                                     "3 2\n"
                                     "1.0\n2.0\n3.0\n"
                                     "4.0\n5.0\n6.0\n");
  const Eigen::MatrixXd A = lu_testing::loadMatrixMarketDense(path);
  Eigen::MatrixXd expect(3, 2);
  expect << 1, 4,
            2, 5,
            3, 6;
  const double diff = (A - expect).norm();
  check(diff == 0.0, "array format: column-major order", diff);
}

void testMalformedIsRejected(const ScratchDir& dir) {
  const std::string notMM = dir.write("bad_banner.mtx", "1 2 3\n1 1 1.0\n");
  const std::string truncated = dir.write("truncated.mtx",
                                          "%%MatrixMarket matrix coordinate real general\n"
                                          "3 3 4\n"
                                          "1 1 1.0\n");
  const std::string outOfRange = dir.write("range.mtx",
                                           "%%MatrixMarket matrix coordinate real general\n"
                                           "2 2 1\n"
                                           "5 1 1.0\n");
  auto rejects = [](const std::string& p) {
    try {
      (void)lu_testing::loadMatrixMarket(p);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  checkTrue(rejects(notMM), "malformed: missing banner rejected");
  checkTrue(rejects(truncated), "malformed: truncated entry list rejected");
  checkTrue(rejects(outOfRange), "malformed: out-of-range index rejected");
  checkTrue(rejects("no_such_file_here.mtx"), "malformed: missing file rejected");
}

// Banner tokenizing must tolerate the extra whitespace real files contain
// (testdata/laoss_*/spmatrix.mtx has "coordinate  real") and mixed case.
void testBannerTolerance(const ScratchDir& dir) {
  const std::string path = dir.write("spacing.mtx",
                                     "%%MatrixMarket   MATRIX   coordinate  Real   General\n"
                                     "\n"
                                     "%comment\n"
                                     "2 2 1\n"
                                     "\n"
                                     "2 2 8.0\n");
  const SparseMatrix<double> A = lu_testing::loadMatrixMarket(path);
  checkTrue(A.coeff(1, 1) == 8.0 && A.nonZeros() == 1, "banner: extra spaces / case / blank lines");
}

// The pattern helpers the solvers depend on.
void testPatternHelpers() {
  const SparseMatrix<double> lap = lu_testing::laplacian2d(6, 5);
  checkTrue(lu_testing::patternIsSymmetric(lap), "patternIsSymmetric: true for a Laplacian");

  // Strictly upper-triangular entry -> pattern is not symmetric.
  SparseMatrix<double> A(3, 3);
  A.insert(0, 1) = 2.0;
  A.insert(0, 0) = 1.0;
  A.makeCompressed();
  checkTrue(!lu_testing::patternIsSymmetric(A), "patternIsSymmetric: false for an unsymmetric pattern");

  const SparseMatrix<double> S = lu_testing::symmetrizePattern(A);
  checkTrue(lu_testing::patternIsSymmetric(S), "symmetrizePattern: result has a symmetric pattern");
  // Symmetrization must not change the operator -- only add structural zeros.
  const double diff = (dense(S) - dense(A)).norm();
  check(diff == 0.0, "symmetrizePattern: values unchanged", diff);
  checkTrue(S.nonZeros() == A.nonZeros() + 1, "symmetrizePattern: adds exactly the mirror entries");
}

// Every matrix in the shared registry must load, be square, and be non-empty.
// Cheap, but it is the check that catches a corrupted or renamed testdata file
// before a solver suite reports a confusing numerical failure.
void testRegistryLoads() {
  for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices()) {
    const std::string path = lu_testing::testdataPath(m.relative);
    try {
      lu_testing::MatrixMarketHeader h;
      const SparseMatrix<double> A = lu_testing::loadMatrixMarket(path, &h);
      checkTrue(A.rows() == A.cols() && A.rows() > 0 && A.nonZeros() > 0,
                std::string("registry loads: ") + m.label);
    } catch (const std::exception& e) {
      lu_testing::fail(std::string("registry loads: ") + m.label + ": " + e.what());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("MatrixMarket reader + pattern helper tests\n");

  bool skipRegistry = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--no-testdata") skipRegistry = true;

  const ScratchDir dir;
  testGeneralCoordinate(dir);
  testSymmetric(dir);
  testSkewSymmetric(dir);
  testHermitian(dir);
  testComplexIntoRealIsRejected(dir);
  testPatternField(dir);
  testArrayFormat(dir);
  testMalformedIsRejected(dir);
  testBannerTolerance(dir);
  testPatternHelpers();

  if (!skipRegistry) {
    std::printf("Benchmark matrix registry (%s):\n", lu_testing::testdataDir().c_str());
    testRegistryLoads();
  }

  return lu_testing::summarize("MatrixMarket");
}
