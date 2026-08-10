// MatrixMarket reader shared by the DirectLUSolvers test suites.
//
// Replaces two divergent private copies (compare_testdata.cpp's loader and
// test_parallel_lu.cpp's terser one), and fixes three defects both had:
//
//   1. SYMMETRY WAS DETECTED BY SUBSTRING SEARCH. `banner.find("symmetric")`
//      also matches "skew-symmetric", so a skew file was mirrored as A(j,i)=+v
//      instead of -v -- silently the wrong matrix. "hermitian" matched nothing
//      at all, so only the lower triangle was read. Neither symmetry appears in
//      this project's curated testdata/, but both are common in the SuiteSparse
//      collection, so this bites as soon as the corpus is widened.
//   2. The `array` (dense) format was unhandled; the coordinate parser would
//      misread such a file rather than reject it. testdata/*/spmatrix_b.mtx are
//      array files.
//   3. A `complex` field was read as if real, consuming the real part and
//      dropping the imaginary one without a word.
//
// The banner is now tokenized properly and every field/symmetry combination is
// either handled or rejected with a message naming the file.

#ifndef DIRECTLUSOLVERS_TEST_TESTING_MATRIXMARKET_H
#define DIRECTLUSOLVERS_TEST_TESTING_MATRIXMARKET_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cctype>
#include <complex>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lu_testing {

enum class MMFormat { Coordinate, Array };
enum class MMField { Real, Integer, Complex, Pattern };
enum class MMSymmetry { General, Symmetric, SkewSymmetric, Hermitian };

struct MatrixMarketHeader {
  MMFormat format = MMFormat::Coordinate;
  MMField field = MMField::Real;
  MMSymmetry symmetry = MMSymmetry::General;
  Eigen::Index rows = 0;
  Eigen::Index cols = 0;
  long long entries = 0;  // declared nnz (coordinate) or rows*cols (array)
};

namespace detail {

// Read one scalar value from a stream. Specialized so a complex Scalar consumes
// MatrixMarket's two-column "re im" representation.
template <typename Scalar>
struct MMValue {
  static bool read(std::istream& s, Scalar& v) {
    double re;
    if (!(s >> re)) return false;
    v = static_cast<Scalar>(re);
    return true;
  }
  static bool acceptsComplexFile() { return false; }
};

template <typename Real>
struct MMValue<std::complex<Real>> {
  static bool read(std::istream& s, std::complex<Real>& v) {
    double re, im;
    if (!(s >> re >> im)) return false;
    v = std::complex<Real>(static_cast<Real>(re), static_cast<Real>(im));
    return true;
  }
  static bool acceptsComplexFile() { return true; }
};

inline std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Next line that is neither empty nor a comment ('%'). Returns false at EOF.
inline bool nextDataLine(std::istream& in, std::string& line) {
  while (std::getline(in, line)) {
    // tolerate CRLF files read in text mode on a POSIX-ish toolchain
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos) continue;  // blank
    if (line[first] == '%') continue;          // comment
    return true;
  }
  return false;
}

// Parse "%%MatrixMarket matrix <format> <field> <symmetry>". Missing trailing
// tokens fall back to the MatrixMarket defaults (real / general).
inline MatrixMarketHeader parseBanner(const std::string& bannerLine, const std::string& path) {
  MatrixMarketHeader h;
  std::istringstream iss(toLower(bannerLine));
  std::vector<std::string> tok;
  for (std::string t; iss >> t;) tok.push_back(t);

  if (tok.size() < 2 || tok[0] != "%%matrixmarket")
    throw std::runtime_error("not a MatrixMarket file (bad banner): " + path);
  if (tok[1] != "matrix") throw std::runtime_error("unsupported MatrixMarket object '" + tok[1] + "': " + path);

  if (tok.size() > 2) {
    if (tok[2] == "coordinate") h.format = MMFormat::Coordinate;
    else if (tok[2] == "array") h.format = MMFormat::Array;
    else throw std::runtime_error("unsupported MatrixMarket format '" + tok[2] + "': " + path);
  }
  if (tok.size() > 3) {
    if (tok[3] == "real" || tok[3] == "double") h.field = MMField::Real;
    else if (tok[3] == "integer") h.field = MMField::Integer;
    else if (tok[3] == "complex") h.field = MMField::Complex;
    else if (tok[3] == "pattern") h.field = MMField::Pattern;
    else throw std::runtime_error("unsupported MatrixMarket field '" + tok[3] + "': " + path);
  }
  if (tok.size() > 4) {
    // Exact token comparison, NOT substring: "skew-symmetric" must not be read
    // as "symmetric", which would mirror it with the wrong sign.
    if (tok[4] == "general") h.symmetry = MMSymmetry::General;
    else if (tok[4] == "symmetric") h.symmetry = MMSymmetry::Symmetric;
    else if (tok[4] == "skew-symmetric" || tok[4] == "skew") h.symmetry = MMSymmetry::SkewSymmetric;
    else if (tok[4] == "hermitian") h.symmetry = MMSymmetry::Hermitian;
    else throw std::runtime_error("unsupported MatrixMarket symmetry '" + tok[4] + "': " + path);
  }
  return h;
}

// The value stored at the mirrored position (j,i) given the value at (i,j).
template <typename Scalar>
Scalar mirroredValue(const Scalar& v, MMSymmetry symmetry) {
  switch (symmetry) {
    case MMSymmetry::Symmetric: return v;
    case MMSymmetry::SkewSymmetric: return -v;
    case MMSymmetry::Hermitian: return Eigen::numext::conj(v);
    case MMSymmetry::General: break;
  }
  return v;
}

}  // namespace detail

// Read a MatrixMarket file into a sparse matrix. `header`, if non-null, receives
// the parsed banner and dimensions.
//
// Handled: coordinate and array formats; real / integer / pattern fields (any
// Scalar) and complex fields (complex Scalar only); general, symmetric,
// skew-symmetric and hermitian symmetries. Off-diagonal entries of a
// non-general file are mirrored with the sign/conjugation that symmetry implies.
template <typename Scalar>
Eigen::SparseMatrix<Scalar> loadMatrixMarketAs(const std::string& path,
                                               MatrixMarketHeader* header = nullptr) {
  typedef Eigen::SparseMatrix<Scalar> SpMat;
  typedef Eigen::Triplet<Scalar, typename SpMat::StorageIndex> Trip;

  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty file " + path);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  MatrixMarketHeader h = detail::parseBanner(line, path);

  if (h.field == MMField::Complex && !detail::MMValue<Scalar>::acceptsComplexFile())
    throw std::runtime_error("complex MatrixMarket file needs a complex Scalar: " + path);

  if (!detail::nextDataLine(in, line)) throw std::runtime_error("unexpected EOF (no size line) in " + path);

  long long rows = 0, cols = 0, declared = 0;
  {
    std::istringstream iss(line);
    if (!(iss >> rows >> cols)) throw std::runtime_error("malformed size line in " + path);
    if (h.format == MMFormat::Coordinate && !(iss >> declared))
      throw std::runtime_error("malformed size line (missing nnz) in " + path);
  }
  h.rows = static_cast<Eigen::Index>(rows);
  h.cols = static_cast<Eigen::Index>(cols);

  const bool mirror = (h.symmetry != MMSymmetry::General);
  std::vector<Trip> triplets;

  if (h.format == MMFormat::Coordinate) {
    h.entries = declared;
    triplets.reserve(static_cast<std::size_t>(mirror ? 2 * declared : declared));
    for (long long k = 0; k < declared; ++k) {
      if (!detail::nextDataLine(in, line))
        throw std::runtime_error("unexpected EOF reading entries in " + path);
      std::istringstream iss(line);
      long long i = 0, j = 0;
      if (!(iss >> i >> j)) throw std::runtime_error("malformed entry in " + path);
      Scalar v = Scalar(1);
      if (h.field != MMField::Pattern && !detail::MMValue<Scalar>::read(iss, v))
        throw std::runtime_error("malformed value in " + path);
      --i;
      --j;  // MatrixMarket indices are 1-based
      if (i < 0 || j < 0 || i >= rows || j >= cols)
        throw std::runtime_error("entry out of range in " + path);
      triplets.emplace_back(static_cast<typename SpMat::StorageIndex>(i),
                            static_cast<typename SpMat::StorageIndex>(j), v);
      if (mirror && i != j)
        triplets.emplace_back(static_cast<typename SpMat::StorageIndex>(j),
                              static_cast<typename SpMat::StorageIndex>(i),
                              detail::mirroredValue(v, h.symmetry));
    }
  } else {
    // Array (dense) format: values in column-major order, one per line. A
    // non-general array file stores only the lower triangle, packed.
    h.entries = rows * cols;
    triplets.reserve(static_cast<std::size_t>(rows * cols));
    auto readInto = [&](long long i, long long j) {
      if (!detail::nextDataLine(in, line))
        throw std::runtime_error("unexpected EOF reading array values in " + path);
      std::istringstream iss(line);
      Scalar v = Scalar(1);
      if (h.field != MMField::Pattern && !detail::MMValue<Scalar>::read(iss, v))
        throw std::runtime_error("malformed value in " + path);
      if (v == Scalar(0)) return;  // keep the sparse result sparse
      triplets.emplace_back(static_cast<typename SpMat::StorageIndex>(i),
                            static_cast<typename SpMat::StorageIndex>(j), v);
      if (mirror && i != j)
        triplets.emplace_back(static_cast<typename SpMat::StorageIndex>(j),
                              static_cast<typename SpMat::StorageIndex>(i),
                              detail::mirroredValue(v, h.symmetry));
    };
    for (long long j = 0; j < cols; ++j)
      for (long long i = (mirror ? j : 0); i < rows; ++i) readInto(i, j);
  }

  SpMat A(h.rows, h.cols);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  if (header) *header = h;
  return A;
}

// Convenience overload for the common double case.
inline Eigen::SparseMatrix<double> loadMatrixMarket(const std::string& path,
                                                    MatrixMarketHeader* header = nullptr) {
  return loadMatrixMarketAs<double>(path, header);
}

// Read a MatrixMarket file as a dense matrix -- for the right-hand-side files
// (testdata/*/spmatrix_b.mtx), which are `array real general`.
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> loadMatrixMarketDenseAs(
    const std::string& path, MatrixMarketHeader* header = nullptr) {
  MatrixMarketHeader h;
  const Eigen::SparseMatrix<Scalar> S = loadMatrixMarketAs<Scalar>(path, &h);
  if (header) *header = h;
  return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(S);
}

inline Eigen::MatrixXd loadMatrixMarketDense(const std::string& path,
                                             MatrixMarketHeader* header = nullptr) {
  return loadMatrixMarketDenseAs<double>(path, header);
}

}  // namespace lu_testing

#endif  // DIRECTLUSOLVERS_TEST_TESTING_MATRIXMARKET_H
