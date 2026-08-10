// The benchmark-matrix registry shared by the DirectLUSolvers test suites.
//
// One place for both the location of "testdata/<x>.mtx" (baked in at configure
// time, env-overridable, so no suite has to assume a working directory) and the
// list itself, so a matrix added here reaches every suite at once.
//
// The `Tier` classification is what lets CTest run a fast subset: Small matrices
// factor in well under a second each; Large ones take seconds to minutes; Huge
// ones are the pathological cases these solvers deliberately decline.

#ifndef DIRECTLUSOLVERS_TEST_TESTING_TESTDATA_H
#define DIRECTLUSOLVERS_TEST_TESTING_TESTDATA_H

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef DLU_TESTDATA_DIR
// Fallback for a hand-compiled build without CMake: assume the repository root
// is the working directory.
#define DLU_TESTDATA_DIR "testdata"
#endif

#ifndef DLU_MATRIX_DIR
// Where fetch_suitesparse.py keeps the manifest and its cache/ directory.
#define DLU_MATRIX_DIR "DirectLUSolvers/test/matrices"
#endif

namespace lu_testing {

// Where the MatrixMarket benchmark matrices live. Baked in by CMake; the
// DLU_TESTDATA_DIR environment variable overrides it at run time.
inline std::string testdataDir() {
  if (const char* env = std::getenv("DLU_TESTDATA_DIR")) {
    if (env[0] != '\0') return std::string(env);
  }
  return std::string(DLU_TESTDATA_DIR);
}

inline std::string testdataPath(const std::string& relative) {
  return testdataDir() + "/" + relative;
}

enum class Tier {
  Small,  // < ~50k rows: seconds total, safe for every suite
  Large,  // the big 3D FEM systems: seconds to a minute each
  Huge,   // n >> 100k with no good separators: these solvers decline by design
};

struct BenchmarkMatrix {
  const char* label;     // short name, also the baseline key
  const char* relative;  // path under testdataDir()
  Tier tier;
  const char* note;
};

// The curated corpus. `relative` carries the real filename because several
// directories name their matrix spmatrix.mtx rather than <dir>.mtx.
inline const std::vector<BenchmarkMatrix>& benchmarkMatrices() {
  static const std::vector<BenchmarkMatrix> matrices = {
      {"bcsstm13", "bcsstm13/bcsstm13.mtx", Tier::Small,
       "singular mass matrix; Eigen::SparseLU fails outright on it"},
      {"dendrimer", "dendrimer/dendrimer.mtx", Tier::Small, "symmetric, well behaved"},
      {"gemat11", "gemat11/gemat11.mtx", Tier::Small, "unsymmetric power-network matrix"},
      {"rdb2048_noL", "rdb2048_noL/rdb2048_noL.mtx", Tier::Small, "unsymmetric, reaction-diffusion"},
      // Near-tridiagonal: bandwidth 2, ~3 nonzeros per row, full diagonal, and
      // only 10 of 3049 entries lack their transpose. The one matrix here with a
      // CHAIN elimination tree, which is why it is worth keeping: AMD amalgamates
      // that chain into 9 dense blocks (43x fill), while METIS keeps it sparse
      // (2.7x). See the amalgamation note in README.md.
      {"setfos", "setfos/spmatrix.mtx", Tier::Small, "near-tridiagonal; chain elimination tree"},
      // setfos's opposite, and the corpus's stress case for a symmetric-pattern
      // solver: 7.8% dense, and only 44% of entries have their transpose, so
      // symmetrizing adds 56% more nonzeros before factorization even starts.
      // Dense columns (median 258 nnz) over sparse rows (median 9). Solves
      // accurately either way, but the ORDERING decides whether the symmetric
      // pattern costs anything: AMD gives 3.93M fill (2x Eigen::SparseLU's
      // 1.94M), METIS 1.63M (below it). The one matrix here that shows that gap
      // is an ordering choice, not an inherent cost of forcing symmetry.
      {"setfos_2", "setfos_2/spmatrix.mtx", Tier::Small, "dense-ish, strongly pattern-unsymmetric"},
      {"sherman1", "sherman1/sherman1.mtx", Tier::Small, "oil reservoir simulation"},
      {"tomography", "tomography/tomography.mtx", Tier::Small, "well-conditioned"},
      {"YaleB_10NN", "YaleB_10NN/YaleB_10NN.mtx", Tier::Small, "10-nearest-neighbour graph"},
      {"bayer05", "bayer05/bayer05.mtx", Tier::Small, "near-singular pathological chemical-process matrix"},
      {"laoss_3", "laoss_3/spmatrix.mtx", Tier::Small, "4180-row 3D FEM"},
      {"laoss_2", "laoss_2/spmatrix.mtx", Tier::Large, "100k-row 3D FEM"},
      {"laoss_1", "laoss_1/spmatrix.mtx", Tier::Large, "251k-row 3D FEM"},
      {"pre2", "pre2/pre2.mtx", Tier::Huge,
       "659k circuit matrix; symmetric-pattern fill is ruinous -- expected to be declined"},
  };
  return matrices;
}

// The matrices at or below `maxTier`, as full paths.
inline std::vector<std::string> benchmarkPaths(Tier maxTier) {
  std::vector<std::string> paths;
  for (const BenchmarkMatrix& m : benchmarkMatrices())
    if (static_cast<int>(m.tier) <= static_cast<int>(maxTier))
      paths.push_back(testdataPath(m.relative));
  return paths;
}

// ---------------------------------------------------------------------------
//  SuiteSparse corpus (optional, downloaded on demand)
// ---------------------------------------------------------------------------
//
// test/matrices/suitesparse.manifest is a checked-in, human-curated list; the
// .mtx files it names are fetched by fetch_suitesparse.py into a git-ignored
// cache/. The manifest is read at RUN time, not baked in at configure time, so
// adopting a matrix is a one-line edit with no rebuild.
//
// Entries whose file has not been downloaded are still returned, with
// `available == false`, so a suite can report "3 of 26 present, run the fetch
// script" rather than silently testing less than the reader assumes.

struct SuiteSparseMatrix {
  std::string group;
  std::string name;
  Tier tier = Tier::Small;
  long long n = 0;
  long long nnz = 0;
  double patternSymmetry = 0.0;  // 1.00 == already symmetric, < 0.5 == stress case
  bool positiveDefinite = false;
  std::string kind;
  std::string path;        // absolute path to the .mtx
  bool available = false;  // has it actually been downloaded?

  std::string label() const { return group + "/" + name; }
};

inline std::string matrixDir() {
  if (const char* env = std::getenv("DLU_MATRIX_DIR")) {
    if (env[0] != '\0') return std::string(env);
  }
  return std::string(DLU_MATRIX_DIR);
}

inline std::string suitesparseManifestPath() {
  return matrixDir() + "/suitesparse.manifest";
}

// Parse the manifest. Returns an empty vector when it cannot be read, which the
// callers treat as "corpus not set up" rather than as an error.
inline std::vector<SuiteSparseMatrix> suitesparseMatrices() {
  std::vector<SuiteSparseMatrix> out;
  std::ifstream in(suitesparseManifestPath());
  if (!in) return out;

  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') continue;

    std::istringstream iss(line);
    SuiteSparseMatrix m;
    std::string tier, spd;
    if (!(iss >> m.group >> m.name >> tier >> m.n >> m.nnz >> m.patternSymmetry >> spd))
      continue;
    std::getline(iss, m.kind);
    if (!m.kind.empty() && m.kind[0] == ' ') m.kind.erase(0, 1);

    m.tier = (tier == "large") ? Tier::Huge : (tier == "standard" ? Tier::Large : Tier::Small);
    m.positiveDefinite = (spd == "y");
    // The tarball expands to <Name>/<Name>.mtx under cache/<Group>/.
    m.path = matrixDir() + "/cache/" + m.group + "/" + m.name + "/" + m.name + ".mtx";
    std::ifstream probe(m.path);
    m.available = static_cast<bool>(probe);
    out.push_back(m);
  }
  return out;
}

// Short label for a matrix path: its parent directory name (so
// "…/setfos/spmatrix.mtx" reads as "setfos"). Falls back to the stem.
inline std::string matrixLabel(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::string dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash);
  const std::size_t slash2 = dir.find_last_of("/\\");
  const std::string parent = (slash2 == std::string::npos) ? dir : dir.substr(slash2 + 1);
  if (!parent.empty()) return parent;
  const std::string file = (slash == std::string::npos) ? path : path.substr(slash + 1);
  const std::size_t dot = file.find_last_of('.');
  return dot == std::string::npos ? file : file.substr(0, dot);
}

}  // namespace lu_testing

#endif  // DIRECTLUSOLVERS_TEST_TESTING_TESTDATA_H
