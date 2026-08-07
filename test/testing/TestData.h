// The benchmark-matrix registry shared by the DirectLUSolvers test suites.
//
// Three suites previously each hard-coded their own list of "testdata/<x>.mtx"
// paths, relative to a working directory they silently assumed. This centralizes
// both the location (baked in at configure time, env-overridable) and the list,
// so a matrix added here reaches every suite at once.
//
// The `Tier` classification is what lets CTest run a fast subset: Small matrices
// factor in well under a second each; Large ones take seconds to minutes; Huge
// ones are the pathological cases these solvers deliberately decline.

#ifndef DIRECTLUSOLVERS_TEST_TESTING_TESTDATA_H
#define DIRECTLUSOLVERS_TEST_TESTING_TESTDATA_H

#include <cstdlib>
#include <string>
#include <vector>

#ifndef DLU_TESTDATA_DIR
// Fallback for a hand-compiled build without CMake: assume the repository root
// is the working directory.
#define DLU_TESTDATA_DIR "testdata"
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
      // NOTE: this file is only 2x2 with 4 nonzeros -- it has been in the
      // benchmark list since the start but exercises essentially nothing.
      // Probably a truncated or placeholder export; worth replacing.
      {"setfos", "setfos/spmatrix.mtx", Tier::Small, "2x2 (!) -- degenerate, see comment"},
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
