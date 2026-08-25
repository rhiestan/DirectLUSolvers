// VTune profiling driver for the header-only METIS port (src/HeaderOnlyMetis/).
//
// profile_driver.cpp splits a SOLVER into analyze/factorize/solve; the ordering
// is only a slice of analyzePattern there and cannot be attributed on its own.
// This driver profiles the ordering itself, and -- when built with METIS --
// times it head to head against the linked C library on the same graph, which
// is the number that actually matters: the port is only worth using if it is
// competitive with the thing it replaces.
//
//   cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON -DDLU_BUILD_PROFILE_DRIVER=ON
//   cmake --build build --target profile_header_only_metis
//   build/profile_header_only_metis --reps 3
//
// The symmetrized-graph construction is done ONCE per matrix and hoisted out of
// the timed region: it is Eigen-side code shared verbatim with
// Eigen::MetisOrdering, so it is not what this port can make faster, and
// leaving it in would dilute the measurement.
//
// With DLU_WITH_ITT=ON each call is wrapped in a VTune task marker ("port" /
// "reference"), so a single run can be split by Task Type in the GUI.

#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "HeaderOnlyMetis/NestedDissection.h"
#include "HeaderOnlyMetis/NestedDissectionParallel.h"
#include "SupernodalLUExecutor.h"
#include "testing/MatrixMarket.h"
#include "testing/MetisGraph.h"
#include "testing/TestData.h"
#include "testing/TestMatrices.h"

#ifdef HAVE_METIS
#include <metis.h>
#endif

#ifdef DLU_WITH_ITT
#include <ittnotify.h>
namespace {
__itt_domain* ittDomain() {
  static __itt_domain* d = __itt_domain_create("HeaderOnlyMetis");
  return d;
}
struct IttTask {
  explicit IttTask(const char* name) {
    __itt_string_handle* h = __itt_string_handle_create(name);
    __itt_task_begin(ittDomain(), __itt_null, __itt_null, h);
  }
  ~IttTask() { __itt_task_end(ittDomain()); }
};
}  // namespace
#define DLU_ITT_TASK(name) IttTask dlu_itt_scoped_task(name)
#else
#define DLU_ITT_TASK(name) ((void)0)
#endif

using Eigen::SparseMatrix;
using Clock = std::chrono::steady_clock;

namespace {

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Graph {
  std::string label;
  std::vector<int> xadj;
  std::vector<int> adjncy;
  int n = 0;
};

// Best-of-reps: the metric of interest is the algorithm's cost, and the
// minimum is the sample least polluted by scheduling noise on a shared laptop.
struct Timing {
  double portMs = 0.0;
  double refMs = 0.0;
  double parMs = 0.0;   // parallel path; 0 when not requested
  bool identical = true;
};

Timing profileOne(const Graph& g, int reps, bool wantPort, bool wantRef, int threads) {
  Timing t;
  t.portMs = 1e300;
  t.refMs = 1e300;
  t.parMs = 1e300;

  std::vector<int> permPort(g.n), ipermPort(g.n);
  std::vector<int> permRef(g.n), ipermRef(g.n);

  for (int r = 0; r < reps; ++r) {
    if (wantPort) {
      DLU_ITT_TASK("port");
      const auto t0 = Clock::now();
      header_only_metis::nodeND<int, float>(g.n, g.xadj.data(), g.adjncy.data(), nullptr, permPort.data(),
                                            ipermPort.data());
      t.portMs = std::min(t.portMs, ms(t0, Clock::now()));
    }
#ifdef HAVE_METIS
    if (wantRef) {
      DLU_ITT_TASK("reference");
      idx_t n = g.n;
      std::vector<idx_t> xadj(g.xadj.begin(), g.xadj.end());
      std::vector<idx_t> adjncy(g.adjncy.begin(), g.adjncy.end());
      std::vector<idx_t> p(g.n), ip(g.n);
      const auto t0 = Clock::now();
      METIS_NodeND(&n, xadj.data(), adjncy.data(), nullptr, nullptr, p.data(), ip.data());
      t.refMs = std::min(t.refMs, ms(t0, Clock::now()));
      for (int i = 0; i < g.n; ++i) permRef[i] = static_cast<int>(p[i]), ipermRef[i] = static_cast<int>(ip[i]);
    }
#endif
  }

#ifdef HAVE_METIS
  t.identical = (!wantPort || !wantRef) || (permPort == permRef && ipermPort == ipermRef);
#endif

  if (threads > 0) {
    const Eigen::supernodal_lu::StdThreadExecutor exec(static_cast<unsigned>(threads));
    std::vector<int> p(g.n), ip(g.n);
    for (int r = 0; r < reps; ++r) {
      DLU_ITT_TASK("parallel");
      const auto t0 = Clock::now();
      header_only_metis::nodeNDParallel<int, float>(g.n, g.xadj.data(), g.adjncy.data(), nullptr, p.data(),
                                                    ip.data(), exec);
      t.parMs = std::min(t.parMs, ms(t0, Clock::now()));
    }
  } else {
    t.parMs = 0.0;
  }
  return t;
}

std::vector<Graph> buildGraphs(std::size_t maxN, bool syntheticOnly) {
  std::vector<Graph> out;
  auto add = [&out, maxN](const std::string& label, const SparseMatrix<double>& A) {
    if (A.rows() != A.cols() || A.rows() == 0) return;
    if (static_cast<std::size_t>(A.rows()) > maxN) return;
    lu_testing::SymmetrizedGraph<int> g = lu_testing::buildSymmetrizedGraph<int>(A);
    Graph gr;
    gr.label = label;
    gr.n = static_cast<int>(g.xadj.size()) - 1;
    gr.xadj = std::move(g.xadj);
    gr.adjncy = std::move(g.adjncy);
    out.push_back(std::move(gr));
  };

  add("lap2d_200x200", lu_testing::laplacian2d(200, 200));
  add("lap3d_40x40x40", lu_testing::laplacian3d(40, 40, 40));

  // Reading the testdata matrices costs more wall time than ordering them, and
  // under a sampler that I/O buries the algorithm being profiled. --synthetic
  // keeps the generated graphs only, so a hotspot report is all ordering.
  if (syntheticOnly) return out;

  for (const lu_testing::BenchmarkMatrix& m : lu_testing::benchmarkMatrices()) {
    try {
      add(m.label, lu_testing::loadMatrixMarket(lu_testing::testdataPath(m.relative)));
    } catch (const std::exception&) {
      // matrix not present in this checkout -- skip it
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  int reps = 3;
  std::size_t maxN = 300000;
  std::string only;
  bool syntheticOnly = false;
  int threads = 0;   // >0 also times the parallel path
  bool wantPort = true, wantRef = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (a == "--max-n" && i + 1 < argc) maxN = static_cast<std::size_t>(std::atoll(argv[++i]));
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--synthetic") syntheticOnly = true;
    else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (a == "--side" && i + 1 < argc) {
      const std::string sd = argv[++i];
      if (sd == "port") wantRef = false;
      else if (sd == "ref") wantPort = false;
      else if (sd == "par") { wantPort = false; wantRef = false; }
      else if (sd != "both") { std::printf("unknown side '%s'\n", sd.c_str()); return 2; }
    } else {
      std::printf("usage: %s [--reps N] [--max-n N] [--only SUBSTRING] [--synthetic]\n"
                  "          [--side port|ref|both]\n", argv[0]);
      return 2;
    }
  }

  const std::vector<Graph> graphs = buildGraphs(maxN, syntheticOnly);
  if (threads > 0)
    std::printf("%-18s %9s %11s %10s %10s %10s %9s\n", "matrix", "n", "nnz", "exact_ms", "par_ms", "ref_ms",
                "speedup");
  else
    std::printf("%-18s %9s %11s %10s %10s %8s %s\n", "matrix", "n", "nnz", "port_ms", "ref_ms", "ratio", "identical");

  double portTotal = 0.0, refTotal = 0.0, parTotal = 0.0;
  bool allIdentical = true;
  for (const Graph& g : graphs) {
    if (!only.empty() && g.label.find(only) == std::string::npos) continue;
    const Timing t = profileOne(g, reps, wantPort, wantRef, threads);
    portTotal += t.portMs;
    refTotal += t.refMs;
    parTotal += t.parMs;
    allIdentical = allIdentical && t.identical;
    if (threads > 0)
      std::printf("%-18s %9d %11zu %10.2f %10.2f %10.2f %8.2fx\n", g.label.c_str(), g.n, g.adjncy.size(),
                  t.portMs, t.parMs, t.refMs, t.parMs > 0 ? t.portMs / t.parMs : 0.0);
    else
      std::printf("%-18s %9d %11zu %10.2f %10.2f %8.2fx %s\n", g.label.c_str(), g.n, g.adjncy.size(), t.portMs,
                  t.refMs, t.refMs > 0 ? t.portMs / t.refMs : 0.0, t.identical ? "yes" : "NO");
  }
  if (threads > 0)
    std::printf("%-18s %9s %11s %10.2f %10.2f %10.2f %8.2fx\n", "TOTAL", "", "", portTotal, parTotal, refTotal,
                parTotal > 0 ? portTotal / parTotal : 0.0);
  else
    std::printf("%-18s %9s %11s %10.2f %10.2f %8.2fx %s\n", "TOTAL", "", "", portTotal, refTotal,
                refTotal > 0 ? portTotal / refTotal : 0.0, allIdentical ? "yes" : "NO");
  return allIdentical ? 0 : 1;
}
