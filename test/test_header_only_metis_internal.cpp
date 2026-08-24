// White-box tests for the header-only METIS port: compare each module's
// internal behaviour against the reference METIS/GKlib implementation
// directly, not just the final METIS_NodeND perm/iperm (see
// test_header_only_metis.cpp for that top-level oracle). A single early
// tie-break divergence inside the multilevel recursion cascades into a
// completely different separator tree with no useful diff at the top level,
// so each module needs to be locked in here BEFORE the next one is built on
// top of it -- see the Phase 2 plan in DirectLUSolvers/src/HeaderOnlyMetis/.
//
// Build + run via CTest (from the DirectLUSolvers directory), requires METIS:
//   cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON && cmake --build build
//   ctest --test-dir build -R test_header_only_metis_internal --output-on-failure

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "testing/Check.h"

#ifdef HAVE_METIS
#include <metis.h>

#include "HeaderOnlyMetis/Compress.h"
#include "HeaderOnlyMetis/Graph.h"
#include "HeaderOnlyMetis/MinimumDegree.h"
#include "HeaderOnlyMetis/Random.h"
#include "HeaderOnlyMetis/Sorting.h"
#endif

using lu_testing::checkTrue;
using lu_testing::note;

namespace {

#ifdef HAVE_METIS

// GK_MKRANDOM(i, idx_t, idx_t) in libmetis/gklib.c generates these under their
// plain names, but libmetis/gklib_rename.h (NOT libmetis/rename.h -- a
// separate file, easy to miss) prefixes them like everything else in
// libmetis: isrand -> libmetis__isrand, etc. Confirmed against the actual
// compiled metis.lib via `dumpbin /symbols`.
extern "C" {
void libmetis__isrand(idx_t seed);
idx_t libmetis__irand(void);
idx_t libmetis__irandInRange(idx_t max);
void libmetis__irandArrayPermute(idx_t n, idx_t* p, idx_t nshuffles, int flag);
void libmetis__irandArrayPermuteFine(idx_t n, idx_t* p, int flag);
}
inline void isrand(idx_t seed) { libmetis__isrand(seed); }
inline idx_t irand() { return libmetis__irand(); }
inline idx_t irandInRange(idx_t max) { return libmetis__irandInRange(max); }
inline void irandArrayPermute(idx_t n, idx_t* p, idx_t nshuffles, int flag) {
  libmetis__irandArrayPermute(n, p, nshuffles, flag);
}
inline void irandArrayPermuteFine(idx_t n, idx_t* p, int flag) {
  libmetis__irandArrayPermuteFine(n, p, flag);
}

void checkRandSequence(idx_t seed, int count) {
  const std::string name = "Random: rand() seed=" + std::to_string(seed);
  isrand(seed);
  header_only_metis::randSeed<idx_t>(seed);

  bool allMatch = true;
  for (int i = 0; i < count; ++i) {
    const idx_t ref = irand();
    const idx_t port = header_only_metis::randNext<idx_t>();
    if (ref != port) {
      allMatch = false;
      break;
    }
  }
  checkTrue(allMatch, name + " (" + std::to_string(count) + " draws)");
}

void checkRandInRange(idx_t seed, idx_t max, int count) {
  const std::string name =
      "Random: randInRange() seed=" + std::to_string(seed) + " max=" + std::to_string(max);
  isrand(seed);
  header_only_metis::randSeed<idx_t>(seed);

  bool allMatch = true;
  bool allInRange = true;
  for (int i = 0; i < count; ++i) {
    const idx_t ref = irandInRange(max);
    const idx_t port = header_only_metis::randInRange<idx_t>(max);
    if (ref != port) allMatch = false;
    if (port < 0 || port >= max) allInRange = false;
  }
  checkTrue(allMatch, name + " (" + std::to_string(count) + " draws)");
  checkTrue(allInRange, name + ": port stays in [0,max)");
}

void checkArrayPermute(idx_t seed, idx_t n, idx_t nshuffles) {
  const std::string name = "Random: randArrayPermute() seed=" + std::to_string(seed) +
                           " n=" + std::to_string(n) + " nshuffles=" + std::to_string(nshuffles);

  std::vector<idx_t> ref(static_cast<std::size_t>(n));
  std::vector<idx_t> port(static_cast<std::size_t>(n));

  isrand(seed);
  irandArrayPermute(n, ref.data(), nshuffles, 1);

  header_only_metis::randSeed<idx_t>(seed);
  header_only_metis::randArrayPermute<idx_t>(n, port.data(), nshuffles, 1);

  checkTrue(ref == port, name);
}

void checkArrayPermuteFine(idx_t seed, idx_t n) {
  const std::string name =
      "Random: randArrayPermuteFine() seed=" + std::to_string(seed) + " n=" + std::to_string(n);

  std::vector<idx_t> ref(static_cast<std::size_t>(n));
  std::vector<idx_t> port(static_cast<std::size_t>(n));

  isrand(seed);
  irandArrayPermuteFine(n, ref.data(), 1);

  header_only_metis::randSeed<idx_t>(seed);
  header_only_metis::randArrayPermuteFine<idx_t>(n, port.data(), 1);

  checkTrue(ref == port, name);
}

void checkRandomModule() {
  // 4321 is METIS's actual default seed (options=NULL -> ctrl->seed=-1 ->
  // InitRandom maps -1 to 4321, libmetis/util.c). The others exercise seed
  // values a real caller might pass explicitly via METIS_OPTION_SEED.
  const idx_t seeds[] = {4321, 0, 1, 42, 123456789};
  for (idx_t seed : seeds) checkRandSequence(seed, 2000);

  const idx_t ranges[] = {1, 2, 3, 7, 1000, 1000000};
  for (idx_t seed : seeds)
    for (idx_t max : ranges) checkRandInRange(seed, max, 500);

  // n<10 takes the "full random swap" branch; n>=10 takes the 4-way blocked
  // swap branch -- both must be checked, straddling the n=10 boundary.
  const idx_t sizes[] = {1, 2, 5, 9, 10, 11, 20, 37, 100, 577};
  for (idx_t seed : seeds) {
    for (idx_t n : sizes) {
      checkArrayPermute(seed, n, n < 10 ? n : n * 2);  // nshuffles unused on the n<10 path
      checkArrayPermuteFine(seed, n);
    }
  }
}

// libmetis/rename.h prefixes the whole mmd.c family, confirmed the same way
// as Random's isrand/irand above.
extern "C" void libmetis__genmmd(idx_t neqns, idx_t* xadj, idx_t* adjncy, idx_t* invp, idx_t* perm,
                                 idx_t delta, idx_t* head, idx_t* qsize, idx_t* list, idx_t* marker,
                                 idx_t maxint, idx_t* ncsub);

// Runs both genmmds on the same graph and compares perm/invp/ncsub. genmmd
// expects xadj/adjncy already shifted to 1-based VALUES (every stored index
// incremented by one) -- exactly what MMDOrder (ometis.c) does before calling
// it -- and destroys adjncy in place, so each side gets its own copy.
void checkGenmmd(const std::string& name, idx_t neqns, std::vector<idx_t> xadj,
                 std::vector<idx_t> adjncy) {
  if (neqns <= 0) return;

  const idx_t nz = xadj[static_cast<std::size_t>(neqns)];
  for (idx_t i = 0; i < nz; ++i) adjncy[static_cast<std::size_t>(i)]++;
  for (idx_t i = 0; i <= neqns; ++i) xadj[static_cast<std::size_t>(i)]++;

  auto ws = [&] { return std::vector<idx_t>(static_cast<std::size_t>(neqns) + 5, 0); };

  std::vector<idx_t> xadjRef = xadj, adjRef = adjncy;
  std::vector<idx_t> invpRef = ws(), permRef = ws(), headRef = ws(), qsizeRef = ws(), listRef = ws(),
                     markerRef = ws();
  idx_t ncsubRef = 0;
  libmetis__genmmd(neqns, xadjRef.data(), adjRef.data(), invpRef.data(), permRef.data(), idx_t(1),
                   headRef.data(), qsizeRef.data(), listRef.data(), markerRef.data(), IDX_MAX,
                   &ncsubRef);

  std::vector<idx_t> xadjPort = xadj, adjPort = adjncy;
  std::vector<idx_t> invpPort = ws(), permPort = ws(), headPort = ws(), qsizePort = ws(),
                     listPort = ws(), markerPort = ws();
  idx_t ncsubPort = 0;
  header_only_metis::genmmd<idx_t>(neqns, xadjPort.data(), adjPort.data(), invpPort.data(),
                                   permPort.data(), idx_t(1), headPort.data(), qsizePort.data(),
                                   listPort.data(), markerPort.data(), IDX_MAX, &ncsubPort);

  // Only indices [0, neqns) are meaningful (node k's result lands at index
  // k-1 after genmmd's internal 1-based-indexing adjustment); the rest of the
  // neqns+5 workspace is scratch. This is exactly the range MMDOrder itself
  // reads (ometis.c: `order[label[i]] = firstvtx+iperm[i]-1` for i<nvtxs).
  const std::vector<idx_t> permRefUsed(permRef.begin(), permRef.begin() + neqns);
  const std::vector<idx_t> permPortUsed(permPort.begin(), permPort.begin() + neqns);
  const std::vector<idx_t> invpRefUsed(invpRef.begin(), invpRef.begin() + neqns);
  const std::vector<idx_t> invpPortUsed(invpPort.begin(), invpPort.begin() + neqns);

  checkTrue(permRefUsed == permPortUsed, name + ": genmmd perm matches");
  checkTrue(invpRefUsed == invpPortUsed, name + ": genmmd invp matches");
  checkTrue(ncsubRef == ncsubPort, name + ": genmmd ncsub matches");
}

// Deterministic Erdos-Renyi-style graph (test-only RNG, unrelated to
// header_only_metis's own generator) for structural variety beyond the fixed
// hand-built shapes.
std::vector<idx_t> randomGraphXadj(idx_t n, double density, unsigned seed, std::vector<idx_t>& adjncy) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::vector<idx_t>> adj(static_cast<std::size_t>(n));
  for (idx_t i = 0; i < n; ++i)
    for (idx_t j = i + 1; j < n; ++j)
      if (u(rng) < density) {
        adj[static_cast<std::size_t>(i)].push_back(j);
        adj[static_cast<std::size_t>(j)].push_back(i);
      }
  std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
  adjncy.clear();
  for (idx_t i = 0; i < n; ++i) {
    xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
    for (idx_t v : adj[static_cast<std::size_t>(i)]) adjncy.push_back(v);
  }
  xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
  return xadj;
}

void checkMmdModule() {
  // Path graph: worst case for MMD's supernode-merging logic (every node has
  // degree <=2, so ties are common).
  {
    const idx_t n = 80;
    std::vector<idx_t> adjncy;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    for (idx_t i = 0; i < n; ++i) {
      xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
      if (i > 0) adjncy.push_back(i - 1);
      if (i + 1 < n) adjncy.push_back(i + 1);
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
    checkGenmmd("mmd: path (n=80)", n, xadj, adjncy);
  }
  // Star graph: one hub, extreme degree imbalance.
  {
    const idx_t n = 80;
    std::vector<idx_t> adjncy;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    for (idx_t i = 0; i < n; ++i) {
      xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
      if (i == 0) {
        for (idx_t k = 1; k < n; ++k) adjncy.push_back(k);
      } else {
        adjncy.push_back(0);
      }
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
    checkGenmmd("mmd: star (n=80)", n, xadj, adjncy);
  }
  // Complete graph: every node merges into one supernode almost immediately.
  {
    const idx_t n = 40;
    std::vector<idx_t> adjncy;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    for (idx_t i = 0; i < n; ++i) {
      xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
      for (idx_t k = 0; k < n; ++k)
        if (k != i) adjncy.push_back(k);
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<idx_t>(adjncy.size());
    checkGenmmd("mmd: complete K40", n, xadj, adjncy);
  }
  // Fully disconnected: every node isolated, all eliminated in the
  // "isolated nodes" fast path before the main loop even starts.
  {
    const idx_t n = 50;
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    std::vector<idx_t> adjncy;
    checkGenmmd("mmd: fully disconnected (n=50)", n, xadj, adjncy);
  }
  // n=1 edge case.
  {
    std::vector<idx_t> xadj = {0, 0};
    std::vector<idx_t> adjncy;
    checkGenmmd("mmd: single isolated vertex", idx_t(1), xadj, adjncy);
  }
  // Random graphs at several sizes and densities, including ones that
  // straddle MMDSWITCH=120 even though genmmd itself doesn't know about that
  // threshold -- it's tested here purely as a standalone algorithm.
  const idx_t sizes[] = {5, 20, 50, 119, 120, 121, 250, 500};
  const double densities[] = {0.02, 0.1, 0.3};
  unsigned seed = 1;
  for (idx_t n : sizes) {
    for (double density : densities) {
      std::vector<idx_t> adjncy;
      std::vector<idx_t> xadj = randomGraphXadj(n, density, seed++, adjncy);
      const std::string name = "mmd: random n=" + std::to_string(n) +
                               " density=" + std::to_string(density);
      checkGenmmd(name, n, xadj, adjncy);
    }
  }
}

// BucketSortKeysInc's reference signature needs an opaque ctrl_t* (used only
// for its workspace arena, iwspacemalloc(ctrl, max+2)) that isn't available
// through the public metis.h -- metislib.h's internal struct/ctrl machinery
// isn't part of the installed interface this test links against. Rather than
// build that scaffolding for one 12-line counting sort, verify against
// std::stable_sort instead: BucketSortKeysInc stably sorts the SEQUENCE
// tperm[0..n) by keys[tperm[ii]] (each element is placed into the next free
// slot of its key's bucket, visited in tperm order), which has exactly one
// correct output -- so matching std::stable_sort on the same comparison is as
// strong a correctness proof as diffing against the compiled reference. The
// actual bit-identical behaviour *in situ*, called through a real ctrl_t,
// gets covered end-to-end once Coarsen.h's white-box tests (which do exercise
// the real call) land.
void checkBucketSortKeysInc(const std::string& name, idx_t n, idx_t max, const std::vector<idx_t>& keys,
                            const std::vector<idx_t>& tperm) {
  std::vector<idx_t> expected = tperm;
  std::stable_sort(expected.begin(), expected.end(),
                   [&](idx_t a, idx_t b) { return keys[static_cast<std::size_t>(a)] < keys[static_cast<std::size_t>(b)]; });

  std::vector<idx_t> perm(static_cast<std::size_t>(n));
  header_only_metis::bucketSortKeysInc<idx_t>(n, max, keys.data(), tperm.data(), perm.data());

  checkTrue(perm == expected, name);
}

void checkSortingModule() {
  std::mt19937 rng(7);
  const idx_t sizes[] = {0, 1, 5, 50, 500};
  const idx_t maxKeys[] = {1, 3, 20};
  for (idx_t n : sizes) {
    for (idx_t max : maxKeys) {
      std::uniform_int_distribution<int> keyDist(0, static_cast<int>(max));
      std::vector<idx_t> keys(static_cast<std::size_t>(n));
      for (idx_t& k : keys) k = static_cast<idx_t>(keyDist(rng));

      std::vector<idx_t> tperm(static_cast<std::size_t>(n));
      for (idx_t i = 0; i < n; ++i) tperm[static_cast<std::size_t>(i)] = i;
      std::shuffle(tperm.begin(), tperm.end(), rng);

      const std::string name =
          "bucketSortKeysInc: n=" + std::to_string(n) + " max=" + std::to_string(max);
      checkBucketSortKeysInc(name, n, max, keys, tperm);
    }
  }
}

// gklib_rename.h prefixes these too (confirmed via grep, same as the Random
// and MMD families above). ikv_t itself isn't in the public metis.h, but
// header_only_metis::IndexKeyValue<idx_t> has the identical (key, val) POD
// layout (GK_MKKEYVALUE_T in GKlib/include/gk_struct.h), so it's ABI-safe to
// use directly for this call.
extern "C" {
void libmetis__ikvsorti(std::size_t n, header_only_metis::IndexKeyValue<idx_t>* base);
void libmetis__isorti(std::size_t n, idx_t* base);
void libmetis__isortd(std::size_t n, idx_t* base);
}

// gkQsort is NOT stable, so the real point of this test is duplicate keys --
// a passing "produces a sorted array" check alone would not catch a
// tie-break divergence from the reference's exact quicksort partitioning.
void checkQsortModule() {
  std::mt19937 rng(11);
  const idx_t sizes[] = {0, 1, 2, 5, 7, 8, 9, 16, 50, 500, 2000};
  // Small key ranges force heavy duplication; the widest range still repeats
  // at n=2000 since it's drawn from only 0..999.
  const idx_t keyRanges[] = {1, 2, 4, 1000};

  for (idx_t n : sizes) {
    for (idx_t range : keyRanges) {
      std::uniform_int_distribution<int> keyDist(0, static_cast<int>(range - 1));

      std::vector<header_only_metis::IndexKeyValue<idx_t>> ref(static_cast<std::size_t>(n));
      for (auto& kv : ref) {
        kv.key = static_cast<idx_t>(keyDist(rng));
        kv.val = static_cast<idx_t>(&kv - ref.data());
      }
      std::vector<header_only_metis::IndexKeyValue<idx_t>> port = ref;

      libmetis__ikvsorti(static_cast<std::size_t>(n), ref.data());
      header_only_metis::ikvSortInc<idx_t>(static_cast<std::size_t>(n), port.data());

      bool match = true;
      for (std::size_t i = 0; i < ref.size(); ++i)
        if (ref[i].key != port[i].key || ref[i].val != port[i].val) match = false;

      const std::string name =
          "ikvSortInc: n=" + std::to_string(n) + " keyRange=" + std::to_string(range);
      checkTrue(match, name);
    }
  }

  // Plain idx_t sortInc/sortDec, same duplicate-key emphasis.
  for (idx_t n : sizes) {
    for (idx_t range : keyRanges) {
      std::uniform_int_distribution<int> keyDist(0, static_cast<int>(range - 1));

      std::vector<idx_t> refInc(static_cast<std::size_t>(n)), refDec(static_cast<std::size_t>(n));
      for (idx_t& k : refInc) k = static_cast<idx_t>(keyDist(rng));
      refDec = refInc;
      std::vector<idx_t> portInc = refInc, portDec = refDec;

      libmetis__isorti(static_cast<std::size_t>(n), refInc.data());
      header_only_metis::sortInc<idx_t>(static_cast<std::size_t>(n), portInc.data());
      libmetis__isortd(static_cast<std::size_t>(n), refDec.data());
      header_only_metis::sortDec<idx_t>(static_cast<std::size_t>(n), portDec.data());

      const std::string base = "n=" + std::to_string(n) + " keyRange=" + std::to_string(range);
      checkTrue(refInc == portInc, "sortInc: " + base);
      checkTrue(refDec == portDec, "sortDec: " + base);
    }
  }
}

// Bridge into METIS's internal ctrl_t/graph_t/CompressGraph, implemented in
// metis_internal_bridge.cpp (its own translation unit -- see that file for
// why). graph_t is kept opaque here: only the accessor functions touch it.
extern "C" {
struct graph_t;
int metis_bridge_CompressGraph(idx_t nvtxs, idx_t* xadj, idx_t* adjncy, idx_t* vwgt, idx_t* cptr,
                                idx_t* cind, graph_t** outGraph);
idx_t metis_bridge_graph_nvtxs(graph_t* g);
idx_t metis_bridge_graph_nedges(graph_t* g);
idx_t* metis_bridge_graph_xadj(graph_t* g);
idx_t* metis_bridge_graph_vwgt(graph_t* g);
idx_t* metis_bridge_graph_adjncy(graph_t* g);
idx_t* metis_bridge_graph_adjwgt(graph_t* g);
idx_t metis_bridge_graph_tvwgt(graph_t* g);
real_t metis_bridge_graph_invtvwgt(graph_t* g);
idx_t* metis_bridge_graph_label(graph_t* g);
void metis_bridge_FreeGraph(graph_t** g);
}

void checkCompressGraph(const std::string& name, idx_t nvtxs, std::vector<idx_t> xadj,
                        std::vector<idx_t> adjncy) {
  std::vector<idx_t> cptrRef(static_cast<std::size_t>(nvtxs) + 1), cindRef(static_cast<std::size_t>(nvtxs));
  graph_t* refGraph = nullptr;
  const int refCompressed =
      metis_bridge_CompressGraph(nvtxs, xadj.data(), adjncy.data(), nullptr, cptrRef.data(),
                                 cindRef.data(), &refGraph);

  std::vector<idx_t> cptrPort(static_cast<std::size_t>(nvtxs) + 1), cindPort(static_cast<std::size_t>(nvtxs));
  auto portGraph = header_only_metis::compressGraph<idx_t, real_t>(
      nvtxs, xadj.data(), adjncy.data(), nullptr, cptrPort.data(), cindPort.data());

  checkTrue((refCompressed != 0) == static_cast<bool>(portGraph), name + ": compress/no-compress agrees");

  if (refCompressed && portGraph) {
    const idx_t cnvtxs = metis_bridge_graph_nvtxs(refGraph);
    const idx_t cnedges = metis_bridge_graph_nedges(refGraph);
    checkTrue(cnvtxs == portGraph->nvtxs, name + ": cnvtxs matches");
    checkTrue(cnedges == portGraph->nedges, name + ": cnedges matches");

    const std::vector<idx_t> xadjRef(metis_bridge_graph_xadj(refGraph),
                                     metis_bridge_graph_xadj(refGraph) + cnvtxs + 1);
    const std::vector<idx_t> vwgtRef(metis_bridge_graph_vwgt(refGraph),
                                     metis_bridge_graph_vwgt(refGraph) + cnvtxs);
    const std::vector<idx_t> adjncyRef(metis_bridge_graph_adjncy(refGraph),
                                       metis_bridge_graph_adjncy(refGraph) + cnedges);
    const std::vector<idx_t> adjwgtRef(metis_bridge_graph_adjwgt(refGraph),
                                       metis_bridge_graph_adjwgt(refGraph) + cnedges);
    const std::vector<idx_t> labelRef(metis_bridge_graph_label(refGraph),
                                      metis_bridge_graph_label(refGraph) + cnvtxs);
    const idx_t tvwgtRef = metis_bridge_graph_tvwgt(refGraph);
    const real_t invtvwgtRef = metis_bridge_graph_invtvwgt(refGraph);

    checkTrue(xadjRef == portGraph->xadj, name + ": xadj matches");
    checkTrue(vwgtRef == portGraph->vwgt, name + ": vwgt matches");
    checkTrue(adjncyRef == portGraph->adjncy, name + ": adjncy matches");
    checkTrue(adjwgtRef == portGraph->adjwgt, name + ": adjwgt matches");
    checkTrue(labelRef == portGraph->label, name + ": label matches");
    checkTrue(tvwgtRef == portGraph->tvwgt, name + ": tvwgt matches");
    checkTrue(invtvwgtRef == portGraph->invtvwgt, name + ": invtvwgt matches (exact)");

    const std::vector<idx_t> cptrRefUsed(cptrRef.begin(), cptrRef.begin() + cnvtxs + 1);
    const std::vector<idx_t> cptrPortUsed(cptrPort.begin(), cptrPort.begin() + cnvtxs + 1);
    checkTrue(cptrRefUsed == cptrPortUsed, name + ": cptr matches");

    const std::vector<idx_t> cindRefUsed(cindRef.begin(), cindRef.begin() + cptrRef[cnvtxs]);
    const std::vector<idx_t> cindPortUsed(cindPort.begin(), cindPort.begin() + cptrRef[cnvtxs]);
    checkTrue(cindRefUsed == cindPortUsed, name + ": cind matches");
  }

  if (refGraph) metis_bridge_FreeGraph(&refGraph);
}

void checkCompressModule() {
  // Compressible: nLeaves groups, each group's vertices share IDENTICAL
  // adjacency (all leaves in a class connect to the same fixed hub set), so
  // CompressGraph should merge each class into one representative.
  for (idx_t nHubs : {2, 5, 10}) {
    for (idx_t leavesPerClass : {5, 20}) {
      for (idx_t nClasses : {3, 10}) {
        const idx_t nLeaves = leavesPerClass * nClasses;
        const idx_t nvtxs = nHubs + nLeaves;
        std::vector<std::vector<idx_t>> adj(static_cast<std::size_t>(nvtxs));
        for (idx_t c = 0; c < nClasses; ++c) {
          // Each class attaches to a distinct, deterministic subset of hubs
          // so different classes are NOT identical to each other.
          std::vector<idx_t> hubset;
          for (idx_t h = 0; h < nHubs; ++h)
            if (((c + h) % nHubs) < (nHubs + 1) / 2) hubset.push_back(h);
          if (hubset.empty()) hubset.push_back(0);
          for (idx_t l = 0; l < leavesPerClass; ++l) {
            const idx_t leaf = nHubs + c * leavesPerClass + l;
            for (idx_t h : hubset) {
              adj[static_cast<std::size_t>(leaf)].push_back(h);
              adj[static_cast<std::size_t>(h)].push_back(leaf);
            }
          }
        }
        std::vector<idx_t> xadj(static_cast<std::size_t>(nvtxs) + 1, 0);
        std::vector<idx_t> adjncy;
        for (idx_t i = 0; i < nvtxs; ++i) {
          xadj[static_cast<std::size_t>(i)] = static_cast<idx_t>(adjncy.size());
          for (idx_t v : adj[static_cast<std::size_t>(i)]) adjncy.push_back(v);
        }
        xadj[static_cast<std::size_t>(nvtxs)] = static_cast<idx_t>(adjncy.size());

        const std::string name = "compress: hubs=" + std::to_string(nHubs) +
                                 " leavesPerClass=" + std::to_string(leavesPerClass) +
                                 " classes=" + std::to_string(nClasses);
        checkCompressGraph(name, nvtxs, xadj, adjncy);
      }
    }
  }

  // Not compressible: distinct-degree random graphs (reusing the mmd-module
  // generator), plus edge cases.
  unsigned seed = 100;
  for (idx_t n : {5, 50, 500}) {
    for (double density : {0.02, 0.1, 0.3}) {
      std::vector<idx_t> adjncy;
      std::vector<idx_t> xadj = randomGraphXadj(n, density, seed++, adjncy);
      const std::string name =
          "compress: random n=" + std::to_string(n) + " density=" + std::to_string(density);
      checkCompressGraph(name, n, xadj, adjncy);
    }
  }
  {
    std::vector<idx_t> xadj = {0, 0};
    std::vector<idx_t> adjncy;
    checkCompressGraph("compress: single isolated vertex", idx_t(1), xadj, adjncy);
  }
}

#endif  // HAVE_METIS

}  // namespace

int main() {
#ifdef HAVE_METIS
  checkRandomModule();
  checkMmdModule();
  checkSortingModule();
  checkQsortModule();
  checkCompressModule();
#else
  note("built without DLU_WITH_METIS -- nothing to check, pass by default");
#endif
  return lu_testing::summarize("test_header_only_metis_internal");
}
