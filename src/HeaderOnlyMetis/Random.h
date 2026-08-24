// Faithful, bit-identical port of GKlib's MT19937-64 random-number generator
// (GKlib/src/random.c, the USE_GKRAND path) and the idx_t-typed wrappers
// METIS itself uses (GK_MKRANDOM(i, idx_t, idx_t), instantiated in
// libmetis/gklib.c via GKlib/include/gk_mkrandom.h).
//
// State is process-global, matching GKlib's file-scope static mt[]/mti
// exactly: METIS resets it once per top-level METIS_NodeND call
// (InitRandom(ctrl->seed) in libmetis/options.c, seed==-1 mapped to the fixed
// constant 4321 in libmetis/util.c -- not time-based, so this is reproducible
// by construction) and everything nested inside that one call draws from the
// same stream. This single global stream is exactly what the recursion's
// tie-breaking depends on for bit-identical output -- do not turn this into
// per-instance state without re-verifying every downstream comparison test.
//
// IndexT is the caller's index type (idx_t on the METIS side, usually
// int32_t). The generator's internal state is always 64-bit regardless of
// IndexT's width, matching GKlib exactly.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_RANDOM_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_RANDOM_H

#include <cstdint>

namespace header_only_metis {

namespace detail {

constexpr int kMTStateSize = 312;
constexpr int kMTPeriodM = 156;
constexpr std::uint64_t kMTMatrixA = 0xB5026F5AA96619E9ULL;
constexpr std::uint64_t kMTUpperMask = 0xFFFFFFFF80000000ULL;  // most significant 33 bits
constexpr std::uint64_t kMTLowerMask = 0x7FFFFFFFULL;          // least significant 31 bits

inline std::uint64_t g_mt[kMTStateSize];
inline int g_mti = kMTStateSize + 1;  // == NN+1 in GKlib: "not yet initialized" sentinel

// Matches gk_randinit(uint64_t seed) (GKlib/src/random.c) exactly.
inline void randInit(std::uint64_t seed) {
  g_mt[0] = seed;
  for (g_mti = 1; g_mti < kMTStateSize; ++g_mti)
    g_mt[g_mti] = (6364136223846793005ULL * (g_mt[g_mti - 1] ^ (g_mt[g_mti - 1] >> 62)) +
                  static_cast<std::uint64_t>(g_mti));
}

// Matches gk_randint64(void) exactly, including the top-bit mask on the
// return value (the generator is used as a 63-bit source, not full 64-bit --
// this is GKlib's choice, not a truncation bug, and must be preserved).
inline std::uint64_t randInt64() {
  static const std::uint64_t mag01[2] = {0ULL, kMTMatrixA};

  if (g_mti >= kMTStateSize) {
    if (g_mti == kMTStateSize + 1) randInit(5489ULL);  // default seed if never seeded

    int i = 0;
    for (; i < kMTStateSize - kMTPeriodM; ++i) {
      const std::uint64_t x = (g_mt[i] & kMTUpperMask) | (g_mt[i + 1] & kMTLowerMask);
      g_mt[i] = g_mt[i + kMTPeriodM] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1ULL)];
    }
    for (; i < kMTStateSize - 1; ++i) {
      const std::uint64_t x = (g_mt[i] & kMTUpperMask) | (g_mt[i + 1] & kMTLowerMask);
      g_mt[i] = g_mt[i + (kMTPeriodM - kMTStateSize)] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1ULL)];
    }
    const std::uint64_t x = (g_mt[kMTStateSize - 1] & kMTUpperMask) | (g_mt[0] & kMTLowerMask);
    g_mt[kMTStateSize - 1] = g_mt[kMTPeriodM - 1] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1ULL)];

    g_mti = 0;
  }

  std::uint64_t x = g_mt[g_mti++];
  x ^= (x >> 29) & 0x5555555555555555ULL;
  x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
  x ^= (x << 37) & 0xFFF7EEE000000000ULL;
  x ^= (x >> 43);

  return x & 0x7FFFFFFFFFFFFFFFULL;
}

// Matches gk_randint32(void) exactly.
inline std::uint32_t randInt32() { return static_cast<std::uint32_t>(randInt64() & 0x7FFFFFFFULL); }

}  // namespace detail

// Matches GK_MKRANDOM's FPRFX##srand: re-seeds the global stream.
template <typename IndexT>
void randSeed(IndexT seed) {
  detail::randInit(static_cast<std::uint64_t>(seed));
}

// Matches GK_MKRANDOM's FPRFX##rand: for IndexT no wider than int32_t (idx_t's
// usual case), draws from the 32-bit generator; wider IndexT draws the full
// 64-bit value. Both consume the same underlying stream state.
template <typename IndexT>
IndexT randNext() {
  if (sizeof(IndexT) <= sizeof(std::int32_t)) return static_cast<IndexT>(detail::randInt32());
  return static_cast<IndexT>(detail::randInt64());
}

// Matches GK_MKRANDOM's FPRFX##randInRange with the (default, non-legacy)
// GK_RNG_LEGACY_WIDTH=0 behaviour: ranges that fit in 31 bits always draw from
// the 32-bit generator regardless of IndexT's width, so 32- and 64-bit builds
// stay in sync as long as every range used fits in 31 bits (true throughout
// METIS's nested-dissection path).
template <typename IndexT>
IndexT randInRange(IndexT max) {
  if (static_cast<std::uint64_t>(max) <= 0x7fffffffULL)
    return static_cast<IndexT>(detail::randInt32() % static_cast<std::uint32_t>(max));
  return static_cast<IndexT>(randNext<IndexT>() % max);
}

// Matches GK_MKRANDOM's FPRFX##randArrayPermute EXACTLY, including the
// non-obvious 4-way "blocked swap" pattern used once n>=10. That pattern is
// not an arbitrary choice to be "improved" -- it is the specific sequence of
// swaps that determines METIS's tie-breaking, and must be copied verbatim.
template <typename IndexT>
void randArrayPermute(IndexT n, IndexT* p, IndexT nshuffles, int flag) {
  if (flag == 1) {
    for (IndexT i = 0; i < n; ++i) p[i] = i;
  }

  if (n < 10) {
    for (IndexT i = 0; i < n; ++i) {
      const IndexT v = randInRange<IndexT>(n);
      const IndexT u = randInRange<IndexT>(n);
      const IndexT tmp = p[v];
      p[v] = p[u];
      p[u] = tmp;
    }
  } else {
    for (IndexT i = 0; i < nshuffles; ++i) {
      const IndexT v = randInRange<IndexT>(n - 3);
      const IndexT u = randInRange<IndexT>(n - 3);
      IndexT tmp;
      tmp = p[v + 0];
      p[v + 0] = p[u + 2];
      p[u + 2] = tmp;
      tmp = p[v + 1];
      p[v + 1] = p[u + 3];
      p[u + 3] = tmp;
      tmp = p[v + 2];
      p[v + 2] = p[u + 0];
      p[u + 0] = tmp;
      tmp = p[v + 3];
      p[v + 3] = p[u + 1];
      p[u + 1] = tmp;
    }
  }
}

// Matches GK_MKRANDOM's FPRFX##randArrayPermuteFine exactly.
template <typename IndexT>
void randArrayPermuteFine(IndexT n, IndexT* p, int flag) {
  if (flag == 1) {
    for (IndexT i = 0; i < n; ++i) p[i] = i;
  }
  for (IndexT i = 0; i < n; ++i) {
    const IndexT v = randInRange<IndexT>(n);
    const IndexT tmp = p[i];
    p[i] = p[v];
    p[v] = tmp;
  }
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_RANDOM_H
