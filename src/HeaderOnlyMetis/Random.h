// Faithful, bit-identical port of GKlib's MT19937-64 random-number generator
// (GKlib/src/random.c, the USE_GKRAND path) and the idx_t-typed wrappers
// METIS itself uses (GK_MKRANDOM(i, idx_t, idx_t), instantiated in
// libmetis/gklib.c via GKlib/include/gk_mkrandom.h).
//
// GKlib keeps mt[]/mti as file-scope statics; this port keeps the identical
// algorithm but hands the state around explicitly as a RandomState, normally
// the one living on the Ctrl. Behaviour is unchanged: METIS reseeds once per
// top-level METIS_NodeND call (InitRandom(ctrl->seed) in libmetis/options.c,
// with seed==-1 mapped to the fixed constant 4321 in libmetis/util.c -- not
// time-based, so it is reproducible by construction), and every draw nested
// inside that call comes from that one stream. Threading the state through
// explicitly rather than hiding it in a global is what makes it possible to
// give independent subtrees independent streams later; while a single
// RandomState is shared by the whole recursion, the draw sequence -- and so
// the tie-breaking, and so the output -- is exactly what it was.
//
// A consequence worth stating plainly: the ordering depends on the ORDER in
// which draws are taken, so any change that adds, removes or reorders a draw
// changes the result even though every individual value is still correct.
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

}  // namespace detail

// GKlib's mt[]/mti pair, made explicit. Default-constructed it carries
// GKlib's "not yet seeded" sentinel, so the first draw self-seeds with 5489
// exactly as gk_randint64 does.
struct RandomState {
  std::uint64_t mt[detail::kMTStateSize];
  int mti = detail::kMTStateSize + 1;  // == NN+1 in GKlib
};

namespace detail {

// Matches gk_randinit(uint64_t seed) (GKlib/src/random.c) exactly.
inline void randInit(RandomState& s, std::uint64_t seed) {
  s.mt[0] = seed;
  for (s.mti = 1; s.mti < kMTStateSize; ++s.mti)
    s.mt[s.mti] = (6364136223846793005ULL * (s.mt[s.mti - 1] ^ (s.mt[s.mti - 1] >> 62)) +
                  static_cast<std::uint64_t>(s.mti));
}

// Matches gk_randint64(void) exactly, including the top-bit mask on the
// return value (the generator is used as a 63-bit source, not full 64-bit --
// this is GKlib's choice, not a truncation bug, and must be preserved).
inline std::uint64_t randInt64(RandomState& s) {
  static const std::uint64_t mag01[2] = {0ULL, kMTMatrixA};

  if (s.mti >= kMTStateSize) {
    if (s.mti == kMTStateSize + 1) randInit(s, 5489ULL);  // default seed if never seeded

    int i = 0;
    for (; i < kMTStateSize - kMTPeriodM; ++i) {
      const std::uint64_t x = (s.mt[i] & kMTUpperMask) | (s.mt[i + 1] & kMTLowerMask);
      s.mt[i] = s.mt[i + kMTPeriodM] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1ULL)];
    }
    for (; i < kMTStateSize - 1; ++i) {
      const std::uint64_t x = (s.mt[i] & kMTUpperMask) | (s.mt[i + 1] & kMTLowerMask);
      s.mt[i] = s.mt[i + (kMTPeriodM - kMTStateSize)] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1ULL)];
    }
    const std::uint64_t x = (s.mt[kMTStateSize - 1] & kMTUpperMask) | (s.mt[0] & kMTLowerMask);
    s.mt[kMTStateSize - 1] = s.mt[kMTPeriodM - 1] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1ULL)];

    s.mti = 0;
  }

  std::uint64_t x = s.mt[s.mti++];
  x ^= (x >> 29) & 0x5555555555555555ULL;
  x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
  x ^= (x << 37) & 0xFFF7EEE000000000ULL;
  x ^= (x >> 43);

  return x & 0x7FFFFFFFFFFFFFFFULL;
}

// Matches gk_randint32(void) exactly.
inline std::uint32_t randInt32(RandomState& s) { return static_cast<std::uint32_t>(randInt64(s) & 0x7FFFFFFFULL); }

}  // namespace detail

// Matches GK_MKRANDOM's FPRFX##srand: re-seeds the global stream.
template <typename IndexT>
void randSeed(RandomState& s, IndexT seed) {
  detail::randInit(s, static_cast<std::uint64_t>(seed));
}

// Matches GK_MKRANDOM's FPRFX##rand: for IndexT no wider than int32_t (idx_t's
// usual case), draws from the 32-bit generator; wider IndexT draws the full
// 64-bit value. Both consume the same underlying stream state.
// GKlib's isrand() takes an idx_t; per-subtree seeding needs the full 64-bit
// width, so this exposes gk_randinit directly. Not part of the METIS-identical
// path -- nodeND() seeds with 4321 through randSeed() exactly as the reference
// does.
inline void randSeed64(RandomState& s, std::uint64_t seed) { detail::randInit(s, seed); }

template <typename IndexT>
IndexT randNext(RandomState& s) {
  if (sizeof(IndexT) <= sizeof(std::int32_t)) return static_cast<IndexT>(detail::randInt32(s));
  return static_cast<IndexT>(detail::randInt64(s));
}

// Matches GK_MKRANDOM's FPRFX##randInRange with the (default, non-legacy)
// GK_RNG_LEGACY_WIDTH=0 behaviour: ranges that fit in 31 bits always draw from
// the 32-bit generator regardless of IndexT's width, so 32- and 64-bit builds
// stay in sync as long as every range used fits in 31 bits (true throughout
// METIS's nested-dissection path).
template <typename IndexT>
IndexT randInRange(RandomState& s, IndexT max) {
  if (static_cast<std::uint64_t>(max) <= 0x7fffffffULL)
    return static_cast<IndexT>(detail::randInt32(s) % static_cast<std::uint32_t>(max));
  return static_cast<IndexT>(randNext<IndexT>(s) % max);
}

// Matches GK_MKRANDOM's FPRFX##randArrayPermute EXACTLY, including the
// non-obvious 4-way "blocked swap" pattern used once n>=10. That pattern is
// not an arbitrary choice to be "improved" -- it is the specific sequence of
// swaps that determines METIS's tie-breaking, and must be copied verbatim.
template <typename IndexT>
void randArrayPermute(RandomState& s, IndexT n, IndexT* p, IndexT nshuffles, int flag) {
  if (flag == 1) {
    for (IndexT i = 0; i < n; ++i) p[i] = i;
  }

  if (n < 10) {
    for (IndexT i = 0; i < n; ++i) {
      const IndexT v = randInRange<IndexT>(s, n);
      const IndexT u = randInRange<IndexT>(s, n);
      const IndexT tmp = p[v];
      p[v] = p[u];
      p[u] = tmp;
    }
  } else {
    for (IndexT i = 0; i < nshuffles; ++i) {
      const IndexT v = randInRange<IndexT>(s, n - 3);
      const IndexT u = randInRange<IndexT>(s, n - 3);
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
void randArrayPermuteFine(RandomState& s, IndexT n, IndexT* p, int flag) {
  if (flag == 1) {
    for (IndexT i = 0; i < n; ++i) p[i] = i;
  }
  for (IndexT i = 0; i < n; ++i) {
    const IndexT v = randInRange<IndexT>(s, n);
    const IndexT tmp = p[i];
    p[i] = p[v];
    p[v] = tmp;
  }
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_RANDOM_H
