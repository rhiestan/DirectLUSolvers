// Faithful port of libmetis/bucketsort.c's BucketSortKeysInc: a plain counting
// sort, used three times in coarsen.c to sort vertices by degree before
// matching. Deterministic, no RNG involved.
//
// The reference threads this through GKlib's workspace-stack allocator
// (WCOREPUSH/iwspacemalloc/WCOREPOP); this port uses a local std::vector
// instead, which only changes when the scratch memory is freed, not the
// arithmetic or iteration order, so it cannot affect the result.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_SORTING_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_SORTING_H

#include <cstddef>
#include <utility>
#include <vector>

namespace header_only_metis {

// Faithful port of GKlib's GK_MKQSORT (GKlib/include/gk_mksort.h) -- the
// classic Schmidt/glibc hybrid quicksort+insertion-sort (median-of-three
// pivot, explicit stack instead of recursion, switches to insertion sort
// below an 8-element threshold). GKlib uses this for EVERY *sort* routine
// (ikvsorti, isorti, ...), and it is NOT a stable sort. Several METIS call
// sites sort by a key that is not always unique (e.g. CompressGraph's
// adjacency-sum key), so matching the reference bit-for-bit on ties requires
// this exact algorithm, not just "any correct sort" like std::sort/
// std::stable_sort -- a different tie-break order changes which vertex ends
// up "representative" of a merged group, which is a real output difference.
//
// `lt(a, b)` takes pointers, matching GK_MKQSORT's GKQSORT_LT(a,b) macro
// exactly (it compares *a and *b through the pointers, never by value).
template <typename T, typename Compare>
void gkQsort(T* base, std::size_t nelt, Compare lt) {
  constexpr std::ptrdiff_t kMaxThresh = 8;
  const std::size_t elems = nelt;
  T hold;

  if (elems < 1) return;

  if (elems > static_cast<std::size_t>(kMaxThresh)) {
    T* lo = base;
    T* hi = lo + elems - 1;
    struct StackEntry {
      T* hi;
      T* lo;
    };
    constexpr std::size_t kStackSize = 8 * sizeof(std::size_t);
    StackEntry stack[kStackSize];
    StackEntry* top = stack + 1;

    while (stack < top) {
      T* left_ptr;
      T* right_ptr;

      T* mid = lo + ((hi - lo) >> 1);

      if (lt(mid, lo)) std::swap(*mid, *lo);
      if (lt(hi, mid)) {
        std::swap(*mid, *hi);
        if (lt(mid, lo)) std::swap(*mid, *lo);
      }
      // else: GK_MKQSORT's "goto _jump_over" -- falls straight through to here.

      left_ptr = lo + 1;
      right_ptr = hi - 1;

      do {
        while (lt(left_ptr, mid)) ++left_ptr;
        while (lt(mid, right_ptr)) --right_ptr;

        if (left_ptr < right_ptr) {
          std::swap(*left_ptr, *right_ptr);
          if (mid == left_ptr)
            mid = right_ptr;
          else if (mid == right_ptr)
            mid = left_ptr;
          ++left_ptr;
          --right_ptr;
        } else if (left_ptr == right_ptr) {
          ++left_ptr;
          --right_ptr;
          break;
        }
      } while (left_ptr <= right_ptr);

      if (right_ptr - lo <= kMaxThresh) {
        if (hi - left_ptr <= kMaxThresh) {
          --top;
          lo = top->lo;
          hi = top->hi;  // POP
        } else {
          lo = left_ptr;
        }
      } else if (hi - left_ptr <= kMaxThresh) {
        hi = right_ptr;
      } else if (right_ptr - lo > hi - left_ptr) {
        top->lo = lo;
        top->hi = right_ptr;
        ++top;  // PUSH
        lo = left_ptr;
      } else {
        top->lo = left_ptr;
        top->hi = hi;
        ++top;  // PUSH
        hi = right_ptr;
      }
    }
  }

  // Insertion sort over the (now mostly-sorted, every partition <=8 elements)
  // array. base points to the first element, end_ptr to the LAST element
  // (inclusive), matching the reference exactly.
  {
    T* const end_ptr = base + elems - 1;
    T* tmp_ptr = base;
    T* run_ptr;
    T* thresh = base + kMaxThresh;
    if (thresh > end_ptr) thresh = end_ptr;

    // Find the smallest element in the first threshold and place it at the
    // array's beginning -- speeds up (and simplifies) the scan below.
    for (run_ptr = tmp_ptr + 1; run_ptr <= thresh; ++run_ptr)
      if (lt(run_ptr, tmp_ptr)) tmp_ptr = run_ptr;

    if (tmp_ptr != base) std::swap(*tmp_ptr, *base);

    run_ptr = base + 1;
    while (++run_ptr <= end_ptr) {
      tmp_ptr = run_ptr - 1;
      while (lt(run_ptr, tmp_ptr)) --tmp_ptr;

      ++tmp_ptr;
      if (tmp_ptr != run_ptr) {
        T* trav = run_ptr + 1;
        while (--trav >= run_ptr) {
          hold = *trav;
          T* hiP;
          T* loP;
          for (hiP = loP = trav; --loP >= tmp_ptr; hiP = loP) *hiP = *loP;
          *hiP = hold;
        }
      }
    }
  }
}

// Matches GKlib's ikv_t: a (key, val) pair, sorted by key.
template <typename IndexT>
struct IndexKeyValue {
  IndexT key;
  IndexT val;
};

// Matches gklib.c's ikvsorti (GK_MKQSORT with ikey_lt: (a)->key < (b)->key).
template <typename IndexT>
void ikvSortInc(std::size_t n, IndexKeyValue<IndexT>* base) {
  gkQsort(base, n, [](const IndexKeyValue<IndexT>* a, const IndexKeyValue<IndexT>* b) {
    return a->key < b->key;
  });
}

// Matches gklib.c's isorti (GK_MKQSORT with i_lt: (*a) < (*b)).
template <typename IndexT>
void sortInc(std::size_t n, IndexT* base) {
  gkQsort(base, n, [](const IndexT* a, const IndexT* b) { return *a < *b; });
}

// Matches gklib.c's isortd (GK_MKQSORT with i_gt: (*a) > (*b)).
template <typename IndexT>
void sortDec(std::size_t n, IndexT* base) {
  gkQsort(base, n, [](const IndexT* a, const IndexT* b) { return *a > *b; });
}

// This function uses simple counting sort to return a permutation array
// corresponding to the sorted order. The keys are assumed to start from 0 and
// be positive. This sorting is used during matching.
template <typename IndexT>
void bucketSortKeysInc(IndexT n, IndexT max, const IndexT* keys, const IndexT* tperm, IndexT* perm) {
  std::vector<IndexT> counts(static_cast<std::size_t>(max) + 2, IndexT(0));

  for (IndexT i = 0; i < n; i++) counts[static_cast<std::size_t>(keys[i])]++;

  // MAKECSR(i, max+1, counts) (GKlib/include/gk_macros.h)
  for (IndexT i = 1; i < max + 1; i++)
    counts[static_cast<std::size_t>(i)] += counts[static_cast<std::size_t>(i - 1)];
  for (IndexT i = max + 1; i > 0; i--) counts[static_cast<std::size_t>(i)] = counts[static_cast<std::size_t>(i - 1)];
  counts[0] = 0;

  for (IndexT ii = 0; ii < n; ii++) {
    const IndexT i = tperm[ii];
    perm[counts[static_cast<std::size_t>(keys[i])]++] = i;
  }
}

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_SORTING_H
