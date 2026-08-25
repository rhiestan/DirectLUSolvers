// Reusable scratch storage for the hot inner routines, the counterpart of
// METIS's workspace core (ctrl->mcore + iwspacemalloc, bracketed by
// WCOREPUSH/WCOREPOP in libmetis/wspace.c).
//
// The refinement and matching routines each need a handful of nvtxs-sized
// index arrays that live exactly as long as one call. Expressing those as
// local std::vectors -- the obvious thing, and what this port did first --
// costs a malloc, a zero-fill and a free per array per call. Those calls are
// not rare: FM_2WayCutRefine alone runs once per initial-partition trial per
// bisection, so on a deep nested-dissection recursion the allocator traffic
// became a measurable fraction of total ordering time (malloc/free/memset
// together were the largest non-algorithmic cost left after the coarsening
// fixes).
//
// This hands out those arrays from buffers that persist on the Ctrl and grow
// monotonically, so after the first few calls there is no allocation at all
// and no zero-fill ever. Contents are deliberately INDETERMINATE on handout,
// exactly as iwspacemalloc's are -- every user below either fills the array
// before reading it or explicitly sets it (the reference relies on the same
// property).
//
// Slots are handed out in stack order, released by a scope guard. Each slot
// keeps its own heap block, so growing one slot never moves another: pointers
// handed out earlier in the same scope stay valid.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_WORKSPACE_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_WORKSPACE_H

#include <cstddef>
#include <memory>
#include <vector>

namespace header_only_metis {

template <typename T>
class Workspace {
 public:
  // Returns an array of at least n elements with INDETERMINATE contents.
  T* take(std::size_t n) {
    if (depth_ == slots_.size()) slots_.emplace_back();
    Slot& s = slots_[depth_++];
    if (s.cap < n) {
      s.data.reset(new T[n]);  // trivial T: default-init, no zero-fill
      s.cap = n;
    }
    return s.data.get();
  }

  std::size_t mark() const { return depth_; }
  void release(std::size_t m) { depth_ = m; }

  // RAII bracket, the equivalent of WCOREPUSH/WCOREPOP.
  class Scope {
   public:
    explicit Scope(Workspace& w) : w_(w), mark_(w.mark()) {}
    ~Scope() { w_.release(mark_); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    Workspace& w_;
    std::size_t mark_;
  };

 private:
  struct Slot {
    std::unique_ptr<T[]> data;
    std::size_t cap = 0;
  };
  std::vector<Slot> slots_;
  std::size_t depth_ = 0;
};

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_WORKSPACE_H
