// Faithful port of GKlib's GK_MKPQUEUE (GKlib/include/gk_mkpqueue.h), a
// binary heap with an O(1) node->index locator, specialized to the
// KEY_LT=key_gt instantiation METIS actually uses everywhere (rpq_t/ipq_t,
// GKlib/src/random.c... no -- libmetis/gklib.c: `#define key_gt(a,b)
// ((a)>(b))`). Passing key_gt as the macro's "KEY_LT" parameter makes every
// `KEY_LT(x,y)` in the template body literally `x>y`, turning this into a
// MAX-heap: GetTop returns the LARGEST key. That matches its use for FM gain
// values, where "highest priority" means "highest gain" -- so every `a>b` in
// this file traces directly back to a `KEY_LT(a,b)` in the macro, not an
// independent design choice.
//
// ASSERT-only in the reference (CheckHeap) is not ported: assertions compile
// out under NDEBUG, which is how the reference build used for comparison is
// built (CMAKE_BUILD_TYPE=Release), so CheckHeap never executes in the
// binary this port is validated against.

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_PQUEUE_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_PQUEUE_H

#include <cstddef>
#include <vector>

namespace header_only_metis {

template <typename KeyT, typename ValT>
class PQueue {
 public:
  explicit PQueue(std::size_t maxnodes) : heap_(maxnodes), locator_(maxnodes, ValT(-1)), nnodes_(0) {}

  std::size_t length() const { return nnodes_; }

  void reset() {
    for (std::size_t i = nnodes_; i-- > 0;) locator_[static_cast<std::size_t>(heap_[i].val)] = ValT(-1);
    nnodes_ = 0;
  }

  void insert(ValT node, KeyT key) {
    std::size_t i = nnodes_++;
    while (i > 0) {
      const std::size_t j = (i - 1) >> 1;
      if (key > heap_[j].key) {
        heap_[i] = heap_[j];
        locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
        i = j;
      } else {
        break;
      }
    }
    heap_[i].key = key;
    heap_[i].val = node;
    locator_[static_cast<std::size_t>(node)] = static_cast<ValT>(i);
  }

  // Matches GK_MKPQUEUE's FPRFX##Delete (renamed to avoid shadowing
  // std::vector::erase-style expectations; behaviour is identical).
  void erase(ValT node) {
    std::size_t i = static_cast<std::size_t>(locator_[static_cast<std::size_t>(node)]);
    locator_[static_cast<std::size_t>(node)] = ValT(-1);

    if (--nnodes_ > 0 && heap_[nnodes_].val != node) {
      const ValT movedNode = heap_[nnodes_].val;
      const KeyT newkey = heap_[nnodes_].key;
      const KeyT oldkey = heap_[i].key;

      if (newkey > oldkey) { /* Filter-up */
        while (i > 0) {
          const std::size_t j = (i - 1) >> 1;
          if (newkey > heap_[j].key) {
            heap_[i] = heap_[j];
            locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
            i = j;
          } else {
            break;
          }
        }
      } else { /* Filter down */
        const std::size_t nnodes = nnodes_;
        std::size_t j;
        while ((j = (i << 1) + 1) < nnodes) {
          if (heap_[j].key > newkey) {
            if (j + 1 < nnodes && heap_[j + 1].key > heap_[j].key) j++;
            heap_[i] = heap_[j];
            locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
            i = j;
          } else if (j + 1 < nnodes && heap_[j + 1].key > newkey) {
            j++;
            heap_[i] = heap_[j];
            locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
            i = j;
          } else {
            break;
          }
        }
      }

      heap_[i].key = newkey;
      heap_[i].val = movedNode;
      locator_[static_cast<std::size_t>(movedNode)] = static_cast<ValT>(i);
    }
  }

  void update(ValT node, KeyT newkey) {
    std::size_t i = static_cast<std::size_t>(locator_[static_cast<std::size_t>(node)]);
    const KeyT oldkey = heap_[i].key;
    if (!(newkey > oldkey) && !(oldkey > newkey)) return;  // equal keys: no-op

    if (newkey > oldkey) { /* Filter-up */
      while (i > 0) {
        const std::size_t j = (i - 1) >> 1;
        if (newkey > heap_[j].key) {
          heap_[i] = heap_[j];
          locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
          i = j;
        } else {
          break;
        }
      }
    } else { /* Filter down */
      const std::size_t nnodes = nnodes_;
      std::size_t j;
      while ((j = (i << 1) + 1) < nnodes) {
        if (heap_[j].key > newkey) {
          if (j + 1 < nnodes && heap_[j + 1].key > heap_[j].key) j++;
          heap_[i] = heap_[j];
          locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
          i = j;
        } else if (j + 1 < nnodes && heap_[j + 1].key > newkey) {
          j++;
          heap_[i] = heap_[j];
          locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
          i = j;
        } else {
          break;
        }
      }
    }

    heap_[i].key = newkey;
    heap_[i].val = node;
    locator_[static_cast<std::size_t>(node)] = static_cast<ValT>(i);
  }

  ValT getTop() {
    if (nnodes_ == 0) return ValT(-1);

    --nnodes_;
    const ValT vtx = heap_[0].val;
    locator_[static_cast<std::size_t>(vtx)] = ValT(-1);

    if (nnodes_ > 0) {
      const KeyT key = heap_[nnodes_].key;
      const ValT node = heap_[nnodes_].val;
      std::size_t i = 0;
      std::size_t j;
      while ((j = 2 * i + 1) < nnodes_) {
        if (heap_[j].key > key) {
          if (j + 1 < nnodes_ && heap_[j + 1].key > heap_[j].key) j++;
          heap_[i] = heap_[j];
          locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
          i = j;
        } else if (j + 1 < nnodes_ && heap_[j + 1].key > key) {
          j++;
          heap_[i] = heap_[j];
          locator_[static_cast<std::size_t>(heap_[i].val)] = static_cast<ValT>(i);
          i = j;
        } else {
          break;
        }
      }
      heap_[i].key = key;
      heap_[i].val = node;
      locator_[static_cast<std::size_t>(node)] = static_cast<ValT>(i);
    }

    return vtx;
  }

  ValT seeTopVal() const { return nnodes_ == 0 ? ValT(-1) : heap_[0].val; }
  KeyT seeTopKey(KeyT kmax) const { return nnodes_ == 0 ? kmax : heap_[0].key; }
  KeyT seeKey(ValT node) const { return heap_[static_cast<std::size_t>(locator_[static_cast<std::size_t>(node)])].key; }

 private:
  struct Entry {
    KeyT key{};
    ValT val{};
  };
  std::vector<Entry> heap_;
  std::vector<ValT> locator_;
  std::size_t nnodes_;
};

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_PQUEUE_H
