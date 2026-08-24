// Minimal ctrl_t equivalent (libmetis/struct.h) for the header-only METIS
// port. Like Graph.h, deliberately pared to what's proven necessary so far --
// extend as later modules need more of ctrl_t's fields rather than
// front-loading everything now. Dropped entirely (dead on the default-options
// node-ND path, per options.c's OMETIS defaults and the earlier scoping
// analysis): dbglvl/timers (dbglvl==0 always -- Eigen's call never sets debug
// options), ondisk (defaults 0), minconn/contig (k-way only), mcore/workspace
// (std::vector replaces GKlib's workspace arena), nbrpool/cnbrsqrt/adids/...
// (k-way refinement only), pid (out-of-core only).

#ifndef DIRECTLUSOLVERS_HEADER_ONLY_METIS_CTRL_H
#define DIRECTLUSOLVERS_HEADER_ONLY_METIS_CTRL_H

namespace header_only_metis {

// METIS_CTYPE_RM / METIS_CTYPE_SHEM (metis.h's mctype_et), the only two
// values genmmd's caller can dispatch to on the default-options path.
enum class CType { RM, SHEM };

template <typename IndexT>
struct Ctrl {
  IndexT CoarsenTo = 0;  // target coarsest-graph vertex count for this bisection
  IndexT maxvwgt = 0;    // max allowed coarsened-vertex weight (ncon=1, so scalar not array)
  CType ctype = CType::SHEM;  // options.c OMETIS default
  bool no2hop = false;        // options.c OMETIS default: 2-hop matching enabled
};

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_CTRL_H
