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

#include "Workspace.h"

namespace header_only_metis {

// METIS_CTYPE_RM / METIS_CTYPE_SHEM (metis.h's mctype_et), the only two
// values genmmd's caller can dispatch to on the default-options path.
enum class CType { RM, SHEM };

// METIS_RTYPE_SEP1SIDED / METIS_RTYPE_SEP2SIDED (metis.h's mrtype_et) -- the
// only two values reachable from Refine2WayNode's rtype switch on the
// default node-ND path (METIS_RTYPE_FM/GREEDY are the kmetis/pmetis values).
enum class RType { SEP1SIDED, SEP2SIDED };

template <typename IndexT, typename RealT>
struct Ctrl {
  IndexT CoarsenTo = 0;  // target coarsest-graph vertex count for this bisection
  IndexT maxvwgt = 0;    // max allowed coarsened-vertex weight (ncon=1, so scalar not array)
  CType ctype = CType::SHEM;  // options.c OMETIS default
  bool no2hop = false;        // options.c OMETIS default: 2-hop matching enabled
  IndexT niter = 10;           // options.c OMETIS default: refinement iterations per pass
  // I2RUBFACTOR(OMETIS_DEFAULT_UFACTOR=200) = 1.0+0.001*200 = 1.2, stored into
  // a real_t array by SetupCtrl -- and then nudged by a fixed epsilon:
  //
  //   ctrl->ubfactors[i] += 0.0000499;      (options.c:120-121)
  //
  // That epsilon is load-bearing, not cosmetic. Every balance threshold on
  // this path is an integer truncation of a product involving ubfactor --
  // e.g. badmaxpwgt = (idx_t)(0.5*ubfactor*(pwgts[0]+pwgts[1])) in
  // FM_2WayNodeBalance -- so whenever 0.6*pwgtsum lands in the narrow band
  // just below an integer, dropping the epsilon shifts the threshold down by
  // one, admits a different vertex, and (because the FM loops draw from the
  // shared RNG stream per boundary vertex) desynchronizes every random draw
  // that follows.
  //
  // The two-step rounding is reproduced deliberately: narrow 1.2 to RealT
  // first, then add the double-typed epsilon and narrow again, matching C's
  // `float += double` semantics on ctrl->ubfactors[i] exactly.
  RealT ubfactor = static_cast<RealT>(static_cast<RealT>(1.0 + 0.001 * 200) + 0.0000499);
  bool compress = true;         // options.c OMETIS default: ctrl->compress
  RealT pijbm[2] = {0, 0};      // Setup2WayBalMultipliers' output: invtvwgt/tpwgts[i], ncon=1
  RType rtype = RType::SEP1SIDED;  // options.c OMETIS default
  IndexT nseps = 1;  // options.c OMETIS default; ometis.c bumps this to 2 if compression achieves >1.5x

  // Per-call scratch for the refinement/matching routines, standing in for
  // ctrl->mcore + iwspacemalloc. Lives here (rather than as locals at each
  // call site) purely so the buffers survive between calls; it carries no
  // algorithmic state, and nothing reads a value from it that the same call
  // did not first write.
  Workspace<IndexT> wspace;
};

}  // namespace header_only_metis

#endif  // DIRECTLUSOLVERS_HEADER_ONLY_METIS_CTRL_H
