// Bridge between test_header_only_metis_internal.cpp and METIS's internal
// (non-public) functions/structs (ctrl_t, graph_t, CompressGraph,
// CoarsenGraph, ...). Deliberately its own translation unit: METIS's internal
// headers (libmetis/rename.h, libmetis/gklib_rename.h) #define blanket
// symbol-renaming macros for hundreds of names -- CompressGraph,
// CoarsenGraph, genmmd, isrand, ikvsorti, and so on -- and header_only_metis::
// deliberately mirrors many of those same names for traceability back to the
// reference source. Macro expansion doesn't respect C++ namespaces, so
// including metislib.h in the same translation unit as the port's own code
// would silently rewrite the port's own calls too. Keeping this in its own
// .cpp file means those macros only ever apply to the reference calls made
// right here, and the exported extern "C" surface below is what the test
// file actually links against -- plain functions, no renaming markup.
// metislib.h's own proto.h (unlike the public metis.h) has no __cplusplus
// guard around its declarations -- it's written purely for consumption from
// libmetis's own .c files. Compiled from this .cpp without extern "C", these
// declarations would get C++ name mangling while the actual compiled
// metis.lib symbols (compiled as plain C) do not, producing "undefined
// symbol" at link time for names that visibly exist in the archive.
extern "C" {
#include "metislib.h"
}

extern "C" {

// Returns 1 if compression succeeded (matching CompressGraph returning
// non-NULL) and writes the result graph_t* to *outGraph; 0 otherwise
// (*outGraph left null). Caller must release a non-null result via
// metis_bridge_FreeGraph.
int metis_bridge_CompressGraph(idx_t nvtxs, idx_t* xadj, idx_t* adjncy, idx_t* vwgt, idx_t* cptr,
                                idx_t* cind, graph_t** outGraph) {
  ctrl_t* ctrl = SetupCtrl(METIS_OP_OMETIS, NULL, 1, 3, NULL, NULL);
  graph_t* graph = CompressGraph(ctrl, nvtxs, xadj, adjncy, vwgt, cptr, cind);
  FreeCtrl(&ctrl);
  *outGraph = graph;
  return graph != NULL;
}

idx_t metis_bridge_graph_nvtxs(graph_t* g) { return g->nvtxs; }
idx_t metis_bridge_graph_nedges(graph_t* g) { return g->nedges; }
idx_t* metis_bridge_graph_xadj(graph_t* g) { return g->xadj; }
idx_t* metis_bridge_graph_vwgt(graph_t* g) { return g->vwgt; }
idx_t* metis_bridge_graph_adjncy(graph_t* g) { return g->adjncy; }
idx_t* metis_bridge_graph_adjwgt(graph_t* g) { return g->adjwgt; }
idx_t metis_bridge_graph_tvwgt(graph_t* g) { return g->tvwgt[0]; }
real_t metis_bridge_graph_invtvwgt(graph_t* g) { return g->invtvwgt[0]; }
idx_t* metis_bridge_graph_label(graph_t* g) { return g->label; }

void metis_bridge_FreeGraph(graph_t** g) { FreeGraph(g); }

// --- Coarsen.h support -------------------------------------------------

// Builds a standalone graph_t by copying the given arrays (ncon=1, matching
// this port's scope). Caller must release via metis_bridge_FreeGraph.
graph_t* metis_bridge_MakeGraph(idx_t nvtxs, idx_t* xadj, idx_t* adjncy, idx_t* vwgt, idx_t* adjwgt) {
  graph_t* graph = CreateGraph();
  const idx_t nedges = xadj[nvtxs];
  graph->nvtxs = nvtxs;
  graph->ncon = 1;
  graph->nedges = nedges;
  graph->xadj = imalloc(nvtxs + 1, "bridge: xadj");
  icopy(nvtxs + 1, xadj, graph->xadj);
  graph->vwgt = imalloc(nvtxs, "bridge: vwgt");
  icopy(nvtxs, vwgt, graph->vwgt);
  graph->adjncy = imalloc(nedges > 0 ? nedges : 1, "bridge: adjncy");
  icopy(nedges, adjncy, graph->adjncy);
  graph->adjwgt = imalloc(nedges > 0 ? nedges : 1, "bridge: adjwgt");
  icopy(nedges, adjwgt, graph->adjwgt);
  graph->cmap = imalloc(nvtxs, "bridge: cmap");
  SetupGraph_tvwgt(graph);
  SetupGraph_label(graph);
  return graph;
}

// ctypeIsSHEM: 0 -> METIS_CTYPE_RM, nonzero -> METIS_CTYPE_SHEM. Also calls
// AllocateWorkSpace(ctrl, graph), which Match_RM/Match_SHEM/CreateCoarseGraph
// all depend on (iwspacemalloc against ctrl->mcore) -- matching the one-time
// setup METIS_NodeND itself does before any algorithm runs.
ctrl_t* metis_bridge_MakeCtrlForCoarsen(graph_t* graph, idx_t coarsenTo, idx_t maxvwgt,
                                        int ctypeIsSHEM, int no2hop) {
  ctrl_t* ctrl = SetupCtrl(METIS_OP_OMETIS, NULL, 1, 3, NULL, NULL);
  ctrl->CoarsenTo = coarsenTo;
  ctrl->maxvwgt[0] = maxvwgt;
  ctrl->ctype = ctypeIsSHEM ? METIS_CTYPE_SHEM : METIS_CTYPE_RM;
  ctrl->no2hop = no2hop;
  AllocateWorkSpace(ctrl, graph);
  return ctrl;
}

void metis_bridge_FreeCtrl(ctrl_t** ctrl) { FreeCtrl(ctrl); }

idx_t metis_bridge_MatchRM(ctrl_t* ctrl, graph_t* graph) { return Match_RM(ctrl, graph); }
idx_t metis_bridge_MatchSHEM(ctrl_t* ctrl, graph_t* graph) { return Match_SHEM(ctrl, graph); }

graph_t* metis_bridge_graph_coarser(graph_t* g) { return g->coarser; }
idx_t* metis_bridge_graph_cmap(graph_t* g) { return g->cmap; }

// Runs the full multi-level CoarsenGraph driver (its own eqewgts computation,
// maxvwgt computation, and do-while termination condition -- not exercised by
// the single-level Match_RM/Match_SHEM bridge above). Returns the coarsest
// graph reached; every intermediate level is still reachable by walking
// ->finer from it, and everything is freed by walking that chain in the
// caller (metis_bridge_FreeGraph is single-level, matching FreeGraph itself).
graph_t* metis_bridge_CoarsenGraph(ctrl_t* ctrl, graph_t* graph) { return CoarsenGraph(ctrl, graph); }

graph_t* metis_bridge_graph_finer(graph_t* g) { return g->finer; }

// --- SeparatorRefinement.h support --------------------------------------

ctrl_t* metis_bridge_MakeCtrlForSepRefine(graph_t* graph, int compress) {
  ctrl_t* ctrl = SetupCtrl(METIS_OP_OMETIS, NULL, 1, 3, NULL, NULL);
  ctrl->compress = compress;
  AllocateWorkSpace(ctrl, graph);
  return ctrl;
}

void metis_bridge_Allocate2WayNodePartitionMemory(ctrl_t* ctrl, graph_t* graph) {
  Allocate2WayNodePartitionMemory(ctrl, graph);
}
void metis_bridge_SetWhere(graph_t* graph, idx_t* where) { icopy(graph->nvtxs, where, graph->where); }
void metis_bridge_Compute2WayNodePartitionParams(ctrl_t* ctrl, graph_t* graph) {
  Compute2WayNodePartitionParams(ctrl, graph);
}
void metis_bridge_FM_2WayNodeRefine2Sided(ctrl_t* ctrl, graph_t* graph, idx_t niter) {
  FM_2WayNodeRefine2Sided(ctrl, graph, niter);
}
void metis_bridge_FM_2WayNodeRefine1Sided(ctrl_t* ctrl, graph_t* graph, idx_t niter) {
  FM_2WayNodeRefine1Sided(ctrl, graph, niter);
}
void metis_bridge_FM_2WayNodeBalance(ctrl_t* ctrl, graph_t* graph) { FM_2WayNodeBalance(ctrl, graph); }

idx_t* metis_bridge_graph_where(graph_t* g) { return g->where; }
idx_t* metis_bridge_graph_pwgts(graph_t* g) { return g->pwgts; }
idx_t metis_bridge_graph_mincut(graph_t* g) { return g->mincut; }
idx_t metis_bridge_graph_nbnd(graph_t* g) { return g->nbnd; }

// --- InitialSeparator.h support ------------------------------------------

ctrl_t* metis_bridge_MakeCtrlForInitSep(graph_t* graph, int compress) {
  ctrl_t* ctrl = SetupCtrl(METIS_OP_OMETIS, NULL, 1, 3, NULL, NULL);
  ctrl->compress = compress;
  AllocateWorkSpace(ctrl, graph);
  return ctrl;
}

void metis_bridge_InitSeparator(ctrl_t* ctrl, graph_t* graph, idx_t niparts) {
  InitSeparator(ctrl, graph, niparts);
}

// Isolation helper: GrowBisection/RandomBisection alone, without the
// ConstructSeparator step that follows in InitSeparator.
void metis_bridge_GrowBisection(ctrl_t* ctrl, graph_t* graph, real_t* ntpwgts, idx_t niparts) {
  GrowBisection(ctrl, graph, ntpwgts, niparts);
}
void metis_bridge_RandomBisection(ctrl_t* ctrl, graph_t* graph, real_t* ntpwgts, idx_t niparts) {
  RandomBisection(ctrl, graph, ntpwgts, niparts);
}
void metis_bridge_Setup2WayBalMultipliers(ctrl_t* ctrl, graph_t* graph, real_t* ntpwgts) {
  Setup2WayBalMultipliers(ctrl, graph, ntpwgts);
}
void metis_bridge_Compute2WayPartitionParams(ctrl_t* ctrl, graph_t* graph) {
  Compute2WayPartitionParams(ctrl, graph);
}
idx_t* metis_bridge_graph_bndind(graph_t* g) { return g->bndind; }
idx_t* metis_bridge_graph_id(graph_t* g) { return g->id; }
idx_t* metis_bridge_graph_ed(graph_t* g) { return g->ed; }

}  // extern "C"
