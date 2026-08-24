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

}  // extern "C"
