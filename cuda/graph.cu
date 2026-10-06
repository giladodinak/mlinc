/* Copyright (c) 2026 Gilad Odinak */
/* GPU graph execution             */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuctx.h"
#include "float.h"
#include "mem.h"
#include "arraytypes.h"
#include "array.h"
#include "graph.h"

/* Creates an empty graph object */
GRAPH* graph_create(void)
{
    GRAPH* g = (GRAPH*) allocmem(1,1,GRAPH);
    g->recording = -1;
    cudaError_t err;
    err = cudaStreamCreate(&g->stream);
    if (err != cudaSuccess) {
        freemem(g);
        return NULL;
    }
    return g;
}

/* Waits for graph execution to complete and destroys the graph object */
void graph_destroy(GRAPH* g)
{
    if (g == NULL) return;
    cudaError_t err;
    err = cudaStreamSynchronize(g->stream);
    for (int id = 0; id < GRAPH_COUNT; id++) {
        if (g->exec[id] != NULL)
            err = cudaGraphExecDestroy(g->exec[id]);        
    }
    err = cudaStreamDestroy(g->stream);
    freemem(g);
    err = err;
}


/* Starts or stops recording the specified graph */
void graph_record(GRAPH* g, int id, int on)
{
    cudaError_t err = cudaErrorInvalidValue;
    if (g == NULL || id < 0 || id >= GRAPH_COUNT) goto Err;
    
    if (on) {
        if (g->exec[id] != NULL || g->recording >= 0) goto Err;
        g->old_stream = cuda_set_stream(g->stream);

        err = cudaStreamBeginCapture(g->stream,cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess) goto Err;

        g->recording = id;
    }
    else {
        if (g->recording != id) goto Err;
        cudaGraph_t graph = NULL;

        err = cudaStreamEndCapture(g->stream,&graph);
        if (err != cudaSuccess) goto Err;

        g->recording = -1;
        cuda_set_stream(g->old_stream);

        err = cudaGraphInstantiate(&g->exec[id],graph,0);
        if (err != cudaSuccess) goto Err;

        err = cudaGraphDestroy(graph);
        if (err != cudaSuccess) goto Err;
    }
    return;
Err:
    fprintf(stderr,"graph_record failed (%d)\n",(int) err);
    exit(1);
}

/* Returns nonzero if the specified graph has been recorded */
int graph_record_exists(const GRAPH* g, int id)
{
    if (g == NULL || id < 0 || id >= GRAPH_COUNT)
        return 0;
    return g->exec[id] != NULL;
}


/* Executes a recorded graph */
void graph_play(GRAPH* g, int id)
{
    cudaError_t err = cudaErrorInvalidValue;
    if (g == NULL || id < 0 || id >= GRAPH_COUNT || g->exec[id] == NULL)
        goto Err;

    err = cudaGraphLaunch(g->exec[id],g->stream);
    if (err != cudaSuccess) goto Err;
    
    return;
Err:
    fprintf(stderr,"graph_play failed (%d)\n",(int) err);
    exit(1);
}


/* Returns the stream used to record and execute the graph */
cudaStream_t graph_stream(const GRAPH* g)
{
    return g->stream;
}
