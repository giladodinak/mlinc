/* Copyright (c) 2026 Gilad Odinak */
/* GPU graph execution             */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "float.h"
#include "mem.h"
#include "arraytypes.h"
#include "array.h"
#include "graph.h"

/* Creates an empty graph object.
 * Returns NULL on error.
 */
GRAPH* graph_create(void)
{
    GRAPH* g = (GRAPH*) allocmem(1,1,GRAPH);
    if (g == NULL)
        return NULL;

    g->recording = -1;

    cudaError_t err;
    err = cudaStreamCreate(&g->stream);
    if (err != cudaSuccess)
        return NULL;

    return g;
}

/* Waits for graph execution to complete and destroys the graph object.
 */
cudaError_t  graph_destroy(GRAPH* g)
{
    if (g == NULL)
        return cudaSuccess;

    cudaError_t err;

    err = cudaStreamSynchronize(g->stream);
    if (err != cudaSuccess)
        return err;

    for (int id = 0; id < GRAPH_COUNT; id++) {
        if (g->exec[id] != NULL) {
            err = cudaGraphExecDestroy(g->exec[id]);
            if (err != cudaSuccess)
                return err;
        }
    }

    err = cudaStreamDestroy(g->stream);
    if (err != cudaSuccess)
        return err;

    freemem(g);

    return cudaSuccess;
}


/* Starts or stops recording the specified graph.
 */
cudaError_t graph_record(GRAPH* g, int id, int on)
{
    if (g == NULL || id < 0 || id >= GRAPH_COUNT)
        return cudaErrorInvalidValue;

    cudaError_t err;
    cublasStatus_t status;
    cublasHandle_t blas = cublas_handle(); /* in array.h */

    if (on) {
        if (g->exec[id] != NULL || g->recording >= 0)
            return cudaErrorInvalidValue;

        status = cublasGetStream(blas,&g->old_stream);
        if (status != CUBLAS_STATUS_SUCCESS)
            return cudaErrorUnknown;

        status = cublasSetStream(blas,g->stream);
        if (status != CUBLAS_STATUS_SUCCESS)
            return cudaErrorUnknown;

        err = cudaStreamBeginCapture(g->stream,cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess)
            return err;

        g->recording = id;

        return cudaSuccess;
    }

    if (g->recording != id)
        return cudaErrorInvalidValue;

    cudaGraph_t graph = NULL;

    err = cudaStreamEndCapture(g->stream,&graph);
    if (err != cudaSuccess)
        return err;

    g->recording = -1;

    status = cublasSetStream(blas,g->old_stream);
    if (status != CUBLAS_STATUS_SUCCESS)
        return cudaErrorUnknown;

    err = cudaGraphInstantiate(&g->exec[id],graph,0);
    if (err != cudaSuccess)
        return err;

    err = cudaGraphDestroy(graph);
    if (err != cudaSuccess)
        return err;

    return cudaSuccess;
}


/* Returns nonzero if the specified graph has been recorded.
 */
int graph_record_exists(const GRAPH* g, int id)
{
    if (g == NULL || id < 0 || id >= GRAPH_COUNT)
        return 0;

    return g->exec[id] != NULL;
}


/* Executes a recorded graph.
 */
cudaError_t graph_play(GRAPH* g, int id)
{
    if (g == NULL || id < 0 || id >= GRAPH_COUNT || g->exec[id] == NULL)
        return cudaErrorInvalidValue;

    cudaError_t err;
    err = cudaGraphLaunch(g->exec[id],g->stream);
    if (err != cudaSuccess)
        return err;

    return cudaSuccess;
}


/* Returns the stream used to record and execute the graph.
 */
cudaStream_t graph_stream(const GRAPH* g)
{
    return g->stream;
}
