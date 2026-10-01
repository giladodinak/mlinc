/* Copyright (c) 2026 Gilad Odinak */
/* GPU graph execution             */
#ifndef GRAPH_H
#define GRAPH_H

#include <cuda_runtime.h>

#define GRAPH_FORWARD   0
#define GRAPH_BACKWARD  1
#define GRAPH_COUNT     2

#define GRAPH_RECORD_OFF 0
#define GRAPH_RECORD_ON  1

typedef struct graph_s {
    cudaStream_t stream;
    cudaStream_t old_stream;
    cudaGraphExec_t exec[GRAPH_COUNT];
    int recording;
} GRAPH;

/* Creates an empty graph object.
 * Returns NULL on error.
 */
GRAPH* graph_create(void);


/* Waits for graph execution to complete and destroys the graph object.
 */
cudaError_t graph_destroy(GRAPH* g);


/* Starts or stops recording the specified graph.
 */
cudaError_t graph_record(GRAPH* g, int id, int on);


/* Returns nonzero if the specified graph has been recorded.
 */
int graph_record_exists(const GRAPH* g, int id);


/* Executes a recorded graph.
 */
cudaError_t graph_play(GRAPH* g, int id);


/* Returns the stream used to record and execute the graph.
 */
cudaStream_t graph_stream(const GRAPH* g);

#endif
