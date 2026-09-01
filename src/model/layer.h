/* Copyright (c) 2023-2024 Gilad Odinak        */
/* Uniform wrappers over the per-layer-type    */
#ifndef LAYER_H
#define LAYER_H
#include <stdio.h>
#include "array.h"
#include "dense.h"
#include "lstm.h"
#include "transformer.h"
#include "negsample.h"

typedef struct layer_s {
    char type;      /* (d)ense (l)stm (t)ransformer (n)egsample */
    union {
        DENSE* dense;
        LSTM* lstm;
        TRANSFORMER* transformer;
        NEGSAMPLE* negsample;
    };
    fArr2D* grads;  /* Array of gradients and adam momentums    */
    int num_grads;  /* Number of entries in grads[]             */
    fArr2D out;     /* Scratch buffer                           */
} LAYER;

/* Reports use of a not-yet-implemented layer type and aborts. */
static inline void layer_unsupported(const char* fn, char type)
{
    fflush(stdout);
    fprintf(stderr,"%s: layer type '%c' not supported\n",fn,type);
    exit(-1);
}

/* Initializes the underlying layer for the given input/batch size.
 *
 * Returns:
 *   The layer's output dimension (used as the next layer's input_dim).
 */
int layer_init(LAYER* l, int input_dim, int batch_size);

/* Returns the layer's output dimension (size of its output vectors). */
static inline int layer_output_dim(const LAYER* l)
{
    switch (l->type) {
        case 'd': return l->dense->S;
        case 'l': return l->lstm->S;
        case 't': return l->transformer->D; /* in == out  */
        case 'n': return l->negsample->E;   /* passthrout */
    }
    layer_unsupported("layer_output_dim",l->type);
    return 0; /* not reached */
}

/* Returns the layer's batch size (rows in its output). */
static inline int layer_batch_size(const LAYER* l)
{
    switch (l->type) {
        case 'd': return l->dense->B;
        case 'l': return l->lstm->B;
        case 't': return l->transformer->BT; /* row count is B*T */
        case 'n': return l->negsample->B;
    }
    layer_unsupported("layer_batch_size",l->type);
    return 0; /* not reached */
}

/* Runs the layer's forward pass.
 *
 * Parameters:
 *   X   - Input array [B][D]
 *   lyr - Ordinal number of this layer in the model
 *
 * Returns:
 *   Pointer to the layer's output array [B][S].
 */
static inline fArr2D layer_forward(LAYER* l, 
                                   const fArr2D X, int training, int lyr)
{
    switch (l->type) {
        case 'd': return dense_forward(l->dense,X,lyr);
        case 'l': return lstm_forward(l->lstm,X,lyr);
        case 't':
            transformer_forward(l->transformer,X,NULL,training,l->out,lyr);
            return l->out;
        case 'n': return negsample_forward(l->negsample,X,lyr);
    }
    layer_unsupported("layer_forward",l->type);
    return NULL; /* Not reached */
}

/* Runs the layer's backward pass.
 *
 * Accumulates weight gradients into l->grads (allocated by
 * layer_alloc_grads()) and, if dx is not NULL, writes the input
 * gradient into dx.
 *
 * Parameters:
 *   dy  - Output gradient [B][S]
 *   X   - The input that produced this layer's output [B][D]
 *   dx  - Input gradient [B][D] (may be NULL for the first layer)
 *   lyr - Ordinal number of this layer in the model
 */
static inline void layer_backward(LAYER* l, fArr2D dy, 
                                  const fArr2D X, fArr2D dx, int lyr)
{
    switch (l->type) {
        case 'd':
            dense_backward(l->dense,dy,X,l->grads[0],dx,lyr);
            return;
        case 'l':
            lstm_backward(l->lstm,dy,X,l->grads,dx,lyr);
            return;
        case 't':
            transformer_backward(l->transformer,dy,X,dx,lyr);
            return;
        case 'n':
            negsample_backward(l->negsample,dy,dx,lyr);
            return;
    }
    layer_unsupported("layer_backward",l->type);
}

/* Resets any state the layer carries across batches. */
void layer_reset(LAYER* l);

/* Frees the underlying layer object (not l->grads; see model_free). */
void layer_free(LAYER* l);

/* Resizes / re-initializes the layer for a new batch size. */
void layer_set_batch_size(LAYER* l, int batch_size);

/* Allocates the layer's gradient (and optimizer-moment) arrays into
 * l->grads / l->num_grads, sized for the given optimizer
 * ('l' linear, 'a' adamw).
 */
void layer_alloc_grads(LAYER* l, char optimizer);

/* Applies one optimizer step to the layer's weights using l->grads. */
void layer_update(LAYER* l, char optimizer,
                  float learning_rate, float weight_decay, int update_cnt);

#endif
