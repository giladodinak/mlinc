/* Copyright (c) 2023-2024 Gilad Odinak */
/* Dense (feed forward) neural network functions */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include "mem.h"
#include "array.h"
#include "random.h"
#include "dense.h"

/* Creates a feed forward neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   activation - Can be one of "none", "sigmoid", "relu", "gelu", or "Softmax"
 *   use_bias   - If set add bias
 *
 * Returns:
 *   Pointer to a dense neural network layer.
 *
 * Notes:
 *   - The neural network needs to be further intialized using dense_init()
 *     before it can be used.
 */
DENSE* dense_create(int units, char* activation, int use_bias)
{
    DENSE* l = allocmem(1,1,DENSE);
    l->S = units;
    l->use_bias = (use_bias) ? 1 : 0;
    if (!strcasecmp("none",activation)) l->activation = 'n';
    if (!strcasecmp("sigmoid",activation)) l->activation = 's';
    if (!strcasecmp("relu",activation)) l->activation = 'r';
    if (!strcasecmp("gelu",activation)) l->activation = 'g';
    if (!strcasecmp("softmax",activation)) l->activation = 'S';
    if (l->activation == 0) {
        freemem(l);
        fflush(stdout);
        fprintf(stderr,"dense_create: invalid activation '%s'\n",activation);
        exit(-1);
    }
    return l;
}

/* Initializes a feed forward neural network created by dense_create().
 *
 *   input_dim  - Size of input vectors
 *   batch_size - Number of input vectors processed simultaneously
 *   training   - 1: allocate backward/gradient buffers, 0 for inference-only
 *
 * Notes:
 *   - The layer's weights are initialized using glorot normal distribution 
 */
void dense_init(DENSE* l, int input_dim, int batch_size, int training)
{
    l->D = input_dim;
    l->B = batch_size;
    l->training = (training) ? 1: 0;
    l->Wx = allocmem(l->D,l->S,float);
    l->h = allocmem(l->B,l->S,float);
    if (l->training && l->activation == 'g')
        l->z = allocmem(l->B,l->S,float);
    if (l->use_bias)
        l->b = allocmem(1,l->S,float);

    typedef float (*ArrDS)[l->S];
    ArrDS Wx = (ArrDS) l->Wx;
    float scale = sqrt(2.0 / (l->D + l->S));
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wx[i][j] = nrand(0.0,scale);

    if (l->training) {
        l->gWx = allocmem(l->D,l->S,float);
        if (l->use_bias)
            l->gb = allocmem(1,l->S,float);
    }
}

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dense_init(), it does nothing.
 *   Otherwise, the network's hidden state is resized and re-initialized
 */
void dense_set_batch_size(DENSE* l, int batch_size)
{
    if (l->B == 0)
        return;
    if (batch_size != l->B) {
        l->B = batch_size;
        freemem(l->h);
        l->h = allocmem(l->B,l->S,float);
        if (l->training && l->activation == 'g') {
            freemem(l->z);
            l->z = allocmem(l->B,l->S,float);
        }
    }
    else
        fltclr(l->h,l->B * l->S);
}

/* Frees the memory allocated by dense_create() / dense_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void dense_free(DENSE* l)
{
    freemem(l->h);
    freemem(l->z);
    freemem(l->Wx);
    freemem(l->gWx);
    freemem(l->b);
    freemem(l->gb);
    freemem(l);
}

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the DENSE neural network layer to be reset
 */
void dense_reset(DENSE* l)
{
    (void) l; /* Do nothing */
}

