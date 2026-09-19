/* Copyright (c) 2026 Gilad Odinak */
/* Dropout layer data functions    */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "float.h"
#include "mem.h"
#include "random.h"
#include "arraytypes.h"
#include "cumem.h"
#include "cuflt.h"
#include "curand.h"
#include "dropout.h"

#define RNG_STATES 1024
#define THDCNT      256

/* Initialize seed per RNG worker */
static __global__ void dropout_init_rng(int* restrict state, int base)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < RNG_STATES) {
        /* Avoid addition integer overflow and 0 seed */
        state[k] = (((long long)base - 1 + k) %  2147483646) + 1;
    }
}

/* Applies dropout in-place. Each active CUDA worker owns one RNG seed and
 * advances that seed while processing its grid-stride sequence of elements.
 */
static __global__ void dropout_forward_kernel(float* restrict mx,
                                              float* restrict mask,
                                              int n, float rate,
                                              int* restrict rng_state)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float scale = 1.0 / (1.0 - rate);

    for (int i = k; i < n; i += stride) {
        float value = cu_urand(&rng_state[k],0,1) >= rate ? scale : 0.0;
        mx[i] *= value;
        mask[i] = value;
    }
}

/* Applies an already-generated dropout mask. */
static __global__ void dropout_backward_kernel(const float* restrict in,
                                               const float* restrict mask,
                                               float* restrict out,
                                               int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = k; i < n; i += stride)
        out[i] = in[i] * mask[i];
}

/* Creates a dropout layer.
 *
 * Parameters:
 *   rate - Dropout rate (0.0 to 1.0)
 *
 * Returns:
 *   Pointer to a dropout
 *
 * Notes:
 *   - The layer needs to be further initialized using dropout_init()
 *     before it can be used.
 */
DROPOUT* dropout_create(float rate)
{
    DROPOUT *l = (DROPOUT*) allocmem(1,1,DROPOUT);
    if (rate < 0) rate = 0;
    else if (rate > 1) rate = 1;
    l->rate = rate;
    return l;
}

/* Initializes a dropout layer created by dropout_create().
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *   input_dim  - Size of input vectors
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(D).
 */
void dropout_init(DROPOUT* l, int input_dim, int batch_size)
{
    l->B = batch_size;
    l->D = input_dim;
    l->mask = (fArr2D) cuallocmem(l->B,l->D,float);

    l->rng_state = cuallocmem(RNG_STATES,1,int);
    int blocks = RNG_STATES / THDCNT;
    dropout_init_rng<<<blocks,THDCNT>>>((int*) l->rng_state,lrng());
}

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dropout_init(), it does nothing.
 */
void dropout_set_batch_size(DROPOUT* l, int batch_size)
{
    if (l->B == 0 || l->B == batch_size) return;
    l->B = batch_size;
    cufreemem(l->mask);
    l->mask = (fArr2D) cuallocmem(l->B,l->D,float);
}

/* Frees the memory allocated by dropout_create() / dropout_init().
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void dropout_free(DROPOUT* l)
{
    if (l == NULL) return;
    cufreemem(l->mask);
    cufreemem(l->rng_state);
    freemem(l);
}

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 */
void dropout_reset(DROPOUT* l)
{
    (void) l; /* do nothing */
}

/* Applies dropout to a 2D array in-place (training only).
 *
 * Parameters:
 *   mx   : Pointer to the 2D array to be processed
 */
void dropout_forward(DROPOUT* restrict l, fArr2D mx_/*[B][D])*/)
{
    int n = l->B * l->D;
    float rate = l->rate;

    if (rate == 1.0f) {
        cufltclr(mx_,n);
        return;
    }

    int blocks = (n + THDCNT - 1) / THDCNT;
    if (blocks > RNG_STATES / THDCNT)
        blocks = RNG_STATES / THDCNT;
    if (blocks < 1)
        blocks = 1;

    dropout_forward_kernel<<<blocks,THDCNT>>>(
        (float*) mx_,(float*) l->mask,n,rate,(int*) l->rng_state);
}

/* Applies a precomputed dropout mask to an input array, writing the
 * result to a separate output array without modifying the input.
 *
 * Parameters:
 *   in  : Pointer to the 2D input array to be masked
 *   out : Pointer to the 2D output array to receive the masked result
 */
void dropout_backward(DROPOUT* restrict l,
                      const fArr2D in_/*[B][D]*/,
                      fArr2D out_/*[B][D]*/)
{
    int n = l->B * l->D;
    int blocks = (n + THDCNT - 1) / THDCNT;
    if (blocks < 1)
        blocks = 1;

    dropout_backward_kernel<<<blocks,THDCNT>>>(
        (const float*) in_,(const float*) l->mask,(float*) out_,n);
}
