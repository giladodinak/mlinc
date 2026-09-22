/* Copyright (c) 2026 Gilad Odinak */
/* Dropout layer data functions    */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include "float.h"
#include "mem.h"
#include "random.h"
#include "arraytypes.h"
#include "cumem.h"
#include "cuflt.h"
#include "dropout.h"

#define THDCNT 256

typedef struct {
    unsigned long long seed;
    unsigned long long counter;
} DROPOUT_RNG;

/* Mixes random seed into a well distributed 64 bit key.
 *
 * References:
 *   - https://prng.di.unimi.it/splitmix64.c
 *   - https://dl.acm.org/doi/epdf/10.1145/2714064.2660195
 */
static unsigned long long dropout_mix_seed(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* Generates a dropout mask and applies it in place.
 *
 * curand_uniform() returns values in (0,1]
 */
static __global__ void dropout_forward_kernel(float* mx,
                                              float* mask,
                                              int n, float rate,
                                              unsigned long long seed,
                                              unsigned long long counter)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 4 * k;
    if (i >= n) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed,(unsigned long long) k,counter,&st);
    float4 r = curand_uniform4(&st);

    float u[4] = { r.x, r.y, r.z, r.w };
    float scale = 1.0f / (1.0f - rate);

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int idx = i + j;
        if (idx < n) {
            float value = (u[j] >= rate) ? scale : 0.0f;
            mx[idx] *= value;
            mask[idx] = value;
        }
    }
}

/* Applies an already-generated dropout mask */
static __global__ void dropout_backward_kernel(const float* in,
                                               const float* restrict mask,
                                               float* out,
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
 *   Allocates the device mask array and seeds this layer's counter based
 *   RNG. The RNG state itself lives in host memory; the device needs only
 *   the seed and the call counter, which are passed as kernel arguments.
 */
void dropout_init(DROPOUT* l, int input_dim, int batch_size)
{
    l->B = batch_size;
    l->D = input_dim;
    l->mask = (fArr2D) cuallocmem(l->B,l->D,float);

    DROPOUT_RNG* rng = (DROPOUT_RNG*) allocmem(1,1,DROPOUT_RNG);
    rng->seed = dropout_mix_seed((unsigned long long) lrng());
    rng->counter = 0;
    l->rng_state = rng;
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
    freemem(l->rng_state);
    freemem(l);
}

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 *
 * Notes:
 *   Rewinds the RNG call counter, so a run repeated from here produces the
 *   same sequence of masks.
 */
void dropout_reset(DROPOUT* l)
{
    if (l == NULL || l->rng_state == NULL) return;
    ((DROPOUT_RNG*) l->rng_state)->counter = 0;
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

    if (n <= 0) return;

    if (rate == 1.0f) {
        cufltclr(mx_,n);
        cufltclr(l->mask,n);
        return;
    }

    DROPOUT_RNG* rng = (DROPOUT_RNG*) l->rng_state;

    /* One thread per four elements; the whole array is covered in one
     * launch, so every element's mask comes from a distinct counter block.
     */
    int quads = (n + 3) / 4;
    int blocks = (quads + THDCNT - 1) / THDCNT;

    dropout_forward_kernel<<<blocks,THDCNT>>>(
        (float*) mx_,(float*) l->mask,n,rate,rng->seed,rng->counter);

    rng->counter += 4; /* Number of random numbers generated */
}

/* Applies a precomputed dropout mask to an input array, writing the
 * result to an output array, which may be the same array as the input
 * (in-place); partial overlap is not supported.
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
    if (n <= 0) return;

    int blocks = (n + THDCNT - 1) / THDCNT;
    if (blocks < 1)
        blocks = 1;

    dropout_backward_kernel<<<blocks,THDCNT>>>(
        (const float*) in_,(const float*) l->mask,(float*) out_,n);
}
