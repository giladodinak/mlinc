/* Copyright (c) 2026 Gilad Odinak */
/* Dropout layer data functions    */

/* dropout.cu implements the same inverted-dropout operation as the
 * textbook dropout.c, but reorganizes the computation for GPU execution.
 *
 * The CPU version generates random values sequentially, applies the
 * corresponding dropout factors, and stores the mask for backward propagation.
 *
 * The CUDA version processes elements in parallel, using the counter-based
 * Philox random number generator. Each thread processes four elements,
 * generating four random values with a single curand_uniform4() call.
 * Both implementations zero dropped elements and scale retained elements
 * by 1/(1 - rate), preserving the same mathematical operation.
 *
 * Unlike the CPU version, CUDA does not store the dropout mask. Instead,
 * it maintains a random seed and a device-resident counter. Forward advances
 * the counter after generating the mask, while backward regenerates the
 * identical mask using the preceding counter value.
 *
 * This requires each dropout layer's forward and backward operations to
 * be paired before another forward on that layer. The mathematical dropout
 * operation is unchanged; the differences are parallel random-number
 * generation, mask regeneration, and CUDA Graph compatibility.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include "cuctx.h"
#include "float.h"
#include "mem.h"
#include "random.h"
#include "arraytypes.h"
#include "cumem.h"
#include "cuflt.h"
#include "dropout.h"

#define THDCNT 256

/* The seed is constant for the life of the layer and may be passed to
 * kernels by value. The call counter lives in device memory so that it
 * advances on every execution, including replays of a recorded CUDA graph.
 */
typedef struct {
    unsigned long long seed;
    unsigned long long* counter; /* Device memory, one element */
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

#define CURAND4 4  /* Number of floats returned by curand_uniform4() */

/* Generates a dropout mask and applies it in place.
 *
 * curand_uniform() returns values in (0,1]
 */
static __global__ void dropout_forward_kernel(float* mx,
                                              int n, float rate,
                                              unsigned long long seed,
                                              const unsigned long long* counter)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = CURAND4 * k;
    if (i >= n) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed,(unsigned long long) k,*counter,&st);
    float4 r = curand_uniform4(&st);

    float u[4] = { r.x, r.y, r.z, r.w };
    float scale = 1.0f / (1.0f - rate);

    #pragma unroll
    for (int j = 0; j < CURAND4; j++) {
        int idx = i + j;
        if (idx < n) {
            float value = (u[j] >= rate) ? scale : 0.0f;
            mx[idx] *= value;
        }
    }
}

/* Advances the device-resident call counter.
 *
 * Launched after dropout_forward_kernel on the same stream, so all threads
 * of the forward kernel have read the counter before it changes.
 */
static __global__ void dropout_advance_kernel(unsigned long long* counter,
                                              unsigned long long step)
{
    *counter += step;
}

/* Regenerates the forward dropout factors from the preceding Philox
 * counter value. Forward advances the counter by four after each call.
 * Forward and backward must be paired before the next forward on this layer.
 */
static __global__ void dropout_backward_kernel(const float* in,
                                               float* out,
                                               int n, float rate,
                                               unsigned long long seed,
                                               const unsigned long long* counter)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = CURAND4 * k;
    if (i >= n) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed,(unsigned long long) k,*counter - 4ULL,&st);
    float4 r = curand_uniform4(&st);
    float u[4] = { r.x, r.y, r.z, r.w };
    float scale = 1.0f / (1.0f - rate);

    #pragma unroll
    for (int j = 0; j < CURAND4; j++) {
        int idx = i + j;
        if (idx < n)
            out[idx] = in[idx] * ((u[j] >= rate) ? scale : 0.0f);
    }
}

/* Copies input to output when dropout is disabled (supports in-place). */
static __global__ void dropout_copy_kernel(const float* in,float* out,int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = k; i < n; i += stride)
        out[i] = in[i];
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
 *   Seeds this layer's counter-based RNG; no mask array is stored.
 *   The seed is kept in host memory and passed to kernels by value;
 *   the call counter is kept in device memory (zero-initialized).
 */
void dropout_init(DROPOUT* l, int input_dim, int batch_size)
{
    l->B = batch_size;
    l->D = input_dim;

    DROPOUT_RNG* rng = (DROPOUT_RNG*) allocmem(1,1,DROPOUT_RNG);
    rng->seed = dropout_mix_seed((unsigned long long) lrng());
    rng->counter = (unsigned long long*) cuallocmem(1,1,unsigned long long);
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
}

/* Frees the memory allocated by dropout_create() / dropout_init().
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void dropout_free(DROPOUT* l)
{
    if (l == NULL) return;
    if (l->rng_state != NULL) {
        DROPOUT_RNG* rng = (DROPOUT_RNG*) l->rng_state;
        cufreemem(rng->counter);
        freemem(rng);
    }
    freemem(l);
}

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 *
 * Notes:
 *   Rewinds the RNG call counter, so a run repeated from here produces the
 *   same sequence of masks. The reset is ordered on the current stream.
 */
void dropout_reset(DROPOUT* l)
{
    if (l == NULL || l->rng_state == NULL) return;
    DROPOUT_RNG* rng = (DROPOUT_RNG*) l->rng_state;
    cudaMemsetAsync(rng->counter,0,sizeof(*rng->counter),cuda_stream());
}

/* Applies dropout to a 2D array in-place (training only).
 *
 * Parameters:
 *   mx   : Pointer to the 2D array to be processed
 *
 * Notes:
 *   In a CUDA graph each replay reads and advances the
 *   device-resident counter, so every execution draws a new mask.
 */
void dropout_forward(DROPOUT* restrict l, fArr2D mx_/*[B][D])*/)
{
    int n = l->B * l->D;
    float rate = l->rate;

    if (n <= 0) return;
    if (rate <= 0.0f) return;

    if (rate >= 1.0f)
        rate = nextafterf(1.0f,0.0f);

    DROPOUT_RNG* rng = (DROPOUT_RNG*) l->rng_state;

    /* One thread per four elements; the whole array is covered in one
     * launch, so every element's mask comes from a distinct counter block.
     */
    int quads = (n + CURAND4 - 1) / CURAND4;
    int blocks = (quads + THDCNT - 1) / THDCNT;

    dropout_forward_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
        (float*) mx_,n,rate,rng->seed,rng->counter);

    /* Number of random numbers generated per thread */
    dropout_advance_kernel<<<1,1,0,cuda_stream()>>>(rng->counter,CURAND4);
}

/* Regenerates the preceding forward dropout mask and applies it to the
 * input, writing to an output array (in-place supported). Partial overlap
 * is not supported. Requires one forward per backward on this dropout layer.
 * This also works with CUDA Graph replay, because the counter is on device.
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

    float rate = l->rate;
    if (rate >= 1.0f)
        rate = nextafterf(1.0f,0.0f);

    if (rate <= 0.0f) {
        if (in_ != out_) {
            int blocks = (n + THDCNT - 1) / THDCNT;
            dropout_copy_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
                (const float*)in_,(float*)out_,n);
        }
        return;
    }

    DROPOUT_RNG* rng = (DROPOUT_RNG*) l->rng_state;
    int quads = (n + CURAND4 - 1) / CURAND4;
    int blocks = (quads + THDCNT - 1) / THDCNT;
    dropout_backward_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
        (const float*) in_,(float*) out_,n,rate,rng->seed,rng->counter);
}
