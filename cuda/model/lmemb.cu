/* Copyright (c) 2026 Gilad Odinak */
 /* Language-model token embedding layer functions - CUDA */
#include <math.h>
#include <cuda_runtime.h>
#include "mem.h"
#include "cumem.h"
#include "float.h"
#include "cuflt.h"
#include "array.h"
#include "random.h"
#include "lmemb.h"

#define THCNT 256

typedef struct devstat_s {
    int* dids;   /* Device copy of current token ids [B*T] */
    int* drows;  /* Device copy of touched rows [B*T]      */
} DEVSTAT;

static __global__ void lmemb_forward_kernel(const int* ids,
                                            const float* Wx,
                                            float* h,
                                            int BT, int E)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = BT * E;
    if (k < n) {
        int i = k / E;
        int e = k % E;
        h[k] = Wx[ids[i] * E + e];
    }
}

static __global__ void clear_rows_kernel(float* gWx, const int* rows,
                                         int nrows, int E)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nrows * E;
    if (k < n) {
        int r = k / E;
        int e = k % E;
        gWx[rows[r] * E + e] = 0.0f;
    }
}

static __global__ void lmemb_backward_kernel(const float* dy,
                                             const int* ids,
                                             float* gWx,
                                             int BT, int E, int padinx)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = BT * E;
    if (k < n) {
        int i = k / E;
        int e = k % E;
        int id = ids[i];
        if (id != padinx)
            atomicAdd(&gWx[id * E + e],dy[k]);
    }
}

static __global__ void lmemb_update_kernel(float* Wx, const float* gWx,
                                           const int* rows, int nrows,
                                           int E, float learning_rate)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nrows * E;
    if (k < n) {
        int r = k / E;
        int e = k % E;
        int row = rows[r];
        int p = row * E + e;
        Wx[p] -= learning_rate * gWx[p];
    }
}

/* Creates a language-model token embedding layer.
 *
 * Parameters:
 *   embedding_dim - Dimension of token embedding vectors (E)
 *   steps         - Sequence length T (number of tokens per sequence)
 *   padinx        - Index value of pad token, -1 if not used
 *
 * Returns:
 *   Pointer to an LMEMB layer.
 *
 * Notes:
 *   - The layer needs to be further initialized using lmemb_init() before
 *     it can be used.
 */
LMEMB* lmemb_create(int embedding_dim, int steps, int padinx)
{
    LMEMB* l = (LMEMB*) allocmem(1,1,LMEMB);
    l->E = embedding_dim;
    l->T = steps;
    l->padinx = padinx;
    l->h = NULL;
    l->Wx = NULL;
    l->gWx = NULL;
    l->mWx = NULL;
    l->vWx = NULL;
    l->seen = NULL;
    l->touched = NULL;
    l->ids = NULL;
    l->devstat = NULL;
    return l;
}

/* Initializes a language-model token embedding layer created by lmemb_create().
 *
 * Parameters:
 *   vocab_size - Number of vocabulary tokens V (including pad, if any)
 *   batch_size - Number of sequences processed simultaneously, B.
 *   training   - 1: allocate backward/optimizer buffers; 0: inference only
 *   tied       - 1: Wx/gWx/moments are borrowed from an output projection and
 *                   are set later with lmemb_set_tied_weights(); 0: owned here
 *
 * Notes:
 *   - When not tied, Wx is allocated here. If training, it is initialized with
 *     a normal distribution scaled by 1/sqrt(E), and the pad row is zeroed;
 *     if not training, Wx is left for the caller to load.
 *   - When tied, no weight/gradient/moment buffers are allocated here.
 *   - The output buffer h and, when training, the sparse touched-row buffers
 *     are always allocated (they are owned regardless of tying).
 */
void lmemb_init(LMEMB* l, int vocab_size, int batch_size, int training, int tied)
{
    l->V = vocab_size;
    l->B = batch_size;
    l->training = training;
    l->tied = tied;
    l->stamp = 0;
    l->ntouched = 0;
    l->BT = l->B * l->T;

    l->h = (fArr2D) cuallocmem(l->BT,l->E,float);

    DEVSTAT* d = (DEVSTAT*) allocmem(1,1,DEVSTAT);
    d->dids = (int*) cuallocmem(l->BT,1,int);
    d->drows = training ? (int*) cuallocmem(l->BT,1,int) : NULL;
    l->devstat = d;

    if (!tied) {
        l->Wx = (fArr2D) cuallocmem(l->V,l->E,float);
        if (training) {
            float* Wx = (float*) allocmem(l->V,l->E,float);
            float scale = 1.0f / sqrtf((float) l->E);
            for (int i = 0; i < l->V * l->E; i++)
                Wx[i] = nrand(0.0,scale);
            if (l->padinx >= 0 && l->padinx < vocab_size)
                fltclr(&Wx[l->padinx * l->E],l->E);
            cufltcpy(l->Wx,Wx,l->V * l->E);
            freemem(Wx);
        }
    }
    /* When tied, Wx/gWx are set later via lmemb_set_tied_weights() */

    if (!training)
        return;

    /* Backward / optimizer buffers */
    if (!tied) {
        l->gWx = (fArr2D) cuallocmem(l->V,l->E,float);
        l->mWx = (fArr2D) cuallocmem(l->V,l->E,float);
        l->vWx = (fArr2D) cuallocmem(l->V,l->E,float);
    }
    /* Sparse touched-row tracking is owned regardless of tying */
    l->seen = (int*) allocmem(l->V,1,int);
    for (int i = 0; i < l->V; i++)
        l->seen[i] = -1;
    l->touched = (int*) allocmem(l->BT,1,int);
}

/* Points a tied layer's Wx/gWx at the output projection's shared buffers.
 * Only valid when the layer was initialized with tied = 1.
 *
 * Parameters:
 *   Wx  - Shared weight matrix   [V][E]
 *   gWx - Shared weight gradient  [V][E]
 */
void lmemb_set_tied_weights(LMEMB* l, fArr2D Wx, fArr2D gWx)
{
    if (!l->tied) return;
    l->Wx = Wx;
    l->gWx = gWx;
    /* mWx/vWx remain NULL: the output layer owns and steps the moments */
}

/* Frees the memory allocated by lmemb_create() / lmemb_init().
 * When tied, the borrowed Wx/gWx/moments are not freed.
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void lmemb_free(LMEMB* l)
{
    if (l == NULL) return;
    cufreemem(l->h);
    if (!l->tied) {
        cufreemem(l->Wx);
        cufreemem(l->gWx);
        cufreemem(l->mWx);
        cufreemem(l->vWx);
    }

    if (l->devstat != NULL) {
        DEVSTAT* d = (DEVSTAT*) l->devstat;
        cufreemem(d->dids);
        cufreemem(d->drows);
        freemem(d);
    }

    freemem(l->seen);
    freemem(l->touched);
    freemem(l);
}

/* Resets any per-batch state the layer carries.
 *
 * Parameters:
 *   l - Pointer to the LMEMB layer to be reset
 */
void lmemb_reset(LMEMB* l)
{
    l->ntouched = 0;
    /* stamp continues to advance; seen[] stays valid across resets */
}

/* Performs the token embedding forward pass: a per-token gather.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   ids - An array of B*T token indices [B*T]
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the output values, h[B*T][E]. Row t is Wx[ids[t]].
 *
 * The ids pointer is retained for the matching backward pass.
 */
fArr2D lmemb_forward(LMEMB* restrict l,
                     const int* restrict ids/*[B*T]*/, int lyr)
{
    (void) lyr;
    DEVSTAT* d = (DEVSTAT*) l->devstat;

    l->ids = ids;
    cudaMemcpy(d->dids,ids,l->BT * sizeof(int),cudaMemcpyHostToDevice);

    int n = l->BT * l->E;
    lmemb_forward_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
        d->dids,(const float*)l->Wx,(float*)l->h,l->BT,l->E);
    return l->h;
}

/* Performs the token embedding backward pass: a per-token scatter-add.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   dy  - The output gradient [B*T][E], from the layer above (the stack input)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Accumulates each token's gradient into its Wx row of gWx, recording the
 * distinct touched rows in l->touched / l->ntouched for a sparse update.
 * The token indices from the last lmemb_forward() are used. There is no
 * input gradient: this is the first layer of the model.
 *
 * Rows are zeroed on first touch only when this layer owns gWx (untied);
 * when tied, the shared gWx is prepared by the output layer and this pass
 * only accumulates. A token index appearing at several positions accumulates.
 */
void lmemb_backward(LMEMB* restrict l,
                    const fArr2D restrict dy/*[B*T][E]*/, int lyr)
{
    (void) lyr;
    DEVSTAT* d = (DEVSTAT*) l->devstat;
    const int* ids = l->ids;

    l->stamp++;
    l->ntouched = 0;

    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        if (id == l->padinx)
            continue;
        if (l->seen[id] == l->stamp)
            continue;
        l->seen[id] = l->stamp;
        l->touched[l->ntouched++] = id;
    }

    if (l->ntouched > 0) {
        cudaMemcpy(d->drows,l->touched,l->ntouched * sizeof(int),
                   cudaMemcpyHostToDevice);

        if (!l->tied) {
            int n = l->ntouched * l->E;
            clear_rows_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
                (float*)l->gWx,d->drows,l->ntouched,l->E);
        }
    }

    int n = l->BT * l->E;
    lmemb_backward_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
        (const float*)dy,d->dids,(float*)l->gWx,
        l->BT,l->E,l->padinx);
}

/* Applies one sparse SGD step to rows touched by lmemb_backward().
 * Weight decay is intentionally omitted for embeddings.
 */
void lmemb_update(LMEMB* restrict l, float learning_rate)
{
    if (l->tied || l->ntouched == 0)
        return;

    DEVSTAT* d = (DEVSTAT*) l->devstat;
    int n = l->ntouched * l->E;
    lmemb_update_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
        (float*)l->Wx,(const float*)l->gWx,d->drows,l->ntouched,
        l->E,learning_rate);
}
