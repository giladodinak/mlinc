/* Copyright (c) 2026 Gilad Odinak */
/* Embedding neural network layer functions - CUDA */
#include <math.h>
#include <cuda_runtime.h>
#include "cuctx.h"
#include "float.h"
#include "mem.h"
#include "cuflt.h"
#include "cumem.h"
#include "array.h"
#include "random.h"
#include "embedding.h"

#define THDCNT 256

typedef struct devstat_s {
    float* dX;   /* Device copy of input contexts [B][M] */
    int* drows;  /* Device copy of touched rows [B*M]    */
} DEVSTAT;

static __global__ void embedding_forward_kernel(const float* X,
                                                const float* Wx,
                                                float* h,
                                                int B, int M, int E)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * E;
    if (k < n) {
        int i = k / E;
        int e = k % E;
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            int w = (int)X[i * M + j];
            sum += Wx[w * E + e];
        }
        h[k] = sum;
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

static __global__ void embedding_backward_kernel(const float* dy,
                                                 const float* X,
                                                 float* gWx,
                                                 int B, int M, int E,
                                                 int padinx)
{
    int i = blockIdx.x;
    int e = threadIdx.x;
    if (i < B && e < E) {
        float g = dy[i * E + e];
        for (int j = 0; j < M; j++) {
            int w = (int)X[i * M + j];
            if (w != padinx)
                atomicAdd(&gWx[w * E + e],g);
        }
    }
}

static __global__ void embedding_update_kernel(float* Wx, const float* gWx,
                                               const int* rows, int nrows,
                                               int E, float learning_rate)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nrows * E;
    if (k < n) {
        int r = k / E;
        int e = k % E;
        int w = rows[r];
        Wx[w * E + e] -= learning_rate * gWx[w * E + e];
    }
}

EMBEDDING* embedding_create(int embedding_dim, int context_len, int padinx)
{
    EMBEDDING* l = (EMBEDDING*) allocmem(1,1,EMBEDDING);
    l->E = embedding_dim;
    l->M = context_len;
    l->padinx = padinx;
    l->h = NULL;
    l->Wx = NULL;
    l->devstat = NULL;
    return l;
}

void embedding_init(EMBEDDING* l, int vocab_size, int batch_size, int training)
{
    l->D = vocab_size;
    l->B = batch_size;
    l->Wx = (fArr2D)cuallocmem(l->D,l->E,float);

    float* Wx = (float*)allocmem(l->D,l->E,float);
    float scale = 1.0f / l->E;
    for (int i = 0; i < l->D * l->E; i++)
        Wx[i] = urand(-scale,scale);
    if (l->padinx >= 0 && l->padinx < vocab_size)
        fltclr(&Wx[l->padinx * l->E],l->E);
    cufltcpy(l->Wx,Wx,l->D * l->E);
    freemem(Wx);

    if (!training)
        return;

    l->h = (fArr2D)cuallocmem(l->B,l->E,float);

    DEVSTAT* d = (DEVSTAT*)allocmem(1,1,DEVSTAT);
    d->dX = (float*)cuallocmem(l->B,l->M,float);
    d->drows = (int*)cuallocmem(l->B * l->M,1,int);
    l->devstat = d;
}

void embedding_free(EMBEDDING* l)
{
    cufreemem(l->Wx);
    cufreemem(l->h);
    if (l->devstat != NULL) {
        DEVSTAT* d = (DEVSTAT*)l->devstat;
        cufreemem(d->dX);
        cufreemem(d->drows);
        freemem(d);
    }
    freemem(l);
}

void embedding_reset(EMBEDDING* l)
{
    (void)l;
}

fArr2D embedding_forward(EMBEDDING* restrict l,
                         fArr2D restrict X, int lyr)
{
    (void)lyr;
    DEVSTAT* d = (DEVSTAT*)l->devstat;
    cufltcpy(d->dX,X,l->B * l->M);

    int n = l->B * l->E;
    int blocks = (n + THDCNT - 1) / THDCNT;
    embedding_forward_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
                   d->dX,(const float*)l->Wx,(float*)l->h,l->B,l->M,l->E);
    return l->h;
}

void embedding_backward(EMBEDDING* restrict l,
                        const fArr2D restrict dy,
                        const fArr2D restrict X_,
                        fArr2D restrict gWx,
                        int* grows, int* grcnt, int lyr)
{
    (void)lyr;
    DEVSTAT* d = (DEVSTAT*)l->devstat;

    int nrows = 0;
    if (grows != NULL && grcnt != NULL) {
        typedef float (*ArrBM)[l->M];
        ArrBM X = (ArrBM)X_;
        for (int i = 0; i < l->B; i++) {
            for (int j = 0; j < l->M; j++) {
                int w = (int)X[i][j];
                if (w == l->padinx)
                    continue;
                int found = 0;
                for (int r = 0; r < nrows; r++) {
                    if (grows[r] == w) {
                        found = 1;
                        break;
                    }
                }
                if (!found)
                    grows[nrows++] = w;
            }
        }
        *grcnt = nrows;

        if (nrows > 0) {
            cudaMemcpy(d->drows,grows,
                       nrows * sizeof(int),cudaMemcpyHostToDevice);
            int n = nrows * l->E;
            int blocks = (n + THDCNT - 1) / THDCNT;
            clear_rows_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
                                           (float*)gWx,d->drows,nrows,l->E);
        }
    }
    else {
        cufltclr(gWx,l->D * l->E);
    }

    embedding_backward_kernel<<<l->B,THDCNT,0,cuda_stream()>>>(
               (const float*)dy,d->dX,(float*)gWx,l->B,l->M,l->E,l->padinx);
}

void embedding_update(EMBEDDING* restrict l,
                      const fArr2D restrict gWx,
                      const int* grows, int grcnt, float learning_rate)
{
    if (grcnt <= 0)
        return;

    DEVSTAT* d = (DEVSTAT*)l->devstat;
    cudaMemcpy(d->drows,grows,grcnt * sizeof(int),cudaMemcpyHostToDevice);

    int n = grcnt * l->E;
    int blocks = (n + THDCNT - 1) / THDCNT;
    embedding_update_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
        (float*)l->Wx,(const float*)gWx,d->drows,grcnt,l->E,learning_rate);
}
