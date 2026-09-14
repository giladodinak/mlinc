/* Copyright (c) 2026 Gilad Odinak */
/* CUDA layer normalization data structure and functions */
#ifndef ADDNORM_H
#define ADDNORM_H

#include <math.h>
#include <cuda_runtime.h>
#include "float.h"
#include "arraytypes.h"

#ifndef THCNT
#define THCNT 256
#endif

typedef struct lnorm_s {
  int B;           /* Number of input vectors   */
  int D;           /* Input vector dimension    */
  fVec mean;       /* Calculated from input [B] */
  fVec sdev;       /* Calculated from input [B] */
  fArr2D xn;       /* Normalized values [B][D]  */
  fVec beta;       /* Learnable parameter [D]   */
  fVec gamma;      /* Learnable parameter [D]   */
} ADDNORM;

#ifdef __cplusplus
extern "C" {
#endif

ADDNORM* addnorm_create();
void addnorm_init(ADDNORM* l, int input_dim, int batch_size);
void addnorm_free(ADDNORM* l);

#ifdef __cplusplus
}
#endif

/* Adds and normalizes each row of x1 and x2, applies learnable gamma/beta,
 * and stores result in y.
 */
static __global__ void addnorm_forward_kernel(const float* x1,
                                              const float* x2,
                                              float* y,
                                              float* xn,
                                              float* mean_,
                                              float* sdev_,
                                              const float* gamma,
                                              const float* beta,
                                              int D)
{
    extern __shared__ float smem[];
    int r = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int j = tid; j < D; j += blockDim.x)
        sum += x1[r * D + j] + x2[r * D + j];

    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float mean = smem[0] / D;
    if (tid == 0)
        mean_[r] = mean;
    __syncthreads();

    float var = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float diff = x1[r * D + j] + x2[r * D + j] - mean;
        var += diff * diff;
    }

    smem[tid] = var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float sdev = sqrtf(smem[0] / D + 1e-9f);
    if (tid == 0)
        sdev_[r] = sdev;
    __syncthreads();

    for (int j = tid; j < D; j += blockDim.x) {
        int k = r * D + j;
        float v = (x1[k] + x2[k] - mean) / sdev;
        xn[k] = v;
        y[k] = gamma[j] * v + beta[j];
    }
}

static inline void addnorm_forward(ADDNORM* l,
                                   const fArr2D x1,
                                   const fArr2D x2,
                                   fArr2D y)
{
    int threads = THCNT;
    addnorm_forward_kernel<<<l->B,threads,threads * sizeof(float)>>>(
        (const float*)x1,(const float*)x2,(float*)y,(float*)l->xn,
        (float*)l->mean,(float*)l->sdev,
        (const float*)l->gamma,(const float*)l->beta,l->D);
}

static __global__ void addnorm_param_grad_kernel(const float* dy,
                                                 const float* xn,
                                                 float* dg,
                                                 float* db,
                                                 int B, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) {
        float gd = 0.0f;
        float bd = 0.0f;
        for (int i = 0; i < B; i++) {
            float d = dy[i * D + j];
            gd += d * xn[i * D + j];
            bd += d;
        }
        dg[j] = gd;
        db[j] = bd;
    }
}

static __global__ void addnorm_backward_kernel(const float* dy,
                                               float* dx,
                                               const float* xn,
                                               const float* gamma,
                                               const float* sdev,
                                               int D)
{
    extern __shared__ float smem[];
    float* s1 = smem;
    float* s2 = smem + blockDim.x;

    int r = blockIdx.x;
    int tid = threadIdx.x;

    float sum_dy = 0.0f;
    float sum_dy_xn = 0.0f;

    for (int j = tid; j < D; j += blockDim.x) {
        int k = r * D + j;
        float dyg = dy[k] * gamma[j];
        sum_dy += dyg;
        sum_dy_xn += dyg * xn[k];
    }

    s1[tid] = sum_dy;
    s2[tid] = sum_dy_xn;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s1[tid] += s1[tid + s];
            s2[tid] += s2[tid + s];
        }
        __syncthreads();
    }

    float mean_dy = s1[0] / D;
    float mean_dy_xn = s2[0] / D;
    float sd = sdev[r];

    for (int j = tid; j < D; j += blockDim.x) {
        int k = r * D + j;
        float dyg = dy[k] * gamma[j];
        dx[k] = (dyg - mean_dy - xn[k] * mean_dy_xn) / sd;
    }
}

/* Computes gradient of layer normalization. */
static inline void addnorm_backward(ADDNORM* l,
                                    const fArr2D dy,
                                    fArr2D dx,
                                    fVec dg,
                                    fVec db)
{
    int threads = THCNT;
    addnorm_param_grad_kernel<<<(l->D + threads - 1) / threads,threads>>>(
        (const float*)dy,(const float*)l->xn,(float*)dg,(float*)db,l->B,l->D);

    addnorm_backward_kernel<<<l->B,threads,2 * threads * sizeof(float)>>>(
        (const float*)dy,(float*)dx,(const float*)l->xn,
        (const float*)l->gamma,(const float*)l->sdev,l->D);
}

#endif
