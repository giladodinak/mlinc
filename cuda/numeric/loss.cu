/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA loss functions and their derivatives */
#include <cuda_runtime.h>
#include <math.h>
#include "loss.h"

#define THCNT 256

static __global__ void cross_entropy_loss_kernel(const float* yp,
                                                 const float* yt,
                                                 float* loss,
                                                 int n)
{
    extern __shared__ float tmp[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x)
        sum += -yt[i] * logf(fmaxf(yp[i],1e-16f));

    tmp[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            tmp[tid] += tmp[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        *loss = tmp[0];
}

extern "C" float cross_entropy_loss(const fArr2D yp_,
                                    const fArr2D yt_,
                                    int T, int K)
{
    float* dloss;
    float loss;

    cudaMalloc(&dloss,sizeof(float));
    cross_entropy_loss_kernel<<<1,THCNT,THCNT * sizeof(float)>>>(
        (const float*) yp_,(const float*) yt_,dloss,T * K);
    cudaMemcpy(&loss,dloss,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dloss);

    return loss;
}


static __global__ void sparse_cross_entropy_loss_kernel(const float* yp,
                                                        const float* yt,
                                                        float* loss,
                                                        int T, int K)
{
    extern __shared__ float tmp[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < T; i += blockDim.x)
        sum += -logf(fmaxf(yp[i * K + (int)yt[i]],1e-16f));

    tmp[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            tmp[tid] += tmp[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        *loss = tmp[0];
}

extern "C" float sparse_cross_entropy_loss(const fArr2D yp_,
                                           const fArr2D yt_,
                                           int T, int K)
{
    float* dloss;
    float loss;

    cudaMalloc(&dloss,sizeof(float));
    sparse_cross_entropy_loss_kernel<<<1,THCNT,THCNT * sizeof(float)>>>(
        (const float*) yp_,(const float*) yt_,dloss,T,K);
    cudaMemcpy(&loss,dloss,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dloss);

    return loss;
}


static __global__ void mean_square_error_kernel(const float* y,
                                                const float* yt,
                                                float* error,
                                                int n)
{
    extern __shared__ float tmp[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float d = y[i] - yt[i];
        sum += d * d;
    }

    tmp[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            tmp[tid] += tmp[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        *error = tmp[0] / n;
}

extern "C" float mean_square_error(const fArr2D y_batch_,
                                   const fArr2D y_true_,
                                   int M, int N)
{
    float* derror;
    float error;

    cudaMalloc(&derror,sizeof(float));
    mean_square_error_kernel<<<1,THCNT,THCNT * sizeof(float)>>>(
        (const float*) y_batch_,(const float*) y_true_,derror,M * N);
    cudaMemcpy(&error,derror,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(derror);

    return error;
}


static __global__ void dLdy_cross_entropy_loss_kernel(const float* yp,
                                                      const float* yt,
                                                      float* dy,
                                                      int n, float invT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        dy[i] = (yp[i] - yt[i]) * invT;
}

extern "C" void dLdy_cross_entropy_loss(const fArr2D yp_,
                                        const fArr2D yt_,
                                        fArr2D dy_,
                                        int T, int K)
{
    int n = T * K;
    int blocks = (n + THCNT - 1) / THCNT;

    dLdy_cross_entropy_loss_kernel<<<blocks,THCNT>>>(
        (const float*) yp_,(const float*) yt_,(float*) dy_,n,1.0f / T);
}


static __global__ void dLdy_sparse_cross_entropy_loss_kernel(const float* yp,
                                                             const float* yt,
                                                             float* dy,
                                                             int T, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = T * K;

    if (i < n) {
        int t = i / K;
        int k = i % K;
        dy[i] = (yp[i] - ((k == (int)yt[t]) ? 1.0f : 0.0f)) / T;
    }
}

extern "C" void dLdy_sparse_cross_entropy_loss(const fArr2D yp_,
                                               const fArr2D yt_,
                                               fArr2D dy_,
                                               int T, int K)
{
    int n = T * K;
    int blocks = (n + THCNT - 1) / THCNT;

    dLdy_sparse_cross_entropy_loss_kernel<<<blocks,THCNT>>>(
        (const float*) yp_,(const float*) yt_,(float*) dy_,T,K);
}


static __global__ void dLdy_mean_square_error_kernel(const float* yp,
                                                     const float* yt,
                                                     float* dy,
                                                     int n,
                                                     float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        dy[i] = 2.0f * (yp[i] - yt[i]) * scale;
}

extern "C" void dLdy_mean_square_error(const fArr2D yp_,
                                       const fArr2D yt_,
                                       fArr2D dy_,
                                       int M, int N)
{
    int n = M * N;
    int blocks = (n + THCNT - 1) / THCNT;

    dLdy_mean_square_error_kernel<<<blocks,THCNT>>>(
        (const float*) yp_,(const float*) yt_,(float*) dy_,
        n,1.0f / (M * N));
}
