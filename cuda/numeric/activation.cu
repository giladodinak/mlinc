/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA activation functions and their derivatives */
#include <math.h>
#include <cuda_runtime.h>
#include "cuctx.h"
#include "arraytypes.h"
#include "activation.h"

#define THDCNT 256

#ifndef M_PI
#define M_PI 3.14159265f
#endif

static __global__ void sigmoid_kernel(float* m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        m[i] = sigmoid1(m[i]);
}

extern "C" void sigmoid(fArr2D m_, int B, int S)
{
    int n = B * S;
    int blocks = (n + THDCNT - 1) / THDCNT;
    float* m = (float*) m_;
    sigmoid_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(m,n);
}

static __global__ void relu_kernel(float* m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && m[i] < 0.0f)
        m[i] = 0.0f;
}

extern "C" void relu(fArr2D m_, int B, int S)
{
    int n = B * S;
    int blocks = (n + THDCNT - 1) / THDCNT;
    float* m = (float*) m_;
    relu_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(m,n);
}

static __global__ void gelu_kernel(float* m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float sqrt2pi = 0.79788456f; /* sqrtf(2.0f / M_PI) */
        float x = m[i];
        float tanh_arg = sqrt2pi * (x + 0.044715f * x * x * x);
        m[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

extern "C" void gelu(fArr2D m_, int B, int S)
{
    int n = B * S;
    int blocks = (n + THDCNT - 1) / THDCNT;
    float* m = (float*) m_;
    gelu_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(m,n);
}

/* One CUDA block handles one row */
static __global__ void softmax_kernel(float* a, int K)
{
    extern __shared__ float tmp[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* p = a + row * K;

    float vmax = -INFINITY;
    for (int i = tid; i < K; i += blockDim.x)
        vmax = fmaxf(vmax,p[i]);

    tmp[tid] = vmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            tmp[tid] = fmaxf(tmp[tid],tmp[tid + s]);
        __syncthreads();
    }

    float m = tmp[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float e = expf(p[i] - m);
        p[i] = e;
        sum += e;
    }

    tmp[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            tmp[tid] += tmp[tid + s];
        __syncthreads();
    }

    float inv = 1.0f / tmp[0];
    for (int i = tid; i < K; i += blockDim.x)
        p[i] *= inv;
}

extern "C" void softmax(fArr2D a_, int B, int K)
{
    int shm = THDCNT * sizeof(float);
    float* a = (float*) a_;
    softmax_kernel<<<B,THDCNT,shm,cuda_stream()>>>(a,K);
}

static __global__ void d_sigmoid_kernel(float* x,
                                        const float* z,
                                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= d_sigmoid1(z[i]);
}

extern "C" void d_sigmoid(fArr2D x_,
                          const fArr2D z_,
                          int B, int D)
{
    int n = B * D;
    int blocks = (n + THDCNT - 1) / THDCNT;
    float* x = (float*) x_;
    const float* z = (const float*) z_;
    d_sigmoid_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(x,z,n);
}

static __global__ void d_relu_kernel(float* x,
                                     const float* z,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= d_relu_1(z[i]);
}

extern "C" void d_relu(fArr2D x_,
                       const fArr2D z_,
                       int B, int D)
{
    int n = B * D;
    int blocks = (n + THDCNT - 1) / THDCNT;
    float* x = (float*) x_;
    const float* z = (const float*) z_;
    d_relu_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(x,z,n);
}

static __global__ void d_gelu_kernel(float* x,
                                     const float* z,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= d_gelu_1(z[i]);
}

extern "C" void d_gelu(fArr2D x_,
                       const fArr2D z_,
                       int B, int D)
{
    int n = B * D;
    int blocks = (n + THDCNT - 1) / THDCNT;
    float* x = (float*) x_;
    const float* z = (const float*) z_;
    d_gelu_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(x,z,n);
}

/* One CUDA block handles one row */
static __global__ void d_softmax_kernel(float* g,
                                        const float* dp,
                                        const float* p,
                                        int D)
{
    extern __shared__ float tmp[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * D;

    float dot = 0.0f;
    for (int i = tid; i < D; i += blockDim.x)
        dot += dp[base + i] * p[base + i];

    tmp[tid] = dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            tmp[tid] += tmp[tid + s];
        __syncthreads();
    }

    dot = tmp[0];
    for (int i = tid; i < D; i += blockDim.x)
        g[base + i] = p[base + i] * (dp[base + i] - dot);
}

extern "C" void d_softmax(fArr2D g_,
                          const fArr2D dp_,
                          const fArr2D p_,
                          int B, int D)
{
    int shm = THDCNT * sizeof(float);
    float* g = (float*) g_;
    const float* dp = (const float*) dp_;
    const float* p = (const float*) p_;
    d_softmax_kernel<<<B,THDCNT,shm,cuda_stream()>>>(g,dp,p,D);
}
