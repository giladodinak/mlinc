/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA activation functions and their derivatives */
#include <cuda_runtime.h>
#include "activation.h"

#define ACT_THREADS 256

static __global__ void sigmoid_kernel(float* m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        m[i] = sigmoid1(m[i]);
}

static __global__ void relu_kernel(float* m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && m[i] < 0.0f)
        m[i] = 0.0f;
}

static __global__ void gelu_kernel(float* m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float x = m[i];
        float tanh_arg =
            sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        m[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

/* One CUDA block handles one row. */
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

static __global__ void d_sigmoid_kernel(float* x,
                                        const float* z,
                                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= d_sigmoid1(z[i]);
}

static __global__ void d_relu_kernel(float* x,
                                     const float* z,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= d_relu_1(z[i]);
}

static __global__ void d_gelu_kernel(float* x,
                                     const float* z,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= d_gelu_1(z[i]);
}

/* One CUDA block handles one row. */
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

extern "C" void sigmoid(fArr2D m, int B, int S)
{
    int n = B * S;
    int blocks = (n + ACT_THREADS - 1) / ACT_THREADS;
    sigmoid_kernel<<<blocks,ACT_THREADS>>>((float*) m,n);
}

extern "C" void relu(fArr2D m, int B, int S)
{
    int n = B * S;
    int blocks = (n + ACT_THREADS - 1) / ACT_THREADS;
    relu_kernel<<<blocks,ACT_THREADS>>>((float*) m,n);
}

extern "C" void gelu(fArr2D m, int B, int S)
{
    int n = B * S;
    int blocks = (n + ACT_THREADS - 1) / ACT_THREADS;
    gelu_kernel<<<blocks,ACT_THREADS>>>((float*) m,n);
}

extern "C" void softmax(fArr2D a, int B, int K)
{
    softmax_kernel<<<B,ACT_THREADS,ACT_THREADS * sizeof(float)>>>(
        (float*) a,K);
}

extern "C" void d_sigmoid(fArr2D x,
                          const fArr2D z,
                          int B, int D)
{
    int n = B * D;
    int blocks = (n + ACT_THREADS - 1) / ACT_THREADS;
    d_sigmoid_kernel<<<blocks,ACT_THREADS>>>(
        (float*) x,(const float*) z,n);
}

extern "C" void d_relu(fArr2D x,
                       const fArr2D z,
                       int B, int D)
{
    int n = B * D;
    int blocks = (n + ACT_THREADS - 1) / ACT_THREADS;
    d_relu_kernel<<<blocks,ACT_THREADS>>>(
        (float*) x,(const float*) z,n);
}

extern "C" void d_gelu(fArr2D x,
                       const fArr2D z,
                       int B, int D)
{
    int n = B * D;
    int blocks = (n + ACT_THREADS - 1) / ACT_THREADS;
    d_gelu_kernel<<<blocks,ACT_THREADS>>>(
        (float*) x,(const float*) z,n);
}

extern "C" void d_softmax(fArr2D g,
                          const fArr2D dp,
                          const fArr2D p,
                          int B, int D)
{
    d_softmax_kernel<<<B,ACT_THREADS,ACT_THREADS * sizeof(float)>>>(
        (float*) g,(const float*) dp,(const float*) p,D);
}
