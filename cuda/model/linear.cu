/* Copyright (c) 2026 Gilad Odinak */
/* Function that updates neural network weights */
#include <cuda_runtime.h>
#include "cuctx.h"
#include "arraytypes.h"
#include "linear.h"

static __global__ void linear_update_kernel(float* w, const float* g,
                                            int n,
                                            float learning_rate,
                                            float weight_decay)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n)
        w[k] -= learning_rate * (g[k] + weight_decay * w[k]);
}

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using linear update.
 * The rate of update is controlled by learning_rate, weight_decay.
 *
 * w and g must point to CUDA device memory.
 */
extern "C" void linear_update(fArr2D w_/*[M][N]*/, fArr2D g_/*[M][N]*/,
                              int M, int N,
                              float learning_rate, float weight_decay)
{
    int n = M * N;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float* w = (float*) w_;
    const float* g = (float*) g_;

    linear_update_kernel<<<blocks,threads,0,cuda_stream()>>>(
                                          w,g,n,learning_rate,weight_decay);
}

