/* Copyright (c) 2023-2024 Gilad Odinak */
/* Function that updates neural network weights */
#ifndef LINEAR_H
#define LINEAR_H

#include <cuda_runtime.h>
#include "arraytypes.h"

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
static inline void linear_update(fArr2D w_/*[M][N]*/, fArr2D g_/*[M][N]*/,
                                 int M, int N,
                                 float learning_rate, float weight_decay)
{
    int n = M * N;
    int threads = 256;

    linear_update_kernel<<<(n + threads - 1) / threads,threads>>>(
        (float*)w_,(const float*)g_,n,learning_rate,weight_decay);
}

#endif
