/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA ADAM optimizer with weight decay functions */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuctx.h"
#include "arraytypes.h"
#include "adamw.h"

/* Adaptive Moment Estimation with weight decay
 * https://arxiv.org/pdf/1711.05101.pdf
 * "Decoupled Weight Decay Regularization" - Algorithm 2 - AdamW
 */

static const float beta1 = 0.9f;
static const float beta2 = 0.999f;
static const float epsilon = 1.0e-7f;

static __global__ void adamw_update_kernel(float* restrict w,
                                           const float* restrict g,
                                           float* restrict m,
                                           float* restrict v,
                                           int n,
                                           float bc1,
                                           float bc2,
                                           float learning_rate,
                                           float weight_decay)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;

    if (v[k] < 0) { /* weight, gradient explosion */
        printf("adamw: weight or gradient explosion\n");
        asm("trap;");
    }

    m[k] = beta1 * m[k] + (1.0f - beta1) * g[k];
    v[k] = beta2 * v[k] + (1.0f - beta2) * g[k] * g[k];

    float mh = m[k] * bc1;
    float vh = v[k] * bc2;
    float ag = mh / (sqrtf(vh) + epsilon);

    /* Weight update with weight decay */
    w[k] -= learning_rate * (ag + weight_decay * w[k]);
}

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using the ADAM optimizer algorithm.  The arrays
 * m[M][N] and v[M][N] stores coefficients used by the algorithm.
 * The rate of update is controlled by learning_rate, weight_decay.
 * w, g, m and v must point to CUDA device memory.
 */
extern "C" void adamw_update(fArr2D w_/*[M][N]*/,fArr2D g_/*[M][N]*/,
                             fArr2D m_/*[M][N]*/,fArr2D v_/*[M][N]*/,
                             int M, int N,
                             float learning_rate, float weight_decay,
                             int update_step)
{
    float bc1 = 1.0f / (1.0f - powf(beta1,update_step));
    float bc2 = 1.0f / (1.0f - powf(beta2,update_step));

    int n = M * N;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float* w = (float*) w_;
    const float* g = (float*) g_;
    float* m = (float*) m_;
    float* v = (float*) v_;

    adamw_update_kernel<<<blocks,threads,0,cuda_stream()>>>(
                              w,g,m,v,n,bc1,bc2,learning_rate,weight_decay);
}
