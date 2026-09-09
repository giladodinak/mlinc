/* Copyright (c) 2026 Gilad Odinak  */
/* Rotary Position Embedding (RoPE) */
#ifndef ROPE_H
#define ROPE_H
#include <math.h>
#include "array.h"

/* Precomputes the RoPE theta table for a given head dimension.
 *
 * Parameters:
 *   theta : Output CUDA device array of length Dh/2;
 *           theta[i] = 10000^(-2i/Dh)
 *   Dh    : Head dimension (must be even)
 *
 * Reference:
 *   Su et al., "RoFormer: Enhanced Transformer with Rotary Position
 *   Embedding", 2021. https://arxiv.org/abs/2104.09864
 */
static __global__ void rope_init_k(float* theta, int Dh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Dh / 2)
        theta[i] = powf(10000.0,-2.0 * i / (float) Dh);
}

static inline void rope_init(float* theta, int Dh)
{
    int n = Dh / 2;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rope_init_k<<<blocks,threads>>>(theta,Dh);
}

/* Applies RoPE in-place to a single head slice of Q or K.
 *
 * Each token at position offset + t is rotated by angle
 * (offset + t) * theta[i] for each consecutive pair of
 * dimensions (2i, 2i+1).
 *
 * Parameters:
 *   x       : CUDA device pointer to the [T][Dh] head slice to be rotated
 *   theta   : CUDA device frequency table of length Dh/2 (from rope_init())
 *   inverse : If non-zero, applies the inverse rotation (for backward pass)
 *   offset  : Position of the first token in x
 *   T       : Sequence length (number of rows)
 *   Dh      : Head dimension (number of columns, must be even)
 */
static __global__ void rope_apply_k(float* x,
                                    const float* theta,
                                    int inverse,
                                    int offset,
                                    int T, int Dh)
{
    float sign = inverse ? -1.0 : 1.0;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int D2 = Dh / 2;
    int n = T * D2;

    if (k < n) {
        int t = k / D2;
        int i = k % D2;

        double angle = ((double)offset + t) * (double)theta[i];
        float cos_a = (float)cos(angle);
        float sin_a = sign * (float)sin(angle);
        float x0 = x[t * Dh + 2 * i];
        float x1 = x[t * Dh + 2 * i + 1];
        x[t * Dh + 2 * i]     = x0 * cos_a - x1 * sin_a;
        x[t * Dh + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}
static inline void rope_apply(fArr2D x/*[T][Dh]*/,
                              const float* theta,
                              int inverse,
                              int offset,
                              int T, int Dh)
{
    int n = T * (Dh / 2);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rope_apply_k<<<blocks,threads>>>((float*)x,theta,inverse,offset,T,Dh);
}

#endif
