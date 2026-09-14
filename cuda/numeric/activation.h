/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA activation functions and their derivatives */
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "array.h"

/* Scalar activation helpers can be called from host or device code. */
static inline __host__ __device__ float sigmoid1(float x)
{
    if (x >= 87.33f) return 1.0f;
    if (x < -87.33f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline __host__ __device__ float d_sigmoid1(float z)
{
    return z * (1.0f - z);
}

static inline __host__ __device__ float d_relu_1(float z)
{
    return (z > 0.0f) ? 1.0f : 0.0f;
}

static inline __host__ __device__ float d_gelu_1(float z)
{
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float z3 = z * z * z;
    float tanh_arg = 0.044715f * z3 + z;
    float tanh_val = tanhf(sqrt_2_over_pi * tanh_arg);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float cubic_deriv = 1.0f + 3.0f * 0.044715f * z * z;
    return 0.5f * (1.0f + tanh_val +
                   z * sech2 * sqrt_2_over_pi * cubic_deriv);
}

static inline __host__ __device__ float d_tanh(float x)
{
    float t = tanhf(x);
    return 1.0f - t * t;
}

static inline __host__ __device__ float d_tanh_x(float z)
{
    return 1.0f - z * z;
}

#ifdef __cplusplus
extern "C" {
#endif

void sigmoid(fArr2D m, int B, int S);
void relu(fArr2D m, int B, int S);
void gelu(fArr2D m, int B, int S);
void softmax(fArr2D a, int B, int K);

void d_sigmoid(fArr2D x, const fArr2D z, int B, int D);
void d_relu(fArr2D x, const fArr2D z, int B, int D);
void d_gelu(fArr2D x, const fArr2D z, int B, int D);

void d_softmax(fArr2D g,
               const fArr2D dp,
               const fArr2D p,
               int B, int D);

#ifdef __cplusplus
}
#endif

#endif
