/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA activation functions and their derivatives */
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "arraytypes.h"

/* Scalar activation helpers can be called from host or device code */

/* Applies the sigmoid activation to a single value (clamped) */
static inline __host__ __device__ float sigmoid1(float x)
{
    if (x >= 87.33f) return 1.0f;
    if (x < -87.33f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* Calculates the derivative of the sigmoid function at point sigmoid_z.
 *
 * Parameters:
 *   z : The output of the sigmoid function
 *
 * Note:
 * The derivative of the sigmoid function, is sigmoid(x) * (1 - sigmoid(x))
 * However the value passed to this function is z = sigmoid(x)
 * sigmoid(x) * (1 - sigmoid(x)) => z * (1 - z)
 */
static inline __host__ __device__ float d_sigmoid1(float z)
{
    return z * (1.0f - z);
}

/* Calculates the derivative of the ReLU function at point z.
 *
 * Parameters:
 *   z : The output of the ReLU function
 *
 * Notice that the drivative of relu(x) and the derivative of relu(relu(x))
 * yield the same values.
 */
static inline __host__ __device__ float d_relu_1(float z)
{
    return (z > 0.0f) ? 1.0f : 0.0f;
}

/* Calculates the derivative of the GELU function at point z using the
 * approximate tanh formulation.
 *
 * Parameters:
 *   z : The input to GELU (pre-activation)
 */
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

/* Calculates the derivative of the hyperbolic tangent (tanh) function
 *
 * Parameters:
 *   x : Input value to the tanh function
 *
 * Note:
 * The canonical definition of d/dx tanh(x) is (1 / cosh(x)^2); this function
 * uses an equivalnet.
 */
static inline __host__ __device__ float d_tanh(float x)
{
    float t = tanhf(x);
    return 1.0f - t * t;
}

/* Calculates the derivative of the hyperbolic tangent (tanh) function given
 * its output i.e. z = tanh(x)
 *
 * Parameters:
 *   z : Output value of the tanh function
 *
 * Note:
 * The derivative of the tanh function, is 1 - tanh(x) * tanh(x)
 * However the value passed to this function is z = tanh(x)
 *  1 - tanh(x) * tanh(x) => 1 - z * z
 */
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

void d_softmax(fArr2D g, const fArr2D dp, const fArr2D p, int B, int D);

#ifdef __cplusplus
}
#endif

#endif
