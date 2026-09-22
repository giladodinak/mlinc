/* Copyright (c) 2023-2024 Gilad Odinak */
/* Activation functions and their derivatives */
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "arraytypes.h"

/* Applies the sigmoid activation to a single value (clamped) */
static inline float sigmoid1(float x)
{
    if (x >= 87.33) return 1;
    if (x < -87.33) return 0;
    return 1.0 / (1.0 + expf(-x));
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
static inline float d_sigmoid1(float z)
{
    return z * (1 - z);
}

/* Calculates the derivative of the ReLU function at point z.
 *
 * Parameters:
 *   z : The output of the ReLU function
 *
 * Notice that the drivative of relu(x) and the derivative of relu(relu(x))
 * yield the same values.
 */
static inline float d_relu_1(float z)
{
    return (z > 0.0) ? 1.0 : 0.0;
}

/* Calculates the derivative of the GELU function at point z using the
 * approximate tanh formulation.
 *
 * Parameters:
 *   z : The input to GELU (pre-activation)
 */
static inline float d_gelu_1(float z)
{
    const float sqrt_2_over_pi = 0.7978845608028654; /* sqrt(2/pi) */
    float z3 = z * z * z;
    float tanh_arg = 0.044715f * z3 + z;
    float tanh_val = tanhf(sqrt_2_over_pi * tanh_arg);
    float sech2 = 1.0 - tanh_val * tanh_val;
    float cubic_deriv = 1.0 + 3.0 * 0.044715 * z * z;
    return 0.5 * (1.0 + tanh_val + z * sech2 * sqrt_2_over_pi * cubic_deriv);
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
static inline float d_tanh(float x)
{
    float t = tanh(x);
    return 1 - t * t;
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
static inline float d_tanh_x(float z)
{
    return 1 - z * z;
}

/* Applies the sigmoid activation function to each element of a 2D array
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 */
void sigmoid(fArr2D m_/*[B][S]*/, int B, int S);

/* Applies the rectified linear unit (ReLU) activation function to each
 * element of a 2D array.
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 */
void relu(fArr2D m_/*[B][S]*/, int B, int S);

/* Applies the Gaussian Error Linear Unit (GELU) activation function to each
 * element of a 2D array.
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 *
 * Reference:
 *   Hendrycks & Gimpel, 2016. "Gaussian Error Linear Units (GELUs)"
 */
void gelu(fArr2D m_/*[B][S]*/, int B, int S);

/* Applies the softmax activation function to each row of a 2D array
 *
 * Parameters:
 *   a  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   K  : Number of columns in the matrix
 *
 * Notes:
 * Converts predictions in vectors that are the rows of the passed in array
 * to probabilities.
 *
 * See https://en.wikipedia.org/wiki/Softmax_function
 *
 * This implementation improves numerical stability by normalizing the input.
 * It subtracts the maximum value from each row to prevent overflow.
 */
void softmax(fArr2D a_/*[B][K]*/, int B, int K);

/* Calculates the derivative of the sigmoid function for a 2D array z, and
 * updates the input array x, by multiplying its original values by the
 * derivative.
 *
 * Parameters:
 *   x : Pointer to the 2D array to be updated
 *   z : Pointer to the 2D array, the output of the sigmoid activation function
 *   B : Number of rows in the matrices
 *   D : Number of columns in the matrices
 */
void d_sigmoid(fArr2D x_/*[B][D]*/, const fArr2D z_/*[B][D]*/, int B, int D);

/* Calculates the derivative of the ReLU function for a 2D array z, and
 * updates the input array x, by multiplying its original values by the
 * derivative.
 *
 * Parameters:
 *   x : Pointer to the 2D array to be updated
 *   z : Pointer to the 2D array, the output of the ReLU function
 *   B : Number of rows in the matrices
 *   D : Number of columns in the matrices
 */
void d_relu(fArr2D x_/*[B][D]*/, const fArr2D z_/*[B][D]*/, int B, int D);

/* Calculates the derivative of GELU for a 2D array z, and updates the input
 * array x by multiplying its original values by the derivative.
 *
 * Parameters:
 *   x : Pointer to the 2D array to be updated
 *   z : Pointer to the 2D array, the input to GELU
 *   B : Number of rows in the matrices
 *   D : Number of columns in the matrices
 */
void d_gelu(fArr2D x_/*[B][D]*/, const fArr2D z_/*[B][D]*/, int B, int D);

/* Computes the gradient of the loss with respect to the input scores of 
 * the softmax, given the gradient with respect to the softmax output:
 * g = J_softmax(z) @ dz
 *
 * Parameters:
 *   g  : Pointer to output 2D array that receives the resulting gradients
 *        of the loss with respect to the input scores before softmax.
 *   dp : Pointer to 2D array of gradients of the loss with respect
 *         to the softmax output
 *   p  : Pointer to 2D array of the output of the softmax function
 *   B  : Number of rows in the matrices (batch size)
 *   D  : Number of columns in the matrices (number of classes)
 *
 * Reference:
 *   https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
 */
void d_softmax(fArr2D g_/*[B][D]*/, const fArr2D dp_/*[B][D]*/,
                                    const fArr2D p_ /*[B][D]*/, int B, int D);

#endif
