/* Copyright (c) 2023-2024 Gilad Odinak */
/* Activation functions and their derivatives */
#include <math.h>
#include "arraytypes.h"
#include "activation.h"

/* Applies the sigmoid activation function to each element of a 2D array
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 */
void sigmoid(fArr2D m_/*[B][S]*/, int B, int S)
{
    typedef float (*ArrBS)[S];
    ArrBS m = (ArrBS) m_;

    for (int i = 0; i < B; i++)
        for (int j = 0; j < S; j++)
            m[i][j] = sigmoid1(m[i][j]);
}

/* Applies the rectified linear unit (ReLU) activation function to each
 * element of a 2D array.
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 */
void relu(fArr2D m_/*[B][S]*/, int B, int S)
{
    typedef float (*ArrBS)[S];
    ArrBS m = (ArrBS) m_;

    for (int i = 0; i < B; i++)
        for (int j = 0; j < S; j++)
            if (m[i][j] < 0.0)
                m[i][j] = 0.0;
}

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
void gelu(fArr2D m_/*[B][S]*/, int B, int S)
{
    typedef float (*ArrBS)[S];
    ArrBS m = (ArrBS) m_;
    const float sqrt_2_over_pi = 0.7978845608028654; /* sqrt(2/pi) */

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < S; j++) {
            float x = m[i][j];
            /* Approximate the Cumulative Distribution Function */
            float tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            m[i][j] = 0.5f * x * (1.0f + tanhf(tanh_arg));
        }
    }
}

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
void softmax(fArr2D a_/*[B][K]*/, int B, int K)
{
    typedef float (*ArrBK)[K];
    ArrBK a = (ArrBK) a_;
    for (int j = 0; j < B; j++) {
        typedef float (*VecK);
        VecK  p = (VecK) a[j];

        float m = p[0]; /* max(p[]) */
        for (int i = 0; i < K; i++) {
            if (m < p[i])
                m = p[i];
        }
        float s = 0.0; /* sum(exp(p[] - m) */
        for (int i = 0; i < K; i++) {
            p[i] = exp(p[i] - m);
            s += p[i];
        }
        for (int i = 0; i < K; i++)
            p[i] /= s;
    }
}

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
void d_sigmoid(fArr2D x_/*[B][D]*/, const fArr2D z_/*[B][D]*/, int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) x_;
    const ArrBD z = (ArrBD) z_;

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            x[i][j] *= d_sigmoid1(z[i][j]);
}

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
void d_relu(fArr2D x_/*[B][D]*/, const fArr2D z_/*[B][D]*/, int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) x_;
    const ArrBD z = (ArrBD) z_;

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            x[i][j] *= d_relu_1(z[i][j]);
}

/* Calculates the derivative of GELU for a 2D array z, and updates the input
 * array x by multiplying its original values by the derivative.
 *
 * Parameters:
 *   x : Pointer to the 2D array to be updated
 *   z : Pointer to the 2D array, the input to GELU
 *   B : Number of rows in the matrices
 *   D : Number of columns in the matrices
 */
void d_gelu(fArr2D x_/*[B][D]*/, const fArr2D z_/*[B][D]*/, int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) x_;
    const ArrBD z = (ArrBD) z_;

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            x[i][j] *= d_gelu_1(z[i][j]);
}

/* Computes the gradient of the loss with respect to the input scores of the softmax,
 * given the gradient with respect to the softmax output: g = J_softmax(z) @ dz
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
                                    const fArr2D p_ /*[B][D]*/, int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD g = (ArrBD) g_;
    const ArrBD dp = (ArrBD) dp_;
    const ArrBD p = (ArrBD) p_;

    for (int i = 0; i < B; i++) {
        float dot = 0;
        for (int k = 0; k < D; k++)
            dot += dp[i][k] * p[i][k];
        for (int j = 0; j < D; j++)
            g[i][j] = p[i][j] * (dp[i][j] - dot);
    }
}
