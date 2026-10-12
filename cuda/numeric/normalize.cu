/* Copyright (c) 2023-2024, 2026 Gilad Odinak */
/* Data normalization functions */
#include <cuda_runtime.h>
#include "cuctx.h"
#include "float.h"
#include "arraytypes.h"
#include "array.h"
#include "normalize.h"

/* Calculates the mean and standard deviation of an array of feature vectors 
 * by feature (i.e., by column) and returns these values in mean and sdev 
 * vectors respectively.
 * 
 * Parameters:
 *   x        - Array of M vectors, each having D elements
 *   M        - Number of vectors in x
 *   D        - Number of elements in each vector 
 *   mean     - Returns a vector of D or D-1 values, each is the mean of 
 *              the values of the corresponding column in x array
 *   sdev     - Returns a vector of D or D-1 values, each is the standard 
 *              deviation from the mean of the corresponding column in x array
 *   exc_last - If not 0, the last column of x (i.e., last element of vectors)
 *              is excluded from the calculation. In that case, mean and sdev
 *              have D-1 elements.
 * 
 * This function calculates element-wise mean and standard deviation for all 
 * feature vectors in the dataset passed in x[][], and returns those mean and 
 * standard deviation arrays. The calculation excludes the last column in x[][]
 * (the bias) if exc_last is not zero.
 */
void calculate_mean_sdev(const fArr2D x_/*[M][D]*/, 
                         int M, int D, 
                         fVec mean/*[D | D-1]*/, 
                         fVec sdev/*[D | D-1]*/, 
                         int exc_last)
{
    typedef float (*ArrMD)[D];
    ArrMD x = (ArrMD) x_;
    int Dx = D - ((exc_last) ? 1 : 0);
    typedef float (*VecDx);
    VecDx sum = (VecDx) mean; /* Calculate sum, convert to mean in place */
    VecDx var = (VecDx) sdev; /* Calculate var, convert to sdev in place */

    fltclr(sum,Dx);
    fltclr(var,Dx);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Dx; j++)
            sum[j] += x[i][j];
    }
    for (int j = 0; j < Dx; j++)
        sum/*mean*/[j] = sum[j] / M;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Dx; j++) {
            float val = x[i][j] - mean[j];
            var[j] += val * val;
        }
    }
    for (int j = 0; j < Dx; j++)
        var/*sdev*/[j] = sqrt(var[j] / M);
}

/* Normalizes an array of feature vectors by feature (column-wise) in place, by
 * subtracting the corresponding mean and dividing by the corresponding sdev.
 *
 * Parameters:
 *   x        - Array of M vectors, each having D elements
 *   B        - Number of vectors in x
 *   D        - Number of elements in each vector
 *   mean     - A vector of D or D-1 values, each is the mean of the values 
 *              of the corresponding column in x array
 *   sdev     - A vector of D or D-1 values, each is the standard deviation
 *              from the mean of the corresponding column in x array
 *   exc_last - If not 0, the last column of x (i.e., last element of vectors)
 *              is excluded from the calculation. In that case, mean and sdev 
 *              have D-1 elements, and only the first D-1  elements in each
 *              vector in x are normalized.
 */
static __global__ void normalize_kernel(float* x, int D, int Dx,
                                        const float* mean, const float* sdev,
                                        int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        int i = k / Dx;
        int j = k % Dx;
        float sd = sdev[j];
        if (sd > 0.0f)
            x[i * D + j] = (x[i * D + j] - mean[j]) / sd;
        else
            x[i * D + j] = 0.0f;
    }
}

void normalize(fArr2D x_/*[B][D]*/,
               int B, int D,
               const fVec mean_/*[D | D-1]*/,
               const fVec sdev_/*[D | D-1]*/,
               int exc_last)
{
    int Dx = D - ((exc_last) ? 1 : 0);
    int n = B * Dx;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float* x = (float*) x_;
    const float* mean = (const float*) mean_;
    const float* sdev = (const float*) sdev_;
    normalize_kernel<<<blocks,threads,0,cuda_stream()>>>(x,D,Dx,mean,sdev,n);
}
