/* Copyright (c) 2023-2024 Gilad Odinak */
/* Accuracy evalutation functions  */
#include <math.h>
#include <cuda_runtime.h>
#include "cuctx.h"
#include "arraytypes.h"
#include "accuracy.h"

#define THDCNT 256

static __global__ void r2_dist_kernel(const float* yp, const float* yt,
                                      float ytmean, float* ypdist,
                                      float* ytdist, int n)
{
    __shared__ float sy[THDCNT];
    __shared__ float st[THDCNT];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    float a = 0.0f;
    float b = 0.0f;
    if (i < n) {
        float dy = yt[i] - yp[i];
        float dt = yt[i] - ytmean;
        a = dy * dy;
        b = dt * dt;
    }

    sy[t] = a;
    st[t] = b;
    __syncthreads();

    for (int d = blockDim.x / 2; d > 0; d >>= 1) {
        if (t < d) {
            sy[t] += sy[t + d];
            st[t] += st[t + d];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(ypdist,sy[0]);
        atomicAdd(ytdist,st[0]);
    }
}

static __global__ void sum_kernel(const float* x, float* sum, int n)
{
    __shared__ float s[THDCNT];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int d = blockDim.x / 2; d > 0; d >>= 1) {
        if (t < d)
            s[t] += s[t + d];
        __syncthreads();
    }

    if (t == 0)
        atomicAdd(sum,s[0]);
}

/* Calculates the R-squared factor between predicted and true values.
 * 
 * This function calculates the numerator of the R-squared factor between M 
 * predicted values and M true values. The result is a value between 0 and M, 
 * where 0 indicates no fit, and M indicates a perfect fit.
 * 
 * Parameters:
 *   yp - Predicted values array of size MxN
 *   yt - True values array of size MxN
 *   M  - Number of samples (rows)
 *   N  - Number of features (columns)
 * 
 * Returns:
 *   R-squared factor numerator value
 * 
 * Reference:
 *   https://en.wikipedia.org/wiki/Coefficient_of_determination
 */
float R2_sum(const fArr2D yp_/*[M][N]*/, 
             const fArr2D yt_/*[M][N]*/, 
             int M, int N)
{
    int n = M * N;
    int threads = THDCNT;
    int blocks = (n + threads - 1) / threads;
    const float* yp = (const float*) yp_;
    const float* yt = (const float*) yt_;

    float* d_sum;
    float* d_ypdist;
    float* d_ytdist;
    cudaMalloc((void**)&d_sum,sizeof(float));
    cudaMalloc((void**)&d_ypdist,sizeof(float));
    cudaMalloc((void**)&d_ytdist,sizeof(float));
    cudaMemset(d_sum,0,sizeof(float));
    cudaMemset(d_ypdist,0,sizeof(float));
    cudaMemset(d_ytdist,0,sizeof(float));

    sum_kernel<<<blocks,threads,0,cuda_stream()>>>(yt,d_sum,n);

    float ytsum;
    cudaMemcpy(&ytsum,d_sum,sizeof(float),cudaMemcpyDeviceToHost);
    float ytmean = ytsum / n;

    r2_dist_kernel<<<blocks,threads,0,cuda_stream()>>>(
                                    yp,yt,ytmean,d_ypdist,d_ytdist,n);

    float ypdist, ytdist;
    cudaMemcpy(&ypdist,d_ypdist,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&ytdist,d_ytdist,sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    cudaFree(d_ypdist);
    cudaFree(d_ytdist);

    return M * (1.0f - (ypdist / ytdist));
}

static __global__ void match_sum_kernel(const float* yp, const float* yt,
                                        int M, int K, int* match_cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int label;
        const float* ypi = yp + i * K;
        const float* yti = yt + i * K;

        for (label = 0; label < K; label++)
            if (yti[label] != 0.0f)
                break;

        int pred = 0;
        for (int j = 0; j < K; j++)
            if (ypi[j] > ypi[pred])
                pred = j;

        if (label == pred)
            atomicAdd(match_cnt,1);
    }
}

/* Calculates the accuracy factor between predicted classes and true labels.
 * 
 * This function calculates the numerator of the accuracy factor between M 
 * predicted classes and M true labels. The result is a value between 0 and M, 
 * where 0 indicates no match and M indicates a perfect match.
 * 
 * Parameters:
 *   yp - Predicted classes array of size MxK
 *   yt - One-hot encdoed true labels array of size MxK
 *   M  - Number of samples (rows)
 *   K  - Number of classes (columns)
 * 
 * Note:
 *   This function assume the one-hot encoded vectors in yt have only one
 *   element set to 1 and all other elements set to 0. It does not check
 *   all values of yt to ensure that is the case.
 *
 * Returns:
 *   Accuracy factor numerator value
 */
float match_sum(const fArr2D yp_/*[M][K]*/,
                const fArr2D yt_/*[M][K]*/, 
                int M, int K)
{
    int* d_match_cnt;
    cudaMalloc((void**)&d_match_cnt,sizeof(int));
    cudaMemset(d_match_cnt,0,sizeof(int));

    int threads = THDCNT;
    int blocks = (M + threads - 1) / threads;
    const float* yp = (const float*) yp_;
    const float* yt = (const float*) yt_;
    
    match_sum_kernel<<<blocks,threads,0,cuda_stream()>>>(yp,yt,M,K,d_match_cnt);

    int match_cnt;
    cudaMemcpy(&match_cnt,d_match_cnt,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_match_cnt);

    return (float) match_cnt;
}
