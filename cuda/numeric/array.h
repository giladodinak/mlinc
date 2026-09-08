/* Copyright (c) 2023-2026 Gilad Odinak */
/* Array data structures and functions  */
#ifndef ARRAY_H
#define ARRAY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuflt.h"

/* Define types of one and two dimensional arrays of unspecified dimensions.
 * These are dynamically cast to explicit dimensions within GPU kernels or
 * device functions.
 *
 * For example:
 *     fArr2D a2d; // a2d is two-dimensional array of unspecified dimensions
 *     typedef float (*ArrNK)[K]; // two-dimensional array NxK (e.g. N=3 K=5)
 *     ArrNK a2dNK = (ArrNK) a2d; // a2dNK is the addrress of 2-D array 3x5
 *     a2dNK[2][4] access an entry in the array
 * Notice that a2dNK memory is contiguous just like when declaring
 * a two-dimensional array  float arr[3][5]
 *
 * When passing an array by reference to a function, only the last dimension
 * can be specified. By convention, the first dimension is noted in the typedef
 * and in comments, as in the CPU implementation.
 *
 * When a2d points to CUDA device memory, a2dNK may be passed through host code
 * and to CUDA/cuBLAS functions, but its elements may only be dereferenced by
 * code executing on the GPU.
 *
 * Functions with __global__ attribute execute on the GPU. The GPU driver
 * loads the compiled functions' code to the GPU memory.
 *
 * The C programming language uses row-major array layout, while cuBLAS
 * routines interpret matrices as column-major. The cuBLAS calls below
 * therefore operate on transposed views of the row-major matrices.
 *
 * Reference:
 * https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf
 */
typedef int (*iVec);
typedef int (*iArr2D)[];
typedef float (*fVec);
typedef float (*fArr2D)[];

#ifdef USE_DOUBLE
#define CUGEMM cublasDgemm
#define CUGEMV cublasDgemv
#define CUGER  cublasDger
#else
#define CUGEMM cublasSgemm
#define CUGEMV cublasSgemv
#define CUGER  cublasSger
#endif

/* Creates a cuBLAS handle when a varibale of this class is instansiated.
 * The handle is released when the variable is destroyed (goes out of scope).
 */
class CublasHandle {
public:
    cublasHandle_t handle;

    CublasHandle() : handle(NULL)
    {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,"cublasCreate failed (%d)\n",(int) status);
            exit(1);
        }
    }

    ~CublasHandle()
    {
        if (handle != NULL)
            cublasDestroy(handle);
    }
};

/* This function initializes one cuBLAS handle per CPU thread that is
 * returned by all invocations of this function from that same thread.
 *
 * Notice that this file is included from .cu files and has C++ semantics:
 * Althogh this function is included by many compilation units there is o
 * only one copy of the handle variable in the calling thread's local storage.
 *
 * This design supports only one (the default) GPU device.
 */
inline cublasHandle_t cublas_handle(void)
{
    static thread_local CublasHandle h;
    return h.handle;
}

/* Multiplies matrix x by matrix y, and returns the result in matrix r.
 * r = x @ y
 * r: resulting matrix NxM
 * x: left matrix Nxd
 * y: right matrix dxM
 * Note that d is the common dimension, not related to D, which usually
 * indicates the size of neural network layer's input vectors dimension.
 */
static inline void matmul(fArr2D restrict r_/*[N][M]*/,
                          const fArr2D restrict x_/*[N][d]*/,
                          const fArr2D restrict y_/*[d][M]*/,
                          int N, int d, int M)
{
    /* Row-major r=x@y is computed by cuBLAS as r.T=y.T@x.T */
    const float alpha = 1.0;
    const float beta = 0.0;
    CUGEMM(cublas_handle(),CUBLAS_OP_N,CUBLAS_OP_N,
           M,N,d,
           &alpha,(const float*)y_,M,
                  (const float*)x_,d,
           &beta, (float*)r_,M);
}

/* Multiplies matrix x by the transpose of matrix y.
 * Adds the result to the matrix r.
 * r = r + x @ y.T
 * r: resulting matrix NxM
 * x: left matrix Nxd
 * y: right matrix Mxd
 */
static inline void addMatmulT(fArr2D restrict r_/*[N][M]*/,
                              const fArr2D restrict x_/*[N][d]*/,
                              const fArr2D restrict y_/*[M][d]*/,
                              int N, int d, int M)
{
    /* Row-major r+=x@y.T is computed by cuBLAS as * r.T += y @ x.T */
    const float alpha = 1.0;
    const float beta = 1.0;
    CUGEMM(cublas_handle(),CUBLAS_OP_T,CUBLAS_OP_N,
           M,N,d,
           &alpha,(const float*)y_,d,
                  (const float*)x_,d,
           &beta, (float*)r_,M);
}

/* Multiplies matrix x by the transpose of matrix y.
 * Returns the result in matrix r.
 * r = x @ y.T
 * r: resulting matrix NxM
 * x: left matrix Nxd
 * y: right matrix Mxd
 */
static inline void matmulT(fArr2D restrict r_/*[N][M]*/,
                           const fArr2D restrict x_/*[N][d]*/,
                           const fArr2D restrict y_/*[M][d]*/,
                           int N, int d, int M)
{
    /* Row-major r=x@y.T is computed as * r.T = y @ x.T */
    const float alpha = 1.0;
    const float beta = 0.0;
    CUGEMM(cublas_handle(),CUBLAS_OP_T,CUBLAS_OP_N,
           M,N,d,
           &alpha,(const float*)y_,d,
                  (const float*)x_,d,
           &beta, (float*)r_,M);
}

/* Multiplies the transpose of matrix x by matrix y.
 * Returns the result in matrix r.
 * r = x.T @ y
 * r: resulting matrix NxM
 * x: left matrix dxN
 * y: right matrix dxM
 */
static inline void Tmatmul(fArr2D restrict r_/*[N][M]*/,
                           const fArr2D restrict x_/*[d][N]*/,
                           const fArr2D restrict y_/*[d][M]*/,
                           int N, int d, int M)
{
    /* Row-major r=x.T@y is computed as r.T = y.T @ x */
    const float alpha = 1.0;
    const float beta = 0.0;
    CUGEMM(cublas_handle(),CUBLAS_OP_N,CUBLAS_OP_T,
           M,N,d,
           &alpha,(const float*)y_,M,
                  (const float*)x_,N,
           &beta, (float*)r_,M);
}

/* Multiplies the vector v by the matrix m and
 * adds the resulting vector to the vector in r.
 * r = r + v @ m
 * r: vector 1xN
 * v: vector 1xM
 * m: matrix MxN
 */
static inline void addvecmatmul(fVec restrict r_/*[N]*/,
                                const fVec restrict v_/*[M]*/,
                                const fArr2D restrict m_/*[M][N]*/,
                                int M, int N)
{
    /* Row-major r+=v@m becomes r += m.T @ v in column-vector form */
    const float alpha = 1.0;
    const float beta = 1.0;
    CUGEMV(cublas_handle(),CUBLAS_OP_N,
           N,M,
           &alpha,(const float*)m_,N,
                  (const float*)v_,1,
           &beta, (float*)r_,1);
}

/* Multiplies the vector w by the transpose of matrix m
 * and adds the resulting vector to the vector in v.
 * v = v + w @ m.T
 * v: vector 1xN
 * w: vector 1xM
 * m: matrix NxM
 */
static inline void addinnermul(fVec restrict v_/*[N]*/,
                               const fVec restrict w_/*[M]*/,
                               const fArr2D restrict m_/*[N][M]*/,
                               int N, int M)
{
    /* Row-major v+=w @ m.T is compute by cuBLAS as v += m @ w */
    const float alpha = 1.0;
    const float beta = 1.0;
    CUGEMV(cublas_handle(),CUBLAS_OP_T,
           M,N,
           &alpha,(const float*)m_,M,
                  (const float*)w_,1,
           &beta, (float*)v_,1);
}

/* Calculates the tensor product (i.e. outer multiplication) of the
 * vector v by the vector w, and adds the resulting matrix to the matrix in m.
 * m = m + (v ⊚ w)
 */
static inline void addoutermul(fArr2D restrict m_/*[N][M]*/,
                               const fVec restrict v_/*[N]*/,
                               const fVec restrict w_/*[M]*/,
                               int N, int M)
{
    /* Row-major m+=v * w.T is computed bu cuBLAS as m.T += w * v.T */
    const float alpha = 1.0;
    CUGER(cublas_handle(),
          M,N,
          &alpha,(const float*)w_,1,
                 (const float*)v_,1,
          (float*)m_,M);
}

static __global__ void transpose_kernel(const float* m, float* mt, int N, int M)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = N * M;
    if (k < n) {
        int i = k / M;
        int j = k % M;
        mt[j * N + i] = m[k];
    }
}

/* Transposes the matrix m and returns the transposed matrix in mt.
 */
static inline void transpose(const fArr2D restrict m_/*[N][M]*/,
                             fArr2D restrict mt_/*[M][N]*/,
                             int N, int M)
{
    int n = N * M;
    int threads = 256;
    transpose_kernel<<<(n + threads - 1) / threads,threads>>>(
        (const float*)m_,(float*)mt_,N,M);
}

static __global__ void matdiag_kernel(const float* m, float* v, int D, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D)
        v[i] = m[i * M + i];
}

/* Returns in v the elements of the main diagonal of m.
 * If N >= M v will contain M elements. Otherwise, it will contain N elements.
 */
static inline void matdiag(const fArr2D restrict m_/*[N][M]*/,
                           fVec restrict v_/*[min(N,M)]*/,
                           int N, int M)
{
    int D = (N < M) ? N : M;
    int threads = 256;
    matdiag_kernel<<<(D + threads - 1) / threads,threads>>>(
        (const float*)m_,(float*)v_,D,M);
}

static __global__ void diagmat_kernel(const float* v, float* m, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        m[i * N + i] = v[i];
}

/* Returns a square matrix m whose diagonal contains the elements of
 * the vector v.
 */
static inline void diagmat(const fVec restrict v_/*[N]*/,
                           fArr2D restrict m_/*[N][N]*/,
                           int N)
{
    cufltclr((void*)m_,0,(size_t)N * N);
    int threads = 256;
    diagmat_kernel<<<(N + threads - 1) / threads,threads>>>(
        (const float*)v_,(float*)m_,N);
}

static __global__ void mateye_kernel(float* m, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        m[i * N + i] = 1.0;
}

/* Initializes the passed in square matrix to the identity matrix.
 */
static inline void mateye(const fArr2D restrict m_/*[N][N]*/, int N)
{
    cufltclr((void*)m_,0,(size_t)N * N);
    int threads = 256;
    mateye_kernel<<<(N + threads - 1) / threads,threads>>>((float*)m_,N);
}

#endif
