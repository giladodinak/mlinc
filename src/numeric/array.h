/* Copyright (c) 2023-2024 Gilad Odinak */
/* Array data structures and functions  */
#ifndef ARRAY_H
#define ARRAY_H
#include <math.h>
#include "float.h"

/* Define types of one and two dimensional arrays of unspecified dimensions
 * These are dynamically cast to explicit dimensions within each function
 *
 * for example
 *     fArr2D a2d; // a2d is two-dimensional array of unspecified dimensions
 *     typedef float (*ArrNK)[K]; // two-dimensional array NxK (e.g. N=3 K=5)
 *     ArrNK a2dNK = (ArrNK) a2d; a2dNK is the addrress of 2-D array 3x5
 *     a2dNK[2][4] access an entry in the array
 * Notice that a2dNK memory is contiguous just like when declaring
 * a two-dimensional array  float arr[3][5]
 *
 * When passing an array by reference to a function, only the last dimension
 * can be sepcified. By convention, the first dimension is noted in typedef,
 * and in comments, see examples below.
 *
 */
typedef int (*iVec);
typedef int (*iArr2D)[];
typedef float (*fVec);
typedef float (*fArr2D)[];

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#ifdef USE_DOUBLE
#define GEMM cblas_dgemm
#else
#define GEMM cblas_sgemm
#endif
#endif

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
    /* r[N][M] = x[N][d] @ y[d][M] */
#ifndef USE_BLAS
    typedef float (*ArrNM)[M]; ArrNM r = (ArrNM) r_;
    typedef float (*ArrNd)[d]; const ArrNd x = (const ArrNd) x_;
    typedef float (*ArrdM)[M]; const ArrdM y = (const ArrdM) y_;
    fltclr(r,N * M);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++)
            for (int j = 0; j < M; j++)
                r[i][j] += x[i][k] * y[k][j];
#else
    GEMM(CblasRowMajor,CblasNoTrans,CblasNoTrans,
         N,M,d,
         1.0,(const float*)x_,d,
         (const float*) y_,M,
         0.0, (float*) r_,M);
#endif
}

/* Multiplies matrix x by the transpose of matrix y.
 * Adds the result to the matrix r.
 * r = r + x @ y.T
 * r: resulting matrix NxM
 * x: left matrix Nxd
 * y: right matrix Mxd
 * Note that d is the common dimension, not related to D, which usually
 * indicates the size of neural network layer's input vectors dimension.
 */
static inline void addMatmulT(fArr2D restrict r_/*[N][M]*/,
                               const fArr2D restrict x_/*[N][d]*/,
                               const fArr2D restrict y_/*[M][d]*/,
                               int N, int d, int M)
{
#ifndef USE_BLAS
    typedef float (*ArrNM)[M]; ArrNM r = (ArrNM) r_;
    typedef float (*ArrNd)[d]; const ArrNd x = (const ArrNd) x_;
    typedef float (*ArrMd)[d]; const ArrMd y = (const ArrMd) y_;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < d; k++)
                r[i][j] += x[i][k] * y[j][k];
#else
    GEMM(CblasRowMajor, CblasNoTrans, CblasTrans,
          N,M,d,
          1.0,(const float*) x_,d,
          (const float*) y_,d,
          1.0,(float*) r_,M);
#endif
}

/* Multiplies matrix x by the transpose of matrix y.
 * Returns the result in matrix r.
 * r = x @ y.T
 * r: resulting matrix NxM
 * x: left matrix Nxd
 * y: right matrix Mxd
 * Note that d is the common dimension, not related to D, which usually
 * indicates the size of neural network layer's input vectors dimension.
 */
static inline void matmulT(fArr2D restrict r_/*[N][M]*/,
                            const fArr2D restrict x_/*[N][d]*/,
                            const fArr2D restrict y_/*[M][d]*/,
                            int N, int d, int M)
{
    /* r[N][M] = x[N][d] @ y[M][d]^T */
#ifndef USE_BLAS
    fltclr((float *) r_,N * M);
    addMatmulT(r_,x_,y_,N,d,M);
#else
    GEMM(CblasRowMajor, CblasNoTrans, CblasTrans,
          N,M,d,
          1.0,(const float*) x_,d,
          (const float*) y_,d,
          0.0,(float*) r_,M);
#endif
}

/* Multiplies the transpose of matrix x by matrix y.
 * Returns the result in matrix r.
 * r = x.T @ y
 * r: resulting matrix NxM
 * x: left matrix dxN
 * y: right matrix dxM
 * Note that d is the common dimension, not related to D, which usually
 * indicates the size of neural network layer's input vectors dimension.
 */
static inline void Tmatmul(fArr2D restrict r_/*[N][M]*/,
                           const fArr2D restrict x_/*[d][N]*/,
                           const fArr2D restrict y_/*[d][M]*/,
                           int N, int d, int M)
{
#ifndef USE_BLAS
    typedef float (*ArrNM)[M]; ArrNM r = (ArrNM) r_;
    typedef float (*ArrdN)[N]; const ArrdN x = (const ArrdN) x_;
    typedef float (*ArrdM)[M]; const ArrdM y = (const ArrdM) y_;
    fltclr(r,N * M);
    for (int k = 0; k < d; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                r[i][j] += x[k][i] * y[k][j];
#else
    GEMM(CblasRowMajor, CblasTrans, CblasNoTrans,
         N,M,d,
         1.0,(const float*) x_,N,
         (const float*) y_,M,
         0.0,(float*) r_,M);
#endif
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
    typedef float (*VecN);
    typedef float (*VecM);
    typedef float (*ArrMN)[N];
    VecN r = (VecN) r_;
    const VecM v = (const VecM) v_;
    const ArrMN m = (const ArrMN) m_;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            r[j] += v[i] * m[i][j];
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
    typedef float (*VecN);
    typedef float (*VecM);
    typedef float (*ArrNM)[M];
    VecN v = (VecN) v_;
    const VecM w = (const VecM) w_;
    const ArrNM m = (const ArrNM)  m_;

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            v[j] += w[i] * m[j][i];
}

/* Calculates the tensor product (i.e. outer multiplication) of the
 * vector v by the vector w, and adds the resulting matrix to the matrix in m.
 * m = m + (v ⊚ w)
 * ⊚  denotes outer multiplication
 * m: matrix NxM
 * v: vector 1xN
 * w: vector 1xM
 */
static inline void addoutermul(fArr2D restrict m_/*[N][M]*/,
                               const fVec restrict v_/*[N]*/,
                               const fVec restrict w_/*[M]*/,
                               int N, int M)
{
    typedef float (*VecN);
    typedef float (*VecM);
    typedef float (*ArrNM)[M];
    const VecN v = (const VecN) v_;
    const VecM w = (const VecM) w_;
    ArrNM m = (ArrNM) m_;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            m[i][j] += v[i] * w[j];
}

/* Transposes the matrix m and returns the tansposed matrix in mt.
 * The transpose of m is obtained by flipping the rows and columns of m.
 */
static inline void transpose(const fArr2D restrict m_/*[N][M]*/,
                             fArr2D restrict mt_/*[M][N]*/,
                             int N, int M)
{
    typedef float (*ArrNM)[M];
    typedef float (*ArrMN)[N];
    ArrNM m = (ArrNM) m_;
    ArrMN mt = (ArrMN) mt_;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            mt[j][i] = m[i][j];
}

/* Returns in v the elements of the main diagonal of m.
 * If N >= M v will contain M elements. Otherwise, it will contain N elements.
 */
static inline void matdiag(const fArr2D restrict m_/*[N][M]*/,
                           fVec restrict v_/*[min(N,M)]*/,
                           int N, int M)
{
    int D = (N < M) ? N : M;
    typedef float (*ArrNM)[M];
    typedef float (*VecD);
    ArrNM m = (ArrNM) m_;
    VecD v = (VecD) v_;
    for (int i = 0; i < D; i++)
        v[i] = m[i][i];
}

/* Returns a square matrix m whose diagonal contains the elements of
 * the vector v.
 */
static inline void diagmat(const fVec restrict v_/*[N]*/,
                           fArr2D restrict m_/*[N][N]*/,
                           int N)
{
    typedef float (*VecN);
    VecN v = (VecN) v_;
    typedef float (*ArrNN)[N];
    ArrNN m = (ArrNN) m_;
    fltclr(m,N*N);
    for (int i = 0; i < N; i++)
        m[i][i] = v[i];
}

/* Initializes the passed in quare matrix to the identity matrix.
 */
static inline void mateye(const fArr2D restrict m_/*[N][N]*/, int N)
{
    typedef float (*ArrNN)[N];
    ArrNN m = (ArrNN) m_;
    fltclr(m,N * N);
    for (int i = 0; i < N; i++)
        m[i][i] = 1.0;
}
#endif
