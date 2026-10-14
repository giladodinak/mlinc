/* Copyright (c) 2023-2026 Gilad Odinak */
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "array.h"
#include "cumem.h"

#define M 4
#define N 3
#define S 4
#define D 2

int test_matmul()
{
    float r[N][M];
    float x[N][D] = {
        {3.0, 6.0},
        {5.0, 4.0},
        {7.0, 2.0}
    };
    float y[D][M] = {
        { 9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };
    float rr[N][M] = {
        {105.0, 114.0, 123.0, 132.0},
        { 97.0, 106.0, 115.0, 124.0},
        { 89.0,  98.0, 107.0, 116.0}
    };

    fArr2D r_ = (fArr2D) cuallocmem(N,M,float);
    fArr2D x_ = (fArr2D) cuallocmem(N,D,float);
    fArr2D y_ = (fArr2D) cuallocmem(D,M,float);

    cudaMemcpy(x_,x,sizeof(x),cudaMemcpyHostToDevice);
    cudaMemcpy(y_,y,sizeof(y),cudaMemcpyHostToDevice);

    /* r = x @ y */
    matmul(r_,x_,y_,N,D,M);
    cudaMemcpy(r,r_,sizeof(r),cudaMemcpyDeviceToHost);

    float ok = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            ok += fabsf(rr[i][j] - r[i][j]);

    cufreemem(r_);
    cufreemem(x_);
    cufreemem(y_);
    return (ok < 1e-9);
}

int test_matmulT()
{
    float r[N][M];
    float x[N][D] = {
        {3.0, 6.0},
        {5.0, 4.0},
        {7.0, 2.0}
    };
    float y[M][D] = {
        { 9.0, 13.0},
        {10.0, 14.0},
        {11.0, 15.0},
        {12.0, 16.0}
    };
    float rr[N][M] = {
        {105.0, 114.0, 123.0, 132.0},
        { 97.0, 106.0, 115.0, 124.0},
        { 89.0,  98.0, 107.0, 116.0}
    };

    fArr2D r_ = (fArr2D) cuallocmem(N,M,float);
    fArr2D x_ = (fArr2D) cuallocmem(N,D,float);
    fArr2D y_ = (fArr2D) cuallocmem(M,D,float);

    cudaMemcpy(x_,x,sizeof(x),cudaMemcpyHostToDevice);
    cudaMemcpy(y_,y,sizeof(y),cudaMemcpyHostToDevice);

    /* r = x @ y.T */
    matmulT(r_,x_,y_,N,D,M);
    cudaMemcpy(r,r_,sizeof(r),cudaMemcpyDeviceToHost);

    float ok = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            ok += fabsf(rr[i][j] - r[i][j]);

    cufreemem(r_);
    cufreemem(x_);
    cufreemem(y_);
    return (ok < 1e-9);
}

int test_addvecmatmul()
{
    float r[N] = {1.0, 2.0, 3.0};
    const float v[M] = {4.0, 5.0, 6.0, 7.0};
    const float m[M][N] = {
        {8.0, 9.0, 10.0},
        {11.0, 12.0, 13.0},
        {14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0}
    };
    float ur[N] = {291.0, 314.0, 337.0};

    fVec r_ = (fVec) cuallocmem(1,N,float);
    fVec v_ = (fVec) cuallocmem(1,M,float);
    fArr2D m_ = (fArr2D) cuallocmem(M,N,float);

    cudaMemcpy(r_,r,sizeof(r),cudaMemcpyHostToDevice);
    cudaMemcpy(v_,v,sizeof(v),cudaMemcpyHostToDevice);
    cudaMemcpy(m_,m,sizeof(m),cudaMemcpyHostToDevice);

    /* r = r + v @ m */
    addvecmatmul(r_,v_,m_,M,N);
    cudaMemcpy(r,r_,sizeof(r),cudaMemcpyDeviceToHost);

    float ok = 0.0;
    for (int i = 0; i < N; i++)
        ok += fabsf(ur[i] - r[i]);

    cufreemem(r_);
    cufreemem(v_);
    cufreemem(m_);
    return (ok < 1e-9);
}

int test_addinnermul()
{
    /* v = np.array([1.0, 2.0, 3.0, 4.0]) */
    float v[S] = {1.0, 2.0, 3.0, 4.0};
    /* w = np.array([5.0, 6.0, 7.0, 8.0]) */
    float w[S] = {5.0, 6.0, 7.0, 8.0};
    /* m = np.array([[9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0]]) */
    float m[S][S] = {
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0, 20.0},
        {21.0, 22.0, 23.0, 24.0}
    };
    float uv[S] = {279.0, 384.0, 489.0, 594.0};

    fVec v_ = (fVec) cuallocmem(1,S,float);
    fVec w_ = (fVec) cuallocmem(1,S,float);
    fArr2D m_ = (fArr2D) cuallocmem(S,S,float);

    cudaMemcpy(v_,v,sizeof(v),cudaMemcpyHostToDevice);
    cudaMemcpy(w_,w,sizeof(w),cudaMemcpyHostToDevice);
    cudaMemcpy(m_,m,sizeof(m),cudaMemcpyHostToDevice);

    /* v = v + w @ m.T */
    addinnermul(v_,w_,m_,S,S);
    cudaMemcpy(v,v_,sizeof(v),cudaMemcpyDeviceToHost);

    float ok = 0.0;
    for (int i = 0; i < S; i++)
        ok += fabsf(uv[i] - v[i]);

    cufreemem(v_);
    cufreemem(w_);
    cufreemem(m_);
    return (ok < 1e-9);
}

int test_addoutermul()
{
    float m[N][M] = {
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0, 20.0}
    };
    float v[N] = {1.0, 2.0, 3.0};
    float w[M] = {5.0, 6.0, 7.0, 8.0};

    float um[N][M] = {
        {14.0, 16.0, 18.0, 20.0},
        {23.0, 26.0, 29.0, 32.0},
        {32.0, 36.0, 40.0, 44.0}
    };

    fArr2D m_ = (fArr2D) cuallocmem(N,M,float);
    fVec v_ = (fVec) cuallocmem(1,N,float);
    fVec w_ = (fVec) cuallocmem(1,M,float);

    cudaMemcpy(m_,m,sizeof(m),cudaMemcpyHostToDevice);
    cudaMemcpy(v_,v,sizeof(v),cudaMemcpyHostToDevice);
    cudaMemcpy(w_,w,sizeof(w),cudaMemcpyHostToDevice);

    /* m = m + (v ⊚ w) */
    addoutermul(m_,v_,w_,N,M);
    cudaMemcpy(m,m_,sizeof(m),cudaMemcpyDeviceToHost);

    float ok = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            ok += fabsf(um[i][j] - m[i][j]);

    cufreemem(m_);
    cufreemem(v_);
    cufreemem(w_);
    return (ok < 1e-9);
}

int main()
{
    printf("matmul() test %s\n",test_matmul() ? "ok" : "failed");
    printf("matmulT() test %s\n",test_matmulT() ? "ok" : "failed");
    printf("addvecmatmul() test %s\n",test_addvecmatmul() ? "ok" : "failed");
    printf("addoutermul() test %s\n",test_addoutermul() ? "ok" : "failed");
    printf("addinnermul() test %s\n",test_addinnermul() ? "ok" : "failed");
    return 0;
}
