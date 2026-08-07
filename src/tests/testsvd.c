/* Copyright (c) 2023-2024 Gilad Odinak */
/* Test SVD decomposistion              */
#include <stdio.h>
#include <strings.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "arrayio.h"
#include "svd.h"

float A0[4][4] = {
    { 0.0 ,  0.0 ,  0.0 ,  2.0 },
    { 0.0 , -6.0 , -4.0 , -8.0 },
    { 6.0 ,  6.0 ,  2.0 ,  5.0 },
    { 0.0 ,  0.0 , -4.0 , -2.0 }
};

float A1[3][4] = {
    { 1.0 ,  0.0 ,  0.0 ,  2.0 },
    { 0.0 ,  0.0 , -1.0 , -8.0 },
    { 0.0 , -1.0 ,  0.0 ,  5.0 }
};

float A2[2][3] = {
    { 1.0 ,  2.0 , 3.0 },
    { 4.0 ,  5.0 , 6.0 }
};

float A3[4][3] = {
    {  6.0 ,  2.0 ,  5.0 },
    {  0.0 , -4.0 , -8.0 },
    { -1.0 ,  0.0 ,  2.0 },
    {  2.0 ,  2.0 ,  7.0 }
};

float A4[4][4] = {
    {  6.0 ,  2.0 ,  5.0, 0.0 },
    {  0.0 , -4.0 , -8.0, 0.0 },
    { -1.0 ,  0.0 ,  2.0, 0.0 },
    {  2.0 ,  2.0 ,  7.0, 0.0 }
};

float U3[4][3] = {
    { -0.52163, 0.78568,-0.09576 },
    {  0.63338, 0.54119, 0.53601 },
    { -0.10894,-0.29311, 0.61008 },
    { -0.56113,-0.06258, 0.57561 }
};

int is_close(fArr2D A_, fArr2D R_, int m, int n, float tol)
{
    int ok = 1;
    typedef float (*ArrMN)[n];
    ArrMN A = (ArrMN) A_;
    ArrMN R = (ArrMN) R_;
    
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (fabsf(fabsf(R[i][j]) - fabsf(A[i][j])) > tol)
                ok = 0;
    return ok;
}

/* Checks whether A is orthogonal. If it is returns 1; otherwise, returns 0.
 * Calculates the elements of A.T @ T and compares to the corresponding
 * value of and identity matrix of same dimensions.
 */
int is_orthogonal(fArr2D A_, int n, float tol)
{
    typedef float (*ArrNN)[n];
    ArrNN A = (ArrNN) A_;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[k][i] * A[k][j];
            if (i == j && fabsf(s - ((float)1.0)) > tol)
                return 0; /* Diagonal elements should be close to 1 */
            if (i != j && fabsf(s) > tol)
                return 0; /* Off-diagonal elements should be close to 0 */
        }
    }
    return 1;
}

int svd_test(fArr2D A, int m, int n, int quiet, int precision, int index)
{
    typedef float (*VecN);
    typedef float (*VecM);
    typedef float (*ArrNN)[n];
    typedef float (*ArrMN)[n];
    typedef float (*ArrMM)[m];

    float tol = 1 / pow(10,(precision-1));
    int dec, pos, ortho, ok;
    int i;
    char name[16];
    char format[16];
    snprintf(format,sizeof(format),"%%%d.%df",3+precision,precision);

    if (!quiet) {
        snprintf(name,sizeof(name),"A%d",index);
        print_array(A,m,n,name,format,0);
    }

    /* R = U @ diag(S) @ Vt           */
    ArrMN R = allocmem(m,n,float);
    
    if (m >= n) {
        /* results of tall array svd */
        ArrMN U_t = allocmem(m,n,float);
        VecN S_t = allocmem(1,n,float);
        ArrNN Vt_t = allocmem(n,n,float);
        /* intermediate tall calculations */
        ArrNN DS_t = allocmem(n,n,float); /* diag(S)      */
        ArrMN US_t = allocmem(m,n,float); /* U @ diag(S)  */
        
        SVD(A,U_t,S_t,Vt_t,m,n);
        if (!quiet) {
            snprintf(name,sizeof(name),"U%d",index);
            print_array(U_t,m,n,name,format,0);
            snprintf(name,sizeof(name),"S%d",index);
            print_array((fArr2D)S_t,1,n,name,format,0);
            snprintf(name,sizeof(name),"Vt%d",index);
            print_array(Vt_t,n,n,name,format,0);
        }
        diagmat(S_t,DS_t,n);
        matmul(US_t,U_t,DS_t,m,n,n);
        matmul(R,US_t,Vt_t,m,n,n);

        pos = 1;
        for (i = 0; i < n - 1; i++)
            if (S_t[i] < 0.0)
                pos = 0;
        dec = 1;
        for (i = 1; i < n - 1; i++)
            if (S_t[i - 1] < S_t[i])
                dec = 0;
        /* Only check Vt, since if S is ok and Vt is ok and A == R, U must be ok */
        ortho = is_orthogonal(Vt_t,n,tol);

        freemem(U_t);
        freemem(S_t);
        freemem(Vt_t);
        freemem(DS_t);
        freemem(US_t);
    }
    else {
        /* results of wide array svd */
        VecM S_w = allocmem(1,m,float);
        ArrMM U_w = allocmem(m,m,float);
        ArrMN Vt_w = allocmem(m,n,float);
        /* intermediate wide calculations */
        ArrMM DS_w = allocmem(m,m,float); /* diag(S)      */
        ArrMM US_w = allocmem(m,m,float); /* U @ diag(S)  */
        
        SVD(A,U_w,S_w,Vt_w,m,n);
        if (!quiet) {
            snprintf(name,sizeof(name),"U%d",index);
            print_array(U_w,m,m,name,format,0);
            snprintf(name,sizeof(name),"S%d",index);
            print_array((fArr2D)S_w,1,m,name,format,0);
            snprintf(name,sizeof(name),"Vt%d",index);
            print_array(Vt_w,m,n,name,format,0);
        }
        diagmat(S_w,DS_w,m);
        matmul(US_w,U_w,DS_w,m,m,m);
        matmul(R,US_w,Vt_w,m,m,n);

        pos = 1;
        for (i = 0; i < m - 1; i++)
            if (S_w[i] < 0.0)
                pos = 0;
        dec = 1;
        for (i = 1; i < m - 1; i++)
            if (S_w[i - 1] < S_w[i])
                dec = 0;
        /* Only check U, since if S is ok and U is ok and A == R, Vt must be ok */
        ortho = is_orthogonal(U_w,m,tol);

        freemem(S_w);
        freemem(U_w);
        freemem(Vt_w);
        freemem(DS_w);
        freemem(US_w);
    }
    if (!quiet) {
        snprintf(name,sizeof(name),"R%d",index);
        print_array(R,m,n,name,format,0);
    }
    ok = is_close(A,R,m,n,tol);
    if (!ok)
        printf("Original matrix A%d and reconstructed matrix R are not close\n",index);
    if (!dec)
        printf("Vector S%d elements are not in decreasing order (m %d, n %d)\n",index,m,n);
    if (!pos)
        printf("Vector S%d elements are not all non-negative\n",index);
    if (!ortho)
        printf("Matrix %s%d is not orthogonal\n",(m>=n)?"Vt":"U",index);

    freemem(R);
    return ok && dec && pos && ortho;
}

int svd_test_U(fArr2D A, fArr2D U, int inplace, int m, int n, int quiet, int precision, int index)
{
    typedef float (*VecM);
    typedef float (*ArrMN)[n];
    typedef float (*ArrMM)[m];

    float tol = 1 / pow(10,(precision-1));
    int ok;
    char name[16];
    char format[16];
    snprintf(format,sizeof(format),"%%%d.%df",3+precision,precision);

    if (!quiet) {
        snprintf(name,sizeof(name),"A%d",index);
        print_array(A,m,n,name,format,0);
    }
    
    if (m >= n) {
        /* results of tall array svd */
        ArrMN U_t = allocmem(m,n,float);

        if (inplace) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    U_t[i][j] = ((ArrMN) A)[i][j];
            SVD(U_t,NULL,NULL,NULL,m,n);
        }
        else
            SVD(A,U_t,NULL,NULL,m,n);
        if (!quiet) {
            snprintf(name,sizeof(name),"U%d",index);
            print_array(U_t,m,n,name,format,0);
        }
        ok = is_close(U,U_t,m,n,tol);
        freemem(U_t);
    }
    else {
        /* results of wide array svd */
        VecM S_w = allocmem(1,m,float);
        ArrMM U_w = allocmem(m,m,float);
        
        SVD(A,U_w,S_w,NULL,m,n);
        if (!quiet) {
            snprintf(name,sizeof(name),"U%d",index);
            print_array(U_w,m,m,name,format,0);
        }
        ok = is_close(U,U_w,m,m,tol);
        freemem(S_w);
        freemem(U_w);
    }
    if (!ok)
        printf("Expected matrix U%d and calculated matrix U are not close\n",index);
    return ok;
}

int full_test(int min_dim, int max_dim, int num_tests, int quiet, int precision)
{
    int pass = 1;

    for (int i = 0; i < num_tests; i++) {
        int m = urand(min_dim,max_dim);
        int n = urand(min_dim,max_dim);
        printf("running test %d A[%d][%d]   \r",i,m,n);
        fflush(stdout);
        {
            float A[m][n];
            for (int j = 0; j < m; j++)
                for (int k = 0; k < n; k++)
                    A[j][k] = nrand(0,1) * 9.0;
            int ok = svd_test(A,m,n,quiet,precision,i);
            if (!ok)
                pass = 0;
        }
    }
    printf("                                 \r");
    fflush(stdout);
    return pass;
}

int svd_test_null(fArr2D A_, int m, int n, int want_u, int want_s, int want_vt,
                  int quiet, int precision, int index)
{
    float tol = 1 / pow(10,(precision-1));
    int tall = (m >= n);
    int ur = m,            uc = tall ? n : m; /* U  is m x n (tall) or m x m (wide) */
    int ls = tall ? n : m;                    /* S  length                          */
    int vr = tall ? n : m, vc = n;            /* Vt is n x n (tall) or m x n (wide)  */

    /* reference full decomposition (on its own copy of A) */
    fArr2D Aref  = allocmem(m,n,float);
    fltcpy(Aref,A_,m * n);
    fArr2D Uref  = allocmem(ur,uc,float);
    fVec   Sref  = allocmem(1,ls,float);
    fArr2D Vtref = allocmem(vr,vc,float);
    SVD(Aref,Uref,Sref,Vtref,m,n);
    freemem(Aref);

    /* partial decomposition on a fresh copy of A, unrequested outputs NULL */
    fArr2D Apart = allocmem(m,n,float);
    fltcpy(Apart,A_,m * n);
    fArr2D U  = want_u  ? allocmem(ur,uc,float) : NULL;
    fVec   S  = want_s  ? allocmem(1,ls,float)  : NULL;
    fArr2D Vt = want_vt ? allocmem(vr,vc,float) : NULL;
    SVD(Apart,U,S,Vt,m,n);

    int keptA = is_close(A_,Apart,m,n,tol);
    int oku   = !want_u  || is_close(Uref,U,ur,uc,tol);
    int oks   = !want_s  || is_close((fArr2D)Sref,(fArr2D)S,1,ls,tol);
    int okvt  = !want_vt || is_close(Vtref,Vt,vr,vc,tol);

    if (!quiet) {
        char name[16], format[16];
        snprintf(format,sizeof(format),"%%%d.%df",3+precision,precision);
        snprintf(name,sizeof(name),"A%d",index);
        print_array(A_,m,n,name,format,0);
    }
    if (!keptA)
        printf("Test %d: input matrix A was modified by SVD\n",index);
    if (!oku)
        printf("Test %d: U does not match reference decomposition\n",index);
    if (!oks)
        printf("Test %d: S does not match reference decomposition\n",index);
    if (!okvt)
        printf("Test %d: Vt does not match reference decomposition\n",index);

    freemem(Uref); freemem(Sref); freemem(Vtref);
    if (U)  freemem(U);
    if (S)  freemem(S);
    if (Vt) freemem(Vt);
    freemem(Apart);

    return keptA && oku && oks && okvt;
}

int svd_test_null_rand(int m, int n, int want_u, int want_s, int want_vt,
                       int precision, int index)
{
    fArr2D A = allocmem(m,n,float);
    typedef float (*ArrMN)[n];
    ArrMN a = (ArrMN) A;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = nrand(0,1) * 9.0;
    int ok = svd_test_null(A,m,n,want_u,want_s,want_vt,1,precision,index);
    freemem(A);
    return ok;
}

int main()
{
    printf("smoke test 4 x 4 %s\n",svd_test(A0,4,4,1,5,0) ? "ok" : "failed");
    printf("smoke test 3 x 4 %s\n",svd_test(A1,3,4,1,5,1) ? "ok" : "failed");
    printf("smoke test 2 x 3 %s\n",svd_test(A2,2,3,1,5,2) ? "ok" : "failed");
    printf("smoke test 4 x 3 %s\n",svd_test(A3,4,3,1,5,3) ? "ok" : "failed");
    printf("smoke test 4 x 4 %s\n",svd_test(A4,4,4,1,5,4) ? "ok" : "failed");
    printf("smoke test 4 x 3 U only %s\n",svd_test_U(A3,U3,0,4,3,1,5,5) ? "ok" : "failed");
    printf("smoke test 4 x 3 inplace U only %s\n",svd_test_U(A3,U3,1,4,3,1,5,6) ? "ok" : "failed");

    /* Null-output tests, temporaries on the stack (VLA) */
    printf("null test 4 x 3 tall, S=NULL (VLA)        %s\n",
        svd_test_null(A3,4,3,1,0,1,1,5,7)  ? "ok" : "failed");
    printf("null test 4 x 3 tall, U=S=NULL (VLA)      %s\n",
        svd_test_null(A3,4,3,0,0,1,1,5,8)  ? "ok" : "failed");
    printf("null test 2 x 3 wide, Vt=S=NULL (VLA)     %s\n",
        svd_test_null(A2,2,3,1,0,0,1,5,9)  ? "ok" : "failed");
    /* Null-output tests, temporaries on the heap (allocmem) */
    printf("null test 512x512 tall, U=NULL (heap)     %s\n",
        svd_test_null_rand(512,512,0,1,1,4,10) ? "ok" : "failed");
    printf("null test 512x512 tall, U=S=NULL (heap)   %s\n",
        svd_test_null_rand(512,512,0,0,1,4,11) ? "ok" : "failed");
    printf("null test 512x600 wide, Vt=S=NULL (heap)  %s\n",
        svd_test_null_rand(512,600,1,0,0,4,12) ? "ok" : "failed");

    printf("full test %s\n",full_test(2,512,100,1,4) ? "ok" : "failed");

    return 0;
}
