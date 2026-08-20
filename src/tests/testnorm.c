/* Copyright (c) 2026 Gilad Odinak */
/* Simple test to check lyrnorm and grpnorm layer correctness */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "lyrnorm.h"
#include "grpnorm.h"
#include "addnorm.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-finite-math-only", "no-unsafe-math-optimizations")
#endif

#define EPS 1e-3
#define TOL 1e-2

void test_lyrnorm_normalized(LYRNORM* l)
{
    printf("Test: lyrnorm normalized output\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float Y[B][D];

    /* gamma=1, beta=0 from init, so output equals the normalized values */
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-2.0f,2.0f);

    lyrnorm_forward(l,X,Y);

    for (int i = 0; i < B; i++) {
        float mean = 0;
        for (int j = 0; j < D; j++)
            mean += Y[i][j];
        mean /= D;

        float var = 0;
        for (int j = 0; j < D; j++)
            var += Y[i][j] * Y[i][j];
        var /= D;

        if (fabsf(mean) > TOL) {
            printf("FAIL: row %d mean=%g, expected 0\n",i,mean);
            exit(1);
        }
        if (fabsf(var - 1.0f) > TOL) {
            printf("FAIL: row %d variance=%g, expected 1\n",i,var);
            exit(1);
        }
    }

    printf("PASS\n");
}

/* Test affine transform: y = gamma[c] * xn + beta[c]. */
void test_lyrnorm_affine(LYRNORM* l)
{
    printf("Test: lyrnorm affine transform\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float Y[B][D];

    typedef float (*ArrBD)[D];
    ArrBD xn = (ArrBD) l->xn;

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f,1.0f);

    lyrnorm_forward(l,X,Y);

    /* y must equal gamma*xn + beta */
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            float expect = l->gamma[j] * xn[i][j] + l->beta[j];
            if (fabsf(Y[i][j] - expect) > TOL) {
                printf("FAIL: Y[%d][%d]=%g expected %g\n",i,j,Y[i][j],expect);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

void test_lyrnorm_invariance(LYRNORM* l)
{
    printf("Test: lyrnorm scale/shift invariance\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float XT[B][D];
    float Y1[B][D];
    float Y2[B][D];

    /* Pure normalization (gamma=1,beta=0) is invariant to a per-row
     * positive scale and shift of the input */
    for (int j = 0; j < D; j++) {
        l->gamma[j] = 1.0f;
        l->beta[j]  = 0.0f;
    }
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f,1.0f);

    for (int i = 0; i < B; i++) {
        float a = urand(0.5f,2.0f);   /* positive scale, per row */
        float c = urand(-2.0f,2.0f);  /* shift, per row          */
        for (int j = 0; j < D; j++)
            XT[i][j] = a * X[i][j] + c;
    }

    lyrnorm_forward(l,X, Y1);
    lyrnorm_forward(l,XT,Y2);

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            if (fabsf(Y1[i][j] - Y2[i][j]) > TOL) {
                printf("FAIL: Y1[%d][%d]=%g Y2=%g differ\n",i,j,Y1[i][j],Y2[i][j]);
                exit(1);
            }

    printf("PASS\n");
}

void test_lyrnorm_grad_invariants(LYRNORM* l)
{
    printf("Test: lyrnorm gradient invariants\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float Y[B][D];
    float dY[B][D];
    float dX[B][D];
    float dg[D];
    float db[D];

    typedef float (*ArrBD)[D];
    ArrBD xn = (ArrBD) l->xn;

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0f,1.0f);
            dY[i][j] = urand(-1.0f,1.0f);
        }
    }

    lyrnorm_forward(l,X,Y);
    lyrnorm_backward(l,dY,dX,dg,db);

    /* Normalization Jacobian: sum(dx)=0 and sum(dx*xn)=0 over each row */
    for (int i = 0; i < B; i++) {
        float sum = 0;
        float dot = 0;
        for (int j = 0; j < D; j++) {
            sum += dX[i][j];
            dot += dX[i][j] * xn[i][j];
        }
        if (fabsf(sum) > TOL) {
            printf("FAIL: row %d sum(dx)=%g, expected 0\n",i,sum);
            exit(1);
        }
        if (fabsf(dot) > TOL) {
            printf("FAIL: row %d sum(dx*xn)=%g, expected 0\n",i,dot);
            exit(1);
        }
    }

    printf("PASS\n");
}

void test_lyrnorm_idempotent(LYRNORM* l)
{
    printf("Test: lyrnorm backward idempotent\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float Y[B][D];
    float dY[B][D];
    float dX[B][D];
    float dg1[D],db1[D];
    float dg2[D],db2[D];

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0f,1.0f);
            dY[i][j] = urand(-1.0f,1.0f);
        }
    }

    lyrnorm_forward(l,X,Y);
    lyrnorm_backward(l,dY,dX,dg1,db1);
    lyrnorm_backward(l,dY,dX,dg2,db2);   /* must clear, not accumulate */

    for (int j = 0; j < D; j++) {
        if (fabsf(dg1[j] - dg2[j]) > TOL || fabsf(db1[j] - db2[j]) > TOL) {
            printf("FAIL: dg/db not idempotent at %d\n",j);
            exit(1);
        }
    }

    printf("PASS\n");
}

void test_lyrnorm_constant(LYRNORM* l)
{
    printf("Test: lyrnorm constant input\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float Y[B][D];

    for (int j = 0; j < D; j++) {
        l->gamma[j] = 1.0f;
        l->beta[j]  = 0.25f;
    }
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = 3.14f;   /* zero variance per row */

    lyrnorm_forward(l,X,Y);

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            if (!isfinite(Y[i][j])) {
                printf("FAIL: Y[%d][%d] not finite\n",i,j);
                exit(1);
            }
            if (fabsf(Y[i][j] - 0.25f) > TOL) {
                printf("FAIL: Y[%d][%d]=%g, expected beta 0.25\n",i,j,Y[i][j]);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

float lyrnorm_loss(LYRNORM* l, fArr2D X, fArr2D A_)
{
    int B = l->B;
    int D = l->D;

    float Y[B][D];
    lyrnorm_forward(l,X,Y);

    typedef float (*ArrBD)[D];
    ArrBD A = (ArrBD) A_;

    float L = 0;
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            L += A[i][j] * Y[i][j];

    return L;
}

void test_lyrnorm_finite_diff(LYRNORM* l)
{
    printf("Test: lyrnorm finite-difference gradients\n");

    int B = l->B;
    int D = l->D;

    float X[B][D];
    float Y[B][D];
    float A[B][D];
    float dY[B][D];
    float dX[B][D];
    float dg[D];
    float db[D];

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0f,1.0f);
            A[i][j]  = urand(-1.0f,1.0f);   /* dL/dY = A */
            dY[i][j] = A[i][j];
        }
    }

    lyrnorm_forward(l,X,Y);
    lyrnorm_backward(l,dY,dX,dg,db);

    /* Check dX */
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            float old = X[i][j];
            X[i][j] = old + EPS;
            float Lp = lyrnorm_loss(l,X,A);
            X[i][j] = old - EPS;
            float Ln = lyrnorm_loss(l,X,A);
            X[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dX[i][j];
            if (fabsf(num - ana) > TOL) {
                printf("FAIL dX[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                exit(1);
            }
        }
    }

    /* Check dgamma */
    for (int j = 0; j < D; j++) {
        float old = l->gamma[j];
        l->gamma[j] = old + EPS;
        float Lp = lyrnorm_loss(l,X,A);
        l->gamma[j] = old - EPS;
        float Ln = lyrnorm_loss(l,X,A);
        l->gamma[j] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = dg[j];
        if (fabsf(num - ana) > TOL) {
            printf("FAIL dgamma[%d]: expected=%g calculated=%g\n",j,num,ana);
            exit(1);
        }
    }

    /* Check dbeta */
    for (int j = 0; j < D; j++) {
        float old = l->beta[j];
        l->beta[j] = old + EPS;
        float Lp = lyrnorm_loss(l,X,A);
        l->beta[j] = old - EPS;
        float Ln = lyrnorm_loss(l,X,A);
        l->beta[j] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = db[j];
        if (fabsf(num - ana) > TOL) {
            printf("FAIL dbeta[%d]: expected=%g calculated=%g\n",j,num,ana);
            exit(1);
        }
    }

    printf("PASS\n");
}

void test_grpnorm_normalized(GRPNORM* l)
{
    printf("Test: grpnorm normalized output\n");

    int B = l->B;
    int T = l->T;
    int C = l->C;
    int G = l->G;
    int Cg = C / G;

    float X[B*T][C];
    float Y[B*T][C];

    for (int i = 0; i < B*T; i++)
        for (int c = 0; c < C; c++)
            X[i][c] = urand(-2.0f,2.0f);

    grpnorm_forward(l,X,Y);

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < G; g++) {
            int M = T * Cg;
            float mean = 0;
            for (int t = 0; t < T; t++)
                for (int c = g*Cg; c < (g+1)*Cg; c++)
                    mean += Y[b*T+t][c];
            mean /= M;

            float var = 0;
            for (int t = 0; t < T; t++)
                for (int c = g*Cg; c < (g+1)*Cg; c++)
                    var += Y[b*T+t][c] * Y[b*T+t][c];
            var /= M;

            if (fabsf(mean) > TOL) {
                printf("FAIL: batch %d group %d mean=%g, expected 0\n",b,g,mean);
                exit(1);
            }
            if (fabsf(var - 1.0f) > TOL) {
                printf("FAIL: batch %d group %d variance=%g, expected 1\n",b,g,var);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

void test_grpnorm_per_channel(GRPNORM* l)
{
    printf("Test: grpnorm per-channel over time (G==C)\n");

    int B = l->B;
    int T = l->T;
    int C = l->C;

    if (l->G != C) {
        printf("FAIL: test requires G==C (G=%d C=%d)\n",l->G,C);
        exit(1);
    }

    float X[B*T][C];
    float Y[B*T][C];

    typedef float (*ArrRC)[C];
    ArrRC xn = (ArrRC) l->xn;

    for (int i = 0; i < B*T; i++)
        for (int c = 0; c < C; c++)
            X[i][c] = urand(-1.5f,1.5f);

    grpnorm_forward(l,X,Y);

    /* Each channel, per batch item, is normalized over the time axis */
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float mean = 0;
            for (int t = 0; t < T; t++)
                mean += xn[b*T+t][c];
            mean /= T;

            float var = 0;
            for (int t = 0; t < T; t++)
                var += xn[b*T+t][c] * xn[b*T+t][c];
            var /= T;

            if (fabsf(mean) > TOL || fabsf(var - 1.0f) > TOL) {
                printf("FAIL: batch %d channel %d mean=%g var=%g\n",b,c,mean,var);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

void test_grpnorm_grad_invariants(GRPNORM* l)
{
    printf("Test: grpnorm gradient invariants\n");

    int B = l->B;
    int T = l->T;
    int C = l->C;
    int G = l->G;
    int Cg = C / G;

    float X[B*T][C];
    float Y[B*T][C];
    float dY[B*T][C];
    float dX[B*T][C];
    float dg[C];
    float db[C];

    typedef float (*ArrRC)[C];
    ArrRC xn = (ArrRC) l->xn;

    for (int c = 0; c < C; c++) {
        l->gamma[c] = urand(0.5f,1.5f);
        l->beta[c]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B*T; i++) {
        for (int c = 0; c < C; c++) {
            X[i][c]  = urand(-1.0f,1.0f);
            dY[i][c] = urand(-1.0f,1.0f);
        }
    }

    grpnorm_forward(l,X,Y);
    grpnorm_backward(l,dY,dX,dg,db);

    /* sum(dx)=0 and sum(dx*xn)=0 over each (batch, group) */
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < G; g++) {
            float sum = 0;
            float dot = 0;
            for (int t = 0; t < T; t++) {
                for (int c = g*Cg; c < (g+1)*Cg; c++) {
                    sum += dX[b*T+t][c];
                    dot += dX[b*T+t][c] * xn[b*T+t][c];
                }
            }
            if (fabsf(sum) > TOL) {
                printf("FAIL: batch %d group %d sum(dx)=%g, expected 0\n",b,g,sum);
                exit(1);
            }
            if (fabsf(dot) > TOL) {
                printf("FAIL: batch %d group %d sum(dx*xn)=%g, expected 0\n",b,g,dot);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

void test_grpnorm_constant(GRPNORM* l)
{
    printf("Test: grpnorm constant input\n");

    int B = l->B;
    int T = l->T;
    int C = l->C;

    float X[B*T][C];
    float Y[B*T][C];

    for (int c = 0; c < C; c++) {
        l->gamma[c] = 1.0f;
        l->beta[c]  = 0.5f;
    }
    for (int i = 0; i < B*T; i++)
        for (int c = 0; c < C; c++)
            X[i][c] = -1.25f;   /* zero variance per group */

    grpnorm_forward(l,X,Y);

    for (int i = 0; i < B*T; i++) {
        for (int c = 0; c < C; c++) {
            if (!isfinite(Y[i][c])) {
                printf("FAIL: Y[%d][%d] not finite\n",i,c);
                exit(1);
            }
            if (fabsf(Y[i][c] - 0.5f) > TOL) {
                printf("FAIL: Y[%d][%d]=%g, expected beta 0.5\n",i,c,Y[i][c]);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

float grpnorm_loss(GRPNORM* l, fArr2D X, fArr2D A_)
{
    int B = l->B;
    int T = l->T;
    int C = l->C;

    float Y[B*T][C];
    grpnorm_forward(l,X,Y);

    typedef float (*ArrRC)[C];
    ArrRC A = (ArrRC) A_;

    float L = 0;
    for (int i = 0; i < B*T; i++)
        for (int c = 0; c < C; c++)
            L += A[i][c] * Y[i][c];

    return L;
}

void test_grpnorm_finite_diff(GRPNORM* l)
{
    printf("Test: grpnorm finite-difference gradients (B=%d T=%d C=%d G=%d)\n",
           l->B,l->T,l->C,l->G);

    int B = l->B;
    int T = l->T;
    int C = l->C;

    float X[B*T][C];
    float Y[B*T][C];
    float A[B*T][C];
    float dY[B*T][C];
    float dX[B*T][C];
    float dg[C];
    float db[C];

    for (int c = 0; c < C; c++) {
        l->gamma[c] = urand(0.5f,1.5f);
        l->beta[c]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B*T; i++) {
        for (int c = 0; c < C; c++) {
            X[i][c]  = urand(-1.0f,1.0f);
            A[i][c]  = urand(-1.0f,1.0f);   /* dL/dY = A */
            dY[i][c] = A[i][c];
        }
    }

    grpnorm_forward(l,X,Y);
    grpnorm_backward(l,dY,dX,dg,db);

    /* Check dX */
    for (int i = 0; i < B*T; i++) {
        for (int c = 0; c < C; c++) {
            float old = X[i][c];
            X[i][c] = old + EPS;
            float Lp = grpnorm_loss(l,X,A);
            X[i][c] = old - EPS;
            float Ln = grpnorm_loss(l,X,A);
            X[i][c] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dX[i][c];
            if (fabsf(num - ana) > TOL) {
                printf("FAIL dX[%d][%d]: expected=%g calculated=%g\n",i,c,num,ana);
                exit(1);
            }
        }
    }

    /* Check dgamma */
    for (int c = 0; c < C; c++) {
        float old = l->gamma[c];
        l->gamma[c] = old + EPS;
        float Lp = grpnorm_loss(l, X,A);
        l->gamma[c] = old - EPS;
        float Ln = grpnorm_loss(l,X,A);
        l->gamma[c] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = dg[c];
        if (fabsf(num - ana) > TOL) {
            printf("FAIL dgamma[%d]: expected=%g calculated=%g\n",c,num,ana);
            exit(1);
        }
    }

    /* Check dbeta */
    for (int c = 0; c < C; c++) {
        float old = l->beta[c];
        l->beta[c] = old + EPS;
        float Lp = grpnorm_loss(l,X,A);
        l->beta[c] = old - EPS;
        float Ln = grpnorm_loss(l,X,A);
        l->beta[c] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = db[c];
        if (fabsf(num - ana) > TOL) {
            printf("FAIL dbeta[%d]: expected=%g calculated=%g\n",c,num,ana);
            exit(1);
        }
    }

    printf("PASS\n");
}

/* grpnorm(G=1) with T=1 must match lyrnorm on the same [B][C] data. */
void test_equivalence(void)
{
    printf("Test: grpnorm(G=1,T=1) == lyrnorm (foarward)\n");

    const int B = 6;
    const int C = 9;

    float X[B][C];
    float YL[B][C];
    float YG[B][C];

    for (int i = 0; i < B; i++)
        for (int c = 0; c < C; c++)
            X[i][c] = urand(-1.0f,1.0f);

    LYRNORM* ln = lyrnorm_create();
    lyrnorm_init(ln,C,B);
    GRPNORM* gn = grpnorm_create(1);
    grpnorm_init(gn,C,/*steps=*/1,B);

    /* Same non-trivial affine on both layers */
    for (int c = 0; c < C; c++) {
        float g = urand(0.5f,1.5f);
        float b = urand(-0.5f,0.5f);
        ln->gamma[c] = g; ln->beta[c] = b;
        gn->gamma[c] = g; gn->beta[c] = b;
    }

    lyrnorm_forward(ln,X,YL);
    grpnorm_forward(gn,X,YG);

    for (int i = 0; i < B; i++)
        for (int c = 0; c < C; c++)
            if (fabsf(YL[i][c] - YG[i][c]) > TOL) {
                printf("FAIL: [%d][%d] lyrnorm=%g grpnorm=%g\n",i,c,YL[i][c],YG[i][c]);
                exit(1);
            }

    lyrnorm_free(ln);
    grpnorm_free(gn);
    printf("PASS\n");
}

/* grpnorm(G=1,T=1) backward must match lyrnorm backward on the same data. */
void test_backward_equivalence(void)
{
    printf("Test: grpnorm(G=1,T=1) == lyrnorm (backward)\n");

    const int B = 6;
    const int C = 9;

    float X[B][C];
    float YL[B][C];
    float YG[B][C];
    float dY[B][C];
    float dXL[B][C];
    float dXG[B][C];
    float dgL[C],dbL[C];
    float dgG[C],dbG[C];

    for (int i = 0; i < B; i++) {
        for (int c = 0; c < C; c++) {
            X[i][c]  = urand(-1.0f,1.0f);
            dY[i][c] = urand(-1.0f,1.0f);   /* shared upstream gradient */
        }
    }

    LYRNORM* ln = lyrnorm_create();
    lyrnorm_init(ln,C,B);
    GRPNORM* gn = grpnorm_create(1);
    grpnorm_init(gn,C,/*steps=*/1,B);

    /* Same non-trivial affine on both layers */
    for (int c = 0; c < C; c++) {
        float g = urand(0.5f,1.5f);
        float b = urand(-0.5f,0.5f);
        ln->gamma[c] = g; ln->beta[c] = b;
        gn->gamma[c] = g; gn->beta[c] = b;
    }

    lyrnorm_forward(ln,X,YL);
    grpnorm_forward(gn,X,YG);
    lyrnorm_backward(ln,dY,dXL,dgL,dbL);
    grpnorm_backward(gn,dY,dXG,dgG,dbG);

    /* Input gradients must match */
    for (int i = 0; i < B; i++)
        for (int c = 0; c < C; c++)
            if (fabsf(dXL[i][c] - dXG[i][c]) > TOL) {
                printf("FAIL: dX[%d][%d] lyrnorm=%g grpnorm=%g\n",i,c,dXL[i][c],dXG[i][c]);
                exit(1);
            }

    /* Parameter gradients must match */
    for (int c = 0; c < C; c++) {
        if (fabsf(dgL[c] - dgG[c]) > TOL) {
            printf("FAIL: dgamma[%d] lyrnorm=%g grpnorm=%g\n",c,dgL[c],dgG[c]);
            exit(1);
        }
        if (fabsf(dbL[c] - dbG[c]) > TOL) {
            printf("FAIL: dbeta[%d] lyrnorm=%g grpnorm=%g\n",c,dbL[c],dbG[c]);
            exit(1);
        }
    }

    lyrnorm_free(ln);
    grpnorm_free(gn);
    printf("PASS\n");
}

void test_addnorm_normalized(ADDNORM* l)
{
    printf("Test: addnorm normalized output\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float Y[B][D];

    /* gamma=1, beta=0 from init, so output equals the normalized sum */
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-2.0f,2.0f);
            X2[i][j] = urand(-2.0f,2.0f);
        }

    addnorm_forward(l,X1,X2,Y);

    for (int i = 0; i < B; i++) {
        float mean = 0;
        for (int j = 0; j < D; j++)
            mean += Y[i][j];
        mean /= D;

        float var = 0;
        for (int j = 0; j < D; j++)
            var += Y[i][j] * Y[i][j];
        var /= D;

        if (fabsf(mean) > TOL) {
            printf("FAIL: row %d mean=%g, expected 0\n",i,mean);
            exit(1);
        }
        if (fabsf(var - 1.0f) > TOL) {
            printf("FAIL: row %d variance=%g, expected 1\n",i,var);
            exit(1);
        }
    }

    printf("PASS\n");
}

/* Test affine transform: y = gamma[c] * xn + beta[c]. */
void test_addnorm_affine(ADDNORM* l)
{
    printf("Test: addnorm affine transform\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float Y[B][D];

    typedef float (*ArrBD)[D];
    ArrBD xn = (ArrBD) l->xn;

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-1.0f,1.0f);
            X2[i][j] = urand(-1.0f,1.0f);
        }

    addnorm_forward(l,X1,X2,Y);

    /* y must equal gamma*xn + beta */
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            float expect = l->gamma[j] * xn[i][j] + l->beta[j];
            if (fabsf(Y[i][j] - expect) > TOL) {
                printf("FAIL: Y[%d][%d]=%g expected %g\n",i,j,Y[i][j],expect);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

/* Output depends only on x1+x2: addnorm(x1,x2) == addnorm(x1+x2,0) */
void test_addnorm_sum_dependence(ADDNORM* l)
{
    printf("Test: addnorm depends only on x1+x2\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float S[B][D];
    float Z[B][D];
    float Y1[B][D];
    float Y2[B][D];

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-1.0f,1.0f);
            X2[i][j] = urand(-1.0f,1.0f);
            S[i][j]  = X1[i][j] + X2[i][j];
            Z[i][j]  = 0.0f;
        }

    addnorm_forward(l,X1,X2,Y1);
    addnorm_forward(l,S, Z, Y2);

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            if (fabsf(Y1[i][j] - Y2[i][j]) > TOL) {
                printf("FAIL: Y1[%d][%d]=%g Y2=%g differ\n",i,j,Y1[i][j],Y2[i][j]);
                exit(1);
            }

    printf("PASS\n");
}

void test_addnorm_grad_invariants(ADDNORM* l)
{
    printf("Test: addnorm gradient invariants\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float Y[B][D];
    float dY[B][D];
    float dX[B][D];
    float dg[D];
    float db[D];

    typedef float (*ArrBD)[D];
    ArrBD xn = (ArrBD) l->xn;

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-1.0f,1.0f);
            X2[i][j] = urand(-1.0f,1.0f);
            dY[i][j] = urand(-1.0f,1.0f);
        }
    }

    addnorm_forward(l,X1,X2,Y);
    addnorm_backward(l,dY,dX,dg,db);

    /* Normalization Jacobian: sum(dx)=0 and sum(dx*xn)=0 over each row */
    for (int i = 0; i < B; i++) {
        float sum = 0;
        float dot = 0;
        for (int j = 0; j < D; j++) {
            sum += dX[i][j];
            dot += dX[i][j] * xn[i][j];
        }
        if (fabsf(sum) > TOL) {
            printf("FAIL: row %d sum(dx)=%g, expected 0\n",i,sum);
            exit(1);
        }
        if (fabsf(dot) > TOL) {
            printf("FAIL: row %d sum(dx*xn)=%g, expected 0\n",i,dot);
            exit(1);
        }
    }

    printf("PASS\n");
}

void test_addnorm_idempotent(ADDNORM* l)
{
    printf("Test: addnorm backward idempotent\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float Y[B][D];
    float dY[B][D];
    float dX[B][D];
    float dg1[D],db1[D];
    float dg2[D],db2[D];

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-1.0f,1.0f);
            X2[i][j] = urand(-1.0f,1.0f);
            dY[i][j] = urand(-1.0f,1.0f);
        }
    }

    addnorm_forward(l,X1,X2,Y);
    addnorm_backward(l,dY,dX,dg1,db1);
    addnorm_backward(l,dY,dX,dg2,db2);   /* must clear, not accumulate */

    for (int j = 0; j < D; j++) {
        if (fabsf(dg1[j] - dg2[j]) > TOL || fabsf(db1[j] - db2[j]) > TOL) {
            printf("FAIL: dg/db not idempotent at %d\n",j);
            exit(1);
        }
    }

    printf("PASS\n");
}

void test_addnorm_constant(ADDNORM* l)
{
    printf("Test: addnorm constant input\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float Y[B][D];

    for (int j = 0; j < D; j++) {
        l->gamma[j] = 1.0f;
        l->beta[j]  = 0.25f;
    }
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++) {
            X1[i][j] = 1.5f;
            X2[i][j] = 1.64f;   /* constant sum per row */
        }

    addnorm_forward(l,X1,X2,Y);

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            if (!isfinite(Y[i][j])) {
                printf("FAIL: Y[%d][%d] not finite\n",i,j);
                exit(1);
            }
            if (fabsf(Y[i][j] - 0.25f) > TOL) {
                printf("FAIL: Y[%d][%d]=%g, expected beta 0.25\n",i,j,Y[i][j]);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

float addnorm_loss(ADDNORM* l, fArr2D x1, fArr2D x2, fArr2D A_)
{
    int B = l->B;
    int D = l->D;

    float Y[B][D];
    addnorm_forward(l,x1,x2,Y);

    typedef float (*ArrBD)[D];
    ArrBD A = (ArrBD) A_;

    float L = 0;
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            L += A[i][j] * Y[i][j];

    return L;
}

void test_addnorm_finite_diff(ADDNORM* l)
{
    printf("Test: addnorm finite-difference gradients\n");

    int B = l->B;
    int D = l->D;

    float X1[B][D];
    float X2[B][D];
    float Y[B][D];
    float A[B][D];
    float dY[B][D];
    float dX[B][D];
    float dg[D];
    float db[D];

    for (int j = 0; j < D; j++) {
        l->gamma[j] = urand(0.5f,1.5f);
        l->beta[j]  = urand(-0.5f,0.5f);
    }
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-1.0f,1.0f);
            X2[i][j] = urand(-1.0f,1.0f);
            A[i][j]  = urand(-1.0f,1.0f);   /* dL/dY = A */
            dY[i][j] = A[i][j];
        }
    }

    addnorm_forward(l,X1,X2,Y);
    addnorm_backward(l,dY,dX,dg,db);

    /* dx is the gradient for BOTH x1 and x2; check against each */
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < D; j++) {
            float ana = dX[i][j];

            {   /* perturb x1 */
                float old = X1[i][j];
                X1[i][j] = old + EPS;
                float Lp = addnorm_loss(l,X1,X2,A);
                X1[i][j] = old - EPS;
                float Ln = addnorm_loss(l,X1,X2,A);
                X1[i][j] = old;

                float num = (Lp - Ln) / (2 * EPS);
                if (fabsf(num - ana) > TOL) {
                    printf("FAIL dX(x1)[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                    exit(1);
                }
            }
            {   /* perturb x2 */
                float old = X2[i][j];
                X2[i][j] = old + EPS;
                float Lp = addnorm_loss(l,X1,X2,A);
                X2[i][j] = old - EPS;
                float Ln = addnorm_loss(l,X1,X2,A);
                X2[i][j] = old;

                float num = (Lp - Ln) / (2 * EPS);
                if (fabsf(num - ana) > TOL) {
                    printf("FAIL dX(x2)[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                    exit(1);
                }
            }
        }
    }

    /* Check dgamma */
    for (int j = 0; j < D; j++) {
        float old = l->gamma[j];
        l->gamma[j] = old + EPS;
        float Lp = addnorm_loss(l,X1,X2,A);
        l->gamma[j] = old - EPS;
        float Ln = addnorm_loss(l,X1,X2,A);
        l->gamma[j] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = dg[j];
        if (fabsf(num - ana) > TOL) {
            printf("FAIL dgamma[%d]: expected=%g calculated=%g\n",j,num,ana);
            exit(1);
        }
    }

    /* Check dbeta */
    for (int j = 0; j < D; j++) {
        float old = l->beta[j];
        l->beta[j] = old + EPS;
        float Lp = addnorm_loss(l,X1,X2,A);
        l->beta[j] = old - EPS;
        float Ln = addnorm_loss(l,X1,X2,A);
        l->beta[j] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = db[j];
        if (fabsf(num - ana) > TOL) {
            printf("FAIL dbeta[%d]: expected=%g calculated=%g\n",j,num,ana);
            exit(1);
        }
    }

    printf("PASS\n");
}

/* addnorm(x1,x2) forward and backward must match lyrnorm(x1+x2). */
void test_addnorm_equivalence(void)
{
    printf("Test: addnorm(x1,x2) == lyrnorm(x1+x2)\n");

    const int B = 5;
    const int D = 8;

    float X1[B][D];
    float X2[B][D];
    float S[B][D];
    float dY[B][D];
    float YA[B][D];
    float YL[B][D];
    float dXA[B][D];
    float dXL[B][D];
    float dgA[D],dbA[D];
    float dgL[D],dbL[D];

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++) {
            X1[i][j] = urand(-1.0f,1.0f);
            X2[i][j] = urand(-1.0f,1.0f);
            S[i][j]  = X1[i][j] + X2[i][j];
            dY[i][j] = urand(-1.0f,1.0f);
        }

    ADDNORM* an = addnorm_create();
    addnorm_init(an,D,B);
    LYRNORM* ln = lyrnorm_create();
    lyrnorm_init(ln,D,B);

    /* Same non-trivial affine on both layers */
    for (int j = 0; j < D; j++) {
        float g = urand(0.5f,1.5f);
        float b = urand(-0.5f,0.5f);
        an->gamma[j] = g; an->beta[j] = b;
        ln->gamma[j] = g; ln->beta[j] = b;
    }

    addnorm_forward(an,X1,X2,YA);
    lyrnorm_forward(ln,S,YL);
    addnorm_backward(an,dY,dXA,dgA,dbA);
    lyrnorm_backward(ln,dY,dXL,dgL,dbL);

    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++) {
            if (fabsf(YA[i][j] - YL[i][j]) > TOL) {
                printf("FAIL: Y[%d][%d] addnorm=%g lyrnorm=%g\n",i,j,YA[i][j],YL[i][j]);
                exit(1);
            }
            if (fabsf(dXA[i][j] - dXL[i][j]) > TOL) {
                printf("FAIL: dX[%d][%d] addnorm=%g lyrnorm=%g\n",i,j,dXA[i][j],dXL[i][j]);
                exit(1);
            }
        }

    for (int j = 0; j < D; j++) {
        if (fabsf(dgA[j] - dgL[j]) > TOL) {
            printf("FAIL: dgamma[%d] addnorm=%g lyrnorm=%g\n",j,dgA[j],dgL[j]);
            exit(1);
        }
        if (fabsf(dbA[j] - dbL[j]) > TOL) {
            printf("FAIL: dbeta[%d] addnorm=%g lyrnorm=%g\n",j,dbA[j],dbL[j]);
            exit(1);
        }
    }

    addnorm_free(an);
    lyrnorm_free(ln);
    printf("PASS\n");
}

int main(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME,&ts);
    unsigned int seed = (unsigned int)(ts.tv_sec ^ ts.tv_nsec);

    printf("seed %u\n",seed);
    init_lrng(seed);

    const int B = 4;
    const int D = 8;   /* LayerNorm feature dimension */
    const int T = 5;   /* GroupNorm time steps        */
    const int C = 6;   /* GroupNorm channels          */
    const int G = 2;   /* GroupNorm groups            */
    printf("EPS %g TOL %g\n",EPS,TOL);
    printf("B %d D %d  |  T %d C %d G %d\n",B,D,T,C,G);

    LYRNORM* ln;
    GRPNORM* gn;

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_normalized(ln); lyrnorm_free(ln);

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_affine(ln); lyrnorm_free(ln);

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_invariance(ln); lyrnorm_free(ln);

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_grad_invariants(ln); lyrnorm_free(ln);

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_idempotent(ln); lyrnorm_free(ln);

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_constant(ln); lyrnorm_free(ln);

    ln = lyrnorm_create(); lyrnorm_init(ln,D,B);
    test_lyrnorm_finite_diff(ln); lyrnorm_free(ln);

    gn = grpnorm_create(G); grpnorm_init(gn,C,T,B);
    test_grpnorm_normalized(gn); grpnorm_free(gn);

    gn = grpnorm_create(C); grpnorm_init(gn,C,T,B);   /* G==C */
    test_grpnorm_per_channel(gn); grpnorm_free(gn);

    gn = grpnorm_create(G); grpnorm_init(gn,C,T,B);
    test_grpnorm_grad_invariants(gn); grpnorm_free(gn);

    gn = grpnorm_create(G); grpnorm_init(gn,C,T,B);
    test_grpnorm_constant(gn); grpnorm_free(gn);

    gn = grpnorm_create(2); grpnorm_init(gn,6,5,2);   /* general G */
    test_grpnorm_finite_diff(gn); grpnorm_free(gn);

    gn = grpnorm_create(4); grpnorm_init(gn,4,5,2);   /* G==C */
    test_grpnorm_finite_diff(gn); grpnorm_free(gn);

    gn = grpnorm_create(1); grpnorm_init(gn,4,4,3);   /* G==1 */
    test_grpnorm_finite_diff(gn); grpnorm_free(gn);

    test_equivalence();
    test_backward_equivalence();

    ADDNORM* an;

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_normalized(an); addnorm_free(an);

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_affine(an); addnorm_free(an);

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_sum_dependence(an); addnorm_free(an);

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_grad_invariants(an); addnorm_free(an);

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_idempotent(an); addnorm_free(an);

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_constant(an); addnorm_free(an);

    an = addnorm_create(); addnorm_init(an,D,B);
    test_addnorm_finite_diff(an); addnorm_free(an);

    test_addnorm_equivalence();

    printf("\nALL TESTS PASSED\n");
    return 0;
}
