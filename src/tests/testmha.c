/* Copyright (c) 2026 Gilad Odinak */
/* Simple test to check mha layer correctness */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "mha.h"

#define EPS 1e-3
#define TOL 1e-2

void test_mha_zero_forward(MHA* m)
{
    printf("Test: MHA zero forward\n");

    int BT = m->BT;
    int D  = m->D;

    float X[BT][D];
    float Y[BT][D];

    fltclr(X, BT*D);
    fltclr(Y, BT*D);
    fltclr(m->Wq, D*D);
    fltclr(m->Wk, D*D);
    fltclr(m->Wv, D*D);
    fltclr(m->Wo, D*D);

    mha_forward(m, X, NULL, Y, 0, 0, 0);

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            if (fabsf(Y[i][j]) > 1e-9f) {
                printf("Y[%d][%d] should be zero but is %g\n",i,j,Y[i][j]);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}

void test_mha_attention_active(MHA* m)
{
    printf("Test: MHA attention is active\n");

    int B = m->B;
    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];

    typedef float (*ArrDD)[D];
    ArrDD Wq = (ArrDD) m->Wq;
    ArrDD Wk = (ArrDD) m->Wk;
    ArrDD Wv = (ArrDD) m->Wv;
    ArrDD Wo = (ArrDD) m->Wo;

    /* Set Wq, Wk, Wv, Wo to identity */
    fltclr(m->Wq, D * D);
    fltclr(m->Wk, D * D);
    fltclr(m->Wv, D * D);
    fltclr(m->Wo, D * D);
    for (int i = 0; i < D; i++) {
        Wq[i][i] = 1.0f;
        Wk[i][i] = 1.0f;
        Wv[i][i] = 1.0f;
        Wo[i][i] = 1.0f;
    }

    /* Fill X with small random values, then set token 0 of each batch
     * to a large distinctive value so it dominates dot-product scores */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < D; j++)
                X[b * T + t][j] = urand(-0.01f, 0.01f);
        }
        /* Token 0: large uniform value so it attends strongly to itself */
        for (int j = 0; j < D; j++)
            X[b * T + 0][j] = 3.0f;
    }

    fltclr(Y, BT * D);
    mha_forward(m, X, NULL, Y, 0, 0, 0);

    /* With identity weights and token 0 dominant, token 0's output
     * should be close to its own input value (self-attention dominates)
     */
    for (int b = 0; b < B; b++) {
        int r = b * T; /* token 0 of batch b */
        for (int j = 0; j < D; j++) {
            float diff = fabsf(Y[r][j] - X[r][j]);
            if (diff > TOL) {
                printf("FAIL: Y[%d][%d]=%g should be close to X[%d][%d]=%g\n",
                       r, j, Y[r][j], r, j, X[r][j]);
                exit(1);
            }
        }
    }

    /* Also verify output is NOT zero (attention is actually doing something) */
    float norm = 0.0f;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            norm += Y[i][j] * Y[i][j];

    if (norm < 1e-6f) {
        printf("FAIL: output is effectively zero, attention not active\n");
        exit(1);
    }

    printf("  OK\n");
}

float mha_loss(MHA* m, fArr2D X)
{
    int BT = m->BT;
    int D  = m->D;

    float Y[BT][D];
    fltclr(Y, BT * D);

    mha_forward(m, X, NULL, Y, 0, 0, 0);

    float L = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            L += Y[i][j];

    return L;
}

void test_mha_finite_diff(MHA* m)
{
    printf("Test: MHA finite-difference gradients\n");

    int BT = m->BT;
    int D  = m->D;

    float X[BT][D];
    float Y[BT][D];
    float dX[BT][D];
    float dY[BT][D];

    typedef float (*ArrDD)[D];

    ArrDD Wq = (ArrDD) m->Wq;
    ArrDD Wk = (ArrDD) m->Wk;
    ArrDD Wv = (ArrDD) m->Wv;
    ArrDD Wo = (ArrDD) m->Wo;
    ArrDD gWq = (ArrDD) m->gWq;
    ArrDD gWk = (ArrDD) m->gWk;
    ArrDD gWv = (ArrDD) m->gWv;
    ArrDD gWo = (ArrDD) m->gWo;

    fltclr(Y, BT * D);
    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0, 1.0);
            dX[i][j] = 0;
            dY[i][j] = 1;   /* dL/dY = 1 */
        }
    }

    mha_forward(m, X, NULL, Y, 0, 0, 0);
    mha_backward(m, dY, X, dX, 0);

    /* Check dX */
    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            float old = X[i][j];
            X[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            X[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            X[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dX[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL dX[%d][%d]: expected=%g calculated=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    /* Check gWq */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wq[i][j];
            Wq[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            Wq[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            Wq[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWq[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWq[%d][%d]: expected=%g calculated=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    /* Check gWk */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wk[i][j];
            Wk[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            Wk[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            Wk[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWk[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWk[%d][%d]: expected=%g calculated=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    /* Check gWv */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wv[i][j];
            Wv[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            Wv[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            Wv[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWv[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWv[%d][%d]: expected=%g calculated=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    /* Check gWo */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wo[i][j];
            Wo[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            Wo[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            Wo[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWo[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWo[%d][%d]: expected=%g calculated=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}
float softmax_loss(fArr2D Scores, fArr2D dAtt_, int T)
{
    float Att[T][T];
    fltcpy(Att, Scores, T * T);
    softmax(Att, T, T);

    typedef float (*ArrTT)[T];
    const ArrTT dAtt = (ArrTT) dAtt_;

    float L = 0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
        L += Att[i][j] * dAtt[i][j];

    return L;
}

void test_d_softmax(void)
{
    printf("Test: d_softmax\n");

    const int T = 4;
    float Scores[T][T];
    float dAtt[T][T];
    float dScores[T][T];
    float Att[T][T];

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            Scores[i][j] = urand(-1, 1);
            dAtt[i][j]   = urand(-1, 1);
        }
    }

    memcpy(Att, Scores, sizeof(Scores));
    softmax((fArr2D)Att, T, T);

    memcpy(dScores, dAtt, sizeof(dAtt));
    d_softmax(dScores, dAtt, Att, T, T);

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {

            float old = Scores[i][j];
            Scores[i][j] = old + EPS;
            float Lp = softmax_loss(Scores, dAtt, T);
            Scores[i][j] = old - EPS;
            float Ln = softmax_loss(Scores, dAtt, T);
            Scores[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dScores[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL softmax dScores[%d][%d]: expected=%g calculated=%g\n",
                       i, j, num, ana);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}

void test_mask(MHA* m) {
    printf("Test: MHA mask\n");

    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];

    typedef float (*ArrDD)[D];
    typedef float (*ArrTT)[T];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-0.1,0.1);

    fltclr(m->Wq, D * D);
    fltclr(m->Wk, D * D);
    fltclr(m->Wv, D * D);
    fltclr(m->Wo, D * D);

    ArrDD Wq = (ArrDD) m->Wq;
    ArrDD Wk = (ArrDD) m->Wk;    
    for (int i = 0; i < D / 2; i++) {
        Wq[i][i] = 1;
        Wk[i][i] = 1;
    }

    mha_forward(m, X, NULL, Y, 0, 0, 0);

    ArrTT Scores = (ArrTT) m->Scores;

    float masked_sum = 0;
    float unmasked_sum = 0;

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            if (j > i)
                masked_sum += fabsf(Scores[i][j]);
            else
                unmasked_sum += fabsf(Scores[i][j]);
        }
    }

    if (masked_sum != 0)
        printf("FAIL mask: masked positions sum = %g, expected 0\n", masked_sum);
    if (unmasked_sum == 0)
        printf("FAIL mask: unmasked positions sum = 0, attention not active\n");
    if (masked_sum != 0 || unmasked_sum == 0)
        exit(1);
    printf("  OK\n");
}

void test_padding_mask(MHA* m) {
    printf("Test: MHA padding mask\n");

    int B = m->B;
    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] =  urand(-0.1, 0.1);

    // Create pad_mask array - length B*T
    // The last two tokens of the last batch are masked
    int pad_mask[BT];
    for (int i = 0; i < BT; i++)
        pad_mask[i] = 1;
    pad_mask[(B - 1) * T + (T - 2)] = 0;
    pad_mask[(B - 1) * T + (T - 1)] = 0;

    mha_forward(m, X, pad_mask, Y, 0, 0, 0);

    typedef float (*ArrTT)[T];
    ArrTT Scores = (ArrTT)m->Scores;

    float padded_sum = 0;
    float unpadded_sum = 0;

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            if (j >= T - 2)  /* last two tokens are padded */
                padded_sum += fabsf(Scores[i][j]);
            else
                unpadded_sum += fabsf(Scores[i][j]);
        }
    }

    if (padded_sum != 0)
        printf("FAIL pad mask: padded positions sum = %g, expected 0\n", padded_sum);
    if (unpadded_sum == 0)
        printf("FAIL pad mask: unpadded positions sum = 0, attention not active\n");
    if (padded_sum != 0 || unpadded_sum == 0)
        exit(1);
    printf("  OK\n");
}

void test_rope_relative_invariance(MHA* m)
{
    printf("Test: MHA RoPE relative position invariance\n");

    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y1[BT][D];
    float Y2[BT][D];

    typedef float (*ArrBHTT)[T];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f, 1.0f);

    /* Forward with offset=0 */
    fltclr(Y1, BT * D);
    mha_forward(m, X, NULL, Y1, 0, 0, 0);

    /* Save Att from offset=0 run */
    float Att1[m->BHT][T];
    fltcpy(Att1, m->Att, m->BHT * T);

    /* Forward with offset */
    int offset = (int) urand(0,1000000);
    fltclr(Y2, BT * D);
    mha_forward(m, X, NULL, Y2, offset, 0, 0);

    ArrBHTT Att2 = (ArrBHTT) m->Att;

    /* Check Att: relative position invariance means attention weights
     * must be identical regardless of absolute position offset */
    float att_diff = 0;
    for (int i = 0; i < m->BHT * T; i++)
        att_diff += fabsf(((float*)Att1)[i] - ((float*)Att2)[i]);

    if (att_diff > TOL) {
        printf("FAIL RoPE invariance: Att differs with offset, diff=%g\n", att_diff);
        exit(1);
    }

    /* Check Y: output must also be identical */
    float y_diff = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            y_diff += fabsf(Y1[i][j] - Y2[i][j]);

    if (y_diff > TOL) {
        printf("FAIL RoPE invariance: Y differs with offset, diff=%g\n", y_diff);
        exit(1);
    }

    printf("  OK\n");
}

int main(void)
{
//  unsigned int seed = 42;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    unsigned int seed =
        (unsigned int)(ts.tv_sec ^ ts.tv_nsec);

    printf("seed %d\n",seed);
    init_lrng(seed);


    const int batch_size = 2;
    const int seq_len = 4;
    const int input_dim = 8;
    const int num_heads = 2;
    printf("EPS %g TOL %g\n",EPS,TOL);
    printf("batch_size %d seq_len %d\n",batch_size,seq_len);
    printf("input_dim %d num_heads %d\n",input_dim,num_heads);

    MHA* m;
    
    test_d_softmax();

    m = mha_create(num_heads,seq_len,/*lookahead=*/-1);
    mha_init(m,input_dim,batch_size,1,0);
    test_mha_zero_forward(m);
    mha_free(m);

    m = mha_create(num_heads,seq_len,/*lookahead=*/-1);
    mha_init(m,input_dim,batch_size,1,0);
    test_mha_attention_active(m);
    mha_free(m);

    m = mha_create(num_heads,seq_len,/*lookahead=*/-1);
    mha_init(m,input_dim,batch_size,1,0);
    test_mha_finite_diff(m);
    mha_free(m);

    m = mha_create(num_heads,seq_len,/*lookahead=*/0); /* causal for mask test */
    mha_init(m,input_dim,batch_size,1,0);
    test_mask(m);
    mha_free(m);

    m = mha_create(num_heads,seq_len,/*lookahead=*/-1);
    mha_init(m,input_dim,batch_size,1,0);
    test_padding_mask(m);
    mha_free(m);

    m = mha_create(num_heads,seq_len,/*lookahead=*/-1);
    mha_init(m,input_dim,batch_size,1,0);
    test_rope_relative_invariance(m);
    mha_free(m);

    printf("\nALL TESTS PASSED\n");
    return 0;
}

