/* Copyright (c) 2026 Gilad Odinak */
/* Simple test to check mha layer correctness */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "float.h"
#include "mem.h"
#include "random.h"
#include "arraytypes.h"
#include "activation.h"
#include "mha.h"

#define EPS 1e-3
#define TOL 1e-2

void test_mha_zero_forward(MHA* m)
{
    printf("Test: MHA zero forward\n");

    int BT = m->BT;
    int D  = m->D;

    fArr2D nnX = (fArr2D) nnallocmem(BT,D,float);
    fArr2D nnY = (fArr2D) nnallocmem(BT,D,float);
    float Y[BT][D];

    nnfltclr(nnX,BT*D);
    nnfltclr(nnY,BT*D);
    nnfltclr(m->Wq,D*D);
    nnfltclr(m->Wk,D*D);
    nnfltclr(m->Wv,D*D);
    nnfltclr(m->Wo,D*D);

    mha_forward(m,nnX,NULL,nnY,0,0,0);
    nnfltcpy(Y,nnY,BT*D);
    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            if (fabsf(Y[i][j]) > 1e-9f) {
                printf("Y[%d][%d] should be zero but is %g\n",i,j,Y[i][j]);
                exit(1);
            }
        }
    }

    printf("  OK\n");
    nnfreemem(nnX);
    nnfreemem(nnY);
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
    float Wq[D][D];
    float Wk[D][D];
    float Wv[D][D];
    float Wo[D][D];

    /* Set Wq,Wk,Wv,Wo to identity */
    fltclr(Wq,D * D);
    fltclr(Wk,D * D);
    fltclr(Wv,D * D);
    fltclr(Wo,D * D);
    for (int i = 0; i < D; i++) {
        Wq[i][i] = 1.0f;
        Wk[i][i] = 1.0f;
        Wv[i][i] = 1.0f;
        Wo[i][i] = 1.0f;
    }
    nnfltcpy(m->Wq,Wq,D * D);
    nnfltcpy(m->Wk,Wk,D * D);
    nnfltcpy(m->Wv,Wv,D * D);
    nnfltcpy(m->Wo,Wo,D * D);

    /* Fill X with small random values,then set token 0 of each batch
     * to a large distinctive value so it dominates dot-product scores
     */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < D; j++)
                X[b * T + t][j] = urand(-0.01f,0.01f);
        }
        /* Token 0: large uniform value so it attends strongly to itself */
        for (int j = 0; j < D; j++)
            X[b * T + 0][j] = 3.0f;
    }

    fArr2D nnX = nnallocmem(BT,D,float);
    fArr2D nnY = nnallocmem(BT,D,float);
    nnfltcpy(nnX,X,BT * D);
    nnfltclr(nnY,BT * D);
    mha_forward(m,nnX,NULL,nnY,0,0,0);
    nnfltcpy(Y,nnY,BT * D);

    /* With identity weights and token 0 dominant,token 0's output
     * should be close to its own input value (self-attention dominates)
     */
    for (int b = 0; b < B; b++) {
        int r = b * T; /* token 0 of batch b */
        for (int j = 0; j < D; j++) {
            float diff = fabsf(Y[r][j] - X[r][j]);
            if (diff > TOL) {
                printf("FAIL: Y[%d][%d]=%g should be close to X[%d][%d]=%g\n",
                       r,j,Y[r][j],r,j,X[r][j]);
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
        printf("FAIL: output is effectively zero,attention not active\n");
        exit(1);
    }

    nnfreemem(nnX);
    nnfreemem(nnY);
    printf("  OK\n");
}

float mha_loss(MHA* m,fArr2D X)
{
    int BT = m->BT;
    int D  = m->D;

    fArr2D Ynn = nnallocmem(BT,D,float);
    float Y[BT][D];
    nnfltclr(Ynn,BT * D);

    mha_forward(m,X,NULL,Ynn,0,0,0);
    nnfltcpy(Y,Ynn,BT * D);
    nnfreemem(Ynn);

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
    float dX[BT][D];
    float dY[BT][D];
    float Wq[D][D];
    float Wk[D][D];
    float Wv[D][D];
    float Wo[D][D];
    float gWq[D][D];
    float gWk[D][D];
    float gWv[D][D];
    float gWo[D][D];

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0,1.0);
            dY[i][j] = 1;   /* dL/dY = 1 */
        }
    }

    fArr2D nnX  = nnallocmem(BT,D,float);
    fArr2D nnY  = nnallocmem(BT,D,float);
    fArr2D nndX = nnallocmem(BT,D,float);
    fArr2D nndY = nnallocmem(BT,D,float);
    nnfltcpy(nnX,X,BT * D);
    nnfltcpy(nndY,dY,BT * D);
    nnfltclr(nnY,BT * D);
    nnfltclr(nndX,BT * D);

    /* Keep host copies of weights for finite-difference perturbations. */
    nnfltcpy(Wq,m->Wq,D * D);
    nnfltcpy(Wk,m->Wk,D * D);
    nnfltcpy(Wv,m->Wv,D * D);
    nnfltcpy(Wo,m->Wo,D * D);

    mha_forward(m,nnX,NULL,nnY,0,0,0);
    mha_backward(m,nndY,nnX,nndX,0);

    nnfltcpy(dX,nndX,BT * D);
    nnfltcpy(gWq,m->gWq,D * D);
    nnfltcpy(gWk,m->gWk,D * D);
    nnfltcpy(gWv,m->gWv,D * D);
    nnfltcpy(gWo,m->gWo,D * D);

    /* Check dX */
    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            float old = X[i][j];
            X[i][j] = old + EPS;
            nnfltcpy(nnX,X,BT * D);
            float Lp = mha_loss(m,nnX);
            X[i][j] = old - EPS;
            nnfltcpy(nnX,X,BT * D);
            float Ln = mha_loss(m,nnX);
            X[i][j] = old;
            nnfltcpy(nnX,X,BT * D);

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dX[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL dX[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                exit(1);
            }
        }
    }

    /* Check gWq */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wq[i][j];
            Wq[i][j] = old + EPS;
            nnfltcpy(m->Wq,Wq,D * D);
            float Lp = mha_loss(m,nnX);
            Wq[i][j] = old - EPS;
            nnfltcpy(m->Wq,Wq,D * D);
            float Ln = mha_loss(m,nnX);
            Wq[i][j] = old;
            nnfltcpy(m->Wq,Wq,D * D);

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWq[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWq[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                exit(1);
            }
        }
    }

    /* Check gWk */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wk[i][j];
            Wk[i][j] = old + EPS;
            nnfltcpy(m->Wk,Wk,D * D);
            float Lp = mha_loss(m,nnX);
            Wk[i][j] = old - EPS;
            nnfltcpy(m->Wk,Wk,D * D);
            float Ln = mha_loss(m,nnX);
            Wk[i][j] = old;
            nnfltcpy(m->Wk,Wk,D * D);

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWk[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWk[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                exit(1);
            }
        }
    }

    /* Check gWv */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wv[i][j];
            Wv[i][j] = old + EPS;
            nnfltcpy(m->Wv,Wv,D * D);
            float Lp = mha_loss(m,nnX);
            Wv[i][j] = old - EPS;
            nnfltcpy(m->Wv,Wv,D * D);
            float Ln = mha_loss(m,nnX);
            Wv[i][j] = old;
            nnfltcpy(m->Wv,Wv,D * D);

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWv[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWv[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                exit(1);
            }
        }
    }

    /* Check gWo */
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float old = Wo[i][j];
            Wo[i][j] = old + EPS;
            nnfltcpy(m->Wo,Wo,D * D);
            float Lp = mha_loss(m,nnX);
            Wo[i][j] = old - EPS;
            nnfltcpy(m->Wo,Wo,D * D);
            float Ln = mha_loss(m,nnX);
            Wo[i][j] = old;
            nnfltcpy(m->Wo,Wo,D * D);

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWo[i][j];

            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWo[%d][%d]: expected=%g calculated=%g\n",i,j,num,ana);
                exit(1);
            }
        }
    }

    nnfreemem(nnX);
    nnfreemem(nnY);
    nnfreemem(nndX);
    nnfreemem(nndY);
    printf("  OK\n");
}

float softmax_loss(fArr2D Scores,fArr2D dAtt_,int T)
{
    float Att[T][T];
    fArr2D nnAtt = nnallocmem(T,T,float);
    nnfltcpy(nnAtt,Scores,T * T);
    softmax(nnAtt,T,T);
    nnfltcpy(Att,nnAtt,T * T);
    nnfreemem(nnAtt);

    typedef float (*ArrTT)[T];
    const ArrTT dAtt = (ArrTT) dAtt_;

    float L = 0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
            L += Att[i][j] * dAtt[i][j];

    return L;
}

void test_mask(MHA* m)
{
    printf("Test: MHA mask\n");

    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];
    float Wq[D][D];
    float Wk[D][D];
    float Scores[T][T];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-0.1,0.1);

    nnfltclr(m->Wq,D * D);
    nnfltclr(m->Wk,D * D);
    nnfltclr(m->Wv,D * D);
    nnfltclr(m->Wo,D * D);

    fltclr(Wq,D * D);
    fltclr(Wk,D * D);
    for (int i = 0; i < D / 2; i++) {
        Wq[i][i] = 1;
        Wk[i][i] = 1;
    }
    nnfltcpy(m->Wq,Wq,D * D);
    nnfltcpy(m->Wk,Wk,D * D);

    fArr2D nnX = nnallocmem(BT,D,float);
    fArr2D nnY = nnallocmem(BT,D,float);
    nnfltcpy(nnX,X,BT * D);
    mha_forward(m,nnX,NULL,nnY,0,0,0);
    nnfltcpy(Y,nnY,BT * D);
    nnfltcpy(Scores,m->Att,T * T);

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
        printf("FAIL mask: masked positions sum = %g,expected 0\n",masked_sum);
    if (unmasked_sum == 0)
        printf("FAIL mask: unmasked positions sum = 0,attention not active\n");
    if (masked_sum != 0 || unmasked_sum == 0)
        exit(1);

    nnfreemem(nnX);
    nnfreemem(nnY);
    printf("  OK\n");
}

void test_padding_mask(MHA* m)
{
    printf("Test: MHA padding mask\n");

    int B = m->B;
    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];
    float Scores[T][T];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] =  urand(-0.1,0.1);

    // Create pad_mask array - length B*T
    // The last two tokens of the last batch are masked
    int pad_mask[BT];
    for (int i = 0; i < BT; i++)
        pad_mask[i] = 1;
    pad_mask[(B - 1) * T + (T - 2)] = 0;
    pad_mask[(B - 1) * T + (T - 1)] = 0;

    fArr2D nnX = nnallocmem(BT,D,float);
    fArr2D nnY = nnallocmem(BT,D,float);
    iVec nn_pad_mask = (iVec) nnallocmem(BT,1,int);
    nnfltcpy(nnX,X,BT * D);
#ifdef USE_CUDA
    cudaMemcpy(nn_pad_mask,pad_mask,BT * sizeof(int),cudaMemcpyDefault);
#else
    memcpy(nn_pad_mask,pad_mask,BT * sizeof(int));
#endif

    mha_forward(m,nnX,nn_pad_mask,nnY,0,0,0);
    nnfltcpy(Y,nnY,BT * D);
    int b = B - 1;
    int h = m->H - 1;
    int base = (b * m->H + h) * T;
    typedef float (*ArrBHTT)[T];
    nnfltcpy(Scores,&((ArrBHTT)m->Att)[base],T * T);

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
        printf("FAIL pad mask: padded positions sum = %g,expected 0\n",padded_sum);
    if (unpadded_sum == 0)
        printf("FAIL pad mask: unpadded positions sum = 0,attention not active\n");
    if (padded_sum != 0 || unpadded_sum == 0)
        exit(1);

    nnfreemem(nnX);
    nnfreemem(nnY);
    nnfreemem(nn_pad_mask);
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

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f,1.0f);

    fArr2D nnX = nnallocmem(BT,D,float);
    fArr2D nnY1 = nnallocmem(BT,D,float);
    fArr2D nnY2 = nnallocmem(BT,D,float);
    nnfltcpy(nnX,X,BT * D);

    /* Forward with offset=0 */
    nnfltclr(nnY1,BT * D);
    mha_forward(m,nnX,NULL,nnY1,0,0,0);
    nnfltcpy(Y1,nnY1,BT * D);

    /* Save Att from offset=0 run */
    float Att1[m->BHT][T];
    nnfltcpy(Att1,m->Att,m->BHT * T);

    /* Forward with offset */
    int offset = (int) urand(0,300000);
    nnfltclr(nnY2,BT * D);
    mha_forward(m,nnX,NULL,nnY2,offset,0,0);
    nnfltcpy(Y2,nnY2,BT * D);

    float Att2[m->BHT][T];
    nnfltcpy(Att2,m->Att,m->BHT * T);

    /* Check Att: relative position invariance means attention weights
     * must be identical regardless of absolute position offset */
    float att_diff = 0;
    for (int i = 0; i < m->BHT * T; i++)
        att_diff += fabsf(((float*)Att1)[i] - ((float*)Att2)[i]);

    if (att_diff > TOL) {
        printf("FAIL RoPE invariance: Att differs with offset,diff=%g\n",att_diff);
        exit(1);
    }

    /* Check Y: output must also be identical */
    float y_diff = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            y_diff += fabsf(Y1[i][j] - Y2[i][j]);

    if (y_diff > TOL) {
        printf("FAIL RoPE invariance: Y differs with offset,diff=%g\n",y_diff);
        exit(1);
    }

    nnfreemem(nnX);
    nnfreemem(nnY1);
    nnfreemem(nnY2);
    printf("  OK\n");
}


void test_mha_forward_step(MHA* m)
{
    printf("Test: MHA forward step\n");

    int T = m->T;
    int D = m->D;

    float X[T][D];
    float Y[T][D];
    float Ys[T][D];

    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            X[t][d] = urand(-1,1);

    fArr2D nnX = nnallocmem(T,D,float);
    fArr2D nnY = nnallocmem(T,D,float);
    fArr2D nnYs = nnallocmem(T,D,float);
    nnfltcpy(nnX,X,T * D);
    nnfltclr(nnY,T * D);
    nnfltclr(nnYs,T * D);

    /* Full causal forward */
    mha_forward(m,nnX,NULL,nnY,0,0,0);

    /* The cached single-token forward must produce the same output row
     * at every position as the full causal forward above
     */
    for (int t = 0; t < T; t++)
        mha_forward_step(m,
                         (fArr2D)((float*)nnX + t * D),
                         (fArr2D)((float*)nnYs + t * D),t,0);

    nnfltcpy(Y,nnY,T * D);
    nnfltcpy(Ys,nnYs,T * D);

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            float diff = fabsf(Y[t][d] - Ys[t][d]);
            if (diff > TOL) {
                printf("FAIL forward step Y[%d][%d]: full=%g step=%g diff=%g\n",
                       t,d,Y[t][d],Ys[t][d],diff);
                exit(1);
            }
        }
    }

    nnfreemem(nnX);
    nnfreemem(nnY);
    nnfreemem(nnYs);
    printf("  OK\n");
}

int main(void)
{
//  unsigned int seed = 42;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME,&ts);
    unsigned int seed = (unsigned int)(ts.tv_sec ^ ts.tv_nsec);

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

    m = mha_create(num_heads,seq_len,/*lookahead=*/0);
    mha_init(m,input_dim,1,0,0);
    test_mha_forward_step(m);
    mha_free(m);

    printf("\nALL TESTS PASSED\n");
    return 0;
}
