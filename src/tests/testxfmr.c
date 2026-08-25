/* Copyright (c) 2026 Gilad Odinak */
/* Tests for the decoder-only transformer layer */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "loss.h"
#include "transformer.h"
#include "dense.h"

int failures = 0;

/* Loss is L = sum(dY * Y) */
static float transformer_loss(TRANSFORMER* l,
                              float* X_flat,
                              const float* dY_flat,
                              int BT,int D)
{
    typedef float (*ArrBTD)[D];
    ArrBTD X = (ArrBTD) X_flat;

    float Y[BT][D];
    fltclr(Y,BT * D);
    transformer_forward(l,(fArr2D) X,NULL,1,(fArr2D) Y,0);

    float L = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            L += dY_flat[i * D + j] * Y[i][j];
    return L;
}

/* Compute numerical gradient for a single weight element */
static inline float numerical_grad(float* w,TRANSFORMER* l,
                                   float* X,
                                   const float* dY,
                                   int BT,int D)
{
    const int EPS = 1e-3; 
    float old = *w;
    *w = old + EPS;
    float Lp = transformer_loss(l,X,dY,BT,D);
    *w = old - EPS;
    float Ln = transformer_loss(l,X,dY,BT,D);
    *w = old;
    return (Lp - Ln) / (2 * EPS);
}

/* Check all elements of a weight matrix against their analytical gradients */
static int check_weight(fArr2D W,fArr2D gW,int rows,int cols,
                         const char* name,TRANSFORMER* l,float* X,
                         const float* dY,int BT,int D,float TOL)
{
    float* w  = (float*) W;
    float* gw = (float*) gW;
    for (int k = 0; k < rows * cols; k++) {
        float num = numerical_grad(&w[k],l,X,dY,BT,D);
        if (fabsf(num - gw[k]) > TOL) {
            printf("FAIL %s[%d][%d]: expected=%g calculated=%g\n",
                   name,k / cols,k % cols,num,gw[k]);
            failures++;
            return 1;
        }
    }
    return 0;
}

/* Test 1: zero forward
 * Zero input + zero weights -> zero output.
 */
void test_transformer_zero_forward(TRANSFORMER* l)
{
    printf("Test: transformer zero forward\n");

    const int BT = l->BT;
    const int D  = l->D;

    float X[BT][D];
    float Y[BT][D];

    fltclr(X,BT * D);
    fltclr(Y,BT * D);

    fltclr(l->mha->Wq,D * D);
    fltclr(l->mha->Wk,D * D);
    fltclr(l->mha->Wv,D * D);
    fltclr(l->mha->Wo,D * D);
    fltclr(l->ffn1->Wx,D * l->Dff);
    fltclr(l->ffn2->Wx,l->Dff * D);

    transformer_forward(l,(fArr2D) X,NULL,1,(fArr2D) Y,0);

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            if (fabsf(Y[i][j]) > 1e-6f) {
                printf("FAIL: Y[%d][%d] = %g,expected 0\n",i,j,Y[i][j]);
                failures++;
            }
        }
    }
    printf("PASS\n");
}

/* Test 2: finite difference gradient check
 * Validates dX and all weight gradients from transformer_backward.
 *
 * Loss is L = sum(dY * Y) so the numerical and analytical gradients are
 * checking the same objective (see transformer_loss above).
 *
 * Uses a shadow transformer instance (ls) that shares weights with l but
 * has separate scratch buffers. Numerical gradient evaluation runs forward
 * passes on ls so l's internal state (sdev,xhat,etc.) is not corrupted
 * between the backward call and the gradient checks.
 */
void test_transformer_finite_diff(TRANSFORMER* l)
{
    printf("Test: transformer finite-difference gradients\n");

    const int TOL = 0.05;
    const int BT  = l->BT;
    const int D   = l->D;
    const int Dff = l->Dff;
    const int B   = l->B;
    const int T   = l->T;
    const int H   = l->mha->H;

    float X[BT][D];
    float Y[BT][D];
    float dX[BT][D];
    float dY[BT][D];

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0f,1.0f);
            dY[i][j] = urand(-1.0f,1.0f);
            dX[i][j] = 0.0f;
        }
    }

    fltclr(Y,BT * D);
    transformer_forward(l,(fArr2D) X,NULL,1,(fArr2D) Y,0);
    transformer_backward(l,(fArr2D) dY,(fArr2D) X,(fArr2D) dX,0);

    /* Shadow instance for numerical gradient evaluation.
     * Shares weights with l via pointer — perturbations to l's weights
     * are visible here,but internal state is separate so forward passes
     * during finite difference don't corrupt l's buffers.
     */
    TRANSFORMER* ls = transformer_create(H,T,D,Dff,0);
    transformer_init(ls,B,0,0.0);

    /* Point ls weights to l's weights so perturbations are shared */
    freemem(ls->mha->Wq);      ls->mha->Wq      = l->mha->Wq;
    freemem(ls->mha->Wk);      ls->mha->Wk      = l->mha->Wk;
    freemem(ls->mha->Wv);      ls->mha->Wv      = l->mha->Wv;
    freemem(ls->mha->Wo);      ls->mha->Wo      = l->mha->Wo;
    freemem(ls->ffn1->Wx);     ls->ffn1->Wx     = l->ffn1->Wx;
    freemem(ls->ffn2->Wx);     ls->ffn2->Wx     = l->ffn2->Wx;
    freemem(ls->norm1->gamma); ls->norm1->gamma = l->norm1->gamma;
    freemem(ls->norm1->beta);  ls->norm1->beta  = l->norm1->beta;
    freemem(ls->norm2->gamma); ls->norm2->gamma = l->norm2->gamma;
    freemem(ls->norm2->beta);  ls->norm2->beta  = l->norm2->beta;

    float* x_flat  = (float*) X;
    float* dy_flat = (float*) dY;
    float* dx_flat = (float*) dX;

    int failed = 0;

    /* Check dX */
    for (int k = 0; k < BT * D; k++) {
        float num = numerical_grad(&x_flat[k],ls,x_flat,dy_flat,BT,D);
        if (fabsf(num - dx_flat[k]) > TOL) {
            printf("FAIL dX[%d][%d]: expected=%g calculated=%g\n",
                   k / D,k % D,num,dx_flat[k]);
            failures++;
            failed = 1;
            break;
        }
    }

    /* Check weight gradients */
    failed = failed || check_weight(l->mha->Wq, l->mha->gWq,D,  D,  "gWq", ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight(l->mha->Wk, l->mha->gWk,D,  D,  "gWk", ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight(l->mha->Wv, l->mha->gWv,D,  D,  "gWv", ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight(l->mha->Wo, l->mha->gWo,D,  D,  "gWo", ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight(l->ffn1->Wx,l->gWx1,    D,  Dff,"gWx1",ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight(l->ffn2->Wx,l->gWx2,    Dff,D,  "gWx2",ls,x_flat,dy_flat,BT,D,TOL);

    failed = failed || check_weight((fArr2D) l->norm1->gamma,(fArr2D) l->dg1,1,D,"dg1",ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight((fArr2D) l->norm1->beta, (fArr2D) l->db1,1,D,"db1",ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight((fArr2D) l->norm2->gamma,(fArr2D) l->dg2,1,D,"dg2",ls,x_flat,dy_flat,BT,D,TOL);
    failed = failed || check_weight((fArr2D) l->norm2->beta, (fArr2D) l->db2,1,D,"db2",ls,x_flat,dy_flat,BT,D,TOL);

    /* Null out shared pointers before freeing ls to avoid double-free */
    ls->mha->Wq = ls->mha->Wk = ls->mha->Wv = ls->mha->Wo = NULL;
    ls->ffn1->Wx = ls->ffn2->Wx = NULL;
    ls->norm1->gamma = ls->norm1->beta = NULL;
    ls->norm2->gamma = ls->norm2->beta = NULL;
    transformer_free(ls);

    printf("PASS\n");
}

/* Test 3: causal mask
 * Changing a future token must not affect past token outputs.
 */
void test_transformer_causal_mask(TRANSFORMER* l)
{
    printf("Test: transformer causal mask\n");

    const int BT = l->BT;
    const int T  = l->T;
    const int D  = l->D;

    float X[BT][D];
    float Y1[BT][D];
    float Y2[BT][D];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f,1.0f);

    fltclr(Y1,BT * D);
    transformer_forward(l,(fArr2D) X,NULL,1,(fArr2D) Y1,0);

    /* Perturb the last token of the first batch item */
    int last = T - 1;
    for (int j = 0; j < D; j++)
        X[last][j] += 1.0f;

    fltclr(Y2,BT * D);
    transformer_forward(l,(fArr2D) X,NULL,1,(fArr2D) Y2,0);

    /* Output at position 0 (past) must be unchanged */
    for (int j = 0; j < D; j++) {
        if (fabsf(Y1[0][j] - Y2[0][j]) > 1e-5f) {
            printf("FAIL causal mask: Y[0][%d] changed from %g to %g "
                   "when future token was perturbed\n",j,Y1[0][j],Y2[0][j]);
            failures++;
            return;
        }
    }

    /* Output at position last must have changed */
    float diff = 0;
    for (int j = 0; j < D; j++)
        diff += fabsf(Y1[last][j] - Y2[last][j]);
    if (diff < 1e-6f) {
        printf("FAIL causal mask: perturbing token %d had no effect on its own output\n",last);
        failures++;
        return;
    }

    printf("PASS\n");
}

/* Test 4: dropout
 * With training=1 and dropout_rate>0,two forward passes should differ.
 * With training=0 (inference),two passes should be identical.
 */
void test_transformer_dropout(TRANSFORMER* l_train,TRANSFORMER* l_infer)
{
    printf("Test: transformer dropout\n");

    const int BT = l_train->BT;
    const int D  = l_train->D;

    float X[BT][D];
    float Y1[BT][D];
    float Y2[BT][D];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f,1.0f);

    /* Training: two passes should differ */
    fltclr(Y1,BT * D);
    transformer_forward(l_train,(fArr2D) X,NULL,1,(fArr2D) Y1,0);
    fltclr(Y2,BT * D);
    transformer_forward(l_train,(fArr2D) X,NULL,1,(fArr2D) Y2,0);

    float diff = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            diff += fabsf(Y1[i][j] - Y2[i][j]);
    if (diff < 1e-6f) {
        printf("FAIL dropout: training passes produced identical outputs\n");
        failures++;
        return;
    }

    /* Inference: two passes must be identical */
    fltclr(Y1,BT * D);
    transformer_forward(l_infer,(fArr2D) X,NULL,1,(fArr2D) Y1,0);
    fltclr(Y2,BT * D);
    transformer_forward(l_infer,(fArr2D) X,NULL,1,(fArr2D) Y2,0);

    diff = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            diff += fabsf(Y1[i][j] - Y2[i][j]);
    if (diff > 0.0f) {
        printf("FAIL dropout: inference passes produced different outputs\n");
        failures++;
        return;
    }

    printf("PASS\n");
}

void smoke_test(void)
{
    const int batch_size = 8;
    const int seq_len    = 4;
    const int model_dim  = 12;
    const int ffn_dim    = 36;  /* 4 * model_dim */
    const int num_heads  = 3;

    TRANSFORMER* l;

    /* Test 1: zero forward */
    l = transformer_create(num_heads,seq_len,model_dim,ffn_dim,0);
    transformer_init(l,batch_size,1,0.0);
    test_transformer_zero_forward(l);
    transformer_free(l);

    /* Test 2: finite difference */
    l = transformer_create(num_heads,seq_len,model_dim,ffn_dim,0);
    transformer_init(l,batch_size,1,0.0);
    test_transformer_finite_diff(l);
    transformer_free(l);

    if (seq_len > 1) {
        /* Test 3: causal mask */
        l = transformer_create(num_heads,seq_len,model_dim,ffn_dim,0);
        transformer_init(l,batch_size,0,0.0);
        test_transformer_causal_mask(l);
        transformer_free(l);
    }

    /* Test 4: dropout */
    TRANSFORMER* l_train = transformer_create(num_heads,seq_len,model_dim,ffn_dim,0);
    transformer_init(l_train,batch_size,1,0.3);

    TRANSFORMER* l_infer = transformer_create(num_heads,seq_len,model_dim,ffn_dim,0);
    transformer_init(l_infer,batch_size,0,0.0);

    /* Copy weights from train to infer so they're comparable */
    memcpy(l_infer->mha->Wq,  l_train->mha->Wq,  model_dim * model_dim * sizeof(float));
    memcpy(l_infer->mha->Wk,  l_train->mha->Wk,  model_dim * model_dim * sizeof(float));
    memcpy(l_infer->mha->Wv,  l_train->mha->Wv,  model_dim * model_dim * sizeof(float));
    memcpy(l_infer->mha->Wo,  l_train->mha->Wo,  model_dim * model_dim * sizeof(float));
    memcpy(l_infer->ffn1->Wx, l_train->ffn1->Wx, model_dim * ffn_dim   * sizeof(float));
    memcpy(l_infer->ffn2->Wx, l_train->ffn2->Wx, ffn_dim   * model_dim * sizeof(float));
    memcpy(l_infer->norm1->gamma,l_train->norm1->gamma,model_dim * sizeof(float));
    memcpy(l_infer->norm1->beta, l_train->norm1->beta, model_dim * sizeof(float));
    memcpy(l_infer->norm2->gamma,l_train->norm2->gamma,model_dim * sizeof(float));
    memcpy(l_infer->norm2->beta, l_train->norm2->beta, model_dim * sizeof(float));

    test_transformer_dropout(l_train,l_infer);
    transformer_free(l_train);
    transformer_free(l_infer);
}

/* Linear weight update: W -= lr * gW */
static void update_vector_weights(fVec V,const fVec gV,int n,float lr)
{
    float* v = (float *) V;
    float* g = (float *) gV;
    for (int i = 0; i < n; i++)
        v[i] -= lr * g[i];
}

static void update_array_weights(fArr2D W,const fArr2D gW,int m,int n,float lr)
{
    float* w = (float *) W;
    float* g = (float *) gW;
    for (int i = 0; i < m * n; i++)
        w[i] -= lr * g[i];
}

/* Update all weights in one transformer layer */
static void transformer_update(TRANSFORMER* l,int D,int DFF,float lr)
{
    update_array_weights(l->mha->Wq,l->mha->gWq,D,D,lr);
    update_array_weights(l->mha->Wk,l->mha->gWk,D,D,lr);
    update_array_weights(l->mha->Wv,l->mha->gWv,D,D,lr);
    update_array_weights(l->mha->Wo,l->mha->gWo,D,D,lr);
    update_array_weights(l->ffn1->Wx,l->gWx1,D,DFF,lr);
    update_array_weights(l->ffn2->Wx,l->gWx2,DFF,D,lr);
    update_vector_weights(l->norm1->gamma,l->dg1,D,lr);
    update_vector_weights(l->norm1->beta,l->db1,D,lr);
    update_vector_weights(l->norm2->gamma,l->dg2,D,lr);
    update_vector_weights(l->norm2->beta,l->db2,D,lr);
}


/* Test 5: Training 
 * Trains a 3-layer transformer to predict the next token in a repeating
 * cyclic sequence of K=32 tokens: 0,1,2,...,31,0,1,2,...
 *
 * Token embeddings are random and fixed (not learned). The transformer
 * must learn to predict the next token purely from attention over context.
 *
 * Architecture per layer:
 *   D=64,H=4,Dff=256,T=32 (one full cycle as context)
 */
void training_test(void)
{
    printf("Test: training multi layer transformer\n");

    const int K = 32;     /* vocabulary size = cycle length    */
    const int T = 32;     /* sequence length (one full cycle)  */
    const int D = 64;     /* model dimension                   */
    const int DFF = 256;  /* FFN hidden dimension              */
    const int H = 4;      /* attention heads                   */
    const int NLYR = 3;   /* number of transformer layers      */
    const int B = 4;      /* batch size                        */
    const int BT = (B*T); /* flattened batch*seq dimension     */

    const int EPOCHS = 1000;
    const float LR = 3e-3f;

    /* Fixed random token embeddings [K][D] */
    float E[K][D];
    for (int k = 0; k < K; k++)
        for (int d = 0; d < D; d++)
            E[k][d] = urand(-0.1,0.1);

    /* Build B sequences,each of length T. The first half of the batch
     * cycles forward (+1),the second half backward (-1). The successor
     * of a token depends on the sequence direction,which is only
     * inferable from context,so attention is required to solve the task.
     * Input token at position t in batch b: (b + dir*t) % K
     * Target token (next token): (b + dir*(t + 1)) % K
     */
    int tok_in[B][T];
    int tok_tgt[B][T];
    for (int b = 0; b < B; b++) {
        int dir = (b < B/2) ? 1 : K - 1; /* +1 or -1 mod K */
        for (int t = 0; t < T; t++) {
            tok_in[b][t]  = (b + dir * t) % K;
            tok_tgt[b][t] = (b + dir * (t + 1)) % K;
        }
    }

    /* Input embeddings X[BT][D] */
    float X[BT][D];
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            fltcpy(X[b*T+t],E[tok_in[b][t]],D);

    /* One-hot targets yt[BT][K] for cross-entropy loss */
    float yt[BT][K];
    fltclr(yt,BT * K);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            yt[b * T + t][tok_tgt[b][t]] = 1.0;

    /* Create 3 transformer layers */
    TRANSFORMER* layers[NLYR];
    for (int i = 0; i < NLYR; i++) {
        layers[i] = transformer_create(H,T,D,DFF,0);
        transformer_init(layers[i],B,1,0.0);
    }

    /* Output projection to vocabulary logits,as a dense softmax layer.
     * Wx has shape [D][K]; forward does matmul + softmax in one call.
     */
    DENSE* out = dense_create(K,"Softmax");
    dense_init(out,D,BT);
    float gWout[D][K];

    /* Intermediate activations */
    float act[NLYR+1][BT][D]; /* act[0] = X,act[i] = output of layer i */
    float dy_out[BT][K];
    float dact[NLYR+1][BT][D];

    fltcpy(act[0],X,BT * D);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        /* Forward pass */

        for (int i = 0; i < NLYR; i++)
            transformer_forward(layers[i],act[i],NULL,1,act[i + 1],0);

        fArr2D yp = dense_forward(out,act[NLYR],0);

        float loss = cross_entropy_loss(yp,yt,BT,K);
        printf("epoch %5d  loss %8.4f\r",epoch + 1,loss / BT);
        fflush(stdout);

        /* Backward pass */

        /* Gradient of softmax + cross-entropy: dy = (yp - yt) / BT */
        dLdy_cross_entropy_loss(yp,yt,dy_out,BT,K);

        /* Projection gradients via dense layer (softmax backward is skipped;
         * dy_out is already yp - yt):
         *   gWout      = act[NLYR].T @ dy_out
         *   dact[NLYR] = dy_out @ Wx.T
         * gWout is cleared first because dense_backward accumulates. */
        fltclr(gWout,D * K);
        dense_backward(out,(fArr2D) dy_out,(fArr2D) act[NLYR],
                       (fArr2D) gWout,(fArr2D) dact[NLYR],0);

        /* Transformer layers backward */
        for (int i = NLYR - 1; i >= 0; i--)
            transformer_backward(layers[i],
                                 (fArr2D) dact[i+1],
                                 (fArr2D) act[i],
                                 (fArr2D) dact[i],
                                 0);

        /* Weight updates */
        update_array_weights(out->Wx,gWout,D,K,LR); /* dense update */
        for (int i = 0; i < NLYR; i++)
            transformer_update(layers[i],D,DFF,LR);
    }

    printf("\n");

    /* Evaluate: forward pass with final weights */
    transformer_forward(layers[0],X,NULL,1,act[1],0);
    for (int i = 1; i < NLYR; i++)
        transformer_forward(layers[i],act[i],NULL,1,act[i + 1],0);
    fArr2D yp = dense_forward(out,act[NLYR],0);

    float probs[BT][K];
    fltcpy(probs,yp,BT * K);

    int correct = 0;
    int counted = 0;
    /* Skip t == 0: with no context the direction is unknowable there */
    for (int b = 0; b < B; b++)
    for (int t = 1; t < T; t++) {
        int bt = b * T + t;
        int pred = 0;
        for (int k = 1; k < K; k++)
            if (probs[bt][k] > probs[bt][pred])
                pred = k;
        int tgt = 0;
        for (int k = 0; k < K; k++)
            if (yt[bt][k] > 0.5) { tgt = k; break; }
        if (pred == tgt) correct++;
        counted++;
    }
    float acc = 100.0 * correct / counted;
    printf("Accuracy: %d / %d (%.1f%%)            \n",correct,counted,acc);

    for (int i = 0; i < NLYR; i++)
        transformer_free(layers[i]);
    dense_free(out);

    if (acc < 99) {
        printf("FAIL: predicted tokens differ from expected tokens\n");
        failures++;
        return;
    }

    printf("PASS\n");
}




int main(void)
{
//  unsigned int seed = 42; // 1782848145 1550305486
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME,&ts);
    unsigned int seed =
        (unsigned int)(ts.tv_sec ^ ts.tv_nsec);

    printf("seed %d\n",seed);
    init_lrng(seed);

    failures = 0;
    smoke_test();
    training_test();

    if (failures == 0)
        printf("\nALL TESTS PASSED\n");
    return 0;
}
