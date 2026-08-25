/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer data structures and functions */
/* References:
 * Attention Is All You Need, 2023. https://arxiv.org/pdf/1706.03762v7 
 * Roformer: Enhanced transformer with Rotary Position Embedding, 2023.
 * https://arxiv.org/pdf/2104.09864
 */
#ifndef MHA_H
#define MHA_H
#include "float.h"
#include "array.h"
#include "activation.h"
#include "dropout.h"
#include "rope.h"

typedef struct {
    /* dimensions */
    int B;      /* batch size */
    int T;      /* sequence length */
    int D;      /* model dim */
    int H;      /* heads */
    int Dh;     /* D / H */
    int BT;     /* B * T */
    int BHT;    /* B * H * T */

    int lookahead;      /* causal masking, set at create time below  */
    int training;       /* 1 if training, 0 if inference             */
    float dropout_rate; /* fraction of attention weights to zero out */

    fArr2D Wq;  /* [D][D] */
    fArr2D Wk;  /* [D][D] */
    fArr2D Wv;  /* [D][D] */
    fArr2D Wo;  /* [D][D] */

    fArr2D Q;   /* [BT][D] */
    fArr2D K;   /* [BT][D] */
    fArr2D V;   /* [BT][D] */

    fVec theta; /* [Dh/2] RoPE frequency table */

    /* Per-(b,h,t) split heads and attention weights, stored for the
     * entire batch so backward can read back exactly what forward
     * computed for each (b,h) pair, rather than recomputing it. 
     */
    fArr2D Qh;      /* [BHT][Dh] row (b*H+h)*T+t */
    fArr2D Kh;      /* [BHT][Dh] row (b*H+h)*T+t */
    fArr2D Vh;      /* [BHT][Dh] row (b*H+h)*T+t */

    fArr2D Att;     /* [BHT][T] row (b*H+h)*T+t */
    fArr2D AttMask; /* [BHT][T] row (b*H+h)*T+t */

    fArr2D Scores;  /* [T][T]   scratch, not persisted */
    fArr2D Oh;      /* [T][Dh]  scratch, not persisted */

    fArr2D Out;     /* [BT][D] */

    /* backward buffers */
    fArr2D dOut;    /* [BT][D] */

    fArr2D dQ;      /* [BT][D] */
    fArr2D dK;      /* [BT][D] */
    fArr2D dV;      /* [BT][D] */

    fArr2D dQh;     /* [T][Dh] */
    fArr2D dKh;     /* [T][Dh] */
    fArr2D dVh;     /* [T][Dh] */

    fArr2D dOh;     /* [T][Dh] */
    fArr2D dAtt;    /* [T][T]  */
    fArr2D dScores; /* [T][T]  */

    /* parameter gradients */
    fArr2D gWq;     /* [D][D] */
    fArr2D gWk;     /* [D][D] */
    fArr2D gWv;     /* [D][D] */
    fArr2D gWo;     /* [D][D] */

} MHA;

/* Creates a Multi-Head Attention layer.
 *
 * Allocates the MHA container and records its structural parameters.
 * Weight matrices and scratch buffers are not allocated until mha_init().
 *
 * Parameters:
 *   heads     - Number of attention heads H. The model dimension passed
 *               to mha_init() must be an integer multiple of heads.
 *   steps     - Sequence length T (number of tokens/frames per sequence).
 *   lookahead - Causal-masking control, fixed for the life of the layer:
 *                 < 0  no masking, fully bidirectional (encoder)
 *                 = 0  strictly causal, no future context (decoder/streaming)
 *                 = L  causal with L frames of future context (bounded latency)
 *               Position i may attend to position j only when
 *               j <= i + lookahead. Independent of the padding mask.
 *
 * Returns:
 *   Pointer to a zero-initialised MHA layer. Call mha_init() before use.
 */
MHA* mha_create(int heads, int steps, int lookahead);

/* Initialises an MHA layer created by mha_create().
 *
 * Sets the model dimension, allocates and randomly initialises the Q/K/V/O
 * projection weights, builds the RoPE frequency table, and allocates the
 * forward scratch buffers. Backward buffers and parameter-gradient arrays
 * are allocated only when training is non-zero.
 *
 * Parameters:
 *   l            - Pointer to the MHA layer from mha_create().
 *   input_dim    - Model dimension D. Must be an integer multiple of the
 *                  head count H; the per-head dimension is Dh = D / H.
 *                  If not divisible, the function prints an error and exits.
 *   batch_size   - Number of sequences processed simultaneously, B.
 *   training     - Non-zero to allocate backward/gradient buffers (required
 *                  before mha_backward()); 0 for inference-only.
 *   dropout_rate - Fraction of attention weights to zero during training
 *                  (0 disables dropout).
 *
 * Notes:
 *   Projection weights are drawn from a normal distribution with standard
 *   deviation sqrt(1/D).
 */
void mha_init(MHA* l, int input_dim, int batch_size, int training, float dropout_rate);

/* Releases all memory owned by an MHA layer.
 *
 * Frees the projection weights, RoPE table, forward scratch buffers, and
 * (when allocated) the backward and parameter-gradient buffers, then frees
 * the layer itself.
 *
 * Parameters:
 *   l - Pointer to the MHA layer to free. Must not be used afterwards.
 */
void mha_free(MHA* l);

/* mha_forward - forward pass of Multi-Head Attention (MHA) layer
 *
 * This function computes the multi-head attention output for a batch
 * of sequences, including optional causal masking and padding masks.
 * It supports training-time backward buffers but does not modify them
 * in this function.
 *
 * Parameters:
 *   l         : Pointer to the MHA layer structure containing weights,
 *               buffers, and dimensions.
 *   X         : Input 2D array of shape [B*T][D], where:
 *               B = batch size, T = sequence length, D = model dimension.
 *   pad_mask  : Optional integer array of length [B*T];
 *               entries are 1 for real tokens, 0 for padding.
 *               If NULL, no padding mask is applied.
 *   Y         : Output 2D array of shape [B*T][D];
                 if NULL, output projection is skipped.
 *   offset    : Position index of the first token in the sequence
 *   lyr       : Layer index.
 *
 * Causal / lookahead masking is fixed per layer and taken from
 * l->lookahead (set in mha_create): < 0 bidirectional, 0 strictly causal,
 * L causal with L frames of future context. Position i may attend to
 * position j only when j <= i + l->lookahead. Independent of pad_mask.
 *
 * Computation (per head h, per batch item b):
 *
 *   Step 1 - Linear projections (in Eq. 1, Sec. 3.2.2):
 *     Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
 *
 *   Step 2 - Split into heads (Sec. 3.2.2):
 *     Qh = Q[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
 *     Kh = K[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
 *     Vh = V[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
 *
 *   Step 3 - Scaled dot-product attention (in Eq. 1, Sec. 3.2.1):
 *     Scores = Qh @ Kh.T / sqrt(Dh)
 *     Scores[i][j] = -1e9 for j > i + lookahead  (causal/lookahead mask)
 *     Att = softmax(Scores)           (row-wise)
 *     Oh  = Att @ Vh
 *
 *   Step 4 - Concatenate heads and project (Eq. 2, Sec. 3.2.2):
 *     Out = Concat(Oh_0, ..., Oh_{H-1})
 *     Y   = Out @ Wo if Y != NULL
 *
 * Notes: 
 *  1. Padding is not allowed at the beginning of a sequence;
 *     only trailing (right-aligned) padding is supported.
 *  2. Offset is used when training overlapping windows:
 *     each batch's start position advances by the window stride.
 *  3. Qh, Kh, Vh, and Att are stored per (b,h) pair for the whole
 *     batch, so that mha_backward can read back exactly what forward
 *     computed for each (b,h) without recomputing it.
 *
 * Reference:
 *   - Vaswani et al., "Attention Is All You Need", 2017
 */
static inline void mha_forward(MHA* restrict l,
                               const fArr2D restrict X/*[BT][D]*/,
                               const iVec restrict pad_mask/*[BT]*/,
                               fArr2D Y/*[BT][D]*/,
                               int offset,
                               int lyr)
{
    (void) lyr;
    const int B = l->B;
    const int T = l->T;
    const int D = l->D;
    const int H = l->H;
    const int Dh = l->Dh;
    const int BT = l->BT;
    const int lookahead = l->lookahead;

    typedef float (*ArrDD)[D];
    typedef float (*ArrBTD)[D];
    typedef float (*ArrBHTDh)[Dh];
    typedef float (*ArrTDh)[Dh];
    typedef float (*ArrTT)[T];
    typedef float (*ArrBHTT)[T];

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;

    ArrBTD Q = (ArrBTD) l->Q;
    ArrBTD K = (ArrBTD) l->K;
    ArrBTD V = (ArrBTD) l->V;

    ArrBHTDh Qh = (ArrBHTDh) l->Qh;
    ArrBHTDh Kh = (ArrBHTDh) l->Kh;
    ArrBHTDh Vh = (ArrBHTDh) l->Vh;

    ArrTT Scores = (ArrTT) l->Scores;
    ArrBHTT Att = (ArrBHTT) l->Att;
    ArrBHTT AttMask = (ArrBHTT) l->AttMask;
    ArrTDh Oh = (ArrTDh) l->Oh;

    ArrBTD Out = (ArrBTD) l->Out;

    /* Step 1 - Linear projections (in Eq. 1, Sec. 3.2.2):
     * Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
     */
    matmul(Q,X,Wq,BT,D,D);
    matmul(K,X,Wk,BT,D,D);
    matmul(V,X,Wv,BT,D,D);
    fltclr(Out,BT * D);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            int base = (b * H + h) * T; /* row offset into [BHT][...] buffers */

            /* Step 2 - Split into heads (Sec. 3.2.2):
             * Qh = Q[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
             * Kh = K[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
             * Vh = V[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
             */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                fltcpy(&Qh[base+t][0],&Q[r][h*Dh],Dh);
                fltcpy(&Kh[base+t][0],&K[r][h*Dh],Dh);
                fltcpy(&Vh[base+t][0],&V[r][h*Dh],Dh);
            }

            rope_apply(&Qh[base],l->theta,0,offset,T,Dh);
            rope_apply(&Kh[base],l->theta,0,offset,T,Dh);

            /* Step 3 - Scaled dot-product attention (in Eq. 1, Sec. 3.2.1):
             * Scores = Qh @ Kh.T / sqrt(Dh)
             * Scores[i][j] = -1e9 for j > i + lookahead  (causal/lookahead mask)
             * Att = softmax(Scores)
             * Oh  = Att @ Vh
             */
            matmulT(Scores,&Qh[base],&Kh[base],T,Dh,T);

            float s = 1.0f / sqrtf((float)Dh);
            for(int i = 0; i < T; i++)
                for(int j = 0; j < T; j++)
                    Scores[i][j] *= s;

            /* Causal / lookahead masking (Sec. 3.2.3, Fig. 2 generalised).
             *   lookahead <  0 : skip - fully bidirectional attention
             *   lookahead == 0 : strictly causal (mask every future frame)
             *   lookahead == L : allow L future frames, mask beyond that
             * Position i attends to j only where j <= i + lookahead. */
            if (lookahead >= 0) {
                for (int i = 0; i < T; i++)
                    for (int j = i + lookahead + 1; j < T; j++)
                        Scores[i][j] = -1e9f;
            }

            if (pad_mask != NULL) {
                for (int i = 0; i < T; i++)
                    if (!pad_mask[b * T + i])
                        for (int j = 0; j < T; j++)
                            Scores[j][i] = -1e9f;
            }

            /* Compute attention probabilities - Softmax - in Eq. 1 */
            softmax(Scores, T, T);

            /* Store this head's attention weights for backward */
            fltcpy(&Att[base], Scores, T * T);

            if (l->training && l->dropout_rate > 0)
                dropout(&Att[base],&AttMask[base],T,T,l->dropout_rate);

            /* in Eq. 1: Attention @ V */
            matmul(Oh,&Att[base],&Vh[base],T,T,Dh);

            /* Step 4 - Concatenate heads and project (Eq. 2, Sec. 3.2.2):
             * Out = Concat(Oh_0, ..., Oh_{H-1})
             */
            for (int t = 0;t < T; t++) {
                int r= b * T + t;
                fltcpy(&Out[r][h * Dh], &Oh[t][0], Dh);
            }
        }
    }
    /* Step 4 continued - output projection (Eq. 2, Sec. 3.2.2):
     * Y = Out @ Wo
     */
    if (Y != NULL)
        matmul(Y,Out,Wo,BT,D,D);
}

/*
 * mha_backward - backward pass of Multi-Head Attention (MHA) layer.
 *
 * Computes gradients of the loss with respect to weights and inputs,
 * reversing each step of mha_forward in order.
 *
 * Reads back the per-(b,h) Qh, Kh, Vh, and Att stored by mha_forward
 * for the same (b,h) pair, rather than recomputing them, so the values
 * used here are guaranteed identical to what forward actually produced.
 *
 * Gradient steps (reverse of forward):
 *
 *   Step 4 backward - output projection (reverse of Y = Out @ Wo):
 *     gWo  = Out.T @ dY
 *     dOut = dY @ Wo.T
 *
 *   Step 3 backward - scaled dot-product attention, per (b,h):
 *     (a) reverse Oh = Att @ Vh:
 *           dVh  = Att.T @ dOh
 *           dAtt = dOh @ Vh.T
 *     (b) reverse Att = softmax(Scores):
 *           dScores = J_softmax(Att).T @ dAtt   (Jacobian, Sec. 3.2.1)
 *           dScores /= sqrt(Dh)                  (reverse scaling)
 *     (c) reverse Scores = Qh @ Kh.T:
 *           dQh = dScores @ Kh
 *           dKh = dScores.T @ Qh
 *
 *   Step 2 backward - accumulate head gradients into full tensors:
 *     dQ[b*T+t][h*Dh+k] += dQh[t][k]
 *     dK[b*T+t][h*Dh+k] += dKh[t][k]
 *     dV[b*T+t][h*Dh+k] += dVh[t][k]
 *
 *   Step 1 backward - linear projections (reverse of Q=X@Wq, K=X@Wk, V=X@Wv):
 *     gWq = X.T @ dQ,  gWk = X.T @ dK,  gWv = X.T @ dV
 *     dX  = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T if dX != NULL
 */
static inline void mha_backward(MHA* restrict l,
                                fArr2D restrict dY /*[BT][D]*/,
                                const fArr2D restrict X/*[BT][D]*/,
                                fArr2D dX,
                                int lyr)
{
    (void) lyr;
    const int B = l->B;
    const int T = l->T;
    const int D = l->D;
    const int H = l->H;
    const int Dh = l->Dh;
    const int BT = l->BT;

    typedef float (*ArrBTD)[D];
    typedef float (*ArrBHTDh)[Dh];
    typedef float (*ArrTDh)[Dh];
    typedef float (*ArrTT)[T];
    typedef float (*ArrBHTT)[T];
    typedef float (*ArrDD)[D];

    ArrBTD dOut = (ArrBTD) l->dOut;

    ArrBTD dQ = (ArrBTD) l->dQ;
    ArrBTD dK = (ArrBTD) l->dK;
    ArrBTD dV = (ArrBTD) l->dV;

    ArrTDh dQh = (ArrTDh) l->dQh;
    ArrTDh dKh = (ArrTDh) l->dKh;
    ArrTDh dVh = (ArrTDh) l->dVh;

    ArrTDh dOh = (ArrTDh) l->dOh;
    ArrTT dAtt = (ArrTT) l->dAtt;
    ArrTT dScores = (ArrTT) l->dScores;

    ArrDD gWq = (ArrDD) l->gWq;
    ArrDD gWk = (ArrDD) l->gWk;
    ArrDD gWv = (ArrDD) l->gWv;
    ArrDD gWo = (ArrDD) l->gWo;

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;

    ArrBHTDh Qh = (ArrBHTDh) l->Qh;
    ArrBHTDh Kh = (ArrBHTDh) l->Kh;
    ArrBHTDh Vh = (ArrBHTDh) l->Vh;

    ArrBHTT Att = (ArrBHTT) l->Att;
    ArrBHTT AttMask = (ArrBHTT) l->AttMask;

    ArrBTD Out = (ArrBTD) l->Out;

    /* Step 4 backward - output projection (reverse of Y = Out @ Wo):
     * gWo  = Out.T @ dY
     * dOut = dY @ Wo.T
     */
    if (dY != NULL) {
        Tmatmul(gWo,Out,dY,D,BT,D);
        matmulT(dOut,dY,Wo,BT,D,D);
    }
    else {
        fltclr(gWo,D * D);
        fltclr(dOut,BT * D);
    }

    fltclr(dQ,BT * D);
    fltclr(dK,BT * D);
    fltclr(dV,BT * D);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            int base = (b * H + h) * T; /* row offset into [BHT][...] buffers */

            /* split dOut */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                fltcpy(&dOh[t][0],&dOut[r][h * Dh],Dh);
            }

            /* Step 3a backward - reverse Oh = Att @ Vh:
             * dVh  = Att.T @ dOh
             * dAtt = dOh @ Vh.T
             */
            Tmatmul(dVh,&Att[base],dOh,T,T,Dh);
            matmulT(dAtt,dOh,&Vh[base],T,Dh,T);

            /* Step 3b backward - reverse Att = softmax(Scores):
             * dScores = J_softmax(Att).T @ dAtt   (Jacobian, Sec. 3.2.1)
             * dScores /= sqrt(Dh)                  (reverse scaling)
             */
            if (l->training && l->dropout_rate > 0)
                for (int i = 0; i < T; i++)
                    for (int j = 0; j < T; j++)
                        dAtt[i][j] *= AttMask[base+i][j];

            d_softmax(dScores,dAtt,&Att[base],T,T);

            float s = 1.0f / sqrtf((float)Dh);
            for(int i = 0; i < T; i++)
              for(int j = 0; j < T; j++)
                dScores[i][j] *= s;

            /* Step 3c backward - reverse Scores = Qh @ Kh.T:
             * dQh = dScores @ Kh
             * dKh = dScores.T @ Qh
             * then apply inverse RoPE to dQh and dKh
             */
            matmul(dQh,dScores,&Kh[base],T,T,Dh);
            Tmatmul(dKh,dScores,&Qh[base],T,T,Dh);

            rope_apply(dQh,l->theta,1,0,T,Dh);
            rope_apply(dKh,l->theta,1,0,T,Dh);

            /* Step 2 backward - accumulate head gradients into full tensors:
             * dQ[b*T+t][h*Dh+k] += dQh[t][k]
             * dK[b*T+t][h*Dh+k] += dKh[t][k]
             * dV[b*T+t][h*Dh+k] += dVh[t][k]
             */
            for (int a = h * Dh, t = 0; t < T; t++) {
                int r=b*T+t;
                for (int k = 0; k < Dh; k++)
                    dQ[r][a + k] += dQh[t][k];
                for (int k = 0; k < Dh; k++)
                    dK[r][a + k] += dKh[t][k];
                for (int k = 0; k < Dh; k++)
                    dV[r][a + k] += dVh[t][k];
            }            
        }
    }

    /* Step 1 backward - reverse Q=X@Wq, K=X@Wk, V=X@Wv:
     * gWq = X.T @ dQ,  gWk = X.T @ dK,  gWv = X.T @ dV
     * dX  = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T if dX != NULL
     */
    fltclr(gWq,D * D);
    fltclr(gWk,D * D);
    fltclr(gWv,D * D);

    Tmatmul(gWq,X,dQ,D,BT,D);
    Tmatmul(gWk,X,dK,D,BT,D);
    Tmatmul(gWv,X,dV,D,BT,D);

    if (dX != NULL) {
        matmulT(dX,dQ,Wq,BT,D,D);
        addMatmulT(dX,dK,Wk,BT,D,D);
        addMatmulT(dX,dV,Wv,BT,D,D);
    }
}

#endif
