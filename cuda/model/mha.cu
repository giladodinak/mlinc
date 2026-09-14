/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer functions */
/* References:
 * Attention Is All You Need, 2023. https://arxiv.org/pdf/1706.03762v7 
 * Roformer: Enhanced transformer with Rotary Position Embedding, 2023.
 * https://arxiv.org/pdf/2104.09864
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include "mem.h"
#include "cuflt.h"
#include "random.h"
#include "activation.h"
#include "dropout.h"
#include "rope.h"
#include "arraytypes.h"
#include "array.h"
#include "mha.h"

static __global__ void scale_scores(float* Scores, const int* pad_mask,
                                    int b, int T, int lookahead, float scale)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = T * T;
    if (k < n) {
        int i = k / T;
        int j = k % T;
        float v = Scores[k] * scale;
        /* Causal / lookahead masking (Sec. 3.2.3, Fig. 2 generalised).
         *   lookahead <  0 : skip - fully bidirectional attention
         *   lookahead == 0 : strictly causal (mask every future frame)
         *   lookahead == L : allow L future frames, mask beyond that
         * Position i attends to j only where j <= i + lookahead
         */
        if (lookahead >= 0 && j > i + lookahead) v = -1e9;
        if (pad_mask != NULL && !pad_mask[b * T + j]) v = -1e9;
        Scores[k] = v;
    }
}

static __global__ void mask_kernel(float* x, const float* mask, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= mask[i];
}

static __global__ void scale_kernel(float* x, int n, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

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
MHA* mha_create(int heads, int steps, int lookahead)
{
    MHA* l = (MHA*)allocmem(1,1,MHA);
    l->H = heads;
    l->T = steps;
    l->lookahead = lookahead;
    return l;
}

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
void mha_init(MHA* l, int input_dim, int batch_size, int training, float dropout_rate)
{
    if (input_dim % l->H != 0) {
        fflush(stdout);
        fprintf(stderr,"mha_init: input_dim %d not an integral multiple of heads %d\n",input_dim,l->H);
        freemem(l);
        exit(-1);
    }
    l->D = input_dim;
    l->Dh = input_dim / l->H;
    l->B = batch_size;
    l->BT = l->B * l->T;
    l->BHT = l->B * l->H * l->T;

    l->training = training;
    l->dropout_rate = dropout_rate;

    l->Wq = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wk = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wv = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wo = (fArr2D) cuallocmem(l->D,l->D,float);

    l->Q = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->K = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->V = (fArr2D) cuallocmem(l->BT,l->D,float);

    l->theta = (fVec) cuallocmem(1,l->Dh / 2,float);

    l->Qh = (fArr2D) cuallocmem(l->BHT,l->Dh,float);
    l->Kh = (fArr2D) cuallocmem(l->BHT,l->Dh,float);
    l->Vh = (fArr2D) cuallocmem(l->BHT,l->Dh,float);

    l->Att = (fArr2D) cuallocmem(l->BHT,l->T,float);
    l->AttMask = (fArr2D) cuallocmem(l->BHT,l->T,float);

    l->Scores = (fArr2D) cuallocmem(l->T,l->T,float);
    l->Oh = (fArr2D) cuallocmem(l->T,l->Dh,float);

    l->Out = (fArr2D) cuallocmem(l->BT,l->D,float);

    int D2 = l->D * l->D;
    float sd = sqrtf(1.0 / l->D);
    float* w = (float*) allocmem(D2,1,float);

    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    cufltcpy(l->Wq,w,D2);
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    cufltcpy(l->Wk,w,D2);
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    cufltcpy(l->Wv,w,D2);
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    cufltcpy(l->Wo,w,D2);
    freemem(w);

    rope_init(l->theta,l->Dh);

    if (!training)
        return;

    l->dOut = (fArr2D) cuallocmem(l->BT,l->D,float);

    l->dQ = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->dK = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->dV = (fArr2D) cuallocmem(l->BT,l->D,float);

    l->dQh = (fArr2D) cuallocmem(l->T,l->Dh,float);
    l->dKh = (fArr2D) cuallocmem(l->T,l->Dh,float);
    l->dVh = (fArr2D) cuallocmem(l->T,l->Dh,float);

    l->dOh = (fArr2D) cuallocmem(l->T,l->Dh,float);
    l->dAtt = (fArr2D) cuallocmem(l->T,l->T,float);
    l->dScores = (fArr2D) cuallocmem(l->T,l->T,float);

    l->gWq = (fArr2D) cuallocmem(l->D,l->D,float);
    l->gWk = (fArr2D) cuallocmem(l->D,l->D,float);
    l->gWv = (fArr2D) cuallocmem(l->D,l->D,float);
    l->gWo = (fArr2D) cuallocmem(l->D,l->D,float);
}

/* Releases all memory owned by an MHA layer.
 *
 * Frees the projection weights, RoPE table, forward scratch buffers, and
 * (when allocated) the backward and parameter-gradient buffers, then frees
 * the layer itself.
 *
 * Parameters:
 *   l - Pointer to the MHA layer to free. Must not be used afterwards.
 */
void mha_free(MHA* l)
{
    cufreemem(l->Wq);
    cufreemem(l->Wk);
    cufreemem(l->Wv);
    cufreemem(l->Wo);

    cufreemem(l->Q);
    cufreemem(l->K);
    cufreemem(l->V);

    cufreemem(l->theta);

    cufreemem(l->Qh);
    cufreemem(l->Kh);
    cufreemem(l->Vh);

    cufreemem(l->Att);
    cufreemem(l->AttMask);

    cufreemem(l->Scores);
    cufreemem(l->Oh);

    cufreemem(l->Out);

    cufreemem(l->dOut);

    cufreemem(l->dQ);
    cufreemem(l->dK);
    cufreemem(l->dV);

    cufreemem(l->dQh);
    cufreemem(l->dKh);
    cufreemem(l->dVh);

    cufreemem(l->dOh);
    cufreemem(l->dAtt);
    cufreemem(l->dScores);

    cufreemem(l->gWq);
    cufreemem(l->gWk);
    cufreemem(l->gWv);
    cufreemem(l->gWo);

    cufreemem(l->Kh_cache);
    cufreemem(l->Vh_cache);

    freemem(l);
}

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
 *   training  : Non-zero during training (applies attention dropout); 0
 *               during inference/generation (no dropout, deterministic).
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
void mha_forward(MHA* restrict l,
                 const fArr2D restrict X/*[BT][D]*/,
                 const iVec restrict pad_mask/*[BT]*/,
                 fArr2D Y/*[BT][D]*/,
                 int offset,
                 int training,
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

    float scale = 1.0 / sqrtf((float) Dh);
    int threads = 256;
    int blocks = (T * T + threads - 1) / threads;
    
    /* Step 1 - Linear projections (in Eq. 1, Sec. 3.2.2):
     * Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
     */
    matmul(Q,X,Wq,BT,D,D);
    matmul(K,X,Wk,BT,D,D);
    matmul(V,X,Wv,BT,D,D);
    cufltclr(Out,BT * D);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            int base = (b * H + h) * T; /* Row offset into [BHT][...] buffers */

            /* Step 2 - Split into heads (Sec. 3.2.2):
             * Qh = Q[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
             * Kh = K[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
             * Vh = V[b*T:(b+1)*T, h*Dh:(h+1)*Dh]
             */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                cufltcpy(&Qh[base + t][0],&Q[r][h * Dh],Dh);
                cufltcpy(&Kh[base + t][0],&K[r][h * Dh],Dh);
                cufltcpy(&Vh[base + t][0],&V[r][h * Dh],Dh);
            }

            rope_apply((fArr2D)&Qh[base],l->theta,0,offset,T,Dh);
            rope_apply((fArr2D)&Kh[base],l->theta,0,offset,T,Dh);

            /* Step 3 - Scaled dot-product attention (in Eq. 1, Sec. 3.2.1):
             * Scores = Qh @ Kh.T / sqrt(Dh)
             * Scores[i][j] = -1e9 for j > i + lookahead  (causal/lookahead mask)
             * Att = softmax(Scores)
             * Oh  = Att @ Vh
             */
            matmulT(Scores,&Qh[base],&Kh[base],T,Dh,T);

            scale_scores<<<blocks,threads>>>(
                   (float*) Scores,(const int *) pad_mask,b,T,lookahead,scale);

            /* Compute attention probabilities - Softmax - in Eq. 1 */
            softmax(Scores,T,T);

            /* Store this head's attention weights for backward */
            cufltcpy(&Att[base],Scores,T * T);

            if (training && l->training && l->dropout_rate > 0)
                dropout(&Att[base],&AttMask[base],T,T,l->dropout_rate);

            /* In Eq. 1: Attention @ V */
            matmul(Oh,&Att[base],&Vh[base],T,T,Dh);

            /* Step 4 - Concatenate heads and project (Eq. 2, Sec. 3.2.2):
             * Out = Concat(Oh_0, ..., Oh_{H-1})
             */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                cufltcpy(&Out[r][h * Dh],&Oh[t][0],Dh);
            }
        }
    }
    /* Step 4 continued - output projection (Eq. 2, Sec. 3.2.2):
     * Y = Out @ Wo
     */
    if (Y != NULL)
        matmul(Y,Out,Wo,BT,D,D);
}

/* mha_forward_step - single-token cached forward (B=1, inference only).
 *
 * Processes one token at absolute position 'offset': projects Q/K/V, RoPE-
 * rotates Q and K, appends the rotated K and V to per-head ring buffers
 * (slot = offset % T), and attends the query over the last min(offset+1, T)
 * cached keys, writing [1][D] to Y.
 *
 * Keys are cached already rotated, so attention over the ring is correct
 * without re-rotating (RoPE depends only on position differences). offset 0
 * starts a fresh sequence. Causal masking is implicit (cached keys are all at
 * positions <= offset).
 *
 * Parameters:
 *   l      - MHA layer
 *   x      - the new token's input row [1][D]
 *   Y      - output row [1][D]
 *   offset - absolute position of this token (0-based)
 *   lyr    - layer index (informational)
 */
void mha_forward_step(MHA* restrict l,
                      const fArr2D restrict x/*[1][D]*/,
                      fArr2D Y/*[1][D]*/,
                      int offset,
                      int lyr)
{
    (void) lyr;
    const int T = l->T;
    const int D = l->D;
    const int H = l->H;
    const int Dh = l->Dh;

    typedef float (*ArrDD)[D];
    typedef float (*Arr1D)[D];
    typedef float (*ArrHTDh)[Dh];

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;
    Arr1D Q = (Arr1D) l->Q;
    Arr1D K = (Arr1D) l->K;
    Arr1D V = (Arr1D) l->V;
    Arr1D Out = (Arr1D) l->Out;

    if (l->Kh_cache == NULL) {
        l->Kh_cache = (fArr2D) cuallocmem(H * T,Dh,float);
        l->Vh_cache = (fArr2D) cuallocmem(H * T,Dh,float);
    }

    ArrHTDh Kc = (ArrHTDh) l->Kh_cache;
    ArrHTDh Vc = (ArrHTDh) l->Vh_cache;
    ArrHTDh Qh = (ArrHTDh) l->Qh;
    ArrHTDh Kh = (ArrHTDh) l->Kh;

    /* Project the single new token */
    matmul(Q,x,Wq,1,D,D);
    matmul(K,x,Wk,1,D,D);
    matmul(V,x,Wv,1,D,D);

    int slot = offset % T; /* Ring position of this token */
    int n = offset + 1;    /* Valid keys so far           */
    if (n > T) n = T;      /* Buffer is full              */

    float scale = 1.0 / sqrtf((float)Dh);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int h = 0; h < H; h++) {
        int hbase = h * T;
        float* qh = (float*) &Qh[hbase];
        float* kh = (float*) &Kh[hbase];
        cufltcpy(qh,&Q[0][h * Dh],Dh);
        cufltcpy(kh,&K[0][h * Dh],Dh);

        rope_apply((fArr2D)qh,l->theta,0,offset,1,Dh);
        rope_apply((fArr2D)kh,l->theta,0,offset,1,Dh);

        cufltcpy(&Kc[hbase + slot][0],kh,Dh);
        cufltcpy(&Vc[hbase + slot][0],&V[0][h * Dh],Dh);

        matmulT(l->Scores,(fArr2D) qh,&Kc[hbase],1,Dh,n);
        scale_kernel<<<blocks,threads>>>((float*)l->Scores,n,scale);
        softmax(l->Scores,1,n);
        matmul(l->Oh,l->Scores,&Vc[hbase],1,n,Dh);

        cufltcpy(&Out[0][h * Dh],l->Oh,Dh);
    }

    if (Y != NULL)
        matmul(Y,Out,Wo,1,D,D);
}

void mha_backward(MHA* restrict l,
                  fArr2D restrict dY/*[BT][D]*/,
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

    int threads = 256;
    int blocks = (T * T + threads - 1) / threads;

    /* Step 4 backward - output projection (reverse of Y = Out @ Wo):
     * gWo  = Out.T @ dY
     * dOut = dY @ Wo.T
     */
    if (dY != NULL) {
        Tmatmul(gWo,Out,dY,D,BT,D);
        matmulT(dOut,dY,Wo,BT,D,D);
    }
    else {
        cufltclr(gWo,D * D);
        cufltclr(dOut,BT * D);
    }

    cufltclr(dQ,BT * D);
    cufltclr(dK,BT * D);
    cufltclr(dV,BT * D);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int base = (b * H + h) * T; /* row offset into [BHT][...] buffers */

            /* split dOut */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                cufltcpy(&dOh[t][0],&dOut[r][h * Dh],Dh);
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
                mask_kernel<<<blocks,threads>>>(
                             (float*)dAtt,(const float*)&AttMask[base],T * T);

            d_softmax(dScores,dAtt,&Att[base],T,T);

            float s = 1.0 / sqrtf((float)Dh);
            scale_kernel<<<blocks,threads>>>((float*)dScores,T * T,s);

            /* Step 3c backward - reverse Scores = Qh @ Kh.T:
             * dQh = dScores @ Kh
             * dKh = dScores.T @ Qh
             * then apply inverse RoPE to dQh and dKh
             */
            matmul(dQh,dScores,&Kh[base],T,T,Dh);
            Tmatmul(dKh,dScores,&Qh[base],T,T,Dh);

            rope_apply((fArr2D)dQh,l->theta,1,0,T,Dh);
            rope_apply((fArr2D)dKh,l->theta,1,0,T,Dh);

            /* Step 2 backward - accumulate head gradients into full tensors:
             * dQ[b*T+t][h*Dh+k] += dQh[t][k]
             * dK[b*T+t][h*Dh+k] += dKh[t][k]
             * dV[b*T+t][h*Dh+k] += dVh[t][k]
             */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                cufltcpy(&dQ[r][h * Dh],&dQh[t][0],Dh);
                cufltcpy(&dK[r][h * Dh],&dKh[t][0],Dh);
                cufltcpy(&dV[r][h * Dh],&dVh[t][0],Dh);
            }
        }
    }

    cufltclr(gWq,D * D);
    cufltclr(gWk,D * D);
    cufltclr(gWv,D * D);

    Tmatmul(gWq,X,dQ,D,BT,D);
    Tmatmul(gWk,X,dK,D,BT,D);
    Tmatmul(gWv,X,dV,D,BT,D);

    if (dX != NULL) {
        matmulT(dX,dQ,Wq,BT,D,D);
        addMatmulT(dX,dK,Wk,BT,D,D);
        addMatmulT(dX,dV,Wv,BT,D,D);
    }
}
