/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer functions */
/* References:
 * Attention Is All You Need, 2023. https://arxiv.org/pdf/1706.03762v7
 * Roformer: Enhanced transformer with Rotary Position Embedding, 2023.
 * https://arxiv.org/pdf/2104.09864
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuctx.h"
#include "mem.h"
#include "cuflt.h"
#include "random.h"
#include "activation.h"
#include "dropout.h"
#include "rope.h"
#include "arraytypes.h"
#include "array.h"
#include "mha.h"
#include "graph.h"

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
    MHA* l = (MHA*) allocmem(1,1,MHA);
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

    l->Wq = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wk = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wv = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wo = (fArr2D) cuallocmem(l->D,l->D,float);

    l->Q = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->K = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->V = (fArr2D) cuallocmem(l->BT,l->D,float);

    l->theta = (fVec) cuallocmem(1,l->Dh / 2,float);

    /* Not used: l->Qh l->Kh l->Vh */

    l->Att = (fArr2D) cuallocmem(l->BHT,l->T,float);

    if (dropout_rate > 0) {
        l->dropout = (DROPOUT**) allocmem(l->B * l->H,1,DROPOUT*);
        for (int i = 0; i < l->B * l->H; i++) {
            l->dropout[i] = dropout_create(dropout_rate);
            dropout_init(l->dropout[i],l->T,l->T);
        }
    }

    l->Scores = (fArr2D) cuallocmem(l->T,l->T,float);
    /* Not used: l->Oh */

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

    if (training) {
        l->dOut = (fArr2D) cuallocmem(l->BT,l->D,float);

        l->dQ = (fArr2D) cuallocmem(l->BT,l->D,float);
        l->dK = (fArr2D) cuallocmem(l->BT,l->D,float);
        l->dV = (fArr2D) cuallocmem(l->BT,l->D,float);

        /* Not used: l->dQh l->dKh l->dVh */

        /* Not used: l->dOh */
        l->dAtt = (fArr2D) cuallocmem(l->T,l->T,float);
        l->dScores = (fArr2D) cuallocmem(l->T,l->T,float);

        l->gWq = (fArr2D) cuallocmem(l->D,l->D,float);
        l->gWk = (fArr2D) cuallocmem(l->D,l->D,float);
        l->gWv = (fArr2D) cuallocmem(l->D,l->D,float);
        l->gWo = (fArr2D) cuallocmem(l->D,l->D,float);
    }
    l->state = graph_create();
}

/* Sets a padding mask
 *
 * Parameters:
 *   l        - Pointer to the MHA layer from mha_create().
 *   pad_mask - An integer array of length [B*T];
 *              entries are 1 for real tokens, 0 for padding.
 *
 * Notes:
 *   - This function may be called only once, after mha_init and before
 *     the first invocation of mha_forward.
 *   - pad_mask content may be set before calling mha_forward.
 */
void mha_set_pad_mask(MHA* l, const iVec pad_mask)
{
    l->pad_mask = pad_mask;
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
    graph_destroy((GRAPH*) l->state);
    cufreemem(l->Wq);
    cufreemem(l->Wk);
    cufreemem(l->Wv);
    cufreemem(l->Wo);

    cufreemem(l->Q);
    cufreemem(l->K);
    cufreemem(l->V);

    cufreemem(l->theta);

    cufreemem(l->Att);

    if (l->dropout != NULL) {
        for (int i = 0; i < l->B * l->H; i++)
            dropout_free(l->dropout[i]);
        freemem(l->dropout);
    }

    cufreemem(l->Scores);

    cufreemem(l->Out);

    /* backward buffers */
    cufreemem(l->dOut);

    cufreemem(l->dQ);
    cufreemem(l->dK);
    cufreemem(l->dV);

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

/* Multiplies all elemensts of x by scale */
static __global__ void scale_scores(float* x, int n, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

/* Multiplies all elemensts of x by scale, then apply masks */
static __global__ void scale_and_mask_scores(float* Scores,
                                             const int* pad_mask,
                                             int b, int T,
                                             int lookahead, float scale)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = T * T;
    if (k < n) {
        int i = k / T;
        int j = k % T;

        float v;

        /* Causal / lookahead masking (Sec. 3.2.3, Fig. 2 generalised):
         *   lookahead <  0 : skip - fully bidirectional attention
         *   lookahead == 0 : strictly causal (mask every future frame)
         *   lookahead == L : allow L future frames, mask beyond that
         * Position i attends to j only where j <= i + lookahead
         */
        if (lookahead >= 0 && j > i + lookahead)
            v = -1e9;
        else
        if (pad_mask != NULL && !pad_mask[b * T + j])
            v = -1e9;
        else
            v = Scores[k] * scale;

        Scores[k] = v;
    }
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
 *   Y         : Output 2D array of shape [B*T][D];
 *               if NULL, output projection is skipped.
 *   offset    : Must be 0, not used
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
 *     Att = Qh @ Kh.T / sqrt(Dh)
 *     mask(Att)
 *     softmax(Att)
 *     [optional] dropout(Att)
 *     Oh = Att @ Vh
 *
 *   Step 4 - Concatenate heads and project (Eq. 2, Sec. 3.2.2):
 *     Out = Concat(Oh_0, ..., Oh_{H-1})
 *     Y   = Out @ Wo if Y != NULL
 *
 * Notes:
 *  1. Padding is not allowed at the beginning of a sequence;
 *     only trailing (right-aligned) padding is supported.
 *  2. Q, K, and V stay in their natural [B*T][D] storage.  A head is a
 *     strided view Q[b,t,h,:], K[b,t,h,:], V[b,t,h,:], where D=H*Dh.
 *     Splitting into heads is therefore a logical view, not a data copy.
 *     Att is stored per (b,h) for backward.
 *
 * Reference:
 *   - Vaswani et al., "Attention Is All You Need", 2017
 */
void mha_forward(MHA* restrict l,
                 const fArr2D restrict X/*[BT][D]*/,
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

    if (offset != 0) {
        fprintf(stderr,"mha_forward: non zero offset not supported\n");
        exit(1);
    }

    typedef float (*ArrDD)[D];
    typedef float (*ArrBTD)[D];
    typedef float (*ArrTT)[T];
    typedef float (*ArrBHTT)[T];

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;

    ArrBTD Q = (ArrBTD) l->Q;
    ArrBTD K = (ArrBTD) l->K;
    ArrBTD V = (ArrBTD) l->V;

    ArrTT AttTemp = (ArrTT) l->Scores;
    ArrBHTT Att = (ArrBHTT) l->Att;
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

    GRAPH* graph = (GRAPH*) l->state;

    int gid = training ? GRAPH_FORWARD_TRAIN : GRAPH_FORWARD_INFER;
    if (graph_record_exists(graph,gid)) {
        graph_play(graph,gid);
    }
    else {
        graph_record(graph,gid,GRAPH_RECORD_ON);

        /* Apply RoPE to the entire batch */
        rope_batch_apply_ld(Q,l->theta,0,0,B,H,T,Dh,D);
        rope_batch_apply_ld(K,l->theta,0,0,B,H,T,Dh,D);

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {

                int base = (b * H + h) * T; /* Row offset into [BHT][...] buffers */

                /* Step 2 - Split into heads (Sec. 3.2.2).
                 *
                 * Notice that Q/K/V are physically [B*T][D], which is also
                 * [B][T][H][Dh].  For this (b,h), Qh/Kh/Vh point at the first
                 * Dh-vector and successive token rows are D elements apart
                 */
                fArr2D Qh = (fArr2D) &Q[b * T][h * Dh];
                fArr2D Kh = (fArr2D) &K[b * T][h * Dh];
                fArr2D Vh = (fArr2D) &V[b * T][h * Dh];

                /* Step 3 - Scaled dot-product Attention (in Eq. 1, Sec. 3.2.1):
                 * Att  = Qh @ Kh.T
                 * Att /= sqrt(Dh)
                 * mask(Att)
                 * softmax(Att)
                 * [optional] dropout(Att)
                 * Oh = Att @ Vh
                 */
                fArr2D Ah = &Att[base];
                matmulT_ld(Ah,Qh,Kh,T,Dh,T,D,D,T);

                const int* pad_mask = (const int*) l->pad_mask;
                scale_and_mask_scores<<<blocks,threads,0,cuda_stream()>>>(
                                     (float*) Ah,pad_mask,b,T,lookahead,scale);

                /* Compute attention probabilities - Softmax - in Eq. 1 */
                softmax(Ah,T,T);

                if (training && l->training)
                    if (l->dropout != NULL && l->dropout[b * H + h]->rate > 0) {
                        cufltcpy(AttTemp,Ah,T * T);
                        dropout_forward(l->dropout[b * H + h],AttTemp);
                        Ah = AttTemp;
                    }

                /* In Eq. 1: Attention @ V */
                /* Write this head directly into its slice of Out.  The output
                 * is logically [T][Dh] with row stride D, so concatenation is
                 * again just a view: no Oh -> Out copies are required.
                 */
                fArr2D Oh = (fArr2D) &Out[b * T][h * Dh];
                matmul_ld(Oh,Ah,Vh,T,T,Dh,T,D,D);
            }
        }
        graph_record(graph,gid,GRAPH_RECORD_OFF);
        graph_play(graph,gid);
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
 *
 * Note that mha_forward_step work only with models trained with lookahead=0.
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
    typedef float (*Arr1Dh)[Dh];

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

    fArr2D AttTemp = l->Scores;

    /* Project the single new token */
    matmul(Q,x,Wq,1,D,D);
    matmul(K,x,Wk,1,D,D);
    matmul(V,x,Wv,1,D,D);

    int slot = offset % T; /* Ring position of this token */
    int n = offset + 1;    /* Valid keys so far           */
    if (n > T) n = T;      /* Buffer is full              */

    float scale = 1.0 / sqrtf((float) Dh);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int h = 0; h < H; h++) {

        int hbase = h * T; /* This head's cache offset */

        /* Q/K/V are [1][D] arrays for the current token.
         * Qh/Kh/Vh/Oh are [1][Dh] views for head h.
         */
        Arr1Dh Qh = (Arr1Dh) &Q[0][h * Dh];
        Arr1Dh Kh = (Arr1Dh) &K[0][h * Dh];
        Arr1Dh Vh = (Arr1Dh) &V[0][h * Dh];
        Arr1Dh Oh = (Arr1Dh) &Out[0][h * Dh];

        /* RoPE-rotate the new token's Q and K at absolute position offset */
        rope_apply(Qh,l->theta,0,offset,1,Dh);
        rope_apply(Kh,l->theta,0,offset,1,Dh);

        /* Store rotated K and raw V into the ring at this slot */
        cufltcpy(&Kc[hbase + slot][0],Kh,Dh);
        cufltcpy(&Vc[hbase + slot][0],Vh,Dh);

        /* Attention of the new token over the n valid cached keys */
        matmulT(AttTemp,Qh,&Kc[hbase],1,Dh,n);
        scale_scores<<<blocks,threads,0,cuda_stream()>>>(
                                            (float*)AttTemp,n,scale);
        softmax(AttTemp,1,n);
        matmul(Oh,AttTemp,&Vc[hbase],1,n,Dh);
    }

    if (Y != NULL)
        matmul(Y,Out,Wo,1,D,D);
}

/* mha_backward - backward pass of Multi-Head Attention (MHA) layer.
 *
 * Computes gradients of the loss with respect to weights and inputs,
 * reversing each step of mha_forward in order.
 *
 * Parameters:
 *   l   - MHA layer
 *   dY  - gradient of loss w.r.t. MHA output [BT][D]
 *   X   - original MHA input [BT][D]
 *   dX  - gradient of loss w.r.t. MHA input [BT][D], or NULL
 *   lyr - layer index (informational)
 *
 * Q, K, and V are read directly from their [B*T][D] storage as strided
 * per-head views.  Att is read back per (b,h) from mha_forward.
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
 *           dScores /= sqrt(Dh)                 (reverse scaling)
 *     (c) reverse Scores = Qh @ Kh.T:
 *           dQh = dScores @ Kh
 *           dKh = dScores.T @ Qh
 *
 *   Step 2 backward - head gradients are written directly into the
 *     corresponding slices of dQ, dK, and dV; no concatenation copy.
 *
 *   Step 1 backward - linear projections (reverse of Q=X@Wq, K=X@Wk, V=X@Wv):
 *     gWq = X.T @ dQ,  gWk = X.T @ dK,  gWv = X.T @ dV
 *     dX  = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T if dX != NULL
 */
void mha_backward(MHA* restrict l,
                  fArr2D restrict dY/*[BT][D]*/,
                  const fArr2D restrict X/*[BT][D]*/,
                  fArr2D dX,
                  int lyr)
{
    (void) lyr;
    if (!l->training) return;

    const int B = l->B;
    const int T = l->T;
    const int D = l->D;
    const int H = l->H;
    const int Dh = l->Dh;
    const int BT = l->BT;

    typedef float (*ArrBTD)[D];
    typedef float (*ArrTT)[T];
    typedef float (*ArrBHTT)[T];
    typedef float (*ArrDD)[D];

    ArrBTD dOut = (ArrBTD) l->dOut;

    ArrBTD dQ = (ArrBTD) l->dQ;
    ArrBTD dK = (ArrBTD) l->dK;
    ArrBTD dV = (ArrBTD) l->dV;

    ArrBTD Q = (ArrBTD) l->Q;
    ArrBTD K = (ArrBTD) l->K;
    ArrBTD V = (ArrBTD) l->V;

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

    ArrBHTT Att = (ArrBHTT) l->Att;

    ArrTT AttTemp = (ArrTT) l->Scores;

    ArrBTD Out = (ArrBTD) l->Out;

    float scale = 1.0 / sqrtf((float) Dh);

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

    GRAPH* graph = (GRAPH*) l->state;

    if (graph_record_exists(graph,GRAPH_BACKWARD)) {
        graph_play(graph,GRAPH_BACKWARD);
    }
    else {
        graph_record(graph,GRAPH_BACKWARD,GRAPH_RECORD_ON);

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {

                int base = (b * H + h) * T; /* row offset into [BHT][...] buffers */

                /* Per-head views.  Each row contains Dh contiguous values;
                 * Q/K/V (and the derivaties) are physically [B*T][D], which
                 * is also [B][T][H][Dh].  For this (b,h), Qh/Kh/Vh (and the
                 * derivaties) point at the first Dh-vector and successive
                 * rows are D elements apart.
                 */
                fArr2D Qh  = (fArr2D) &Q[b * T][h * Dh];
                fArr2D Kh  = (fArr2D) &K[b * T][h * Dh];
                fArr2D Vh  = (fArr2D) &V[b * T][h * Dh];
                fArr2D dOh = (fArr2D) &dOut[b * T][h * Dh];
                fArr2D dQh = (fArr2D) &dQ[b * T][h * Dh];
                fArr2D dKh = (fArr2D) &dK[b * T][h * Dh];
                fArr2D dVh = (fArr2D) &dV[b * T][h * Dh];

                /* Step 3a backward - reverse Oh = Att @ Vh:
                 * dVh  = Att.T @ dOh
                 * dAtt = dOh @ Vh.T
                 *
                 * Note that stored Att contains the pre-dropout values.
                 * With dropout, AttTemp reconstructs the post-dropout values
                 * needed for the dVh calculation.
                 */
                if (l->dropout != NULL && l->dropout[b * H + h]->rate > 0) {
                    /* Use AttTemp as scratch buffer */
                    dropout_backward(l->dropout[b * H + h],&Att[base],AttTemp);
                    Tmatmul_ld(dVh,AttTemp,dOh,T,T,Dh,T,D,D);
                }
                else
                    Tmatmul_ld(dVh,&Att[base],dOh,T,T,Dh,T,D,D);

                matmulT_ld(dAtt,dOh,Vh,T,Dh,T,D,D,T);

                /* Step 3b backward - reverse Att = softmax(Scores):
                 * dScores = J_softmax(Att).T @ dAtt   (Jacobian, Sec. 3.2.1)
                 * dScores /= sqrt(Dh)                  (reverse scaling)
                 */
                if (l->dropout != NULL && l->dropout[b * H + h]->rate > 0)
                    dropout_backward(l->dropout[b * H + h],dAtt,dAtt);

                d_softmax(dScores,dAtt,&Att[base],T,T);

                scale_scores<<<blocks,threads,0,cuda_stream()>>>(
                                             (float*)dScores,T * T,scale);

                /* Step 3c backward - reverse Scores = Qh @ Kh.T:
                 * dQh = dScores @ Kh
                 * dKh = dScores.T @ Qh
                 * then apply inverse RoPE to dQh and dKh
                 */
                matmul_ld(dQh,dScores,Kh,T,T,Dh,T,D,D);
                Tmatmul_ld(dKh,dScores,Qh,T,T,Dh,T,D,D);

                rope_apply_ld(dQh,l->theta,1,0,T,Dh,D);
                rope_apply_ld(dKh,l->theta,1,0,T,Dh,D);

                /* Step 2 backward - dQ/dK/dV were written directly into the
                 * appropriate head slices, so no gradient concatenation copies
                 * are required.
                 */
            }
        }
        graph_record(graph,GRAPH_BACKWARD,GRAPH_RECORD_OFF);
        graph_play(graph,GRAPH_BACKWARD);
    }

    /* Step 1 backward - reverse Q=X@Wq, K=X@Wk, V=X@Wv:
     * gWq = X.T @ dQ,  gWk = X.T @ dK,  gWv = X.T @ dV
     * dX  = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T if dX != NULL
     */
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
