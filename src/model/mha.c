/* Copyright (c) 2026 Gilad Odinak      */
/* Multi-Head Attention layer functions */
/* References:
 * Attention Is All You Need, 2023. https://arxiv.org/pdf/1706.03762v7 
 * Roformer: Enhanced transformer with Rotary Position Embedding, 2023.
 * https://arxiv.org/pdf/2104.09864
 */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "random.h"
#include "activation.h"
#include "dropout.h"
#include "rope.h"
#include "arraytypes.h"
#include "array.h"
#include "mha.h"

typedef struct {
    /* CPU implementation-specific forward buffers */
    fArr2D Qh;      /* [BHT][Dh] */
    fArr2D Kh;      /* [BHT][Dh] */
    fArr2D Vh;      /* [BHT][Dh] */
    DROPOUT** dropout; /* [B*H]  */
    fArr2D Scores;  /* [T][T]    */
    fArr2D Oh;      /* [T][Dh]   */

    /* CPU implementation-specific backward */
    fArr2D dQh;     /* [T][Dh] */
    fArr2D dKh;     /* [T][Dh] */
    fArr2D dVh;     /* [T][Dh] */
    fArr2D dOh;     /* [T][Dh] */
    fArr2D dAtt;    /* [T][T]  */
    fArr2D dScores; /* [T][T]  */

    /* KV cache for single-token generation */
    fArr2D Kh_cache; /* [H*T][Dh] */
    fArr2D Vh_cache; /* [H*T][Dh] */
} MHA_STATE;


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
    MHA* l = allocmem(1,1,MHA);
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
    
    MHA_STATE* s = allocmem(1,1,MHA_STATE);
    l->state = s;

    l->Wq = allocmem(l->D,l->D,float);
    l->Wk = allocmem(l->D,l->D,float);
    l->Wv = allocmem(l->D,l->D,float);
    l->Wo = allocmem(l->D,l->D,float);

    l->Q = allocmem(l->BT,l->D,float);
    l->K = allocmem(l->BT,l->D,float);
    l->V = allocmem(l->BT,l->D,float);

    l->theta = allocmem(1,l->Dh / 2,float);

    s->Qh = allocmem(l->BHT,l->Dh,float);
    s->Kh = allocmem(l->BHT,l->Dh,float);
    s->Vh = allocmem(l->BHT,l->Dh,float);

    l->Att = allocmem(l->BHT,l->T,float);

    if (dropout_rate > 0) {
        s->dropout = allocmem(l->B * l->H,1,DROPOUT*);
        for (int i = 0; i < l->B * l->H; i++) {
            s->dropout[i] = dropout_create(dropout_rate);
            dropout_init(s->dropout[i],l->T,l->T);
        }
    }

    s->Scores = allocmem(l->T,l->T,float);
    s->Oh = allocmem(l->T,l->Dh,float);
    
    l->Out = allocmem(l->BT,l->D,float);

    int D2 = l->D * l->D;
    float sd = sqrtf(1.0 / l->D);
    float *w;
    
    w = (float*) l->Wq;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    w = (float*) l->Wk;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    w = (float*) l->Wv;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    w = (float*) l->Wo;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);

    rope_init(l->theta,l->Dh);

    if (!training) return;

    l->dOut = allocmem(l->BT,l->D,float);

    l->dQ = allocmem(l->BT,l->D,float);
    l->dK = allocmem(l->BT,l->D,float);
    l->dV = allocmem(l->BT,l->D,float);

    s->dQh = allocmem(l->T,l->Dh,float);
    s->dKh = allocmem(l->T,l->Dh,float);
    s->dVh = allocmem(l->T,l->Dh,float);

    s->dOh = allocmem(l->T,l->Dh,float);
    s->dAtt = allocmem(l->T,l->T,float);
    s->dScores = allocmem(l->T,l->T,float);

    l->gWq = allocmem(l->D,l->D,float);
    l->gWk = allocmem(l->D,l->D,float);
    l->gWv = allocmem(l->D,l->D,float);
    l->gWo = allocmem(l->D,l->D,float);    
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
    MHA_STATE* s = (MHA_STATE*) l->state;
    freemem(l->Wq);
    freemem(l->Wk);
    freemem(l->Wv);
    freemem(l->Wo);

    freemem(l->Q);
    freemem(l->K);
    freemem(l->V);

    freemem(l->theta);

    freemem(s->Qh);
    freemem(s->Kh);
    freemem(s->Vh);

    freemem(l->Att);

    if (s->dropout != NULL) {
        for (int i = 0; i < l->B * l->H; i++)
            dropout_free(s->dropout[i]);
        freemem(s->dropout);
    }

    freemem(s->Scores);
    freemem(s->Oh);
    
    freemem(l->Out);

    /* backward buffers */
    freemem(l->dOut);

    freemem(l->dQ);
    freemem(l->dK);
    freemem(l->dV);

    freemem(s->dQh);
    freemem(s->dKh);
    freemem(s->dVh);

    freemem(s->dOh);
    freemem(s->dAtt);
    freemem(s->dScores);

    freemem(l->gWq);
    freemem(l->gWk);
    freemem(l->gWv);
    freemem(l->gWo);

    freemem(s->Kh_cache);
    freemem(s->Vh_cache);

    freemem(s);
    freemem(l);
}

/* Multiplies all elemensts of x by scale */
static inline void scale_scores(fArr2D restrict x_, int M, int T, float scale)
{
    typedef float (*ArrMT)[T];
    ArrMT x = (ArrMT) x_;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < T; j++)
            x[i][j] *= scale;
}

/* Multiplies all elemensts of x by scale, then apply masks */
static inline void scale_and_mask_scores(fArr2D Scores_/*[T][T]*/,
                                         const iVec pad_mask/*[BT]*/,
                                         int b, int T,
                                         int lookahead, float scale)
{
    typedef float (*ArrTT)[T];
    ArrTT Scores = (ArrTT) Scores_;

    scale_scores(Scores,T,T,scale);

    /* Causal / lookahead masking (Sec. 3.2.3, Fig. 2 generalised).
     *   lookahead <  0 : skip - fully bidirectional attention
     *   lookahead == 0 : strictly causal (mask every future frame)
     *   lookahead == L : allow L future frames, mask beyond that
     * Position i attends to j only where j <= i + lookahead
     */
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
    MHA_STATE* s = (MHA_STATE*) l->state;

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

    ArrBHTDh Qh = (ArrBHTDh) s->Qh;
    ArrBHTDh Kh = (ArrBHTDh) s->Kh;
    ArrBHTDh Vh = (ArrBHTDh) s->Vh;

    ArrTT Scores = (ArrTT) s->Scores;
    ArrBHTT Att = (ArrBHTT) l->Att;
    ArrTDh Oh = (ArrTDh) s->Oh;

    ArrBTD Out = (ArrBTD) l->Out;

    float scale = 1.0 / sqrtf((float) Dh);

    /* Step 1 - Linear projections (in Eq. 1, Sec. 3.2.2):
     * Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
     */
    matmul(Q,X,Wq,BT,D,D);
    matmul(K,X,Wk,BT,D,D);
    matmul(V,X,Wv,BT,D,D);
    fltclr(Out,BT * D);

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
                fltcpy(&Qh[base + t][0],&Q[r][h * Dh],Dh);
                fltcpy(&Kh[base + t][0],&K[r][h * Dh],Dh);
                fltcpy(&Vh[base + t][0],&V[r][h * Dh],Dh);
            }

            rope_apply(&Qh[base],l->theta,0,offset,T,Dh);
            rope_apply(&Kh[base],l->theta,0,offset,T,Dh);

            /* Step 3 - Scaled dot-product attention (in Eq. 1, Sec. 3.2.1):
             * Scores  = Qh @ Kh.T
             * Scores /= sqrt(Dh)
             * Scores[i][j] = -1e9 for j > i + lookahead (causal/lookahead mask)
             * Att = softmax(Scores)
             * Oh  = Att @ Vh
             */
            matmulT(Scores,&Qh[base],&Kh[base],T,Dh,T);

            const iVec pad_mask = (const iVec) l->pad_mask;
            scale_and_mask_scores(Scores,pad_mask,b,T,lookahead,scale);

            /* Compute attention probabilities - Softmax - in Eq. 1 */
            softmax(Scores,T,T);

            /* Store this head's attention weights (pre-dropout) for backward */
            fltcpy(&Att[base],Scores,T * T);

            if (training && l->training)
                if (s->dropout != NULL && s->dropout[b * H + h]->rate > 0)
                    dropout_forward(s->dropout[b * H + h],Scores);

            /* In Eq. 1: Attention @ V */
            matmul(Oh,Scores,&Vh[base],T,T,Dh);

            /* Step 4 - Concatenate heads and project (Eq. 2, Sec. 3.2.2):
             * Out = Concat(Oh_0, ..., Oh_{H-1})
             */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                fltcpy(&Out[r][h * Dh],&Oh[t][0],Dh);
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
    MHA_STATE* s = (MHA_STATE*) l->state;

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

    if (s->Kh_cache == NULL) {
        s->Kh_cache = allocmem(H * T,Dh,float);
        s->Vh_cache = allocmem(H * T,Dh,float);
    }

    ArrHTDh Kc = (ArrHTDh) s->Kh_cache;
    ArrHTDh Vc = (ArrHTDh) s->Vh_cache;

    /* Project the single new token */
    matmul(Q,x,Wq,1,D,D);
    matmul(K,x,Wk,1,D,D);
    matmul(V,x,Wv,1,D,D);

    int slot = offset % T; /* Ring position of this token */
    int n = offset + 1;    /* Valid keys so far           */
    if (n > T) n = T;      /* Buffer is full              */

    float scale = 1.0 / sqrtf((float) Dh);

    for (int h = 0; h < H; h++) {

        int hbase = h * T; /* This head's cache offset */

        Arr1Dh Qh = (Arr1Dh) &Q[0][h * Dh];
        Arr1Dh Kh = (Arr1Dh) &K[0][h * Dh];
        Arr1Dh Vh = (Arr1Dh) &V[0][h * Dh];
        Arr1Dh Oh = (Arr1Dh) &Out[0][h * Dh];

        /* RoPE-rotate the new token's Q and K at absolute position offset */
        rope_apply(Qh,l->theta,0,offset,1,Dh);
        rope_apply(Kh,l->theta,0,offset,1,Dh);

        /* Store rotated K and raw V into the ring at this slot */
        fltcpy(&Kc[hbase + slot][0],Kh,Dh);
        fltcpy(&Vc[hbase + slot][0],Vh,Dh);

        /* Attention of the new token over the n valid cached keys */
        matmulT(s->Scores,Qh,&Kc[hbase],1,Dh,n);
        scale_scores(s->Scores,1,n,scale);
        softmax(s->Scores,1,n);
        matmul(Oh,s->Scores,&Vc[hbase],1,n,Dh);
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
    MHA_STATE* s = (MHA_STATE*) l->state;

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

    ArrTDh dQh = (ArrTDh) s->dQh;
    ArrTDh dKh = (ArrTDh) s->dKh;
    ArrTDh dVh = (ArrTDh) s->dVh;

    ArrTDh dOh = (ArrTDh) s->dOh;
    ArrTT dAtt = (ArrTT) s->dAtt;
    ArrTT dScores = (ArrTT) s->dScores;

    ArrDD gWq = (ArrDD) l->gWq;
    ArrDD gWk = (ArrDD) l->gWk;
    ArrDD gWv = (ArrDD) l->gWv;
    ArrDD gWo = (ArrDD) l->gWo;

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;

    ArrBHTDh Qh = (ArrBHTDh) s->Qh;
    ArrBHTDh Kh = (ArrBHTDh) s->Kh;
    ArrBHTDh Vh = (ArrBHTDh) s->Vh;

    ArrBHTT Att = (ArrBHTT) l->Att;

    ArrBTD Out = (ArrBTD) l->Out;

    float scale = 1.0 / sqrtf((float) Dh);

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
             *
             * Note that the stored Att containts the pre-dropout values,
             * whereas dVh calculation uses the post-dropout values, which
             * can be resoncstructed by applying the dropout mask.
             */
            if (s->dropout != NULL && s->dropout[b * H + h]->rate > 0) {
                /* Use Scroes as scratch buffer */
                dropout_backward(s->dropout[b * H + h],&Att[base],s->Scores);
                Tmatmul(dVh,s->Scores,dOh,T,T,Dh);
            }
            else
                Tmatmul(dVh,&Att[base],dOh,T,T,Dh);

            matmulT(dAtt,dOh,&Vh[base],T,Dh,T);

            /* Step 3b backward - reverse Att = softmax(Scores):
             * dScores = J_softmax(Att).T @ dAtt   (Jacobian, Sec. 3.2.1)
             * dScores /= sqrt(Dh)                  (reverse scaling)
             */
            if (s->dropout != NULL && s->dropout[b * H + h]->rate > 0)
                dropout_backward(s->dropout[b * H + h],dAtt,dAtt);

            d_softmax(dScores,dAtt,&Att[base],T,T);

            scale_scores(dScores,T,T,scale);

            /* Step 3c backward - reverse Scores = Qh @ Kh.T:
             * dQh = dScores @ Kh
             * dKh = dScores.T @ Qh
             * then apply inverse RoPE to dQh and dKh
             */
            matmul(dQh,dScores,&Kh[base],T,T,Dh);
            Tmatmul(dKh,dScores,&Qh[base],T,T,Dh);

            rope_apply(dQh,l->theta,1,0,T,Dh);
            rope_apply(dKh,l->theta,1,0,T,Dh);

            /* Step 2 backward - concatenate head gradients
             * dQ dimentsions B x T x D dQh dimensions B x T x Dh
             */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                fltcpy(&dQ[r][h * Dh],&dQh[t][0],Dh);
                fltcpy(&dK[r][h * Dh],&dKh[t][0],Dh);
                fltcpy(&dV[r][h * Dh],&dVh[t][0],Dh);
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
