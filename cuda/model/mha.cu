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

typedef struct {
    fArr2D Att;      /* [BHT][T]  */
    fArr2D Scores;   /* [T][T]    */
    fArr2D dAtt;     /* [T][T]    */
    fArr2D dScores;  /* [T][T]    */
    fVec dAttRowDot; /* [BHT]     */
    fArr2D Kh_cache; /* [H*T][Dh] */
    fArr2D Vh_cache; /* [H*T][Dh] */
    GRAPH* graph;
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
    l->dropout_rate = dropout_rate;

    MHA_STATE* s = (MHA_STATE*) allocmem(1,1,MHA_STATE);
    l->state = s;

    l->Wq = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wk = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wv = (fArr2D) cuallocmem(l->D,l->D,float);
    l->Wo = (fArr2D) cuallocmem(l->D,l->D,float);

    l->Q = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->K = (fArr2D) cuallocmem(l->BT,l->D,float);
    l->V = (fArr2D) cuallocmem(l->BT,l->D,float);

    l->theta = (fVec) cuallocmem(1,l->Dh / 2,float);

    s->Att = (fArr2D) cuallocmem(l->BHT,l->T,float);

    if (dropout_rate > 0) {
        l->dropout = (DROPOUT**) allocmem(l->B * l->H,1,DROPOUT*);
        for (int i = 0; i < l->B * l->H; i++) {
            l->dropout[i] = dropout_create(dropout_rate);
            dropout_init(l->dropout[i],l->T,l->T);
        }
    }

    s->Scores = (fArr2D) cuallocmem(l->T,l->T,float);

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

        s->dAtt = (fArr2D) cuallocmem(l->T,l->T,float);
        s->dScores = (fArr2D) cuallocmem(l->T,l->T,float);
        s->dAttRowDot = (fVec) cuallocmem(1,l->BHT,float);

        l->gWq = (fArr2D) cuallocmem(l->D,l->D,float);
        l->gWk = (fArr2D) cuallocmem(l->D,l->D,float);
        l->gWv = (fArr2D) cuallocmem(l->D,l->D,float);
        l->gWo = (fArr2D) cuallocmem(l->D,l->D,float);
    }
    s->graph = graph_create();
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
 *   - pad_mask content may be set before each invocation of mha_forward.
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
    graph_destroy(s->graph);
    cufreemem(l->Wq);
    cufreemem(l->Wk);
    cufreemem(l->Wv);
    cufreemem(l->Wo);

    cufreemem(l->Q);
    cufreemem(l->K);
    cufreemem(l->V);

    cufreemem(l->theta);

    cufreemem(s->Att);

    if (l->dropout != NULL) {
        for (int i = 0; i < l->B * l->H; i++)
            dropout_free(l->dropout[i]);
        freemem(l->dropout);
    }

    cufreemem(s->Scores);

    cufreemem(l->Out);

    /* backward buffers */
    cufreemem(l->dOut);

    cufreemem(l->dQ);
    cufreemem(l->dK);
    cufreemem(l->dV);

    cufreemem(s->dAtt);
    cufreemem(s->dScores);
    cufreemem(s->dAttRowDot);

    cufreemem(l->gWq);
    cufreemem(l->gWk);
    cufreemem(l->gWv);
    cufreemem(l->gWo);

    cufreemem(s->Kh_cache);
    cufreemem(s->Vh_cache);

    freemem(s);
    freemem(l);
}

/* Warp- and block-level reductions.  blockDim.x must be a multiple of 32
 * (at most 1024).  Every thread of the block receives the result.  The
 * shared buffer must hold 32 floats; it is safe to reuse after return.
 */
static __device__ __forceinline__ float warp_reduce_max(float v)
{
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));
    return v;
}

static __device__ __forceinline__ float block_reduce_max(float v, float* sh)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (lane < nwarps) ? sh[lane] : -INFINITY;
    v = warp_reduce_max(v);
    __syncthreads();
    return v;
}

static __device__ __forceinline__ float warp_reduce_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu,v,o);
    return v;
}

static __device__ __forceinline__ float block_reduce_sum(float v, float* sh)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (lane < nwarps) ? sh[lane] : 0.0f;
    v = warp_reduce_sum(v);
    __syncthreads();
    return v;
}

/* Fused score scaling, causal/padding masks, and row-wise softmax.
 *
 * One CUDA block processes one attention row; each thread holds NV row
 * elements in registers (element j = threadIdx.x + k*blockDim.x, so loads
 * and stores are coalesced).  The row is read from global memory once and
 * written once.  Masked elements are not loaded and are written as 0.
 *
 * If every key of the row is masked (only possible when the whole sequence
 * is padding), the row is set to the uniform distribution 1/T, matching the
 * previous -1e9 masking behavior.
 *
 * Uses only device-resident inputs, so the launch is CUDA Graph capturable.
 */
#define NUM_THREADS 1024
#define EPT 8 /* Elements Per Thread */
#define MAX_CONTEXT (EPT * NUM_THREADS)

static __global__ void scale_mask_softmax_kernel(float* Scores,
                                                 const int* pad_mask,
                                                 int b, int T,
                                                 int lookahead, float scale)
{
    __shared__ float sh[32];
    const int row = blockIdx.x;
    float* p = Scores + (size_t) row * T;
    const int* pm = (pad_mask != NULL) ? pad_mask + (size_t) b * T : NULL;

    int limit = T;
    if (lookahead >= 0 && row + lookahead + 1 < T)
        limit = row + lookahead + 1;

    float v[EPT];
    float vmax = -INFINITY;
#pragma unroll
    for (int k = 0; k < EPT; k++) {
        const int j = threadIdx.x + k * blockDim.x;
        float x = -INFINITY;
        if (j < limit && (pm == NULL || pm[j]))
            x = p[j] * scale;
        v[k] = x;
        vmax = fmaxf(vmax,x);
    }
    vmax = block_reduce_max(vmax,sh);

    if (vmax == -INFINITY) {              /* Fully masked row */
        const float u = 1.0f / (float) T;
#pragma unroll
        for (int k = 0; k < EPT; k++) {
            const int j = threadIdx.x + k * blockDim.x;
            if (j < T) p[j] = u;
        }
        return;
    }

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < EPT; k++) {
        const float e = (v[k] == -INFINITY) ? 0.0f : expf(v[k] - vmax);
        v[k] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum,sh);
    const float invsum = 1.0f / sum;

#pragma unroll
    for (int k = 0; k < EPT; k++) {
        const int j = threadIdx.x + k * blockDim.x;
        if (j < T) p[j] = v[k] * invsum;
    }
}

/* Launches scale_mask_softmax_kernel over the T rows of one [T][T]
 * attention matrix.  blockDim.x = ceil(T/EPT) rounded up to a multiple of 32.
 */
static void scale_mask_softmax(float* Scores, const int* pad_mask,
                               int b, int T, int lookahead, float scale)
{
    if (T > MAX_CONTEXT) {
        fflush(stdout);
        fprintf(stderr,"scale_mask_softmax: T %d exceeds %d\n",T,MAX_CONTEXT);
        exit(-1);
    }
    int threads = (T + EPT - 1) / EPT;
    threads = (threads + 31) / 32 * 32;
    const int blocks = T;
    scale_mask_softmax_kernel<<<blocks,threads,0,cuda_stream()>>>(
        Scores,pad_mask,b,T,lookahead,scale);
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
 *  2. Q, K, and V stay in their [B*T][D] storage.  A head is a
 *     strided view Q[b,t,h,:], K[b,t,h,:], V[b,t,h,:], where D=H*Dh.
 *     Splitting into heads is therefore a logical view, not a data copy.
 *     Att is stored per (b,h) for backward.
 *  3. Out is not cleared: every head's Att @ Vh GEMM (beta=0) overwrites
 *     its full [T][Dh] slice.
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
    typedef float (*ArrTT)[T];
    typedef float (*ArrBHTT)[T];

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;

    ArrBTD Q = (ArrBTD) l->Q;
    ArrBTD K = (ArrBTD) l->K;
    ArrBTD V = (ArrBTD) l->V;

    ArrTT AttTemp = (ArrTT) s->Scores;
    ArrBHTT Att = (ArrBHTT) s->Att;
    ArrBTD Out = (ArrBTD) l->Out;

    float scale = 1.0 / sqrtf((float) Dh);

    /* Step 1 - Linear projections (in Eq. 1, Sec. 3.2.2):
     * Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
     */
    matmul(Q,X,Wq,BT,D,D);
    matmul(K,X,Wk,BT,D,D);
    matmul(V,X,Wv,BT,D,D);

    rope_batch_apply_ld(Q,l->theta,0,offset,B,H,T,Dh,D);
    rope_batch_apply_ld(K,l->theta,0,offset,B,H,T,Dh,D);

    GRAPH* graph = s->graph;
    int gid = training ? GRAPH_FORWARD_TRAIN : GRAPH_FORWARD_INFER;
    if (graph_record_exists(graph,gid)) {
        graph_play(graph,gid);
    }
    else {
        graph_record(graph,gid,GRAPH_RECORD_ON);

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
                scale_mask_softmax((float*) Ah,pad_mask,b,T,lookahead,scale);

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

/* Multiplies all elemensts of x by scale */
static __global__ void scale_scores(float* x, int n, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
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
        s->Kh_cache = (fArr2D) cuallocmem(H * T,Dh,float);
        s->Vh_cache = (fArr2D) cuallocmem(H * T,Dh,float);
    }

    ArrHTDh Kc = (ArrHTDh) s->Kh_cache;
    ArrHTDh Vc = (ArrHTDh) s->Vh_cache;

    fArr2D AttTemp = s->Scores;

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
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        scale_scores<<<blocks,threads,0,cuda_stream()>>>(
                                            (float*)AttTemp,n,scale);
        softmax(AttTemp,1,n);
        matmul(Oh,AttTemp,&Vc[hbase],1,n,Dh);
    }

    if (Y != NULL)
        matmul(Y,Out,Wo,1,D,D);
}

/* Kernel for attn_rowdot().  One warp per (b,t,h) row, h varying fastest,
 * so consecutive warps read consecutive Dh-slices of dOut and Out.
 * blockDim.x must be a multiple of 32.
 */
static __global__ void attn_rowdot_kernel(float* __restrict__ RowDot,
                                          const float* __restrict__ dOut,
                                          const float* __restrict__ Out,
                                          int B, int T, int H, int Dh)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= B * T * H) return;  /* Past last row */
    const int h = warp % H;
    const int bt = warp / H;        /* Token row in [B*T][D]: b*T + t */
    const float* a = dOut + (size_t) bt * H * Dh + h * Dh;
    const float* o = Out  + (size_t) bt * H * Dh + h * Dh;
    float sum = 0.0f;
    for (int k = lane; k < Dh; k += 32)
        sum += a[k] * o[k];
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        const int b = bt / T;
        const int t = bt % T;
        RowDot[(b * H + h) * T + t] = sum;
    }
}

/* attn_rowdot - softmax-backward row sums for all B*H*T attention rows.
 *
 * Computes, for every batch item b, head h and token t:
 *
 *   RowDot[(b*H + h)*T + t] = dO[t] . O[t]
 *
 * where O[t] = Out[b*T+t][h*Dh:(h+1)*Dh] is head h's attention output for
 * token t, dO[t] is the same slice of dOut, and "." is the dot product
 * over Dh elements.
 *
 * Softmax backward needs, per attention row t, the sum
 *   sum_j P[t][j] * dP[t][j]
 * which would otherwise require reading two T-length rows.  It equals
 * dO[t] . O[t], with or without attention dropout:
 *
 *   P[t][j]   = softmax output (pre-dropout Att)
 *   M[t][j]   = dropout mask, including the 1/(1-p) scale (M = 1 if no dropout)
 *   P'[t][j]  = P[t][j] * M[t][j]          (post-dropout attention)
 *   O[t]      = sum_j P'[t][j] * V[j]
 *   dP'[t][j] = dO[t] . V[j]
 *   dP[t][j]  = dP'[t][j] * M[t][j]        (dropout backward)
 *
 *   sum_j P[t][j] * dP[t][j]
 *     = sum_j P'[t][j] * dP'[t][j]
 *     = sum_j P'[t][j] * (dO[t] . V[j])
 *     = dO[t] . (sum_j P'[t][j] * V[j])
 *     = dO[t] . O[t]
 *
 * so dScores[t][j] = scale * P[t][j] * (dP[t][j] - RowDot[t]).
 *
 * Out must hold the post-dropout attention output from mha_forward.
 *
 * References:
 *   - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
 *     with IO-Awareness", 2022. https://arxiv.org/abs/2205.14135
 *     (introduces D_i = rowsum(dO * O) in the backward pass)
 *   - Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
 *     Work Partitioning", 2023. https://arxiv.org/abs/2307.08691
 *     (computes D_i in a separate pass before the backward loop, as here)
 */
static void attn_rowdot(float* RowDot, const float* dOut, const float* Out,
                        int B, int T, int H, int Dh)
{
    const int threads = 256;
    const long long nthreads = (long long) B * T * H * 32;
    const int blocks = (int) ((nthreads + threads - 1) / threads);
    attn_rowdot_kernel<<<blocks,threads,0,cuda_stream()>>>(
        RowDot,dOut,Out,B,T,H,Dh);
}


/* Backward of the fused scale/mask/softmax, elementwise:
 *
 *     dScores[t][j] = scale * Att[t][j] * (dAtt[t][j] - dAttRowDot)
 *
 * dAttRowDot comes from attn_rowdot_kernel, so no row reduction is needed.
 * One CUDA block processes one row.  Masked elements (Att == 0) are not
 * loaded and are written as 0.  For a fully padded sequence (pad_mask of
 * its first token is 0) the forward pass produced uniform rows, so no
 * elements are skipped for that sequence.
 */
static __global__ void d_softmax_scaled_kernel(float* __restrict__ dScores,
                                               const float* __restrict__ dAtt,
                                               const float* __restrict__ Att,
                                               const float* __restrict__ Drow,
                                               const int* __restrict__ pad_mask,
                                               int b, int T,
                                               int lookahead, float scale)
{
    const int row = blockIdx.x;
    const size_t base = (size_t) row * T;

    const int* pm = (pad_mask != NULL) ? pad_mask + (size_t) b * T : NULL;
    const bool all_padding = (pm != NULL && pm[0] == 0);
    if (all_padding) pm = NULL;

    int limit = T;
    if (!all_padding && lookahead >= 0 && row + lookahead + 1 < T)
        limit = row + lookahead + 1;

    const float d = Drow[row];
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float g = 0.0f;
        if (j < limit && (pm == NULL || pm[j]))
            g = scale * Att[base + j] * (dAtt[base + j] - d);
        dScores[base + j] = g;
    }
}

/* mha_backward - backward pass of Multi-Head Attention (MHA) layer.
 *
 * Computes gradients of the loss with respect to weights and inputs,
 * reversing each step of mha_forward in order.
 *
 * Parameters:
 *   l      - MHA layer
 *   dY     - gradient of loss w.r.t. MHA output [BT][D]
 *   X      - original MHA input [BT][D]
 *   dX     - gradient of loss w.r.t. MHA input [BT][D], or NULL
 *   offset - Position index of the first token in the sequence
 *   lyr    - layer index (informational)
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
 *           dScores = Att * (dAtt - rowdot(dOh,Oh))   (Jacobian, Sec. 3.2.1)
 *           dScores /= sqrt(Dh)                       (reverse scaling)
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
 *
 * dQ, dK, dV, gWq, gWk and gWv are not cleared: the GEMMs that produce
 * them use beta=0 and overwrite their full outputs.
 */
void mha_backward(MHA* restrict l,
                  fArr2D restrict dY/*[BT][D]*/,
                  const fArr2D restrict X/*[BT][D]*/,
                  fArr2D dX,
                  int offset,
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
    const int lookahead = l->lookahead;
    MHA_STATE* s = (MHA_STATE*) l->state;

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

    ArrTT dAtt = (ArrTT) s->dAtt;
    ArrTT dScores = (ArrTT) s->dScores;
    float* dAttRowDot = (float*) s->dAttRowDot;

    ArrDD gWq = (ArrDD) l->gWq;
    ArrDD gWk = (ArrDD) l->gWk;
    ArrDD gWv = (ArrDD) l->gWv;
    ArrDD gWo = (ArrDD) l->gWo;

    ArrDD Wq = (ArrDD) l->Wq;
    ArrDD Wk = (ArrDD) l->Wk;
    ArrDD Wv = (ArrDD) l->Wv;
    ArrDD Wo = (ArrDD) l->Wo;

    ArrBHTT Att = (ArrBHTT) s->Att;

    ArrTT AttTemp = (ArrTT) s->Scores;

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
        cufltclr(gWo,D * D);
        cufltclr(dOut,BT * D);
    }

    /* Softmax-backward row sums for all B*H*T row */
    attn_rowdot(dAttRowDot,(const float*) dOut,(const float*) Out,B,T,H,Dh);

    GRAPH* graph = s->graph;

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
                 * dScores = Att * (dAtt - dAttRowDot) (Jacobian, Sec. 3.2.1)
                 * dScores /= sqrt(Dh)                 (reverse scaling)
                 */
                if (l->dropout != NULL && l->dropout[b * H + h]->rate > 0)
                    dropout_backward(l->dropout[b * H + h],dAtt,dAtt);

                const int* pad_mask = (const int*) l->pad_mask;
                const int threads = 256; /* Tunable: multiple of 32, maximum 1024 */
                const int blocks = T;    /* One block per row                     */
                d_softmax_scaled_kernel<<<blocks,threads,0,cuda_stream()>>>(
                      (float*)dScores,(const float*)dAtt,
                      (const float*)&Att[base],dAttRowDot + base,
                      pad_mask,b,T,lookahead,scale);

                /* Step 3c backward - reverse Scores = Qh @ Kh.T:
                 * dQh = dScores @ Kh
                 * dKh = dScores.T @ Qh
                 * then apply inverse RoPE to dQh and dKh
                 */
                matmul_ld(dQh,dScores,Kh,T,T,Dh,T,D,D);
                Tmatmul_ld(dKh,dScores,Qh,T,T,Dh,T,D,D);

                /* Step 2 backward - dQ/dK/dV were written directly into the
                 * appropriate head slices, so no gradient concatenation copies
                 * are required.
                 */
            }
        }
        graph_record(graph,GRAPH_BACKWARD,GRAPH_RECORD_OFF);
        graph_play(graph,GRAPH_BACKWARD);
    }

    rope_batch_apply_ld(dQ,l->theta,1,offset,B,H,T,Dh,D);
    rope_batch_apply_ld(dK,l->theta,1,offset,B,H,T,Dh,D);

    /* Step 1 backward - reverse Q=X@Wq, K=X@Wk, V=X@Wv:
     * gWq = X.T @ dQ,  gWk = X.T @ dK,  gWv = X.T @ dV
     * dX  = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T if dX != NULL
     */
    Tmatmul(gWq,X,dQ,D,BT,D);
    Tmatmul(gWk,X,dK,D,BT,D);
    Tmatmul(gWv,X,dV,D,BT,D);

    if (dX != NULL) {
        matmulT(dX,dQ,Wq,BT,D,D);
        addMatmulT(dX,dK,Wk,BT,D,D);
        addMatmulT(dX,dV,Wv,BT,D,D);
    }
}
