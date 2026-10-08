/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer data structures and functions */
#ifndef MHA_H
#define MHA_H
#include "float.h"
#include "mem.h"
#include "arraytypes.h"
#include "activation.h"
#include "dropout.h"
#include "rope.h"

typedef struct {
    /* dimensions */
    int B;      /* Batch size      */
    int T;      /* Sequence length */
    int D;      /* Model dimension */
    int H;      /* Number of heads */
    int Dh;     /* D / H     */
    int BT;     /* B * T     */
    int BHT;    /* B * H * T */

    int lookahead;      /* Causal masking, set at create time below  */
    int training;       /* 1 if training, 0 if inference             */
    float dropout_rate; /* Attention dropout rate                    */

    fArr2D Wq;  /* [D][D] */
    fArr2D Wk;  /* [D][D] */
    fArr2D Wv;  /* [D][D] */
    fArr2D Wo;  /* [D][D] */

    fArr2D Q;   /* [BT][D] */
    fArr2D K;   /* [BT][D] */
    fArr2D V;   /* [BT][D] */

    fVec theta; /* [Dh/2] RoPE frequency table */

    /* Attention weights stored by forward for backward/reference tests. */
    fArr2D Att;     /* [BHT][T] row (b*H+h)*T+t */

    fArr2D Out;     /* [BT][D] */

    iVec pad_mask;  /* [BT]    */

    /* backward buffers */
    fArr2D dOut;    /* [BT][D] */

    fArr2D dQ;      /* [BT][D] */
    fArr2D dK;      /* [BT][D] */
    fArr2D dV;      /* [BT][D] */

    /* parameter gradients */
    fArr2D gWq;     /* [D][D] */
    fArr2D gWk;     /* [D][D] */
    fArr2D gWv;     /* [D][D] */
    fArr2D gWo;     /* [D][D] */


    void* state; /* Implementation specific (NULL if not used) */
} MHA;

#ifdef __cplusplus
extern "C" { /* nvcc is a c++ compiler */
#endif

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
void mha_set_pad_mask(MHA* l, const iVec pad_mask);

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
 *   Y         : Output 2D array of shape [B*T][D];
                 if NULL, output projection is skipped.
 *   offset    : Position index of the first token in the sequence
 *   training  : Non-zero during training (applies attention dropout); 0
 *               during inference/generation (no dropout, deterministic).
 *   lyr       : Layer index.
 *
 * Notes: 
 *  1. Padding is not allowed at the beginning of a sequence;
 *     only trailing (right-aligned) padding is supported.
 *  2. Offset is used when training overlapping windows:
 *     each batch's start position advances by the window stride.
 *
 * Reference:
 *   - Vaswani et al., "Attention Is All You Need", 2017
 */
void mha_forward(MHA* restrict l,
                 const fArr2D restrict X/*[BT][D]*/,
                 fArr2D Y/*[BT][D]*/,
                 int offset,
                 int training,
                 int lyr);

/* mha_forward_step - single-token cached forward (B=1, inference only).
 *
 * Processes one token at absolute position 'offset'.
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
                      int lyr);

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
 */
void mha_backward(MHA* restrict l,
                  fArr2D restrict dY /*[BT][D]*/,
                  const fArr2D restrict X/*[BT][D]*/,
                  fArr2D dX,
                  int lyr);

#ifdef __cplusplus
}
#endif

#endif
