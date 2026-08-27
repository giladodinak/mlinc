/* Copyright (c) 2026 Gilad Odinak */
/* Decoder-only Transformer layer data structures and functions            */
/* Reference: Attention Is All You Need https://arxiv.org/pdf/1706.03762v7 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include "float.h"
#include "array.h"
#include "dropout.h"
#include "mha.h"
#include "addnorm.h"
#include "dense.h"

/* Decoder-only transformer layer.
 *
 * Architecture (per token, per layer):
 *
 *   X ---> MaskedMHA --> dropout --> AddNorm1 ---> FFN --> dropout --> AddNorm2
 *      |                                |      |                          |
 *      +--------------------------------+      +--------------------------+
 *        (residual)                               (residual)
 *
 * The FFN is a two-layer feed-forward network:
 *   ffn1: [D -> Dff] with activation
 *   ffn2: [Dff -> D] with no activation
 *
 * This layer implements one decoder block.
 * A complete decoder-only Transformer stacks multiple such layers,
 * preceded by token and positional embeddings and followed by a linear
 * projection to vocabulary logits.
 *
 * Note: The original paper uses ReLU. This implementation uses GELU instead.
 */
typedef struct {
    int B;              /* Batch size                                    */
    int T;              /* Sequence length                               */
    int D;              /* Model dimension                               */
    int Dff;            /* FFN hidden dimension (typically 4*D)          */
    int BT;             /* B * T                                         */
    int training;       /* 1 if training, 0 if inference                 */
    float dropout_rate; /* Sub-layer output dropout rate                 */
    MHA* mha;           /* Masked self-attention [BT][D]  -> [BT][D]     */
    DENSE* ffn1;        /* FFN first  layer      [BT][D]  -> [BT][Dff]   */
    DENSE* ffn2;        /* FFN second layer      [BT][Dff]-> [BT][D]     */
    ADDNORM* norm1;     /* Add-Norm after MHA                            */
    ADDNORM* norm2;     /* Add-Norm after FFN                            */
    fArr2D mha_out;     /* MHA projection output      [BT][D]            */
    fArr2D norm1_out;   /* Output of norm1            [BT][D]            */
    fArr2D drop_mask1;  /* Dropout mask after MHA     [BT][D]            */
    fArr2D drop_mask2;  /* Dropout mask after FFN     [BT][D]            */
    fArr2D d_norm2_in;  /* Grad w.r.t. ffn2 output / norm2 input [BT][D] */
    fArr2D d_ffn1_in;   /* Grad w.r.t. ffn1 input                [BT][D] */
    fArr2D d_norm1_in;  /* Grad w.r.t. mha output / norm1 input  [BT][D] */
    fArr2D d_mha_out;   /* Grad w.r.t. mha output                [BT][D] */
    fArr2D d_ffn2_in;   /* Masked grad into FFN branch (post-drop2) [BT][D] */
    fArr2D d_mha_masked;/* Masked grad into MHA branch (post-drop1) [BT][D] */
    fArr2D gWx1;        /* Gradient of ffn1->Wx  [D][Dff]                */
    fArr2D gWx2;        /* Gradient of ffn2->Wx  [Dff][D]                */
    fVec dg1;           /* Gradient of norm1->gamma  [D]                 */
    fVec db1;           /* Gradient of norm1->beta   [D]                 */
    fVec dg2;           /* Gradient of norm2->gamma  [D]                 */
    fVec db2;           /* Gradient of norm2->beta   [D]                 */
} TRANSFORMER;

/* transformer_create - allocates a TRANSFORMER and its sub-layers.
 *
 * Parameters:
 *   heads     - number of attention heads
 *   steps     - sequence length T
 *   model_dim - D, the model dimension
 *   ffn_dim   - FFN hidden dimension Dff (typically 4 * model_dim)
 *   lookahead - self-attention masking: <0 bidirectional, 0 causal,
 *               L causal with L frames of future context
 *
 * Returns:
 *   Pointer to a zero-initialised TRANSFORMER. Call transformer_init() next.
 */
TRANSFORMER* transformer_create(int heads, int steps, 
                                int model_dim, int ffn_dim,
                                int lookahead);

/* transformer_init - initialises weights and allocates scratch buffers.
 *
 * Parameters:
 *   l            - pointer to TRANSFORMER returned by transformer_create()
 *   batch_size   - B
 *   training     - non-zero if backward pass will be used
 *   dropout_rate - fraction of sub-layer outputs to zero (0 = no dropout)
 */
void transformer_init(TRANSFORMER* l, int batch_size, int training, float dropout_rate);

/* transformer_free - releases all memory owned by the layer. */
void transformer_free(TRANSFORMER* l);

/* transformer_forward - forward pass of a decoder-only transformer layer.
 *
 * Implements the decoder sub-layer stack from Vaswani et al. (2017),
 * "Attention Is All You Need", https://arxiv.org/pdf/1706.03762v7
 *
 * Note: The original paper uses ReLU. This implementation uses GELU instead.
 *
 * Parameters:
 *   l        - pointer to the TRANSFORMER layer
 *   X        - input  [B*T][D]
 *   pad_mask - optional padding mask [B*T]; 1 = real token, 0 = pad.
 *   training - non zero if in training path, zero if in inference path
 *   Y        - output [B*T][D]
 *   lyr      - layer index (informational)
 *
 * Note: if the transformer was initialized not in training mode, the
 * value of this training parameter is ignotred.
 *
 * Computation (Sec. 3.1, p.3 and Sec. 3.2, p.4):
 *
 *   Step 1 - Masked multi-head self-attention (Sec. 3.2.3):
 *     mha_out = MaskedMHA(X)
 *     mha_out = dropout(mha_out)               (Sec. 5.4)
 *
 *   Step 2 - First residual add + layer norm (Eq. after Sec. 3.1):
 *     norm1_out = LayerNorm(X + mha_out)
 *
 *   Step 3 - Position-wise feed-forward network (Sec. 3.3):
 *     ffn1_out = gelu(norm1_out @ Wx1)
 *     ffn2_out = ffn1_out @ Wx2
 *     ffn2_out = dropout(ffn2_out)             (Sec. 5.4)
 *
 *   Step 4 - Second residual add + layer norm (Eq. after Sec. 3.1):
 *     Y = LayerNorm(norm1_out + ffn2_out)
 */
static inline void transformer_forward(TRANSFORMER* restrict l,
                                       const fArr2D restrict X  /*[BT][D]*/,
                                       const iVec restrict pad_mask /*[BT]*/,
                                       int training,
                                       fArr2D Y /*[BT][D]*/,
                                       int lyr)
{
    const int BT = l->BT;
    const int D  = l->D;

    typedef float (*ArrBTD)[D];

    ArrBTD mha_out = (ArrBTD) l->mha_out;
    ArrBTD drop_mask1 = (ArrBTD) l->drop_mask1;
    ArrBTD norm1_out = (ArrBTD) l->norm1_out;
    ArrBTD drop_mask2 = (ArrBTD) l->drop_mask2;

    /* Step 1 - Masked multi-head self-attention (Sec. 3.2.3):
     * mha_out = MaskedMHA(X)
     * mha_out = dropout(mha_out)
     */
    mha_forward(l->mha, X, pad_mask,mha_out,0,training,lyr);
    if (training && l->training && l->dropout_rate > 0)
        dropout(mha_out,drop_mask1,BT,D,l->dropout_rate);

    /* Step 2 - First residual add + layer norm (Sec. 3.1):
     * norm1_out = LayerNorm(X + mha_out)
     */
    addnorm_forward(l->norm1,X,mha_out,norm1_out);

    /* Step 3 - Position-wise feed-forward network (Sec. 3.3):
     * ffn1_out = gelu(norm1_out @ Wx1)
     * ffn2_out = ffn1_out @ Wx2
     * ffn2_out = dropout(ffn2_out)
     */
    fArr2D ffn1_out = dense_forward(l->ffn1,norm1_out,lyr);
    fArr2D ffn2_out = dense_forward(l->ffn2,ffn1_out,lyr);

    if (training && l->training && l->dropout_rate > 0)
        dropout(ffn2_out,drop_mask2,BT,D,l->dropout_rate);

    /* Step 4 - Second residual add + layer norm (Sec. 3.1):
     * Y = LayerNorm(norm1_out + ffn2_out)
     */
    addnorm_forward(l->norm2,norm1_out,ffn2_out,Y);
}

/* transformer_backward - backward pass of a decoder-only transformer layer.
 *
 * Computes gradients of the loss with respect to weights and inputs,
 * reversing each step of transformer_forward in order.
 *
 * Parameters:
 *   l   - pointer to the TRANSFORMER layer (must have run forward first)
 *   dY  - gradient of loss w.r.t. layer output  [BT][D]  (read)
 *   X   - original input to this layer          [BT][D]  (read)
 *   dX  - optional gradient of loss w.r.t. layer input [BT][D]  (written)
 *   lyr - layer index (informational)
 *
 * Gradient steps (reverse of forward):
 *
 *   Step 4 backward - second residual add + layer norm
 *     (reverse of Y = LayerNorm(norm1_out + ffn2_out)):
 *     d_norm2_in = addnorm_backward(dY)
 *       same gradient flows into both ffn2_out branch and norm1_out branch
 *
 *   Step 3 backward - FFN
 *     (reverse of ffn2_out = dropout(ffn2_out @ Wx2)):
 *     d_ffn2_in  = d_norm2_in * drop_mask2   (dropout, residual branch preserved)
 *     (reverse of ffn2_out = ffn1_out @ Wx2):
 *     gWx2       = ffn1_out.T @ d_ffn2_in
 *     d_ffn1_in  = d_ffn2_in @ Wx2.T
 *     (reverse of ffn1_out = gelu(norm1_out @ Wx1)):
 *     gWx1       = norm1_out.T @ d_ffn1_in
 *     d_norm1_in = d_ffn1_in @ Wx1.T        (through gelu derivative)
 *
 *   Residual accumulation for norm2 skip connection:
 *     d_norm1_in += d_norm2_in
 *       (norm1_out feeds both the FFN branch and the AddNorm2 residual)
 *
 *   Step 2 backward - first residual add + layer norm
 *     (reverse of norm1_out = LayerNorm(X + mha_out)):
 *     d_mha_out = addnorm_backward(d_norm1_in)
 *       same gradient flows into both mha_out branch and X branch
 *
 *   Step 1 backward - masked MHA
 *     (reverse of mha_out = dropout(MaskedMHA(X))):
 *     d_mha_masked = d_mha_out * drop_mask1  (dropout, residual branch preserved)
 *     dX_mha = mha_backward(d_mha_masked)
 *
 *   Residual accumulation for norm1 skip connection:
 *     dX = dX_mha + d_mha_out
 *       (X feeds both the MHA branch and the AddNorm1 residual)
 *
 * Weight gradients written to:
 *   l->gWx1, l->gWx2          (ffn1, ffn2 weights)
 *   l->mha->gWq/gWk/gWv/gWo  (MHA weights)
 *   l->dg1/db1, l->dg2/db2   (norm1, norm2 gamma/beta)
 */
static inline void transformer_backward(TRANSFORMER* restrict l,
                                        fArr2D restrict dY  /*[BT][D]*/,
                                        const fArr2D restrict X /*[BT][D]*/,
                                        fArr2D dX /*[BT][D]*/,
                                        int lyr)
{
    const int BT = l->BT;
    const int D = l->D;

    typedef float (*ArrBTD)[D];

    ArrBTD norm1_out = (ArrBTD) l->norm1_out;
    ArrBTD d_norm2_in = (ArrBTD) l->d_norm2_in;
    ArrBTD d_ffn1_in = (ArrBTD) l->d_ffn1_in;
    ArrBTD d_norm1_in = (ArrBTD) l->d_norm1_in;
    ArrBTD d_mha_out  = (ArrBTD) l->d_mha_out;

    /* Step 4 backward - second residual add + layer norm
     * (reverse of Y = LayerNorm(norm1_out + ffn2_out)):
     * d_norm2_in = addnorm_backward(dY)
     *   same gradient flows into both ffn2_out branch and norm1_out branch
     */
    addnorm_backward(l->norm2,dY,d_norm2_in,l->dg2,l->db2);

    /* Step 3 backward - FFN
     * (reverse of ffn2_out = dropout(ffn1_out @ Wx2)):
     * d_ffn2_in = d_norm2_in * drop_mask2   (residual branch preserved)
     */
    fArr2D d_ffn2_in = l->d_ffn2_in;
    if (l->training && l->dropout_rate > 0)
        apply_dropout_mask(d_norm2_in,l->drop_mask2,d_ffn2_in,BT,D);
    else
        d_ffn2_in = (fArr2D) d_norm2_in;

    /* Step 3 backward continued
     * (reverse of ffn2_out = ffn1_out @ Wx2):
     * gWx2 = ffn1_out.T @ d_ffn2_in
     * d_ffn1_in = d_ffn2_in @ Wx2.T
     * (reverse of ffn1_out = gelu(norm1_out @ Wx1)):
     * gWx1 = norm1_out.T @ d_ffn1_in
     * d_norm1_in = d_ffn1_in @ Wx1.T (through gelu derivative)
     */
    dense_backward(l->ffn2,d_ffn2_in,l->ffn1->h,l->gWx2,(fArr2D) d_ffn1_in,lyr);

    dense_backward(l->ffn1,(fArr2D) d_ffn1_in,norm1_out,l->gWx1,d_norm1_in,lyr);

    /* Residual accumulation for norm2 skip connection:
     * d_norm1_in += d_norm2_in
     * (norm1_out feeds both the FFN branch and the AddNorm2 residual)
     */
    for (int i = 0; i < BT * D; i++)
        ((float*) d_norm1_in)[i] += ((float*) d_norm2_in)[i];

    /* Step 2 backward - first residual add + layer norm
     * (reverse of norm1_out = LayerNorm(X + mha_out)):
     * d_mha_out = addnorm_backward(d_norm1_in)
     *   same gradient flows into both mha_out branch and X branch
     */
    addnorm_backward(l->norm1,d_norm1_in,d_mha_out,l->dg1,l->db1);

    /* Step 1 backward - masked MHA
     * (reverse of mha_out = dropout(MaskedMHA(X))):
     * d_mha_masked = d_mha_out * drop_mask1  (residual branch preserved)
     * dX_mha = mha_backward(d_mha_masked)
     * dX = dX_mha + d_mha_out
     *   (X feeds both the MHA branch and the AddNorm1 residual)
     */
    fArr2D d_mha_masked = l->d_mha_masked;
    if (l->training && l->dropout_rate > 0)
        apply_dropout_mask(d_mha_out,l->drop_mask1,d_mha_masked,BT,D);
    else
        d_mha_masked = d_mha_out;

    mha_backward(l->mha,d_mha_masked,X,dX,lyr);
    if (dX != NULL) {
        for (int i = 0; i < BT * D; i++)
            ((float*) dX)[i] += ((float*) d_mha_out)[i];
    }
}

#endif /* TRANSFORMER_H */
