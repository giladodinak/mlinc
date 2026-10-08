/* Copyright (c) 2026 Gilad Odinak */
/* Decoder-only Transformer layer implementation
 *
 * Reference: Attention Is All You Need https://arxiv.org/pdf/1706.03762v7 
 *
 * Notes:
 *   - The original paper uses ReLU. This implementation uses GELU instead.
 *   - In the paper the FFN use a bias term (Sec. 3.3. Eq. 2),
 *     This implementation omits the bias.
 */
#include <string.h>
#include "float.h"
#include "mem.h"
#include "arraytypes.h"
#include "array.h"
#include "dropout.h"
#include "mha.h"
#include "addnorm.h"
#include "dense.h"
#include "transformer.h"

/* Allocates a TRANSFORMER and its sub-layers.
 *
 * Weight buffers and scratch arrays are not allocated until transformer_init().
 *
 * Parameters:
 *   heads     - number of attention heads
 *   steps     - sequence length T
 *   model_dim - D
 *   ffn_dim   - FFN hidden dimension Dff (typically 4 * model_dim)
 *   lookahead - causal masking for self-attention (fixed for the layer):
 *                 < 0  bidirectional (encoder)
 *                 = 0  strictly causal (decoder / streaming)
 *                 = L  causal with L frames of future context
 *
 * Note: The original paper uses ReLU. This implementation uses GELU instead.
 */
TRANSFORMER* transformer_create(int heads, int steps,
                                int model_dim, int ffn_dim,
                                int lookahead)
{
    TRANSFORMER* l = allocmem(1, 1, TRANSFORMER);
    l->T = steps;
    l->D = model_dim;
    l->Dff = ffn_dim;
    l->mha = mha_create(heads, steps, lookahead);
    l->ffn1 = dense_create(ffn_dim,"gelu",0);
    l->ffn2 = dense_create(model_dim,"none",0);
    l->norm1 = addnorm_create();
    l->norm2 = addnorm_create();
    return l;
}

/* Initialises weights and allocates all scratch buffers.
 *
 * Must be called after transformer_create() and before any forward/backward.
 *
 * Parameters:
 *   l            - pointer to the TRANSFORMER
 *   batch_size   - B
 *   training     - non-zero if the backward pass will be used
 *   dropout_rate - fraction of sub-layer outputs to zero (0 = no dropout)
 */
void transformer_init(TRANSFORMER* l, int batch_size, int training, float dropout_rate)
{
    const int B = batch_size;
    const int T = l->T;
    const int Dff = l->Dff;
    const int D = l->D;
    const int BT = B * T;

    l->B = B;
    l->BT = BT;
    l->training = training ? 1 : 0;

    mha_init(l->mha,D,B,training,0);
    addnorm_init(l->norm1,D,BT);
    addnorm_init(l->norm2,D,BT);
    dense_init(l->ffn1,D,BT,training);
    dense_init(l->ffn2,Dff,BT,training);

    l->mha_out = allocmem(BT,D,float);
    l->norm1_out = allocmem(BT,D,float);

    if (!training) return;

    l->d_norm2_in = allocmem(BT,D,float);
    l->d_ffn1_in = allocmem(BT,Dff,float);
    l->d_norm1_in = allocmem(BT,D,float);
    l->d_mha_out = allocmem(BT,D,float);
    l->d_ffn2_in = allocmem(BT,D,float);
    l->d_mha_masked = allocmem(BT,D,float);

    l->dg1 = allocmem(D,1,float);
    l->db1 = allocmem(D,1,float);
    l->dg2 = allocmem(D,1,float);
    l->db2 = allocmem(D,1,float);

    if (dropout_rate > 0) {
        l->dropout1 = dropout_create(dropout_rate);
        l->dropout2 = dropout_create(dropout_rate);
        dropout_init(l->dropout1,D,BT);
        dropout_init(l->dropout2,D,BT);
    }
}

/* Sets a padding mask
 *
 * Parameters:
 *   l        - Pointer to the TRANSFORMER layer from transformer_create().
 *   pad_mask - An integer array of length [B*T];
 *              entries are 1 for real tokens, 0 for padding.
 *
 * Notes:
 *   - This function may be called only once, after transformer_init and
 *     before the first invocation of transformer_forward.
 *   - pad_mask content may be set before calling mha_forward.
 */
void transformer_set_pad_mask(TRANSFORMER* l, const iVec pad_mask)
{
    mha_set_pad_mask(l->mha,pad_mask);
}

/* Releases all memory owned by the TRANSFORMER.
 */
void transformer_free(TRANSFORMER* l)
{
    if (l == NULL) return;

    dense_free(l->ffn1);
    dense_free(l->ffn2);
    addnorm_free(l->norm1);
    addnorm_free(l->norm2);
    dropout_free(l->dropout1);
    dropout_free(l->dropout2);
    mha_free(l->mha);

    freemem(l->mha_out);
    freemem(l->norm1_out);
    freemem(l->d_norm2_in);
    freemem(l->d_ffn1_in);
    freemem(l->d_norm1_in);
    freemem(l->d_mha_out);
    freemem(l->d_ffn2_in);
    freemem(l->d_mha_masked);
    freemem(l->dg1);
    freemem(l->db1);
    freemem(l->dg2);
    freemem(l->db2);

    freemem(l);
}

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
 *   training - non zero if in training path, zero if in inference path
 *   Y        - output [B*T][D]
 *   lyr      - layer index (informational)
 *
 * Note: if the transformer was initialized not in training mode, the
 * value of this training parameter is ignored.
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
void transformer_forward(TRANSFORMER* restrict l,
                         const fArr2D restrict X  /*[BT][D]*/,
                         int training,
                         fArr2D Y /*[BT][D]*/,
                         int lyr)
{
    /* Step 1 - Masked multi-head self-attention (Sec. 3.2.3):
     * mha_out = MaskedMHA(X)
     * mha_out = dropout(mha_out)
     */
    mha_forward(l->mha,X,l->mha_out,0,training,lyr);
    if (training && l->training && l->dropout1 != NULL)
        dropout_forward(l->dropout1,l->mha_out);

    /* Step 2 - First residual add + layer norm (Sec. 3.1):
     * norm1_out = LayerNorm(X + mha_out)
     */
    addnorm_forward(l->norm1,X,l->mha_out,l->norm1_out);

    /* Step 3 - Position-wise feed-forward network (Sec. 3.3):
     * ffn1_out = gelu(norm1_out @ Wx1)
     * ffn2_out = ffn1_out @ Wx2
     * ffn2_out = dropout(ffn2_out)
     */
    fArr2D ffn1_out = dense_forward(l->ffn1,l->norm1_out,lyr);
    fArr2D ffn2_out = dense_forward(l->ffn2,ffn1_out,lyr);

    if (training && l->training && l->dropout2 != NULL)
        dropout_forward(l->dropout2,ffn2_out);

    /* Step 4 - Second residual add + layer norm (Sec. 3.1):
     * Y = LayerNorm(norm1_out + ffn2_out)
     */
    addnorm_forward(l->norm2,l->norm1_out,ffn2_out,Y);
}

/* transformer_forward_step - single-token cached forward for decoding.
 *
 * Modifed version of transformer_forward that processes one new token [1][D]
 * at absolute position 'offset', using the MHA KV cache.
 *
 * Parameters:
 *   l   - pointer to the TRANSFORMER layer
 *   X   - input  [1][D]
 *   off - absolute position of this token (0-based)
 *   Y   - output [1][D]
 *   lyr - layer index (informational)
 */
void transformer_forward_step(TRANSFORMER* restrict l,
                              const fArr2D restrict X /*[1][D]*/,
                              int offset,
                              fArr2D Y /*[1][D]*/,
                              int lyr)
{
    /* Masked self-attention for the single token via the KV cache */
    mha_forward_step(l->mha,X,l->mha_out,offset,lyr);

    /* Residual + norm (one row; sub-layer B must be 1) */
    addnorm_forward(l->norm1,X,l->mha_out,l->norm1_out);

    /* FFN (one row) */
    fArr2D ffn1_out = dense_forward(l->ffn1,l->norm1_out,lyr);
    fArr2D ffn2_out = dense_forward(l->ffn2,ffn1_out,lyr);

    /* Residual + norm */
    addnorm_forward(l->norm2,l->norm1_out,ffn2_out,Y);
}

static inline void transformer_add(fArr2D a1, fArr2D a2, int BT, int D)
{
    for (int i = 0; i < BT * D; i++)
        ((float*) a1)[i] += ((float*) a2)[i];
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
 *   l->ffn1->gWx, l->ffn2->gWx  (ffn1, ffn2 weights)
 *   l->mha->gWq/gWk/gWv/gWo  (MHA weights)
 *   l->dg1/db1, l->dg2/db2   (norm1, norm2 gamma/beta)
 */
void transformer_backward(TRANSFORMER* restrict l,
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
    if (l->training && l->dropout2 != NULL)
        dropout_backward(l->dropout2,d_norm2_in,d_ffn2_in);
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
    dense_backward(l->ffn2,d_ffn2_in,l->ffn1->h,(fArr2D) d_ffn1_in,lyr);

    dense_backward(l->ffn1,(fArr2D) d_ffn1_in,norm1_out,d_norm1_in,lyr);

    /* Residual accumulation for norm2 skip connection:
     * d_norm1_in += d_norm2_in
     * (norm1_out feeds both the FFN branch and the AddNorm2 residual)
     */
    transformer_add(l->d_norm1_in,l->d_norm2_in,BT,D);

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
    if (l->training && l->dropout1 != NULL)
        dropout_backward(l->dropout1,d_mha_out,d_mha_masked);
    else
        d_mha_masked = l->d_mha_out;

    mha_backward(l->mha,d_mha_masked,X,dX,0,lyr);

    if (dX != NULL) {
        for (int i = 0; i < BT * D; i++)
            ((float*) dX)[i] += ((float*) d_mha_out)[i];
    }
}
