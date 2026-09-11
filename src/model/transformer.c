/* Copyright (c) 2026 Gilad Odinak */
/* Decoder-only Transformer layer implementation                           */
#include <string.h>
#include "float.h"
#include "mem.h"
#include "array.h"
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
    l->dropout_rate = dropout_rate;
    l->training = training;

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
        l->drop_mask1 = allocmem(BT,D,float);
        l->drop_mask2 = allocmem(BT,D,float);
    }
}

/* Releases all memory owned by the TRANSFORMER.
 */
void transformer_free(TRANSFORMER* l)
{
    dense_free(l->ffn1);
    dense_free(l->ffn2);
    addnorm_free(l->norm1);
    addnorm_free(l->norm2);
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
    freemem(l->drop_mask1);
    freemem(l->drop_mask2);
    freemem(l);
}
