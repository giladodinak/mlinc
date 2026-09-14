/* Copyright (c) 2026 Gilad Odinak */
/* CUDA decoder-only Transformer layer implementation */
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuflt.h"
#include "cumem.h"
#include "arraytypes.h"
#include "array.h"
#include "dropout.h"
#include "mha.h"
#include "addnorm.h"
#include "dense.h"
#include "transformer.h"

#define THCNT 256

/* dst += src, both device-resident. */
static __global__ void transformer_add_kernel(float* dst,
                                              const float* src,
                                              int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] += src[i];
}

/* Allocates a TRANSFORMER control structure and its sub-layer controls.
 * Numeric buffers are allocated by transformer_init() in CUDA device memory.
 */
extern "C" TRANSFORMER* transformer_create(int heads, int steps,
                                           int model_dim, int ffn_dim,
                                           int lookahead)
{
    TRANSFORMER* l = (TRANSFORMER*)calloc(1,sizeof(TRANSFORMER));
    if (l == NULL)
        return NULL;

    l->T = steps;
    l->D = model_dim;
    l->Dff = ffn_dim;
    l->mha = mha_create(heads,steps,lookahead);
    l->ffn1 = dense_create(ffn_dim,(char*)"gelu",0);
    l->ffn2 = dense_create(model_dim,(char*)"none",0);
    l->norm1 = addnorm_create();
    l->norm2 = addnorm_create();

    return l;
}

/* Initialises sub-layers and allocates Transformer-owned device buffers. */
extern "C" void transformer_init(TRANSFORMER* l,
                                 int batch_size,
                                 int training,
                                 float dropout_rate)
{
    const int B = batch_size;
    const int T = l->T;
    const int Dff = l->Dff;
    const int D = l->D;
    const int BT = B * T;

    l->B = B;
    l->BT = BT;
    l->dropout_rate = dropout_rate;
    l->training = training ? 1 : 0;

    mha_init(l->mha,D,B,training,0);
    addnorm_init(l->norm1,D,BT);
    addnorm_init(l->norm2,D,BT);
    dense_init(l->ffn1,D,BT,training);
    dense_init(l->ffn2,Dff,BT,training);

    l->mha_out = (fArr2D)cuallocmem(BT,D,float);
    l->norm1_out = (fArr2D)cuallocmem(BT,D,float);

    if (!l->training)
        return;

    l->d_norm2_in = (fArr2D)cuallocmem(BT,D,float);
    l->d_ffn1_in = (fArr2D)cuallocmem(BT,Dff,float);
    l->d_norm1_in = (fArr2D)cuallocmem(BT,D,float);
    l->d_mha_out = (fArr2D)cuallocmem(BT,D,float);
    l->d_ffn2_in = (fArr2D)cuallocmem(BT,D,float);
    l->d_mha_masked = (fArr2D)cuallocmem(BT,D,float);

    l->dg1 = (fVec)cuallocmem(1,D,float);
    l->db1 = (fVec)cuallocmem(1,D,float);
    l->dg2 = (fVec)cuallocmem(1,D,float);
    l->db2 = (fVec)cuallocmem(1,D,float);

    if (dropout_rate > 0) {
        l->drop_mask1 = (fArr2D)cuallocmem(BT,D,float);
        l->drop_mask2 = (fArr2D)cuallocmem(BT,D,float);
    }
}

extern "C" void transformer_free(TRANSFORMER* l)
{
    if (l == NULL)
        return;

    dense_free(l->ffn1);
    dense_free(l->ffn2);
    addnorm_free(l->norm1);
    addnorm_free(l->norm2);
    mha_free(l->mha);

    cufreemem(l->mha_out);
    cufreemem(l->norm1_out);
    cufreemem(l->d_norm2_in);
    cufreemem(l->d_ffn1_in);
    cufreemem(l->d_norm1_in);
    cufreemem(l->d_mha_out);
    cufreemem(l->d_ffn2_in);
    cufreemem(l->d_mha_masked);
    cufreemem(l->dg1);
    cufreemem(l->db1);
    cufreemem(l->dg2);
    cufreemem(l->db2);
    cufreemem(l->drop_mask1);
    cufreemem(l->drop_mask2);

    free(l);
}

extern "C" void transformer_forward(TRANSFORMER* restrict l,
                                    const fArr2D restrict X,
                                    const iVec restrict pad_mask,
                                    int training,
                                    fArr2D Y,
                                    int lyr)
{
    const int BT = l->BT;
    const int D = l->D;

    /* Step 1 - masked multi-head self-attention + dropout. */
    mha_forward(l->mha,X,pad_mask,l->mha_out,0,training,lyr);

    if (training && l->training && l->dropout_rate > 0)
        dropout(l->mha_out,l->drop_mask1,BT,D,l->dropout_rate);

    /* Step 2 - first residual add + layer norm. */
    addnorm_forward(l->norm1,X,l->mha_out,l->norm1_out);

    /* Step 3 - position-wise FFN + dropout. */
    fArr2D ffn1_out = dense_forward(l->ffn1,l->norm1_out,lyr);
    fArr2D ffn2_out = dense_forward(l->ffn2,ffn1_out,lyr);

    if (training && l->training && l->dropout_rate > 0)
        dropout(ffn2_out,l->drop_mask2,BT,D,l->dropout_rate);

    /* Step 4 - second residual add + layer norm. */
    addnorm_forward(l->norm2,l->norm1_out,ffn2_out,Y);
}

extern "C" void transformer_forward_step(TRANSFORMER* restrict l,
                                         const fArr2D restrict X,
                                         int offset,
                                         fArr2D Y,
                                         int lyr)
{
    /* Preserve the original implementation's single-token path. */
    mha_forward_step(l->mha,X,l->mha_out,offset,lyr);

    addnorm_forward(l->norm1,X,l->mha_out,l->norm1_out);

    fArr2D ffn1_out = dense_forward(l->ffn1,l->norm1_out,lyr);
    fArr2D ffn2_out = dense_forward(l->ffn2,ffn1_out,lyr);

    addnorm_forward(l->norm2,l->norm1_out,ffn2_out,Y);
}

extern "C" void transformer_backward(TRANSFORMER* restrict l,
                                     fArr2D restrict dY,
                                     const fArr2D restrict X,
                                     fArr2D dX,
                                     int lyr)
{
    const int BT = l->BT;
    const int D = l->D;
    const int n = BT * D;

    /* Step 4 backward - second residual add + layer norm. */
    addnorm_backward(l->norm2,dY,l->d_norm2_in,l->dg2,l->db2);

    /* Reverse dropout after FFN. */
    fArr2D d_ffn2_in = l->d_ffn2_in;
    if (l->training && l->dropout_rate > 0)
        apply_dropout_mask(l->d_norm2_in,l->drop_mask2,
                           d_ffn2_in,BT,D);
    else
        d_ffn2_in = l->d_norm2_in;

    /* Step 3 backward - FFN. */
    dense_backward(l->ffn2,d_ffn2_in,l->ffn1->h,
                   l->d_ffn1_in,lyr);

    dense_backward(l->ffn1,l->d_ffn1_in,l->norm1_out,
                   l->d_norm1_in,lyr);

    /* norm1_out also feeds AddNorm2's residual branch. */
    transformer_add_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
        (float*)l->d_norm1_in,(const float*)l->d_norm2_in,n);

    /* Step 2 backward - first residual add + layer norm. */
    addnorm_backward(l->norm1,l->d_norm1_in,l->d_mha_out,
                     l->dg1,l->db1);

    /* Reverse dropout after MHA. */
    fArr2D d_mha_masked = l->d_mha_masked;
    if (l->training && l->dropout_rate > 0)
        apply_dropout_mask(l->d_mha_out,l->drop_mask1,
                           d_mha_masked,BT,D);
    else
        d_mha_masked = l->d_mha_out;

    /* Step 1 backward - MHA. */
    mha_backward(l->mha,d_mha_masked,X,dX,lyr);

    /* X also feeds AddNorm1's residual branch. */
    if (dX != NULL)
        transformer_add_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
            (float*)dX,(const float*)l->d_mha_out,n);
}
