/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer functions */
#include <stdio.h>
#include "mem.h"
#include "random.h"
#include "rope.h"
#include "mha.h"

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
    l->dropout_rate  = dropout_rate;
    
    l->Wq = allocmem(l->D,l->D,float);
    l->Wk = allocmem(l->D,l->D,float);
    l->Wv = allocmem(l->D,l->D,float);
    l->Wo = allocmem(l->D,l->D,float);

    l->Q = allocmem(l->BT,l->D,float);
    l->K = allocmem(l->BT,l->D,float);
    l->V = allocmem(l->BT,l->D,float);

    l->theta = allocmem(1,l->Dh / 2,float);

    l->Qh = allocmem(l->BHT,l->Dh,float);
    l->Kh = allocmem(l->BHT,l->Dh,float);
    l->Vh = allocmem(l->BHT,l->Dh,float);

    l->Att = allocmem(l->BHT,l->T,float);
    l->AttMask = allocmem(l->BHT,l->T,float);

    l->Scores = allocmem(l->T,l->T,float);
    l->Oh = allocmem(l->T,l->Dh,float);
    
    l->Out = allocmem(l->BT,l->D,float);

    float* w;
    int D2 = l->D * l->D;
    float sd = sqrtf(1.0 / l->D);
    w = (float*) l->Wq;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    w = (float*) l->Wk;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    w = (float*) l->Wv;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);
    w = (float*) l->Wo;
    for (int i = 0; i < D2; i++) w[i] = nrand(0,sd);

    rope_init(l->theta,l->Dh);

    if (!training)
        return;

    l->dOut = allocmem(l->BT,l->D,float);

    l->dQ = allocmem(l->BT,l->D,float);
    l->dK = allocmem(l->BT,l->D,float);
    l->dV = allocmem(l->BT,l->D,float);

    l->dQh = allocmem(l->T,l->Dh,float);
    l->dKh = allocmem(l->T,l->Dh,float);
    l->dVh = allocmem(l->T,l->Dh,float);

    l->dOh = allocmem(l->T,l->Dh,float);
    l->dAtt = allocmem(l->T,l->T,float);
    l->dScores = allocmem(l->T,l->T,float);

    l->gWq = allocmem(l->D,l->D,float);
    l->gWk = allocmem(l->D,l->D,float);
    l->gWv = allocmem(l->D,l->D,float);
    l->gWo = allocmem(l->D,l->D,float);    
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
    freemem(l->Wq);
    freemem(l->Wk);
    freemem(l->Wv);
    freemem(l->Wo);

    freemem(l->Q);
    freemem(l->K);
    freemem(l->V);

    freemem(l->theta);

    freemem(l->Qh);
    freemem(l->Kh);
    freemem(l->Vh);

    freemem(l->Att);
    freemem(l->AttMask);

    freemem(l->Scores);
    freemem(l->Oh);
    
    freemem(l->Out);

    /* backward buffers */
    freemem(l->dOut);

    freemem(l->dQ);
    freemem(l->dK);
    freemem(l->dV);

    freemem(l->dQh);
    freemem(l->dKh);
    freemem(l->dVh);

    freemem(l->dOh);
    freemem(l->dAtt);
    freemem(l->dScores);

    freemem(l->gWq);
    freemem(l->gWk);
    freemem(l->gWv);
    freemem(l->gWo);

    freemem(l->Kh_cache);
    freemem(l->Vh_cache);

    freemem(l);
}
