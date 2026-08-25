/* Copyright (c) 2026 Gilad Odinak            */
/* Decoder-only language model data structure */
#ifndef LM_H
#define LM_H
#include "array.h"
#include "lmemb.h"
#include "layer.h"
#include "smsftmax.h"

/* The trained model assembled by lmtrain             */
typedef struct lm_s {
    int V;              /* Vocabulary size                        */
    int E;              /* Enbedding dimension (=model dimension) */
    int T;              /* Sequence length                        */
    int B;              /* Batch size (number of sequences)       */
    int N;              /* Number of transformer layers           */
    int BT;             /* B * T                                  */
    int n_neg;
    LMEMB* emb;
    LAYER* tr;          /* N transformer LAYERs */
    SMSFTMAX* head;
    fArr2D* gHead;      /* Head gradient buffer [1] of [K][E]     */

    /* Activations between layers: 
     * acts[0] = emb output, acts[k+1] = out * of transformer k. 
     * Each is [BT][E]. acts[N] is the head input.
     */
    fArr2D* acts;       /* N+1 buffers                            */
    /* gradient buffers flowing back down the stack, [BT][E] each */
    fArr2D dtop;        /* grad at head input (dh from smsftmax)  */
    fArr2D dcur;        /* scratch grad passed between layers     */
    fArr2D dnext;

    iVec pad_mask;      /* [BT]  1 real / 0 pad                   */
    fArr2D labels;      /* [BT][1] next-token targets (as floats) */
    iVec ids;           /* [BT]  input token ids                  */
} LM;

#endif
