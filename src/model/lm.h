/* Copyright (c) 2026 Gilad Odinak */
/* Decoder-only language model data structure and functions */
#ifndef LM_H
#define LM_H
#include "array.h"
#include "lmemb.h"
#include "layer.h"
#include "smsftmax.h"

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

LM* lm_create(int vocab, int model_dim, int heads, int seq_len,
              int batch, int layers, int ffn_dim, int n_neg,
              float dropout, char optimizer);

void lm_free(LM* m);

fArr2D lm_forward(LM* m, int training);

void lm_backward(LM* m);

void lm_update(LM* m, char optimizer, float lr, float wd, int update_cnt);

int lm_generate(LM* m, HASHMAP* hmap,
                const int* seed, int seedlen,
                int steps, char *buffer, int buflen,
                float temperature, int top_k,
                float rep_penalty, int rep_win_len);

#endif
