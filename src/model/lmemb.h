/* Copyright (c) 2026 Gilad Odinak */
/* Language-model token embedding layer data structures and functions */
#ifndef LMEMB_H
#define LMEMB_H
#include "array.h"

typedef struct lmemb_s {
  int V;       /* Vocabulary size                                 */
  int B;       /* Batch size                                      */
  int T;       /* Sequence length                                 */
  int E;       /* Embedding dimension                             */
  int padinx;  /* Pad index, -1 if not used                       */
  int training;/* 1: backward/optimizer buffers allocated         */
  int tied;    /* 1: Wx/gWx/moments borrowed, not owned           */
  int BT;      /* B * T                                           */
  fArr2D h;    /* Output matrix [B*T][E]                          */
  fArr2D Wx;   /* Weights matrix [V][E] (owned or borrowed)       */
  fArr2D gWx;  /* Weight gradients [V][E] (owned or borrowed)     */
  fArr2D mWx;  /* AdamW 1st moment [V][E] (owned; NULL if tied)   */
  fArr2D vWx;  /* AdamW 2nd moment [V][E] (owned; NULL if tied)   */
  int* seen;   /* Per-row batch stamp [V], first-touch detect     */
  int stamp;   /* Current batch stamp                             */
  int* touched;/* Distinct Wx rows touched this batch [B*T]       */
  int ntouched;/* Number of valid entries in touched[]            */
  const int* ids; /* Current batch token indices [B*T] (borrowed) */
} LMEMB;

/* Creates a language-model token embedding layer.
 *
 * Parameters:
 *   embedding_dim - Dimension of token embedding vectors (E)
 *   steps         - Sequence length T (number of tokens per sequence)
 *   padinx        - Index value of pad token, -1 if not used
 *
 * Returns:
 *   Pointer to an LMEMB layer.
 *
 * Notes:
 *   - The layer needs to be further initialized using lmemb_init() before
 *     it can be used.
 */
LMEMB* lmemb_create(int embedding_dim, int steps, int padinx);

/* Initializes a language-model token embedding layer created by lmemb_create().
 *
 * Parameters:
 *   vocab_size - Number of vocabulary tokens V (including pad, if any)
 *   batch_size - Number of sequences processed simultaneously, B.
 *   training   - 1: allocate backward/optimizer buffers; 0: inference only
 *   tied       - 1: Wx/gWx/moments are borrowed from an output projection and
 *                   are set later with lmemb_set_tied_weights(); 0: owned here
 *
 * Notes:
 *   - When not tied, Wx is allocated here. If training, it is initialized with
 *     a normal distribution scaled by 1/sqrt(E), and the pad row is zeroed;
 *     if not training, Wx is left for the caller to load.
 *   - When tied, no weight/gradient/moment buffers are allocated here.
 *   - The output buffer h and, when training, the sparse touched-row buffers
 *     are always allocated (they are owned regardless of tying).
 */
void lmemb_init(LMEMB* l, int vocab_size, int batch_size, int training, int tied);

/* Points a tied layer's Wx/gWx at the output projection's shared buffers.
 * Only valid when the layer was initialized with tied = 1.
 *
 * Parameters:
 *   Wx  - Shared weight matrix  [V][E]
 *   gWx - Shared weight gradient [V][E]
 */
void lmemb_set_tied_weights(LMEMB* l, fArr2D Wx, fArr2D gWx);

/* Frees the memory allocated by lmemb_create() / lmemb_init().
 * When tied, the borrowed Wx/gWx/moments are not freed.
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void lmemb_free(LMEMB* l);

/* Resets any per-batch state the layer carries.
 *
 * Parameters:
 *   l - Pointer to the LMEMB layer to be reset
 */
void lmemb_reset(LMEMB* l);

/* Marks a Wx gradient row for accumulation this batch.
 *
 * On the first touch of 'row' in the current batch (detected via the per-row
 * stamp), the row is zeroed (when this layer owns gWx) and appended to
 * l->touched. Subsequent touches in the same batch are no-ops, so gradients
 * accumulate correctly. O(1) per touch, unlike a linear scan.
 *
 * When tied, the shared gWx has already been zeroed and populated by the
 * output layer's backward, so this layer must not re-zero the row; it only
 * records it. Hence the row is zeroed only when the layer owns gWx.
 */
static inline void lmemb_prep_row(LMEMB* restrict l, int row)
{
    if (l->seen[row] == l->stamp)
        return;
    l->seen[row] = l->stamp;
    if (!l->tied) {
        typedef float (*ArrVE)[l->E];
        ArrVE gWx = (ArrVE) l->gWx;
        fltclr(gWx[row],l->E);
    }
    l->touched[l->ntouched++] = row;
}

/* Performs the token embedding forward pass: a per-token gather.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   ids - An array of B*T token indices [B*T]
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the output values, h[B*T][E]. Row t is Wx[ids[t]].
 *
 * The ids pointer is retained for the matching backward pass.
 */
static inline fArr2D lmemb_forward(LMEMB* restrict l,
                                   const int* restrict ids/*[B*T]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBTE)[l->E];
    typedef float (*ArrVE)[l->E];
    ArrBTE h = (ArrBTE) l->h;
    ArrVE Wx = (ArrVE) l->Wx;

    l->ids = ids;
    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        for (int k = 0; k < l->E; k++)
            h[i][k] = Wx[id][k];
    }
    return l->h;
}

/* Performs the token embedding backward pass: a per-token scatter-add.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   dy  - The output gradient [B*T][E], from the layer above (the stack input)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Accumulates each token's gradient into its Wx row of gWx, recording the
 * distinct touched rows in l->touched / l->ntouched for a sparse update.
 * The token indices from the last lmemb_forward() are used. There is no
 * input gradient: this is the first layer of the model.
 *
 * Rows are zeroed on first touch only when this layer owns gWx (untied);
 * when tied, the shared gWx is prepared by the output layer and this pass
 * only accumulates. A token index appearing at several positions accumulates.
 */
static inline void lmemb_backward(LMEMB* restrict l,
                              const fArr2D restrict dy_/*[B*T][E]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBTE)[l->E];
    typedef float (*ArrVE)[l->E];
    ArrBTE dy = (ArrBTE) dy_;
    ArrVE gWx = (ArrVE) l->gWx;
    const int* ids = l->ids;

    l->stamp++;
    l->ntouched = 0;
    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        if (id == l->padinx)
            continue;
        lmemb_prep_row(l,id);
        for (int k = 0; k < l->E; k++)
            gWx[id][k] += dy[i][k];
    }
}

#endif
