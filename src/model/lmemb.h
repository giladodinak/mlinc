/* Copyright (c) 2026 Gilad Odinak */
/* Language-model token embedding layer data structures and functions */
#ifndef LMEMB_H
#define LMEMB_H
#include "arraytypes.h"

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
  void* devstat;  /* Additional implementation state (NULL if not used) */
} LMEMB;

#ifdef __cplusplus
extern "C" {
#endif

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
fArr2D lmemb_forward(LMEMB* restrict l,
                     const int* restrict ids/*[B*T]*/, int lyr);

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
void lmemb_backward(LMEMB* restrict l,
                    const fArr2D restrict dy/*[B*T][E]*/, int lyr);

/* Applies one sparse SGD step to rows touched by lmemb_backward().
 * Weight decay is intentionally omitted for embeddings.
 */
void lmemb_update(LMEMB* restrict l, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif
