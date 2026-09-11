/* Copyright (c) 2026 Gilad Odinak                      */
/* Negative-sampling layer data structure and functions */
#ifndef NEGSAMPLE_H
#define NEGSAMPLE_H
#include "array.h"

/* This layer replaces a dense(K,"Softmax") + cross-entropy output when the
 * vocabulary K is too large for a full softmax. Its forward pass is the
 * identity: it passes the E-dimensional input h through unchanged, so the
 * model's output dimension equals its input dimension E. The word-scoring
 * weights Wo[K][E] are used only by negsample_loss(), which scores the true
 * next word plus a few sampled negatives.
 */
typedef struct negsample_s {
  int E;         /* Input vector dimension (= output dimension, identity) */
  int K;         /* Vocabulary size (number of words)                     */
  int B;         /* Number of input vectors in a batch                    */
  int n_neg;     /* Number of negative samples drawn per position         */
  int training;  /* 1 - training mode, 0 - inference only                 */
  fArr2D Wo;     /* Output weight matrix [K][E]                           */
  fArr2D gWo;    /* Output weight gradients [K][E]                        */
  fArr2D h;      /* Identity output passthrough [B][E]                    */
  const int* dist; /* Unigram negative-sampling table (not owned)         */
  int dist_size; /* Number of entries in dist[]                           */
  int* touched;  /* Distinct Wo rows updated by the last negsample_loss() */
  int ntouched;  /* Number of valid entries in touched[]                  */
  int* seen;     /* Per-row batch stamp [K], for first-touch detection    */
  int stamp;     /* Current batch stamp                                   */
} NEGSAMPLE;

/* Creates a negative-sampling layer.
 *
 * Parameters:
 *   vocab_size    - Number of words (K)
 *   num_negatives - Number of negative samples per position
 *
 * Returns:
 *   Pointer to a negative-sampling layer.
 *
 * Notes:
 *   - The layer needs to be further initialized using negsample_init()
 *     before it can be used.
 */
NEGSAMPLE* negsample_create(int vocab_size, int num_negatives);

/* Initializes a negative-sampling layer created by negsample_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors (E)
 *   batch_size - Number of input vectors processed simultaneously
 *   training   - 1: allocate backward/gradient buffers, 0 for inference-only
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(E).
 */
void negsample_init(NEGSAMPLE* l, int input_dim, int batch_size, int training);

/* Provides the unigram negative-sampling table (referenced, not owned). */
void negsample_set_dist(NEGSAMPLE* l, const int* dist_table, int dist_table_size);

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before negsample_init(), it does nothing.
 */
void negsample_set_batch_size(NEGSAMPLE* l, int batch_size);

/* Frees the memory allocated by negsample_create() / negsample_init().
 * The sampling table passed to negsample_set_dist() is not freed.
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void negsample_free(NEGSAMPLE* l);

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 */
void negsample_reset(NEGSAMPLE* l);

/* Performs the layer's forward pass: identity passthrough.
 *
 * Parameters:
 *   l   - Pointer to the layer's data
 *   X   - Input array [B][E] (the stack's output)
 *   lyr - Ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the output array [B][E] (a copy of X).
 */
static inline fArr2D negsample_forward(NEGSAMPLE* restrict l,
                                       const fArr2D restrict X/*[B][E]*/,
                                       int lyr)
{
    (void) lyr;
    fltcpy(l->h,X,l->B * l->E);
    return l->h;
}

/* Performs the layer's backward pass: identity (dx = dy).
 *
 * The gradient into the input h is produced by negsample_loss() and passed
 * in as dy; this pass simply forwards it to dx.
 *
 * Parameters:
 *   l   - Pointer to the layer
 *   dy  - Output gradient [B][E] (grad w.r.t. h, from negsample_loss)
 *   dx  - Input gradient [B][E] (written if not NULL)
 *   lyr - Ordinal number of this layer in a model (not used)
 */
static inline void negsample_backward(NEGSAMPLE* restrict l,
                                      const fArr2D restrict dy/*[B][E]*/,
                                      fArr2D restrict dx/*[B][E]*/,
                                      int lyr)
{
    (void) lyr;
    if (dx != NULL)
        fltcpy(dx,dy,l->B * l->E);
}

/* Computes the negative-sampling loss and its gradients for a batch.
 *
 * For each of the first 'cnt' rows, scores the target word plus n_neg
 * sampled negatives, accumulating the gradient into h (dh) and into the
 * output weights (gWo). Only the Wo rows actually scored are zeroed (on
 * first touch this batch) and recorded in l->touched / l->ntouched, so a
 * subsequent sparse update touches only those rows.
 *
 * Parameters:
 *   l       - Pointer to the layer
 *   h       - Input embeddings [B][E]
 *   labels  - Target word indices [B][1] (stored as floats)
 *   dh      - Gradient w.r.t. h [B][E]
 *   cnt     - Number of valid rows in this batch (<= B)
 *   correct - If not NULL, incremented by the number of positions whose
 *             positive score exceeds all their sampled negatives.
 *
 * Returns:
 *   The summed loss over the batch.
 */
float negsample_loss(NEGSAMPLE* restrict l,
                     const fArr2D restrict h/*[B][E]*/,
                     const fArr2D restrict labels/*[B][1]*/,
                     fArr2D restrict dh/*[B][E]*/,
                     int cnt, int* correct);

/* Applies one sparse SGD step to the rows of Wo touched by the last
 * negsample_loss(), with weight decay.
 *
 * Parameters:
 *   l             - Pointer to the layer
 *   learning_rate - Gradient multiplier
 *   weight_decay  - Weight magnitude suppressor (0 to disable)
 */
void negsample_update(NEGSAMPLE* restrict l,
                      float learning_rate, float weight_decay);

/* Scores h against the entire vocabulary for generation.
 *
 * Computes logits[i][k] = h[i] . Wo[k] for i in 0..cnt-1, k in 0..K-1.
 * The caller applies softmax and samples. This is a full-vocabulary pass.
 *
 * Parameters:
 *   l      - Pointer to the layer
 *   h      - Input embeddings [cnt][E]
 *   logits - Output scores [cnt][K] (written)
 *   cnt    - Number of rows
 */
void negsample_logits(NEGSAMPLE* restrict l,
                      const fArr2D restrict h/*[cnt][E]*/,
                      fArr2D restrict logits/*[cnt][K]*/,
                      int cnt);

#endif
