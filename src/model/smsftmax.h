/* Copyright (c) 2026 Gilad Odinak                    */
/* Sampled-softmax layer data structure and functions */
#ifndef SMSFTMAX_H
#define SMSFTMAX_H
#include "array.h"

/* This layer replaces a dense(K,"Softmax") + cross-entropy output when the
 * vocabulary K is too large for a full softmax. Its forward pass is the
 * identity: it passes the E-dimensional input h through unchanged, so the
 * model's output dimension equals its input dimension E. The word-scoring
 * weights Wo[K][E] are used only by smsftmax_loss(), which forms a softmax
 * over the true next word plus a few sampled negatives.
 */
typedef struct smsftmax_s {
  int E;         /* Input vector dimension (= output dimension, identity) */
  int K;         /* Vocabulary size (number of words)                     */
  int B;         /* Number of input vectors in a batch                    */
  int n_neg;     /* Number of negative samples drawn per position         */
  fArr2D Wo;     /* Output weight matrix [K][E]                           */
  fArr2D h;      /* Identity output passthrough [B][E]                    */
  int* dist;     /* Unigram sampling table (not owned)                    */
  int dist_size; /* Number of entries in dist[]                           */
  int* touched;  /* Distinct Wo rows updated by the last smsftmax_loss()  */
  int ntouched;  /* Number of valid entries in touched[]                  */
  int* seen;     /* Per-row batch stamp [K], for first-touch detection    */
  int stamp;     /* Current batch stamp                                   */
  float* logq;   /* Precomputed ln(q_c) per word [K], q from dist[]       */
} SMSFTMAX;

/* Creates a sampled-softmax layer.
 *
 * Parameters:
 *   vocab_size    - Number of words (K)
 *   num_negatives - Number of negative samples per position (k)
 *
 * Returns:
 *   Pointer to a sampled-softmax layer. Must be initialised with
 *   smsftmax_init() before use.
 */
SMSFTMAX* smsftmax_create(int vocab_size, int num_negatives);

/* Initializes a sampled-softmax layer created by smsftmax_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors (E)
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(E).
 */
void smsftmax_init(SMSFTMAX* l, int input_dim, int batch_size);

/* Provides the unigram sampling table (referenced, not owned) and precomputes
 * ln(q_c) for every word from the frequencies implied by the table. Must be
 * called after smsftmax_init() (needs K). q_c = count_in_dist(c) / dist_size.
 */
void smsftmax_set_dist(SMSFTMAX* l, int* dist_table, int dist_table_size);

/* Sets a new batch size.
 *
 * Notes:
 *   If called before smsftmax_init(), does nothing. Otherwise resizes the
 *   passthrough and touched buffers.
 */
void smsftmax_set_batch_size(SMSFTMAX* l, int batch_size);

/* Frees memory allocated by smsftmax_create() / smsftmax_init().
 * The sampling table passed to smsftmax_set_dist() is not freed.
 */
void smsftmax_free(SMSFTMAX* l);

/* Resets any state the layer carries across batches. */
void smsftmax_reset(SMSFTMAX* l);

/* Forward pass: identity passthrough (output = input) */
static inline fArr2D smsftmax_forward(SMSFTMAX* restrict l,
                                      const fArr2D restrict X/*[B][E]*/,
                                      int lyr)
{
    (void) lyr;
    fltcpy(l->h,X,l->B * l->E);
    return l->h;
}

/* Backward pass: identity (dx = dy). The gradient into h is produced by
 * smsftmax_loss() and passed in as dy.
 */
static inline void smsftmax_backward(SMSFTMAX* restrict l,
                                     const fArr2D restrict dy/*[B][E]*/,
                                     fArr2D restrict dx/*[B][E]*/,
                                     int lyr)
{
    (void) lyr;
    if (dx != NULL)
        fltcpy(dx,dy,l->B * l->E);
}

/* Computes the sampled-softmax loss and its gradients for a batch.
 *
 * For each of the first 'cnt' rows, forms a softmax over the target plus
 * n_neg sampled negatives (drawn with replacement, positive rejected),
 * with the -ln(k*q_c) correction on every candidate. Accumulates the
 * gradient into h (dh) and into the output weights (gWo). Only the Wo rows
 * actually scored are zeroed on first touch this batch and recorded in
 * l->touched / l->ntouched for a subsequent sparse update.
 *
 * The loss per position is -ln p'(target), where p' is the corrected softmax
 * over the candidate set. This is the standard sampled-softmax objective and
 * is calibrated (unlike negative sampling).
 *
 * Parameters:
 *   l       - Pointer to the layer
 *   h       - Input embeddings [B][E]
 *   labels  - Target word indices [B][1] (stored as floats)
 *   gWo     - Output-weight gradients [K][E]
 *   dh      - Gradient w.r.t. h [B][E]
 *   cnt     - Number of valid rows in this batch (<= B)
 *   correct - If not NULL, incremented per position whose target is the
 *             argmax over its candidate set (top-1 among sampled candidates).
 *
 * Returns:
 *   The summed loss over the batch.
 */
float smsftmax_loss(SMSFTMAX* restrict l,
                    const fArr2D restrict h/*[B][E]*/,
                    const fArr2D restrict labels/*[B][1]*/,
                    fArr2D restrict gWo/*[K][E]*/,
                    fArr2D restrict dh/*[B][E]*/,
                    int cnt, int* correct);

/* Applies one sparse SGD step to the rows of Wo touched by the last
 * smsftmax_loss(), with weight decay.
 */
void smsftmax_update(SMSFTMAX* restrict l, fArr2D gWo,
                     float learning_rate, float weight_decay);

/* Scores h against the entire vocabulary for generation / full-vocab eval.
 * logits[i][k] = h[i] . Wo[k]. The caller applies softmax and samples, or
 * takes the argmax for true top-1 accuracy.
 */
void smsftmax_logits(SMSFTMAX* restrict l,
                     const fArr2D restrict h/*[cnt][E]*/,
                     fArr2D restrict logits/*[cnt][K]*/,
                     int cnt);

#endif
