/* Copyright (c) 2026 Gilad Odinak                */
/* Negative-sampling output layer layer functions */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mem.h"
#include "array.h"
#include "random.h"
#include "activation.h"
#include "negsample.h"

/* This layer replaces a dense(K,"Softmax") + cross-entropy output when the
 * vocabulary K is too large for a full softmax. Its forward pass is the
 * identity: it passes the E-dimensional input h through unchanged, so the
 * model's output dimension equals its input dimension E. The word-scoring
 * weights Wo[K][E] are used only by negsample_loss(), which scores the true
 * next word plus a few sampled negatives.
 *
 * Because the loss needs the target word index, it is computed by
 * negsample_loss() (called from the model's loss step) rather than in the
 * layer's backward pass. The backward pass is the identity (dx = dy); the
 * gradient into h is produced by negsample_loss() and handed back as dy.
 *
 * For generation, negsample_logits() scores h against all K words so the
 * result can be softmaxed and sampled.
 *
 * Reference:
 *   Distributed Representations of Words and Phrases and their
 *   Compositionality, Mikolov et al., 2013, https://arxiv.org/pdf/1310.4546
 */

/* Prepares a Wo gradient row for accumulation this batch.
 *
 * On the first touch of 'row' in this batch (per-row stamp), the row is zeroed
 * and appended to l->touched. Later touches are no-ops so gradients add up.
 */
static inline void prep_row(NEGSAMPLE* l, fArr2D gWo_, int row)
{
    if (l->seen[row] == l->stamp)
        return;
    l->seen[row] = l->stamp;
    typedef float (*ArrE)[l->E];
    ArrE gWo = (ArrE) gWo_;
    for (int j = 0; j < l->E; j++)
        gWo[row][j] = 0.0f;
    l->touched[l->ntouched++] = row;
}

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
NEGSAMPLE* negsample_create(int vocab_size, int num_negatives)
{
    NEGSAMPLE* l = allocmem(1,1,NEGSAMPLE);
    l->K = vocab_size;
    l->n_neg = num_negatives;
    if (l->K < 1 || l->n_neg < 1) {
        freemem(l);
        fflush(stdout);
        fprintf(stderr,"negsample_create: invalid vocab_size %d "
                "or num_negatives %d\n",vocab_size,num_negatives);
        exit(-1);
    }
    return l;
}

/* Initializes a negative-sampling layer created by negsample_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors (E)
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(E).
 */
void negsample_init(NEGSAMPLE* l, int input_dim, int batch_size)
{
    l->E = input_dim;
    l->B = batch_size;
    l->Wo = allocmem(l->K,l->E,float);
    l->h = allocmem(l->B,l->E,float);
    l->touched = allocmem(l->B * (l->n_neg + 1),1,int);
    l->ntouched = 0;
    l->seen = allocmem(l->K,1,int);
    for (int i = 0; i < l->K; i++)
        l->seen[i] = -1;
    l->stamp = 0;
    l->dist = NULL;
    l->dist_size = 0;

    typedef float (*ArrKE)[l->E];
    ArrKE Wo = (ArrKE) l->Wo;
    float scale = 1.0 / sqrtf((float) l->E);
    for (int i = 0; i < l->K; i++)
        for (int j = 0; j < l->E; j++)
            Wo[i][j] = nrand(0.0,scale);
}

/* Provides the unigram negative-sampling table (referenced, not owned). */
void negsample_set_dist(NEGSAMPLE* l, const int* dist_table, int dist_table_size)
{
    l->dist = dist_table;
    l->dist_size = dist_table_size;
}

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before negsample_init(), it does nothing.
 *   Otherwise, the passthrough and touched buffers are resized.
 */
void negsample_set_batch_size(NEGSAMPLE* l, int batch_size)
{
    if (l->B == 0)
        return;
    if (batch_size != l->B) {
        l->B = batch_size;
        freemem(l->h);
        l->h = allocmem(l->B,l->E,float);
        freemem(l->touched);
        l->touched = allocmem(l->B * (l->n_neg + 1),1,int);
    }
    else
        fltclr(l->h,l->B * l->E);
    l->ntouched = 0;
}

/* Frees the memory allocated by negsample_create() / negsample_init().
 * The sampling table passed to negsample_set_dist() is not freed.
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void negsample_free(NEGSAMPLE* l)
{
    freemem(l->h);
    freemem(l->Wo);
    freemem(l->touched);
    freemem(l->seen);
    freemem(l);
}

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 */
void negsample_reset(NEGSAMPLE* l)
{
    (void) l; /* Do nothing */
}

/* Computes the negative-sampling loss and its gradients for a batch.
 *
 * Scores each target and sampled negatives, accumulating gradients into h
 * and Wo. Records only touched Wo rows, enabling a subsequent sparse update.
 *
 * Negative Wo/gWo rows are scattered and cache-unfriendly. Draw and prefetch
 * all negatives first, then process them while prefetching ahead. This changes
 * only access order, not results.
 *
 * Parameters:
 *   l       - Pointer to the layer
 *   h       - Input embeddings [B][E]
 *   labels  - Target word indices [B][1] (stored as floats)
 *   gWo     - Output-weight gradients [K][E]
 *   dh      - Gradient w.r.t. h [B][E]
 *   cnt     - Number of valid rows in this batch (<= B)
 *   correct - If not NULL, incremented by the number of positions whose
 *             positive score exceeds all their sampled negatives.
 *
 * Returns:
 *   The summed loss over the batch.
 */
float negsample_loss(NEGSAMPLE* restrict l,
                     const fArr2D restrict h_/*[B][E]*/,
                     const fArr2D restrict labels_/*[B][1]*/,
                     fArr2D restrict gWo_/*[K][E]*/,
                     fArr2D restrict dh_/*[B][E]*/,
                     int cnt, int* correct)
{
    const int E = l->E;
    const int K = l->K;
    const int n_neg = l->n_neg;

    typedef float (*ArrBE)[E];
    typedef float (*ArrKE)[E];
    ArrBE h = (ArrBE) h_;
    ArrBE dh = (ArrBE) dh_;
    ArrKE Wo = (ArrKE) l->Wo;
    ArrKE gWo = (ArrKE) gWo_;
    const float* labels = (const float*) labels_;

    if (l->dist == NULL) {
        fflush(stdout);
        fprintf(stderr,"negsample_loss: sampling table not set "
                "(call negsample_set_dist)\n");
        exit(-1);
    }

    l->stamp++;
    l->ntouched = 0;
    fltclr(dh_,cnt * E);

    int negbuf[n_neg];
    float loss = 0.0;
    for (int i = 0; i < cnt; i++) {
        int target = (int) labels[i];
        if (target <= 0 || target >= K) /* Index 0 reserved (PAD); skip */
            continue;

        /* Positive sample */
        float dot = 0.0;
        for (int j = 0; j < E; j++)
            dot += Wo[target][j] * h[i][j];
        float p = sigmoid1(dot);
        loss -= logf(p + 1e-8);
        float grad = p - 1.0; /* d loss / d dot */

        prep_row(l,gWo_,target);
        for (int j = 0; j < E; j++) {
            gWo[target][j] += grad * h[i][j];
            dh[i][j] += grad * Wo[target][j];
        }

        float pos_dot = dot;
        int beaten = 1; /* Positive scored above all its negatives */

        /* Draw all negatives up front and prefetch their rows     */
        for (int k = 0; k < n_neg;) {
            int neg = l->dist[(int) urand(0,l->dist_size)];
            if (neg == target)
                continue;
            negbuf[k] = neg;
            __builtin_prefetch(Wo[neg], 0, 1);
            __builtin_prefetch(gWo[neg], 1, 1);
            k++;
        }

        /* Process the negatives
         * Prefetch the next row while computing on the current one
         */
        for (int k = 0; k < n_neg; k++) {
            int neg = negbuf[k];
            if (k + 1 < n_neg) {
                int nxt = negbuf[k + 1];
                __builtin_prefetch(Wo[nxt], 0, 1);
                __builtin_prefetch(gWo[nxt], 1, 1);
            }

            dot = 0.0f;
            for (int j = 0; j < E; j++)
                dot += Wo[neg][j] * h[i][j];
            p = sigmoid1(-dot);
            loss -= logf(p + 1e-8f);
            grad = 1.0f - p; /* d loss / d dot */

            prep_row(l,gWo_,neg);
            for (int j = 0; j < E; j++) {
                gWo[neg][j] += grad * h[i][j];
                dh[i][j]    += grad * Wo[neg][j];
            }
            if (dot >= pos_dot)
                beaten = 0;
        }
        if (correct != NULL && beaten)
            (*correct)++;
    }
    return loss;
}

/* Applies one sparse SGD step to the rows of Wo touched by the last
 * negsample_loss(), with weight decay.
 *
 * Parameters:
 *   l             - Pointer to the layer
 *   gWo           - Output-weight gradients [K][E]
 *   learning_rate - Gradient multiplier
 *   weight_decay  - Weight magnitude suppressor (0 to disable)
 */
void negsample_update(NEGSAMPLE* restrict l, fArr2D gWo_,
                      float learning_rate, float weight_decay)
{
    const int E = l->E;
    typedef float (*ArrKE)[E];
    ArrKE Wo  = (ArrKE) l->Wo;
    ArrKE gWo = (ArrKE) gWo_;

    for (int r = 0; r < l->ntouched; r++) {
        int i = l->touched[r];
        for (int j = 0; j < E; j++)
            Wo[i][j] -= learning_rate * (gWo[i][j] + weight_decay * Wo[i][j]);
    }
}

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
                      int cnt)
{
    matmulT(logits,h,l->Wo,cnt,l->E,l->K);
}
