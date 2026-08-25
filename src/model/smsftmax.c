/* Copyright (c) 2026 Gilad Odinak            */
/* Sampled-softmax output layer functions     */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mem.h"
#include "array.h"
#include "random.h"
#include "smsftmax.h"

/* This layer replaces a dense(K,"Softmax") + cross-entropy output when the
 * vocabulary K is too large for a full softmax. Its forward pass is the
 * identity: it passes the E-dimensional input h through unchanged, so the
 * model's output dimension equals its input dimension E. The word-scoring
 * weights Wo[K][E] are used by smsftmax_loss() to score the true next word
 * plus a few sampled negatives.
 *
 * Because the loss needs the target word index, it is computed by
 * smsftmax_loss() (called from the model's loss step) rather than in the
 * layer's backward pass. The backward pass is the identity (dx = dy); the
 * gradient into h is produced by smsftmax_loss() and handed back as dy.
 *
 * This layer is derived from the NEGSAMPLE layer. Unlike negative sampling,
 * which scores each sample with an independent sigmoid, this layer uses an
 * importance-corrected sampled approximation to the full-softmax gradient.
 *
 * The loss uses the improved logQ correction of Khrylchenko et al. 2025: the
 * positive is deterministic (present with probability 1), so it is left
 * uncorrected and excluded from the softmax denominator, the negatives use
 * the positive-excluded proposal Q', and the loss is weighted by the model's
 * misclassification probability 1 - P(target|u). See smsftmax_loss() for the
 * exact formula and gradient.
 *
 * For generation, smsftmax_logits() scores h against all K words so the
 * result can be softmaxed and sampled, or argmax'd for true top-1 accuracy.
 *
 * References:
 * 1. Correcting the LogQ Correction: Revisiting Sampled Softmax for
 *    Large-Scale Retrieval, Khrylchenko et al., RecSys 2025,
 *    https://arxiv.org/pdf/2507.09331
 *
 * 2. Quick Training of Probabilistic Neural Nets by Importance Sampling,
 *    Bengio and Senecal, AISTATS 2003,
 *    https://proceedings.mlr.press/r4/bengio03a.html
 */

/* Prepares a Wo gradient row for accumulation this batch.
 *
 * On the first touch of 'row' in this batch (per-row stamp), the row is zeroed
 * and appended to l->touched. Later touches are no-ops so gradients add up.
 */
static inline void prep_row(SMSFTMAX* l, fArr2D gWo_, int row)
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
SMSFTMAX* smsftmax_create(int vocab_size, int num_negatives)
{
    SMSFTMAX* l = allocmem(1,1,SMSFTMAX);
    l->K = vocab_size;
    l->n_neg = num_negatives; /* Ref. #1 Section 3.2 */
    if (l->K < 1 || l->n_neg < 1) {
        freemem(l);
        fflush(stdout);
        fprintf(stderr,"smsftmax_create: invalid vocab_size %d "
                "or num_negatives %d\n",vocab_size,num_negatives);
        exit(-1);
    }
    return l;
}

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
void smsftmax_init(SMSFTMAX* l, int input_dim, int batch_size)
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
    l->logq = allocmem(l->K,1,float);

    typedef float (*ArrKE)[l->E];
    ArrKE Wo = (ArrKE) l->Wo;
    float scale = 1.0 / sqrtf((float) l->E);
    for (int i = 0; i < l->K; i++)
        for (int j = 0; j < l->E; j++)
            Wo[i][j] = nrand(0.0,scale);
}

/* Sets the sampling table and precomputes ln(q_c) per word.
 * q_c = (number of times word c appears in dist[]) / dist_size. Words absent
 * from dist[] are floored to q_c = 1/dist_size so logq is finite; since they
 * cannot be drawn as negatives, this value is only ever read as q(target)
 * (via log(1 - q_target)) when such a word is the positive.
 */
void smsftmax_set_dist(SMSFTMAX* l, int* dist_table, int dist_table_size)
{
    if (dist_table == NULL || dist_table_size <= 0) {
        fflush(stdout);
        fprintf(stderr,"smsftmax_set_dist: invalid sampling table\n");
        exit(-1);
    }

    l->dist = dist_table;
    l->dist_size = dist_table_size;

    /* Count occurrences of each word in the table. */
    int* cnt = allocmem(l->K,1,int);
    for (int i = 0; i < dist_table_size; i++) {
        int w = dist_table[i];
        if (w >= 0 && w < l->K)
            cnt[w]++;
    }
    /* Ref. #1, Sec. 4: Q(d) = #d/N.
     * Ref. #2, Sec. 2.2: Q is the proposal distribution for importance sampling.
     */
    float inv = 1.0 / (float) dist_table_size;
    float min_q = inv; /* finite minimum for q=0; affects positives only */
    for (int c = 0; c < l->K; c++) {
        float q = (cnt[c] > 0) ? (float) cnt[c] * inv : min_q;
        l->logq[c] = logf(q);
    }
    freemem(cnt);
}

/* Sets a new batch size.
 *
 * Notes:
 *   If called before smsftmax_init(), does nothing. Otherwise resizes the
 *   passthrough and touched buffers.
 */
void smsftmax_set_batch_size(SMSFTMAX* l, int batch_size)
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

/* Frees memory allocated by smsftmax_create() / smsftmax_init().
 * The sampling table passed to smsftmax_set_dist() is not freed.
 */
void smsftmax_free(SMSFTMAX* l)
{
    freemem(l->h);
    freemem(l->Wo);
    freemem(l->touched);
    freemem(l->seen);
    freemem(l->logq);
    freemem(l);
}

/* Resets any state the layer carries across batches. */
void smsftmax_reset(SMSFTMAX* l)
{
    (void) l; /* No cross-batch state to reset */
}

/* Computes the corrected sampled-softmax loss and gradients for a batch.
 *
 * For each of the first 'cnt' rows, samples n_neg negatives from Q' =
 * Q(. | word != target), computes the importance-corrected negative
 * distribution, and applies the improved logQ gradient of Ref. #1:
 *
 *   w [ -grad s(target) + sum_c p_c grad s(c) ]
 *
 * where p_c is the softmax over negative scores s(c)-ln Q'(c), and
 * w = 1-P(target|h). The estimate of P(target|h) reuses the same negatives.
 * The weight w is stop-gradient: its derivative is not included.
 *
 * Only the Wo rows actually scored are zeroed on first touch this batch and
 * recorded in l->touched / l->ntouched for a subsequent sparse update.
 *
 * Parameters:
 *   l       - Pointer to the layer
 *   h       - Input embeddings [B][E]
 *   labels  - Target word indices [B][1] (stored as floats)
 *   gWo     - Output-weight gradients [K][E]
 *   dh      - Gradient w.r.t. h [B][E]
 *   cnt     - Number of valid rows in this batch (<= B)
 *   correct - If not NULL, incremented when the target has the largest raw
 *             model score among the sampled candidate set.
 *
 * Returns:
 *   The summed weighted loss over the batch.
 */
float smsftmax_loss(SMSFTMAX* restrict l,
                    const fArr2D restrict h_/*[B][E]*/,
                    const fArr2D restrict labels_/*[B][1]*/,
                    fArr2D restrict gWo_/*[K][E]*/,
                    fArr2D restrict dh_/*[B][E]*/,
                    int cnt, int* correct)
{
    const int E = l->E;
    const int K = l->K;
    const int k = l->n_neg;

    typedef float (*ArrBE)[E];
    typedef float (*ArrKE)[E];
    ArrBE h = (ArrBE) h_;
    ArrBE dh = (ArrBE) dh_;
    ArrKE Wo = (ArrKE) l->Wo;
    ArrKE gWo = (ArrKE) gWo_;
    const float* labels = (const float*) labels_;

    if (l->dist == NULL || l->logq == NULL) {
        fflush(stdout);
        fprintf(stderr,"smsftmax_loss: sampling table not set "
                "(call smsftmax_set_dist)\n");
        exit(-1);
    }

    int   cand[k];            /* sampled negative word ids                 */
    float score[k];           /* raw negative scores                       */
    float logit[k];           /* s(c) - ln Q'(c)                           */
    float prob[k];            /* softmax over corrected negatives          */

    l->stamp++;
    l->ntouched = 0;
    fltclr(dh_,cnt * E);

    const float lnk = logf((float) k);
    float loss = 0.0;
    for (int i = 0; i < cnt; i++) {
        int target = (int) labels[i];
        if (target <= 0 || target >= K)  /* index 0 reserved (PAD); skip  */
            continue;

        float qtarget = expf(l->logq[target]);
        if (qtarget >= 1.0) {
            fflush(stdout);
            fprintf(stderr,"smsftmax_loss: sampling distribution contains "
                    "only target %d\n",target);
            exit(-1);
        }

        /* Ref. #1, Sec. 4: obtain Q' by sampling from Q and discarding
         * the positive item when it appears.
         */
        for (int c = 0; c < k; c++) {
            int neg;
            do {
                neg = l->dist[(int) urand(0,l->dist_size)];
            } while (neg == target);
            cand[c] = neg;
        }

        /* Ref. #1, Sec. 3.1: f(u,d) = <g(u),h(d)> (score function). */
        float pos = 0.0;
        for (int j = 0; j < E; j++)
            pos += Wo[target][j] * h[i][j];

        /* Ref. #1, Sec. 4: Q'(c) = Q(c)/(1-Q(target)).
         * Ref. #1, Sec. 3.3: importance weight uses exp[s(c)-ln Q'(c)].
         * Ref. #2, Sec. 2.2: importance ratio uses P/Q.
         */
        float log1mqt = log1pf(-qtarget);
        float maxl = -INFINITY;
        for (int c = 0; c < k; c++) {
            int w = cand[c];
            float dot = 0.0f;
            for (int j = 0; j < E; j++)
                dot += Wo[w][j] * h[i][j];
            score[c] = dot;
            logit[c] = dot - l->logq[w] + log1mqt;
            if (logit[c] > maxl)
                maxl = logit[c];
        }

        /* Ref. #1, Sec. 3.3, importance-sampling equation:
         *   Normalize exp[s(c)-ln Q'(c)] over sampled negatives.
         * Ref. #2, Sec. 2.2 / Algorithm 3:
         *   Normalized importance sampling estimate of the negative gradient.
         */
        float sum = 0.0;
        for (int c = 0; c < k; c++) {
            prob[c] = expf(logit[c] - maxl);
            sum += prob[c];
        }
        float invsum = 1.0 / sum;
        for (int c = 0; c < k; c++)
            prob[c] *= invsum;

        /* Ref. #1, Sec. 3.3, P(target|h) estimate equation:
         *   P ~= exp(pos) / [exp(pos) + (1/k) sum exp(logit_c)].
         * Ref. #1, Sec. 3.3, weighted-loss equation:
         *   w = sg(1-P(target|h)).
         */
        float logzneg = maxl + logf(sum);
        float logzest = logzneg - lnk;
        float d = pos - logzest;
        float weight;
        if (d >= 0.0) {
            float e = expf(-d);
            weight = e / (1.0 + e);
        }
        else {
            float e = expf(d);
            weight = 1.0f / (1.0f + e);
        }

        /* Ref. #1, Sec. 3.3, weighted-loss equation:
         *   L = -w log[exp(pos) / sum exp(logit_c)],
         *   w = sg(1-P(target|h)).
         */
        loss += weight * (logzneg - pos);

        /* Ref. #1, Sec. 3.3, corrected-gradient decomposition:
         *   w[-grad s(target) + E_{Q'} grad s(negative)].
         * Ref. #2, Eq. (6): positive term plus expected negative term.
         */
        prep_row(l,gWo_,target);
        for (int j = 0; j < E; j++) {
            gWo[target][j] -= weight * h[i][j];
            dh[i][j]       -= weight * Wo[target][j];
        }

        for (int c = 0; c < k; c++) {
            int w = cand[c];
            float g = weight * prob[c];
            prep_row(l,gWo_,w);
            for (int j = 0; j < E; j++) {
                gWo[w][j] += g * h[i][j];
                dh[i][j]  += g * Wo[w][j];
            }
        }

        /* Raw-score sampled top-1, consistent with full-vocabulary inference */
        if (correct != NULL) {
            int c;
            for (c = 0; c < k && score[c] <= pos; c++);
            if (c == k)
                (*correct)++;
        }
    }
    return loss;
}

/* Applies one sparse SGD step to the rows of Wo touched by the last
 * smsftmax_loss(), with weight decay.
 */
void smsftmax_update(SMSFTMAX* restrict l, fArr2D gWo_,
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

/* Scores h against the entire vocabulary for generation / full-vocab eval.
 * logits[i][k] = h[i] . Wo[k]. The caller applies softmax and samples, or
 * takes the argmax for true top-1 accuracy.
 */
void smsftmax_logits(SMSFTMAX* restrict l,
                     const fArr2D restrict h/*[cnt][E]*/,
                     fArr2D restrict logits/*[cnt][K]*/,
                     int cnt)
{
    /* logits[cnt][K] = h[cnt][E] @ Wo[K][E]^T  
     * (raw scores softmax over the full vocabulary)
     */
    matmulT(logits,h,l->Wo,cnt,l->E,l->K);
}
