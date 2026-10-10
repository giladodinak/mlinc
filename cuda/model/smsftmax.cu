/* Copyright (c) 2026 Gilad Odinak            */
/* Sampled-softmax output layer functions - CUDA */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuctx.h"
#include "mem.h"
#include "cumem.h"
#include "cuflt.h"
#include "arraytypes.h"
#include "array.h"
#include "random.h"
#include "smsftmax.h"

#define THDCNT 256

typedef struct devstat_s {
    int* hsamples;    /* Host target/negative samples [B][n_neg+1]  */
    int* dsamples;    /* Device copy of hsamples                    */
    int* dtouched;    /* Device copy of touched[]                   */
    float* dlogq;     /* Device copy of logq[K]                     */
    float* dloss;     /* Device loss accumulator                    */
    int* dcorrect;    /* Device correct-count accumulator           */
    float* dlogits;   /* Device full-vocabulary logits scratch      */
    int logits_rows;  /* Rows currently allocated in dlogits        */
} DEVSTAT;

/* This layer replaces a dense(K,"Softmax") + cross-entropy output when the
 * vocabulary K is too large for a full softmax. Its forward pass is the
 * identity: it passes the E-dimensional input h through unchanged, so the
 * model's output dimension equals its input dimension E. The word-scoring
 * weights Wo[K][E] are used by smsftmax_loss() to score the true next word
 * plus a few sampled negatives.
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

static inline void prep_row(SMSFTMAX* l, int row)
{
    if (l->seen[row] == l->stamp)
        return;
    l->seen[row] = l->stamp;
    l->touched[l->ntouched++] = row;
}

static __global__ void clear_rows_kernel(float* gWo, const int* touched,
                                         int ntouched, int E)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ntouched * E;
    if (k < n) {
        int r = k / E;
        int j = k % E;
        int row = touched[r];
        gWo[row * E + j] = 0.0f;
    }
}

/* Implements the same corrected sampled-softmax equations as smsftmax.c.
 * Each block handles one training row. Dot products use the block as a
 * reduction; the n_neg probabilities are small per-row shared scratch.
 */
static __global__ void smsftmax_loss_kernel(const float* h,
                                             const int* samples,
                                             float* dh,
                                             const float* Wo,
                                             float* gWo,
                                             const float* logq,
                                             int E, int n_neg, int cnt,
                                             float* loss, int* correct)
{
    int i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= cnt)
        return;

    int target = samples[i * (n_neg + 1)];
    if (target < 0)
        return;

    extern __shared__ float shared[];
    float* red   = shared;
    float* score = shared + blockDim.x;
    float* prob  = score + n_neg;

    __shared__ float pos;
    __shared__ float weight;

    float dot = 0.0f;
    for (int j = tid; j < E; j += blockDim.x)
        dot += Wo[target * E + j] * h[i * E + j];

    red[tid] = dot;
    __syncthreads();
    for (int n = blockDim.x / 2; n > 0; n >>= 1) {
        if (tid < n)
            red[tid] += red[tid + n];
        __syncthreads();
    }
    if (tid == 0)
        pos = red[0];
    __syncthreads();

    /* Ref. #1, Sec. 4: Q'(c) = Q(c)/(1-Q(target)).
     * Ref. #1, Sec. 3.3: importance weight uses exp[s(c)-ln Q'(c)].
     */
    float log1mqt = log1pf(-expf(logq[target]));

    for (int c = 0; c < n_neg; c++) {
        int w = samples[i * (n_neg + 1) + c + 1];

        dot = 0.0f;
        for (int j = tid; j < E; j += blockDim.x)
            dot += Wo[w * E + j] * h[i * E + j];

        red[tid] = dot;
        __syncthreads();
        for (int n = blockDim.x / 2; n > 0; n >>= 1) {
            if (tid < n)
                red[tid] += red[tid + n];
            __syncthreads();
        }

        if (tid == 0) {
            score[c] = red[0];
            prob[c] = red[0] - logq[w] + log1mqt;
        }
        __syncthreads();
    }

    if (tid == 0) {
        /* Ref. #1, Sec. 3.3, importance-sampling equation:
         * normalize exp[s(c)-ln Q'(c)] over sampled negatives.
         */
        float maxl = -INFINITY;
        for (int c = 0; c < n_neg; c++)
            if (prob[c] > maxl)
                maxl = prob[c];

        float sum = 0.0f;
        for (int c = 0; c < n_neg; c++) {
            prob[c] = expf(prob[c] - maxl);
            sum += prob[c];
        }
        float invsum = 1.0f / sum;
        for (int c = 0; c < n_neg; c++)
            prob[c] *= invsum;

        /* Ref. #1, Sec. 3.3:
         * P ~= exp(pos) / [exp(pos) + (1/k) sum exp(logit_c)]
         * w = sg(1-P(target|h)).
         */
        float logzneg = maxl + logf(sum);
        float logzest = logzneg - logf((float)n_neg);
        float d = pos - logzest;
        if (d >= 0.0f) {
            float e = expf(-d);
            weight = e / (1.0f + e);
        }
        else {
            float e = expf(d);
            weight = 1.0f / (1.0f + e);
        }

        /* Ref. #1, Sec. 3.3:
         * L = -w log[exp(pos) / sum exp(logit_c)].
         */
        atomicAdd(loss,weight * (logzneg - pos));

        if (correct != NULL) {
            int ok = 1;
            for (int c = 0; c < n_neg; c++)
                if (score[c] > pos) {
                    ok = 0;
                    break;
                }
            if (ok)
                atomicAdd(correct,1);
        }
    }
    __syncthreads();

    /* Ref. #1, Sec. 3.3, corrected-gradient decomposition:
     * w[-grad s(target) + E_Q' grad s(negative)].
     * Ref. #2, Eq. (6): positive term plus expected negative term.
     */
    for (int j = tid; j < E; j += blockDim.x) {
        float hj = h[i * E + j];
        atomicAdd(&gWo[target * E + j],-weight * hj);
        dh[i * E + j] -= weight * Wo[target * E + j];
    }
    __syncthreads();

    for (int c = 0; c < n_neg; c++) {
        int w = samples[i * (n_neg + 1) + c + 1];
        float g = weight * prob[c];

        for (int j = tid; j < E; j += blockDim.x) {
            float hj = h[i * E + j];
            atomicAdd(&gWo[w * E + j],g * hj);
            dh[i * E + j] += g * Wo[w * E + j];
        }
        __syncthreads();
    }
}

static __global__ void smsftmax_update_kernel(float* Wo,
                                               const float* gWo,
                                               const int* touched,
                                               int ntouched, int E,
                                               float learning_rate,
                                               float weight_decay)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ntouched * E;
    if (k < n) {
        int r = k / E;
        int j = k % E;
        int row = touched[r];
        int p = row * E + j;
        Wo[p] -= learning_rate * (gWo[p] + weight_decay * Wo[p]);
    }
}

/* Merges two (max, sum-of-exp, argmax) partial states. Ties on the max
 * keep the lower index, matching a left-to-right strict '>' scan.
 */
static __device__ __forceinline__ void eval_merge(float* m, float* s, int* ix,
                                                  float m2, float s2, int ix2)
{
    if (m2 > *m || (m2 == *m && ix2 < *ix)) {
        if (*m != -INFINITY)
            s2 += *s * expf(*m - m2);
        *m = m2; *s = s2; *ix = ix2;
    }
    else if (m2 != -INFINITY)
        *s += s2 * expf(m2 - *m);
}

/* Full-vocabulary evaluation. One block per row: a single online pass over
 * logits[row][1..K-1] computes max, argmax and sum exp(z - max); index 0
 * (PAD) is excluded. Accumulates -log p(target) and top-1 hits.
 */
static __global__ void smsftmax_eval_kernel(const float* logits,
                                            const int* targets,
                                            int K, int cnt,
                                            float* nll, int* correct)
{
    int i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= cnt)
        return;
    int target = targets[i];
    if (target <= 0 || target >= K)
        return;

    __shared__ float sm[THDCNT];
    __shared__ float ss[THDCNT];
    __shared__ int   si[THDCNT];

    const float* z = logits + (size_t) i * K;
    float m = -INFINITY, s = 0.0f;
    int ix = K;
    for (int k = 1 + tid; k < K; k += blockDim.x) {
        float v = z[k];
        if (v > m) {
            s = (m == -INFINITY) ? 1.0f : s * expf(m - v) + 1.0f;
            m = v;
            ix = k;
        }
        else
            s += expf(v - m);
    }
    sm[tid] = m; ss[tid] = s; si[tid] = ix;
    __syncthreads();

    for (int n = blockDim.x / 2; n > 0; n >>= 1) {
        if (tid < n) {
            float a = sm[tid], b = ss[tid];
            int c = si[tid];
            eval_merge(&a,&b,&c,sm[tid + n],ss[tid + n],si[tid + n]);
            sm[tid] = a; ss[tid] = b; si[tid] = c;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logp = (z[target] - sm[0]) - logf(ss[0]);
        atomicAdd(nll,-logp);
        if (si[0] == target)
            atomicAdd(correct,1);
    }
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
    SMSFTMAX* l = (SMSFTMAX*) allocmem(1,1,SMSFTMAX);
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
    l->Wo = (fArr2D) cuallocmem(l->K,l->E,float);
    l->h = (fArr2D) cuallocmem(l->B,l->E,float);
    l->touched = (int*) allocmem(l->B * (l->n_neg + 1),1,int);
    l->ntouched = 0;
    l->seen = (int*) allocmem(l->K,1,int);
    for (int i = 0; i < l->K; i++)
        l->seen[i] = -1;
    l->stamp = 0;
    l->dist = NULL;
    l->dist_size = 0;
    l->logq = (float*) allocmem(l->K,1,float);

    int ns = l->B * (l->n_neg + 1);
    DEVSTAT* d = (DEVSTAT*) allocmem(1,1,DEVSTAT);
    d->hsamples = (int*) allocmem(ns,1,int);
    d->dsamples = (int*) cuallocmem(ns,1,int);
    d->dtouched = (int*) cuallocmem(ns,1,int);
    d->dlogq = (float*) cuallocmem(l->K,1,float);
    d->dloss = (float*) cuallocmem(1,1,float);
    d->dcorrect = (int*) cuallocmem(1,1,int);
    d->dlogits = NULL;
    d->logits_rows = 0;
    l->devstat = d;

    float* Wo = (float*) allocmem(l->K,l->E,float);
    float scale = 1.0f / sqrtf((float) l->E);
    for (int i = 0; i < l->K * l->E; i++)
        Wo[i] = nrand(0.0,scale);
    cufltcpy(l->Wo,Wo,l->K * l->E);
    freemem(Wo);
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
    int* cnt = (int*) allocmem(l->K,1,int);
    for (int i = 0; i < dist_table_size; i++) {
        int w = dist_table[i];
        if (w >= 0 && w < l->K)
            cnt[w]++;
    }
    /* Ref. #1, Sec. 4: Q(d) = #d/N.
     * Ref. #2, Sec. 2.2: Q is the proposal distribution for importance sampling.
     */
    float inv = 1.0f / (float) dist_table_size;
    float min_q = inv; /* finite minimum for q=0; affects positives only */
    for (int c = 0; c < l->K; c++) {
        float q = (cnt[c] > 0) ? (float) cnt[c] * inv : min_q;
        l->logq[c] = logf(q);
    }
    freemem(cnt);

    DEVSTAT* d = (DEVSTAT*) l->devstat;
    cufltcpy(d->dlogq,l->logq,l->K);
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
        cufreemem(l->h);
        l->h = (fArr2D) cuallocmem(l->B,l->E,float);

        int ns = l->B * (l->n_neg + 1);
        DEVSTAT* d = (DEVSTAT*) l->devstat;

        freemem(l->touched);
        l->touched = (int*) allocmem(ns,1,int);

        freemem(d->hsamples);
        cufreemem(d->dsamples);
        cufreemem(d->dtouched);
        d->hsamples = (int*) allocmem(ns,1,int);
        d->dsamples = (int*) cuallocmem(ns,1,int);
        d->dtouched = (int*) cuallocmem(ns,1,int);
    }
    else
        cufltclr(l->h,l->B * l->E);
    l->ntouched = 0;
}

/* Frees memory allocated by smsftmax_create() / smsftmax_init().
 * The sampling table passed to smsftmax_set_dist() is not freed.
 */
void smsftmax_free(SMSFTMAX* l)
{
    if (l == NULL) return;

    cufreemem(l->h);
    cufreemem(l->Wo);

    if (l->devstat != NULL) {
        DEVSTAT* d = (DEVSTAT*) l->devstat;
        freemem(d->hsamples);
        cufreemem(d->dsamples);
        cufreemem(d->dtouched);
        cufreemem(d->dlogq);
        cufreemem(d->dloss);
        cufreemem(d->dcorrect);
        cufreemem(d->dlogits);
        freemem(d);
    }

    freemem(l->touched);
    freemem(l->seen);
    freemem(l->logq);
    freemem(l);
}

/* Resets any state the layer carries across batches. */
void smsftmax_reset(SMSFTMAX* l)
{
    (void) l;
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
                    const fArr2D restrict h,
                    const fArr2D restrict labels,
                    fArr2D restrict gWo,
                    fArr2D restrict dh,
                    int cnt, int* correct)
{
    const int E = l->E;
    const int K = l->K;
    const int k = l->n_neg;
    DEVSTAT* d = (DEVSTAT*) l->devstat;

    if (l->dist == NULL || l->logq == NULL) {
        fflush(stdout);
        fprintf(stderr,"smsftmax_loss: sampling table not set "
                "(call smsftmax_set_dist)\n");
        exit(-1);
    }

    l->stamp++;
    l->ntouched = 0;
    cufltclr(dh,cnt * E);

    for (int i = 0; i < cnt; i++) {
        int target = (int)((const float*) labels)[i];
        int base = i * (k + 1);

        if (target <= 0 || target >= K) {
            d->hsamples[base] = -1;
            continue;
        }

        float qtarget = expf(l->logq[target]);
        if (qtarget >= 1.0f) {
            fflush(stdout);
            fprintf(stderr,"smsftmax_loss: sampling distribution contains "
                    "only target %d\n",target);
            exit(-1);
        }

        d->hsamples[base] = target;
        prep_row(l,target);

        /* Ref. #1, Sec. 4: obtain Q' by sampling from Q and discarding
         * the positive item when it appears.
         */
        for (int c = 0; c < k; c++) {
            int neg;
            do {
                neg = l->dist[(int)urand(0,l->dist_size)];
            } while (neg == target);

            d->hsamples[base + c + 1] = neg;
            prep_row(l,neg);
        }
    }

    cudaMemcpy(d->dsamples,d->hsamples,
               cnt * (k + 1) * sizeof(int),cudaMemcpyHostToDevice);
    if (l->ntouched > 0) {
        cudaMemcpy(d->dtouched,l->touched,
                   l->ntouched * sizeof(int),cudaMemcpyHostToDevice);

        int n = l->ntouched * E;
        int blocks = (n + THDCNT - 1) / THDCNT;
        clear_rows_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
                              (float*)gWo,d->dtouched,l->ntouched,E);
    }

    cufltclr(d->dloss,1);
    if (correct != NULL)
        cudaMemset(d->dcorrect,0,sizeof(int));

    size_t shm = (THDCNT + 2 * k) * sizeof(float);
    smsftmax_loss_kernel<<<cnt,THDCNT,shm,cuda_stream()>>>(
        (const float*)h,d->dsamples,(float*)dh,
        (const float*)l->Wo,(float*)gWo,d->dlogq,
        E,k,cnt,d->dloss,(correct != NULL) ? d->dcorrect : NULL);

    float loss = 0.0f;
    cudaMemcpy(&loss,d->dloss,sizeof(float),cudaMemcpyDeviceToHost);

    if (correct != NULL) {
        int c = 0;
        cudaMemcpy(&c,d->dcorrect,sizeof(int),cudaMemcpyDeviceToHost);
        *correct += c;
    }

    return loss;
}

void smsftmax_update(SMSFTMAX* restrict l, fArr2D gWo,
                     float learning_rate, float weight_decay)
{
    if (l->ntouched == 0)
        return;

    DEVSTAT* d = (DEVSTAT*) l->devstat;
    int n = l->ntouched * l->E;
    int blocks = (n + THDCNT - 1) / THDCNT;
    smsftmax_update_kernel<<<blocks,THDCNT,0,cuda_stream()>>>(
        (float*)l->Wo,(const float*)gWo,d->dtouched,l->ntouched,l->E,
        learning_rate,weight_decay);
}

/* Full-vocabulary scoring is used by validation/generation, whose caller
 * consumes logits on the CPU. Compute on the GPU and copy the result back.
 */
void smsftmax_logits(SMSFTMAX* restrict l,
                     const fArr2D restrict h,
                     fArr2D restrict logits,
                     int cnt)
{
    DEVSTAT* d = (DEVSTAT*) l->devstat;

    if (cnt > d->logits_rows) {
        cufreemem(d->dlogits);
        d->dlogits = (float*) cuallocmem(cnt,l->K,float);
        d->logits_rows = cnt;
    }

    matmulT((fArr2D)d->dlogits,h,l->Wo,cnt,l->E,l->K);
    cufltcpy(logits,d->dlogits,cnt * l->K);
}

/* Full-vocabulary evaluation for validation. Scores the first 'cnt' rows of
 * h against all of Wo on the GPU and reduces each row on the GPU; only the
 * summed NLL and the hit count are copied back.
 */
float smsftmax_eval(SMSFTMAX* restrict l,
                    const fArr2D restrict h,
                    const fArr2D restrict labels,
                    int cnt, int* correct)
{
    DEVSTAT* d = (DEVSTAT*) l->devstat;
    const int K = l->K;

    if (cnt <= 0)
        return 0.0f;
    if (cnt > l->B) {
        fflush(stdout);
        fprintf(stderr,"smsftmax_eval: cnt %d exceeds batch size %d\n",
                cnt,l->B);
        exit(-1);
    }

    if (cnt > d->logits_rows) {
        cufreemem(d->dlogits);
        d->dlogits = (float*) cuallocmem(cnt,K,float);
        d->logits_rows = cnt;
    }

    /* Targets as ints; hsamples/dsamples hold >= B*(n_neg+1) entries. */
    const float* lab = (const float*) labels;
    for (int i = 0; i < cnt; i++)
        d->hsamples[i] = (int) lab[i];
    cudaMemcpy(d->dsamples,d->hsamples,cnt * sizeof(int),
               cudaMemcpyHostToDevice);

    matmulT((fArr2D)d->dlogits,h,l->Wo,cnt,l->E,K);

    cufltclr(d->dloss,1);
    cudaMemset(d->dcorrect,0,sizeof(int));
    smsftmax_eval_kernel<<<cnt,THDCNT,0,cuda_stream()>>>(
        d->dlogits,d->dsamples,K,cnt,d->dloss,d->dcorrect);

    float nll = 0.0f;
    cudaMemcpy(&nll,d->dloss,sizeof(float),cudaMemcpyDeviceToHost);
    if (correct != NULL) {
        int c = 0;
        cudaMemcpy(&c,d->dcorrect,sizeof(int),cudaMemcpyDeviceToHost);
        *correct += c;
    }
    return nll;
}
