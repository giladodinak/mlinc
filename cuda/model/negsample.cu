/* Copyright (c) 2026 Gilad Odinak                */
/* Negative-sampling output layer layer functions */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "mem.h"
#include "cumem.h"
#include "cuflt.h"
#include "array.h"
#include "random.h"
#include "activation.h"
#include "negsample.h"

#define THCNT 256

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
    cufltclr(&((float*)gWo_)[row * l->E],l->E);
    l->touched[l->ntouched++] = row;
}

static __global__ void negsample_loss_kernel(const float* h,
                                              const int* samples,
                                              float* dh,
                                              const float* Wo,
                                              float* gWo,
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

    extern __shared__ float s[];
    __shared__ float grad;
    __shared__ float pos_dot;
    __shared__ int beaten;

    float dot = 0.0f;
    for (int j = tid; j < E; j += blockDim.x)
        dot += Wo[target * E + j] * h[i * E + j];

    s[tid] = dot;
    __syncthreads();
    for (int n = blockDim.x / 2; n > 0; n >>= 1) {
        if (tid < n)
            s[tid] += s[tid + n];
        __syncthreads();
    }
    dot = s[0];

    if (tid == 0) {
        float p = sigmoid1(dot);
        atomicAdd(loss,-logf(p + 1e-8f));
        grad = p - 1.0f;
        pos_dot = dot;
        beaten = 1;
    }
    __syncthreads();

    for (int j = tid; j < E; j += blockDim.x) {
        atomicAdd(&gWo[target * E + j],grad * h[i * E + j]);
        dh[i * E + j] += grad * Wo[target * E + j];
    }
    __syncthreads();

    for (int k = 0; k < n_neg; k++) {
        int neg = samples[i * (n_neg + 1) + k + 1];

        dot = 0.0f;
        for (int j = tid; j < E; j += blockDim.x)
            dot += Wo[neg * E + j] * h[i * E + j];

        s[tid] = dot;
        __syncthreads();
        for (int n = blockDim.x / 2; n > 0; n >>= 1) {
            if (tid < n)
                s[tid] += s[tid + n];
            __syncthreads();
        }
        dot = s[0];

        if (tid == 0) {
            float p = sigmoid1(-dot);
            atomicAdd(loss,-logf(p + 1e-8f));
            grad = 1.0f - p;
            if (dot >= pos_dot)
                beaten = 0;
        }
        __syncthreads();

        for (int j = tid; j < E; j += blockDim.x) {
            atomicAdd(&gWo[neg * E + j],grad * h[i * E + j]);
            dh[i * E + j] += grad * Wo[neg * E + j];
        }
        __syncthreads();
    }

    if (tid == 0 && correct != NULL && beaten)
        atomicAdd(correct,1);
}

static __global__ void negsample_update_kernel(float* Wo,
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
        int i = touched[r];
        int p = i * E + j;
        Wo[p] -= learning_rate * (gWo[p] + weight_decay * Wo[p]);
    }
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
    NEGSAMPLE* l = (NEGSAMPLE*) allocmem(1,1,NEGSAMPLE);
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
 *   training   - 1: allocate backward/gradient buffers, 0 for inference-only
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(E).
 */
void negsample_init(NEGSAMPLE* l, int input_dim, int batch_size, int training)
{
    l->E = input_dim;
    l->B = batch_size;
    l->h = (fArr2D) cuallocmem(l->B,l->E,float);
    l->Wo = (fArr2D) cuallocmem(l->K,l->E,float);

    training = (training) ? 1 : 0;
    l->training = training;

    if (training) {
        l->gWo = (fArr2D) cuallocmem(l->K,l->E,float);
        l->touched = (int*) allocmem(l->B * (l->n_neg + 1),1,int);
        l->ntouched = 0;
        l->seen = (int*) allocmem(l->K,1,int);
        for (int i = 0; i < l->K; i++)
            l->seen[i] = -1;
    }

    l->stamp = 0;
    l->dist = NULL;
    l->dist_size = 0;

    float* Wo = (float*) allocmem(l->K,l->E,float);
    float scale = 1.0 / sqrtf((float) l->E);
    for (int i = 0; i < l->K * l->E; i++)
        Wo[i] = nrand(0.0,scale);
    cufltcpy(l->Wo,Wo,l->K * l->E);
    freemem(Wo);
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
 */
void negsample_set_batch_size(NEGSAMPLE* l, int batch_size)
{
    if (l->B == 0)
        return;
    if (batch_size != l->B) {
        l->B = batch_size;
        cufreemem(l->h);
        l->h = (fArr2D) cuallocmem(l->B,l->E,float);
        if (l->training) {
            freemem(l->touched);
            l->touched = (int*) allocmem(l->B * (l->n_neg + 1),1,int);
        }
    }
    else
        cufltclr(l->h,l->B * l->E);
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
    cufreemem(l->h);
    cufreemem(l->Wo);
    cufreemem(l->gWo);
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
                     const fArr2D restrict h_/*[B][E]*/,
                     const fArr2D restrict labels_/*[B][1]*/,
                     fArr2D restrict dh_/*[B][E]*/,
                     int cnt, int* correct)
{
    if (!l->training) return 1e30;
    const int E = l->E;
    const int K = l->K;
    const int n_neg = l->n_neg;

    if (l->dist == NULL) {
        fflush(stdout);
        fprintf(stderr,"negsample_loss: sampling table not set "
                "(call negsample_set_dist)\n");
        exit(-1);
    }

    l->stamp++;
    l->ntouched = 0;
    cufltclr(dh_,cnt * E);

    float* labels = (float*) allocmem(cnt,1,float);
    cufltcpy(labels,labels_,cnt);
    int* samples = (int*) allocmem(cnt * (n_neg + 1),1,int);

    for (int i = 0; i < cnt; i++) {
        int target = (int) labels[i];
        if (target <= 0 || target >= K) {
            samples[i * (n_neg + 1)] = -1;
            continue;
        }

        samples[i * (n_neg + 1)] = target;
        prep_row(l,l->gWo,target);

        for (int k = 0; k < n_neg;) {
            int neg = l->dist[(int) urand(0,l->dist_size)];
            if (neg == target)
                continue;
            samples[i * (n_neg + 1) + k + 1] = neg;
            prep_row(l,l->gWo,neg);
            k++;
        }
    }

    int* dsamples = (int*) cuallocmem(cnt * (n_neg + 1),1,int);
    cudaMemcpy(dsamples,samples,cnt * (n_neg + 1) * sizeof(int),cudaMemcpyHostToDevice);

    float* dloss = (float*) cuallocmem(1,1,float);
    int* dcorrect = NULL;
    if (correct != NULL)
        dcorrect = (int*) cuallocmem(1,1,int);

    negsample_loss_kernel<<<cnt,THCNT,THCNT * sizeof(float)>>>(
        (const float*)h_,dsamples,(float*)dh_,
        (const float*)l->Wo,(float*)l->gWo,
        E,n_neg,cnt,dloss,dcorrect);

    float loss = 0.0f;
    cudaMemcpy(&loss,dloss,sizeof(float),cudaMemcpyDeviceToHost);
    if (correct != NULL) {
        int c = 0;
        cudaMemcpy(&c,dcorrect,sizeof(int),cudaMemcpyDeviceToHost);
        *correct += c;
    }

    cufreemem(dsamples);
    cufreemem(dloss);
    cufreemem(dcorrect);
    freemem(samples);
    freemem(labels);
    return loss;
}

/* Applies one sparse SGD step to the rows of Wo touched by the last
 * negsample_loss(), with weight decay.
 *
 * Parameters:
 *   l             - Pointer to the layer
 *   learning_rate - Gradient multiplier
 *   weight_decay  - Weight magnitude suppressor (0 to disable)
 */
void negsample_update(NEGSAMPLE* restrict l,
                      float learning_rate, float weight_decay)
{
    if (!l->training || l->ntouched == 0) return;
    const int E = l->E;

    int* touched = (int*) cuallocmem(l->ntouched,1,int);
    cudaMemcpy(touched,l->touched,l->ntouched * sizeof(int),cudaMemcpyHostToDevice);

    int n = l->ntouched * E;
    negsample_update_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
        (float*)l->Wo,(const float*)l->gWo,touched,l->ntouched,E,
        learning_rate,weight_decay);

    cufreemem(touched);
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
