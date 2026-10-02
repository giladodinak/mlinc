/* Copyright (c) 2026 Gilad Odinak */
/* Connectionist Temporal Classification functions */
/* References:
 * https://www.cs.toronto.edu/~graves/icml_2006.pdf
 * https://www.cs.toronto.edu/~graves/phd.pdf
 */
#include <math.h>
#include <cuda_runtime.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "editdist.h"
#include "ctc.h"

#define THCNT 256


/* Equation 7.18 */
static __device__ __forceinline__ float logsumexp(float a, float b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    return (a >= b) ? a + log1pf(expf(b - a))
                    : b + log1pf(expf(a - b));
}


/* Converts input predictions to log scale and decodes the predicted
 * and true vectors to label indices.
 *
 * The device label buffer is temporarily used as:
 *
 *     [ypc[T] | ytc[T]]
 *
 * It is subsequently overwritten by the padded true-label sequence.
 */
static __global__ void ctc_decode_kernel(const float* yp,
                                         const float* yt,
                                         float* logyp,
                                         int* label,
                                         int T, int L)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        int k = t * L;
        int ypcls = 0;
        int ytcls = 0;
        float ypmax = yp[k];
        float ytmax = yt[k];

        for (int l = 0; l < L; l++) {
            float p = yp[k + l];
            float y = yt[k + l];

            logyp[k + l] = logf(p);

            if (p > ypmax) {
                ypmax = p;
                ypcls = l;
            }
            if (y > ytmax) {
                ytmax = y;
                ytcls = l;
            }
        }

        label[t] = ypcls;
        label[T + t] = ytcls;
    }
}


/* Computes the alpha and beta tables.
 *
 * The time dimension is sequential. Threads in one block operate in
 * parallel over the padded-label dimension.
 */
static __global__ void ctc_forward_backward_kernel(const float* yp,
                                                    const int* label,
                                                    float* alpha,
                                                    float* beta,
                                                    int T, int L,
                                                    int S, int blank)
{
    int tid = threadIdx.x;

    for (int i = tid; i < T * S; i += blockDim.x) {
        alpha[i] = -INFINITY;
        beta[i] = -INFINITY;
    }
    __syncthreads();

    /* Initialize the alpha table */
    if (tid == 0)
        alpha[0] = yp[blank];                         /* Equation 7.5 */
    if (S > 1 && tid == 1)
        alpha[1] = yp[label[1]];                     /* Equation 7.6 */

    __syncthreads();

    for (int t = 1; t < T; t++) {                   /* Equation 7.8 */
        int start = S - (2 * (T - t));              /* Equation 7.9 */
        if (start < 0)
            start = 0;

        int end = 2 * (t + 1);
        if (end > S)
            end = S;

        for (int s = start + tid; s < end; s += blockDim.x) {
            int ls = label[s];
            float ats = alpha[(t - 1) * S + s];

            if (s >= 1)
                ats = logsumexp(
                    ats,
                    alpha[(t - 1) * S + s - 1]);

            if (s >= 2 && !(ls == blank || label[s - 2] == ls))
                ats = logsumexp(
                    ats,
                    alpha[(t - 1) * S + s - 2]);

            alpha[t * S + s] = ats + yp[t * L + ls];
        }

        __syncthreads();
    }

    /* Initialize the beta table */
    if (tid == 0)
        beta[(T - 1) * S + S - 1] = 0.0f;           /* Equation 7.12 */

    if (S > 1 && tid == 1)
        beta[(T - 1) * S + S - 2] = 0.0f;           /* Equation 7.13 */

    __syncthreads();

    for (int t = T - 2; t >= 0; t--) {              /* Equation 7.15 */
        int start = S - (2 * (T - t));
        if (start < 0)
            start = 0;

        int end = 2 * (t + 1);                      /* Equation 7.16 */
        if (end > S)
            end = S;

        for (int s = start + tid; s < end; s += blockDim.x) {
            /* Equation 7.15 has a typo, reads yp[t][] instead
             * of yp[t + 1][]
             */
            float bts = beta[(t + 1) * S + s]
                      + yp[(t + 1) * L + label[s]];

            if (s + 1 < S)
                bts = logsumexp(
                    bts,
                    beta[(t + 1) * S + s + 1]
                    + yp[(t + 1) * L + label[s + 1]]);

            if (s + 2 < S &&
                    !(label[s] == blank || label[s + 2] == label[s]))
                bts = logsumexp(
                    bts,
                    beta[(t + 1) * S + s + 2]
                    + yp[(t + 1) * L + label[s + 2]]);

            beta[t * S + s] = bts;
        }

        __syncthreads();
    }
}


/* Calculates probabilities per time step
 * (see note below Equation 7.29).
 */
static __global__ void ctc_prob_kernel(const float* alpha,
                                       const float* beta,
                                       float* prob,
                                       int T, int S)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        float p = -INFINITY;

        for (int s = 0; s < S; s++)                 /* Equation 7.23 */
            p = logsumexp(
                p,
                alpha[t * S + s] + beta[t * S + s]);

        prob[t] = p;
    }
}


/* Equation 7.29 */
static __global__ void dLdy_ctc_loss_kernel(const float* yp,
                                            const int* label,
                                            const float* alpha,
                                            const float* beta,
                                            const float* prob,
                                            float* dy,
                                            int T, int L, int S)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < T * L) {
        int t = k / L;
        int l = k % L;
        float sum = -INFINITY;

        for (int s = 0; s < S; s++)
            if (l == label[s])                      /* Equation 7.24 */
                sum = logsumexp(
                    sum,
                    alpha[t * S + s] + beta[t * S + s]);

        dy[k] = expf(yp[k]) - expf(sum - prob[t]);
    }
}


/* Merges consecutive identical labels and removes blanks. */
static int collapse_labels(int* y, int T, int blank)
{
    int i, j, n;

    if (T == 0)
        return 0;

    for (i = 1, j = 0; i < T; i++)
        if (y[i] != y[j])
            y[++j] = y[i];

    for (n = j + 1, i = 0, j = 0; i < n; i++)
        if (y[i] != blank)
            y[j++] = y[i];

    return j;
}


/* Creates a Contectionist Temporal Classification loss calculator.
 * 
 * Parameters:
 *  T     - Number of time steps (number of input vectors)
 *  L     - Number of distinct labels (input vectors dimension), including blank
 *  blank - The index of the blank label (0 .. L-1)
 */
extern "C" CTC* ctc_create(int T, int L, int blank)
{
    CTC* ctc = (CTC*) allocmem(1,1,CTC);
    ctc->T = T;
    ctc->L = L;
    ctc->S = 2 * T + 1;
    ctc->blank = blank;
    ctc->yp = (fArr2D) cuallocmem(T,L,float);
    ctc->ypc = (iVec) allocmem(T,1,int);
    ctc->ytc = (iVec) allocmem(T,1,int);
    ctc->label = (iVec) cuallocmem(ctc->S,1,int);
    ctc->alpha = (fArr2D) cuallocmem(T,ctc->S,float);
    ctc->beta = (fArr2D) cuallocmem(T,ctc->S,float);
    ctc->prob = (fVec) cuallocmem(T,1,float);
    return ctc;
}


/* Frees memory allocated by ctc_create()
 */
extern "C" void ctc_free(CTC* ctc)
{
    cufreemem(ctc->yp);
    freemem(ctc->ypc);
    freemem(ctc->ytc);
    cufreemem(ctc->label);
    cufreemem(ctc->alpha);
    cufreemem(ctc->beta);
    cufreemem(ctc->prob);
    freemem(ctc);
}


/* Calculates ctc loss for a batch of probability vectors
 * and correspoding class labels.
 *
 * Parameters:
 *   yp - array of T vectors, each having L class probabilites
 *   yt - array of up to T one-hot encoded labels. If there are
 *        less than T labels, pad with blank labels to the end.
 *
 * Returns:
 *   ctc loss value
 *
 * Notes:
 * 1. ctc_loss stores yp, converted to log scale, to be used by dLdy_ctc_loss
 * 2. It merges consecutive identical labels, and removes blanks
 *    a a a b b c dd -> a b c d ;  a a ^ a b b c dd -> a a b c d ( ^ == blank)
 * 3. It stores the predicted and true labels, to be used by ctc_match_sum
 *
 * 4. This function merges repeated identical labels, so instead of 
 *    padding with blank labels at the end, labels can be duplicated.
 *    For example, assuming blank label index is 0, and there are six
 *    time steps, the (one hot encoded) labels 5 1 2 can be encoded 
 *    as either 5 1 2 0 0 0 or 5 5 1 2 2 0 or 5 5 1 1 2 2.  If the 
 *    alignment is known, the later approch yields better results
 *    when batch size is smaller than sequence length.
 */
extern "C" float ctc_loss(CTC* ctc,
                          const fArr2D yp_/*[T][L]*/,
                          const fArr2D yt_/*[T][L]*/,
                          int T, int L)
{
    const int blank = ctc->blank;
    int i, s;

    if (T == 0)
        return INFINITY;

    /* Convert input predictions to log scale and decode both prediction
     * and true-label vectors.
     *
     * ctc->label is temporarily:
     *     [ypc[T] | ytc[T]]
     */
    int blocks = (T + THCNT - 1) / THCNT;

    ctc_decode_kernel<<<blocks,THCNT>>>(
        (const float*)yp_,
        (const float*)yt_,
        (float*)ctc->yp,
        (int*)ctc->label,
        T,L);

    cudaMemcpy(ctc->ypc,
               ctc->label,
               T * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(ctc->ytc,
               ctc->label + T,
               T * sizeof(int),
               cudaMemcpyDeviceToHost);

    ctc->ypclen = collapse_labels(ctc->ypc,T,blank);
    ctc->ytclen = collapse_labels(ctc->ytc,T,blank);

    /* Create padded label, insert one blank between labels and at the ends */
    int* label = (int*) allocmem(2 * T + 1,1,int);

    label[0] = blank;
    for (i = 0, s = 1; i < ctc->ytclen && s < 2 * T + 1; i++) {
        label[s++] = ctc->ytc[i];
        label[s++] = blank;
    }

    ctc->S = s;                                     /* Actual padded label length */

    cudaMemcpy(ctc->label,
               label,
               ctc->S * sizeof(int),
               cudaMemcpyHostToDevice);

    freemem(label);

    /* Note that the referenced equations use 1-based indexing,
     * while C arrays are 0-based; the below code is adjusted accordingly.
     */
    ctc_forward_backward_kernel<<<1,THCNT>>>(
        (const float*)ctc->yp,
        (const int*)ctc->label,
        (float*)ctc->alpha,
        (float*)ctc->beta,
        T,L,ctc->S,blank);

    blocks = (T + THCNT - 1) / THCNT;

    ctc_prob_kernel<<<blocks,THCNT>>>(
        (const float*)ctc->alpha,
        (const float*)ctc->beta,
        (float*)ctc->prob,
        T,ctc->S);

    float* prob = (float*) allocmem(T,1,float);

    cudaMemcpy(prob,
               ctc->prob,
               T * sizeof(float),
               cudaMemcpyDeviceToHost);

    float loss = 0;
    for (int t = 0; t < T; t++)
        loss += -prob[t];

    freemem(prob);

    return loss / T;
}


/* Calculates the gradient of the ctc loss with respect to prediction (dL/dy)
 * for predicted vectors and their true labels.
 *
 * ctc_loss() must be called before calling this function, as it builds
 * the padded true labels vector and calculates the values of alpha table
 * used by this function.
 *
 * Parameters:
 *   yp - array of T vectors, each having L class probabilites
 *   yt - array of up to T one-hot encoded labels. If there are
 *        less than T labels, blank labels pad the time steps
 *        between them.
 *
 * Returns:
 *   dy - array of T vectors, each having L gradients (output)
 *
 * Note tha the passed in yp is not used. Instead the function uses
 * the log scale values of yp calculated by ctc_loss and stored in ctc.
 * Similarily it does not used the apssed in yt, instead using the
 * padded lables created by ctc_loss and stored in ctc.
 */
extern "C" void dLdy_ctc_loss(CTC* ctc,
                              const fArr2D yp_/*[T][L]*/,
                              const fArr2D yt_/*[T][L]*/,
                              fArr2D dy_/*[T][L]*/,
                              int T, int L)
{
    (void) yp_;
    (void) yt_;

    int n = T * L;
    int blocks = (n + THCNT - 1) / THCNT;

    dLdy_ctc_loss_kernel<<<blocks,THCNT>>>(
        (const float*)ctc->yp,
        (const int*)ctc->label,
        (const float*)ctc->alpha,
        (const float*)ctc->beta,
        (const float*)ctc->prob,
        (float*)dy_,
        T,L,ctc->S);
}


/* Calculates the accuracy factor between predicted classes and true labels.
 * 
 * This function calculates the numerator of the accuracy factor between T 
 * predicted classes and T true labels. The result is a value between 0 and T, 
 * where 0 indicates no match and T indicates a perfect match.
 * This numer is propotional to the edit distance of the two label sequences
 * after consequitve identical labels are merged.
 * 
 * Parameters:
 *   yp - Predicted classes array of size TxL
 *   yt - One-hot encdoed true labels array of size TxL
 *   T  - Number of samples (rows)
 *   L  - Number of classes (columns)
 * 
 * Note:
 *   This function uses the predicted and true labels calculated by ctc_loss,
 *   and ignores the passed in yp, yt arrays.
 *
 * Returns:
 *   Accuracy factor numerator value
 */
extern "C" float ctc_accuracy(CTC* ctc,
                              const fArr2D yp_/*[T][L]*/,
                              const fArr2D yt_/*[T][L]*/, 
                              int T, int L)
{
    (void) yp_;
    (void) yt_;
    (void) L;

    float fact = ((ctc->ypclen > ctc->ytclen)
                    ? ctc->ypclen : ctc->ytclen);

    if (fact == 0)
        return T;

    float dist = edit_dist(
        ctc->ypc,ctc->ypclen,
        ctc->ytc,ctc->ytclen);

    return (1.0f - (dist / fact)) * T;
}
