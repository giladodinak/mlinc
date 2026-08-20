/* Copyright (c) 2023-2024 Gilad Odinak */
/* Connectionist Temporal Classification functions */
/* References:
 * https://www.cs.toronto.edu/~graves/icml_2006.pdf
 * https://www.cs.toronto.edu/~graves/phd.pdf
 */
#include <math.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "onehot.h"
#include "editdist.h"
#include "ctc.h"

/* CTC loss runs in log-space and depends on IEEE semantics that
 * -ffast-math discards. This pragma restores these semantics.
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-finite-math-only", "no-unsafe-math-optimizations")
#endif

/* Creates a Contectionist Temporal Classification loss calculator.
 * 
 * Parameters:
 *  T     - Number of time steps (number of input vectors)
 *  L     - Number of distinct labels (input vectors dimension), including blank
 *  blank - The index of the blank label (0 .. L-1)
 */
CTC* ctc_create(int T, int L, int blank)
{
    CTC* ctc = allocmem(1,1,CTC);
    ctc->T = T;
    ctc->L = L;
    ctc->S = 2 * T + 1;
    ctc->blank = blank;
    ctc->yp = allocmem(T,L,float);
    ctc->ypc = allocmem(T,1,int);
    ctc->ytc = allocmem(T,1,int);
    ctc->label = allocmem(ctc->S,1,int);
    ctc->alpha = allocmem(T,ctc->S,float);
    ctc->beta = allocmem(T,ctc->S,float);
    ctc->prob = allocmem(T,L,float);
    return ctc;
}

/* Frees memory allocated by ctc_create()
 */
void ctc_free(CTC* ctc)
{
    freemem(ctc->yp);
    freemem(ctc->ypc);
    freemem(ctc->ytc);
    freemem(ctc->label);
    freemem(ctc->alpha);
    freemem(ctc->beta);
    freemem(ctc->prob);
    freemem(ctc);
}

/* Equation 7.18 */
static inline float logsumexp(float a, float b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    return (a >= b) ? a + log1pf(expf(b - a)) : b + log1pf(expf(a - b));
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
float ctc_loss(CTC* ctc, const fArr2D yp_/*[T][L]*/,
                         const fArr2D yt_/*[T][L]*/,
                         int T, int L)
{
    const int blank = ctc->blank;
    typedef float (*ArrTL)[L];
    ArrTL yp = (ArrTL) ctc->yp;
    iVec ypc = ctc->ypc;
    iVec ytc = ctc->ytc;
    iVec label = ctc->label;
    int i, j, t, s, S;

    if (T == 0)
        return INFINITY;

    /* Convert input predictions to log scale and store in ctc */
    fltcpy(yp,yp_,T * L);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < L; j++)
            yp[i][j] = log(yp[i][j]);

    onehot_decode(yp_,ypc,T,L);    /* Convert input predictions to labels */
    for (i = 1, j = 0; i < T; i++) /* Merge consecutive identical labels  */
        if (ypc[i] != ypc[j])
            ypc[++j] = ypc[i];
    for (s = j + 1, i = 0, j = 0; i < s; i++) /* Remove blanks */
        if (ypc[i] != blank)
            ypc[j++] = ypc[i];
    ctc->ypclen = j;

    onehot_decode(yt_,ytc,T,L);    /* Convert true vectors to labels      */
    for (i = 1, j = 0; i < T; i++) /* Merge consecutive identical labels  */
        if (ytc[i] != ytc[j])
            ytc[++j] = ytc[i];
    for (s = j + 1, i = 0, j = 0; i < s; i++) /* Remove blanks */
        if (ytc[i] != blank)
            ytc[j++] = ytc[i];
    ctc->ytclen = j;

    /* Create padded label, insert one blank between labels and at the ends */
    label[0] = blank;
    for (i = 0, s = 1; i < j && s < 2 * T + 1; i++) {
        label[s++] = ytc[i];
        label[s++] = blank;
    }
    ctc->S = S = s; /* Actual padded label length */

    typedef float (*ArrTS)[S];
    ArrTS alpha = (ArrTS) ctc->alpha;
    ArrTS beta = (ArrTS) ctc->beta;
    for (t = 0; t < T; t++)
        for (s = 0; s < S; s++)
            alpha[t][s] = beta[t][s] = -INFINITY;

    /* Note that the referenced equations use 1-based indexing, 
     * while C arrays are 0-based; the below code is adjusted accordingly
     */

    /* Initialize the alpha table */
    alpha[0][0] = yp[0][blank];             /* Equation 7.5 */
    if (S > 1)
        alpha[0][1] = yp[0][label[1]];      /* Equation 7.6 */
    else
        alpha[0][1] = yp[0][blank];
    for (s = 2; s < S; s++)                 /* Equation 7.7 */
        alpha[0][s] = -INFINITY;
    for (t = 1; t < T; t++) {               /* Equation 7.8 */
        int start = S - (2 * (T - t));      /* Equation 7.9 */
        if (start < 0) start = 0;
        int end = 2 * (t + 1);
        if (end > S) end = S;
        for (s = 0; s < start; s++)         /* Equation 7.9 */
            alpha[t][s] = -INFINITY;
        for (s = start; s < end; s++) {
            int ls = label[s];
            float ats = alpha[t - 1][s];
            if (s >= 1)
                ats = logsumexp(ats,alpha[t - 1][s - 1]);
            if (s >= 2 && !(ls == blank || label[s - 2] == ls))
                ats = logsumexp(ats,alpha[t - 1][s - 2]);
            alpha[t][s] = ats + yp[t][ls];
        }
    }

    /* Initialize the beta table */
    beta[T - 1][S - 1] = 0;                 /* Equation 7.12 */
    if (S > 1)
        beta[T - 1][S - 2] = 0;             /* Equation 7.13 */
    for (s = S - 3; s >= 0; s--)            /* Equation 7.14 */
        beta[T - 1][s] = -INFINITY;
    for (t = T - 2; t >= 0; t--) {          /* Equation 7.15 */
        int start = S - (2 * (T - t));
        if (start < 0) start = 0;
        int end = 2 * (t + 1);              /* Equation 7.16 */
        if (end > S) end = S;
        for (s = end; s < S; s++)           /* Equation 7.16 */
            beta[t][s] = -INFINITY;
        for (s = start; s < end; s++) {
            /* Equation 7.15 has a typo, reads yp[t][] instead of yp[t + 1][] */
            float bts = beta[t + 1][s] + yp[t + 1][label[s]];
            if (s + 1 < S)
                bts = logsumexp(bts,beta[t + 1][s + 1] + yp[t + 1][label[s + 1]]);
            if (s + 2 < S && !(label[s] == blank || label[s + 2] == label[s]))
                bts = logsumexp(bts,beta[t + 1][s + 2] + yp[t + 1][label[s + 2]]);
            beta[t][s] = bts;
        }
    }

    /* Calculate probabilities per time step (see note below Equation 7.29) */
    for (t = 0; t < T; t++) {
        float prob = -INFINITY;
        for (s = 0; s < S; s++) /* Equation 7.23 */
            prob = logsumexp(prob,alpha[t][s] + beta[t][s]);
        ctc->prob[t] = prob;
    }
    float loss = 0;
    for (t = 0; t < T; t++)
        loss += -ctc->prob[t];
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
void dLdy_ctc_loss(CTC* ctc, const fArr2D yp_/*[T][L]*/,
                             const fArr2D yt_/*[T][L]*/,
                             fArr2D dy_/*[T][L]*/,
                             int T, int L)

{
    (void) yp_;
    (void) yt_;
    const int S = ctc->S;
    typedef float (*ArrTL)[L];
    ArrTL yp = (ArrTL) ctc->yp;
    ArrTL dy = (ArrTL) dy_;
    int t, s, l;

    iVec label = ctc->label;
    typedef float (*ArrTS)[S];
    ArrTS alpha = (ArrTS) ctc->alpha;
    ArrTS beta = (ArrTS) ctc->beta;

    for (t = 0; t < T; t++) {
        for (l = 0; l < L; l++) {
            float sum = -INFINITY;
            for (s = 0; s < S; s++)
                if (l == label[s])  /* Equation 7.24 */
                    sum = logsumexp(sum,alpha[t][s] + beta[t][s]);
            /* Equation 7.29 */
            dy[t][l] = expf(yp[t][l]) - expf(sum - ctc->prob[t]);
        }
    }
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
float ctc_accuracy(CTC* ctc, const fArr2D yp_/*[T][L]*/,
                   const fArr2D yt_/*[T][L]*/, 
                   int T, int L)
{
    (void) yp_;
    (void) yt_;
    (void) L;
    float fact = ((ctc->ypclen > ctc->ytclen) ? ctc->ypclen : ctc->ytclen);
    if (fact == 0) return T;
    float dist = edit_dist(ctc->ypc,ctc->ypclen,ctc->ytc,ctc->ytclen);
    return (1.0 - (dist / fact)) * T;
}
