/* Copyright (c) 2023-2024 Gilad Odinak */
/* Beam search decoders                 */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "beamsrch.h"

struct can_seq_s {
    int* seq;
    float score;
};

static inline int can_seq_cmp(const void* a, const void* b)
{
    float as = ((const struct can_seq_s*) a)->score;
    float bs = ((const struct can_seq_s*) b)->score;
    return (as > bs) - (as < bs);
}

/*  Performs beam search decoding on a set of probabilities over time steps.
 *
 * Parameters:
 *   probabilities - A 2D array [T][C] of probabilities, where T is the number
 *                   of time steps and C is the number of classes (symbols).
 *   T             - The number of time steps.
 *   C             - The number of classes (symbols) at each time step.
 *   beam_width    - The beam width, i.e., the number of sequences to keep
 *                   at each step.
 *   sequences     - A 2D array [B][T+1] to store the resulting sequences,
 *                   where B is the beam width.
 *   scores        - A 1D array [B] to store the scores of the 
 *                   resulting sequences.
 *
 * Description:
 *   This function implements beam search decoding. It iteratively builds
 *   up sequences by selecting the most likely symbols at each time step,
 *   keeping only the top "beam_width" sequences with the highest cumulative
 *   scores.
 *
 *   The function first initializes the sequences and scores. At each time
 *   step, it generates candidate sequences by extending the existing 
 *   sequences with each possible symbol.
 *   The candidates are then sorted by their scores, and the top "beam_width"
 *   sequences are selected for the next time step. The process repeats until
 *   all time steps are processed.
 *
 *   Memory is allocated dynamically for storing new sequences if the total
 *   size exceeds a certain threshold.
 */

void beam_search(fArr2D probabilities_,
                 int T, int C, int beam_width,
                 iArr2D sequences_, fVec scores_)
{
    if (T < 0 || beam_width <= 0)
        return;
    const int B = beam_width;
    typedef float (*fArrTC)[C];
    typedef int (*iArrBT1)[T + 1];
    typedef float (*fVecB);
    fArrTC probabilities = (fArrTC) probabilities_;
    iArrBT1 sequences = (iArrBT1) sequences_;
    fVecB scores = (fVecB) scores_;

    /* Use heap for really large sequences, stack otherwise */
    int useheap = (B * C) * ( T + 1) * sizeof(int) > 65536;
    int ns[(useheap) ? 1 : (B * C)][(useheap) ? 1 : (T + 1)];
    iArrBT1 new_seqs = (iArrBT1) ((useheap) ? allocmem(B * C,T + 1,int) : ns);

    struct can_seq_s can_seqs[B * C];

    for (int i = 0; i < B; i++)
        for (int j = 0; j <= T; j++)
            sequences[i][j] = 0;
    scores[0] = 0.0;
    int num_sequences = 1;

    for (int t = 0; t < T; ++t) {
        int num_can = 0;
        for (int i = 0; i < num_sequences; i++) {
            for (int c = 0; c < C; c++) {
                int* new_seq = new_seqs[num_can];
                memcpy(new_seq,sequences[i],t * sizeof(int));
                new_seq[t] = c;
                float p = fmaxf(probabilities[t][c],1e-30);
                float new_score = scores[i] - logf(p);
                can_seqs[num_can].seq = new_seq;
                can_seqs[num_can].score = new_score;
                num_can++;
            }
        }

        qsort(can_seqs,num_can,sizeof(struct can_seq_s),can_seq_cmp);

        num_sequences = (num_can < beam_width) ? num_can : beam_width;
        for (int i = 0; i < num_sequences; i++) {
            memcpy(sequences[i],can_seqs[i].seq,(t + 1) * sizeof(int));
            scores[i] = can_seqs[i].score;
        }
    }
    if (useheap)
        freemem(new_seqs);
}

/* CTC prefix beam search decoder
 *
 * Reference:
 *   A. Hannun, Sequence Modeling with CTC, Distill, 2017
 *   https://distill.pub/2017/ctc/
 */

/* CTC prefix beam search runs in log-space and depends on IEEE semantics
 * that -ffast-math discards. This pragma restores these semantics.
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-finite-math-only", "no-unsafe-math-optimizations")
#endif

/* A candidate prefix: a beam (parent), optionally extended by one symbol */
struct ctc_can_s {
    int parent;     /* Index of the beam this candidate is derived from   */
    int sym;        /* Appended symbol, or -1 if prefix is unchanged      */
    float pb;       /* Log probability of paths ending in blank           */
    float pnb;      /* Log probability of paths ending in non blank       */
    float p;        /* Log probability of all paths: log(e^pb + e^pnb)    */
};

/* Returns log(e^a + e^b) */
static inline float logsumexp(float a, float b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    return (a >= b) ? a + log1pf(expf(b - a)) : b + log1pf(expf(a - b));
}

static inline int ctc_can_cmp(const void* a, const void* b)
{
    float ap = ((const struct ctc_can_s*) a)->p;
    float bp = ((const struct ctc_can_s*) b)->p;
    return (ap < bp) - (ap > bp); /* Descending order */
}

/*  Performs CTC prefix beam search decoding on a set of probabilities
 *  over time steps. SIL is the CTC blank symbol.
 *
 * Parameters:
 *   probabilities - A 2D array [T][C] of probabilities, where T is the number
 *                   of time steps and C is the number of classes (symbols),
 *                   including the blank (SIL).
 *   T             - The number of time steps.
 *   C             - The number of classes (symbols) at each time step.
 *   beam_width    - The beam width, i.e., the number of prefixes to keep
 *                   at each step.
 *   sequences     - A 2D array [B][T+1] to store the resulting label
 *                   sequences, where B is the beam width. Sequences are
 *                   already collapsed (repeats merged, blanks removed)
 *                   and are terminated (padded) with SIL.
 *   scores        - A 1D array [B] to store the scores (negative log
 *                   probabilities) of the resulting sequences, best first.
 *                   Unused entries are set to INFINITY.
 *
 * Description:
 *   Unlike a plain beam search over frame paths, each beam is a collapsed
 *   label prefix. All frame paths that collapse to the same prefix are
 *   merged, and their probabilities summed. For each prefix two
 *   probabilities are kept: of paths ending in blank (pb) and of paths
 *   ending in a non blank symbol (pnb). At each time step, for a prefix
 *   whose last symbol is l:
 *     blank        - prefix unchanged,  pb  += (pb + pnb) * y(blank)
 *     symbol c = l - prefix unchanged,  pnb += pnb * y(c)
 *                    prefix + c,        pnb += pb * y(c)
 *     symbol c!= l - prefix + c,        pnb += (pb + pnb) * y(c)
 *   The candidates are then sorted by pb + pnb and the top "beam_width"
 *   prefixes are kept for the next time step. All calculations are done
 *   in the log domain.
 *
 *   Memory is allocated dynamically for storing the prefixes if the total
 *   size exceeds a certain threshold.
 */
void ctc_beam_search(fArr2D probabilities_,
                     int T, int C, int beam_width,
                     iArr2D sequences_, fVec scores_)
{
    if (T < 0 || beam_width <= 0)
        return;
    const int B = beam_width;
    typedef float (*fArrTC)[C];
    typedef int (*iArrBT1)[T + 1];
    typedef float (*fVecB);
    fArrTC probabilities = (fArrTC) probabilities_;
    iArrBT1 sequences = (iArrBT1) sequences_;
    fVecB scores = (fVecB) scores_;

    /* Initialize results: empty sequences, infinite (worst) scores */
    for (int i = 0; i < B; i++) {
        for (int j = 0; j <= T; j++)
            sequences[i][j] = SIL;
        scores[i] = INFINITY;
    }
    if (C <= SIL)
        return;

    /* Current and next beams prefixes. Use heap for really large prefixes,
     * stack otherwise
     */
    int useheap = 2 * B * (T + 1) * sizeof(int) > 65536;
    int cs[(useheap) ? 1 : B][(useheap) ? 1 : (T + 1)];
    int ns[(useheap) ? 1 : B][(useheap) ? 1 : (T + 1)];
    iArrBT1 cur = (iArrBT1) ((useheap) ? allocmem(B,T + 1,int) : cs);
    iArrBT1 nxt = (iArrBT1) ((useheap) ? allocmem(B,T + 1,int) : ns);
    int len[B];         /* Current beams prefixes lengths                 */
    int nlen[B];        /* Next beams prefixes lengths                    */
    float pb[B];        /* Current beams log prob. of ending in blank     */
    float pnb[B];       /* Current beams log prob. of ending in non blank */
    int extof[B];       /* Beam i is beam extof[i] + one symbol, or -1    */
    float lp[C];        /* Log probabilities of symbols at time step t    */

    /* Each beam yields at most itself + C - 1 extended prefixes */
    struct ctc_can_s cans[B * C];

    /* Start with a single empty prefix, ending in blank */
    len[0] = 0;
    pb[0] = 0.0f;
    pnb[0] = -INFINITY;
    int nb = 1;

    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; c++)
            lp[c] = logf(fmaxf(probabilities[t][c],1e-30f));

        /* Find beams that are a one symbol extension of another beam;
         * extending that other beam must merge into the existing beam.
         */
        for (int i = 0; i < nb; i++) {
            extof[i] = -1;
            for (int j = 0; j < nb && len[i] > 0; j++) {
                if (len[j] == len[i] - 1 &&
                    memcmp(cur[j],cur[i],len[j] * sizeof(int)) == 0) {
                    extof[i] = j;
                    break;
                }
            }
        }

        /* The first nb candidates are the current beams, unchanged */
        for (int i = 0; i < nb; i++) {
            cans[i].parent = i;
            cans[i].sym = -1;
            cans[i].pb = -INFINITY;
            cans[i].pnb = -INFINITY;
        }
        int nc = nb;

        for (int j = 0; j < nb; j++) {
            float p = logsumexp(pb[j],pnb[j]);
            int last = (len[j] > 0) ? cur[j][len[j] - 1] : -1;

            /* Blank: prefix unchanged, now ends in blank */
            cans[j].pb = logsumexp(cans[j].pb,p + lp[SIL]);

            for (int c = 0; c < C; c++) {
                if (c == SIL)
                    continue;
                float ext;  /* Log probability of extending by c */
                if (c == last) {
                    /* Repeated symbol collapses: prefix unchanged */
                    cans[j].pnb = logsumexp(cans[j].pnb,pnb[j] + lp[c]);
                    /* Only paths ending in blank extend the prefix */
                    ext = pb[j] + lp[c];
                }
                else
                    ext = p + lp[c];

                /* If extended prefix is already a beam, merge into it */
                int k = -1;
                for (int i = 0; i < nb; i++) {
                    if (extof[i] == j && cur[i][len[i] - 1] == c) {
                        k = i;
                        break;
                    }
                }
                if (k < 0) {
                    k = nc++;
                    cans[k].parent = j;
                    cans[k].sym = c;
                    cans[k].pb = -INFINITY;
                    cans[k].pnb = -INFINITY;
                }
                cans[k].pnb = logsumexp(cans[k].pnb,ext);
            }
        }

        /* Keep the beam_width most probable prefixes */
        for (int k = 0; k < nc; k++)
            cans[k].p = logsumexp(cans[k].pb,cans[k].pnb);
        qsort(cans,nc,sizeof(struct ctc_can_s),ctc_can_cmp);

        int nnb = (nc < B) ? nc : B;
        for (int i = 0; i < nnb; i++) {
            int j = cans[i].parent;
            memcpy(nxt[i],cur[j],len[j] * sizeof(int));
            nlen[i] = len[j];
            if (cans[i].sym >= 0)
                nxt[i][nlen[i]++] = cans[i].sym;
            pb[i] = cans[i].pb;
            pnb[i] = cans[i].pnb;
        }
        iArrBT1 tmp = cur;
        cur = nxt;
        nxt = tmp;
        memcpy(len,nlen,nnb * sizeof(int));
        nb = nnb;
    }

    /* Prefix length is at most T, so sequences remain SIL terminated */
    for (int i = 0; i < nb; i++) {
        memcpy(sequences[i],cur[i],len[i] * sizeof(int));
        scores[i] = -logsumexp(pb[i],pnb[i]);
    }
    if (useheap) {
        freemem(cur);
        freemem(nxt);
    }
}
