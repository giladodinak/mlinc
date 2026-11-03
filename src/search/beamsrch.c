/* Copyright (c) 2023-2024 Gilad Odinak */
/* Beam search decoder             */
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
