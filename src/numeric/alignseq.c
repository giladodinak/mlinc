/* Copyright (c) 2023-2024 Gilad Odinak */
/* Sequence alignment function     */
/* References:
 * 1. A General Method Applicable to the Search for Similarities in the 
 *    Amino Acid Sequence of Two Proteins J. Mol. Bwl. (1970) 48, 443-453 
 * 2. https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
 */
#include <string.h>
#include "mem.h"
#include "alignseq.h"

static inline void reverse(int* s, int len)
{
    for (int i = 0, j = len - 1; i < len / 2; i++, j--) {
        int t = s[i];
        s[i] = s[j];
        s[j] = t;
    }
}

/* Aligns two sequences to have equal length and smallest edit distance.
 *
 * This function uses the Needleman Wunsch algorithm to find the optimal
 * alignment.  In order to align the sequences it may need to create gaps
 * in either or both sequences. The gaps are filled with 'blank' value, 
 * that should not be present in the sequences to be aligned.
 *
 * Parameters:
 *   p     - First sequence to be aligned with second sequence
 *   plen  - The length of the first sequence
 *   t     - Second sequence to be abligned with the first sequence
 *   tlen  - The length of the second sequence
 *   rp    - An array that is filled with the aligned first sequence
 *   rt    - An array that is filled with the aligned second sequence
 *   rlen  - The size of rp, rt arrays. Should be twice the size fo the
 *           longer of p, t.
 *   blank - The value to be used to fill gaps in the aligned sequences
 *
 * Returns:
 *   The edit distance between the aligned sequences. Returns -1 if
 *   size of rp, rt buffers, as indicated by rlen, is too small.
 *
 * Note:
 *   If the aligned sequences are shorter than rlen, they are terminated
 *   with a 'blank'.
 */ 
int alignseq(const int* p, int plen, 
             const int* t, int tlen,
             int* rp, int* rt, int rlen, int blank)
{
    int useheap = ((plen + 1) * (tlen + 1) > 10000) ? 1 : 0;

    /* F is the scoring matrix */
    typedef int (*iArrPT)[tlen + 1];
    int F_[useheap ? 1 : plen + 1][useheap ? 1 : tlen + 1];
    iArrPT F = (iArrPT) (useheap ? allocmem(plen + 1,tlen + 1,int) : F_);

    /* Initialize the scoring matrix */
    memset(F,0,(plen + 1) * (tlen + 1) * sizeof(int));
    for (int i = 1; i <= plen; i++)
        F[i][0] = -i;
    for (int j = 0; j <= tlen; j++)
        F[0][j] = -j;

    /* D is the traceback direction matrix */
    typedef int (*cArrPT)[tlen + 1];
    int D_[useheap ? 1 : plen + 1][useheap ? 1 : tlen + 1];
    cArrPT D = (cArrPT) (useheap ? allocmem(plen + 1,tlen + 1,int) : D_);

    /* Initialize the traceback matrix */
    memset(D,0,(plen + 1) * (tlen + 1) * sizeof(int));
    for (int i = 1; i <= plen; i++) /* Left edge, can only go up    */
        D[i][0] = 'U';
    for (int j = 0; j <= tlen; j++) /* Upper edge, can only go left */
        D[0][j] = 'L';

    /* Forward pass, calculate scores */
    for (int i = 0; i < plen; i++) {
        for (int j = 0; j < tlen; j++) {
            /* Calculate scores of different alignment steps */
            int match = F[i][j] + ((p[i] != t[j]) ? -1 : 0); /* (mis)match */
            int pgap = F[i][j + 1] - 1; /* Insert gap in t */
            int tgap = F[i + 1][j] - 1; /* Insert gap in p */
            if (match >= pgap && match >= tgap) {
                F[i + 1][j + 1] = match;
                D[i + 1][j + 1] = 'D';
            }
            else
            if (pgap > match && pgap >= tgap) {
                F[i + 1][j + 1] = pgap;
                D[i + 1][j + 1] = 'U';
            }
            else { /* tgap > match && tgap > pgap */
                F[i + 1][j + 1] = tgap;
                D[i + 1][j + 1] = 'L';
            }
        }
    }

    /* Backward pass, trace best path and calculate edit distance */
    int dist = 0;
    int rinx = 0;
    for (int i = plen, j = tlen; i > 0 || j > 0;) {
        if (rinx >= rlen) { /* Output buffers too small */
            dist = -1;
            break;
        }
        int s = D[i][j];
        if (s == 'D') { /* i > 0 && j > 0; see fwd pass */
            rp[rinx] = p[i - 1]; 
            rt[rinx] = t[j - 1]; 
            if (rp[rinx] != rt[rinx])
                dist++; /* Mismatch is a substitution */
            rinx++; i--; j--;
        }
        else
        if (s == 'U') { /* i > 0; see initialization  */
            rp[rinx] = p[i - 1]; 
            rt[rinx] = blank; 
            dist++; /* Gap in t (extra token in p)    */
            rinx++; i--;
        }
        else { /* s == 'L', j > 0; see initialization */
            rp[rinx] = blank; 
            rt[rinx] = t[j - 1]; 
            dist++; /* Gap in p (missing token in p)  */
            rinx++; j--;
        }
    }
    /* Because of traceing back, sequences are reversed; fix that */
    reverse(rp,rinx);
    reverse(rt,rinx);
    if (rinx < rlen) { /* Terminate sequences if shorter than rlen */
        rp[rinx] = blank;
        rt[rinx] = blank;
    }
    if (useheap) {
        freemem(F);
        freemem(D);
    }
    return dist;
}
