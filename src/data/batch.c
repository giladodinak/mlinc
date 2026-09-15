/* Copyright (c) 2023-2024 Gilad Odinak */
/* Data preparation for model training  */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "random.h"
#include "array.h"
#include "batch.h"

/* Constructs an iterator that returns batches of input vectors, 
 * and optionally their expected output vectors.
 */
BATCH* batch_create(const fArr2D x, int D, const fArr2D y, int N, int B,
                    const int* len, int num, int shuffle)
{
    BATCH* b = allocmem(1,1,BATCH);
    b->x = x;
    b->y = y;
    b->D = D;
    b->N = N;
    b->B = B;
    b->shuffle = (shuffle) ? 1 : 0;
    b->num = num;
    b->shufSeq = NULL;
    b->shufLen = NULL;
    b->shufVec = NULL;
    b->curSeq = 0;
    b->curVec = 0;

    if (len != NULL && num > 1) {
        b->shufSeq = allocmem(1,num,float);
        b->shufLen = allocmem(1,num,float);
        /* shufSeq[i] is the index of the begining of sequence 'i' in x; 
         * the sequence length is in shufLen[i]
         * Since the sequences are contingous in x, sequence 'i+1'
         * starts right after sequence 'i' ends
         */  
        b->shufSeq[0] = 0;
        b->shufLen[0] = len[0];
        for (int i = 1; i < num; i++) {
            b->shufSeq[i] = b->shufSeq[i - 1] + b->shufLen[i - 1];
            b->shufLen[i] = len[i];
        }
    }
    else
    if (shuffle) {
        /* shufVec[i] is the index of a vector in x */
        b->shufVec = allocmem(1,num,float);
        for (int i = 0; i < num; i++)
            b->shufVec[i] = i;
    }
    return b;
}

void batch_free(BATCH* b)
{
    freemem(b->shufSeq);
    freemem(b->shufLen);
    freemem(b->shufVec);
    freemem(b);
}

void batch_shuffle(BATCH* restrict b)
{
    b->curSeq = 0;
    b->curVec = 0;
    if (!b->shuffle)
        return;
    if (b->shufSeq != NULL) {
        for (int k = 0; k < 3; k++) {
            for (int i = b->num - 1; i > 0; i--) {
                int j = (int) urand(0.0,1.0 + i);
                int tmp = b->shufSeq[i];
                b->shufSeq[i] = b->shufSeq[j];
                b->shufSeq[j] = tmp;
                tmp = b->shufLen[i];
                b->shufLen[i] = b->shufLen[j];
                b->shufLen[j] = tmp;
            }
        }
    }
    else 
    if (b->shufVec != NULL) {
        for (int k = 0; k < 3; k++) {
            for (int i = b->num - 1; i > 0; i--) {
                int j = (int) urand(0.0,1.0 + i);
                int tmp = b->shufVec[i];
                b->shufVec[i] = b->shufVec[j];
                b->shufVec[j] = tmp;
            }
        }
    }
}

/* Copies a batch of input samples, and optionally their labels.
 * x is an array thet receives batch_size samples 
 * y is an array that receives batch_size corresponding lables, if it is
 * not NULL and labels were passed to batch_create.
 * 
 * Returns number of actual samples returned. If number of returned samples
 * is less than batch_size pads the returned data with zeros. Returns 0
 * past end of data.
 */
int batch_copy(BATCH* restrict b, fArr2D restrict x, fArr2D restrict y)
{
    int D = b->D;
    int N = b->N;
    int B = b->B;
    int cnt = 0, ycnt = 0;
    typedef float (*ArrBD)[D];
    typedef float (*ArrBN)[b->N];
    ArrBD xs = (ArrBD) b->x;
    ArrBD xd = (ArrBD) x;
    ArrBN ys = (ArrBN) b->y; /* Maybe NULL */
    ArrBN yd = (ArrBN) y;    /* Maybe NULL */
    
    if (b->shufSeq != NULL) {
        if (b->curSeq < b->num) { /* b->num is number of sequences */
            int curSeq = b->curSeq;
            int curVec = b->curVec;
            int seqLen = b->shufLen[curSeq];
            for (cnt = 0; cnt < B && b->curSeq < b->num && 
                                     b->curVec < seqLen; cnt++) {
                int i = b->shufSeq[b->curSeq] + b->curVec++;
                for (int j = 0; j < D; j++)
                    xd[cnt][j] = xs[i][j];
            }
            if (ys != NULL && yd != NULL) {
                for (ycnt = 0; ycnt < cnt; ycnt++) {
                    int i = b->shufSeq[curSeq] + curVec++;
                    for (int j = 0; j < N; j++)
                        yd[ycnt][j] = ys[i][j];
                }
            }
            if (b->curVec >= seqLen) {
                b->curSeq++;
                b->curVec = 0;
            }
        }
    }
    else
    if (b->shufVec != NULL) {
        int curVec = b->curVec;
        for (cnt = 0; cnt < B && b->curVec < b->num; cnt++) {
            int i = b->shufVec[b->curVec++];
            for (int j = 0; j < D; j++)
                xd[cnt][j] = xs[i][j];
        }
        if (ys != NULL && yd != NULL) {
            for (ycnt = 0; ycnt < cnt; ycnt++) {
                int i = b->shufVec[curVec++];
                for (int j = 0; j < N; j++)
                    yd[ycnt][j] = ys[i][j];
            }
        }
    }
    else {
        int curVec = b->curVec;
        for (cnt = 0; cnt < B && b->curVec < b->num; cnt++) {
            int i = b->curVec++;
            for (int j = 0; j < D; j++)
                xd[cnt][j] = xs[i][j];
        }
        if (ys != NULL && yd != NULL) {
            for (ycnt = 0; ycnt < cnt; ycnt++) {
                int i = curVec++;
                for (int j = 0; j < N; j++)
                    yd[ycnt][j] = ys[i][j];
            }
        }
    }
    if (cnt < B) { /* Pad to batch size */
        for (int i = cnt; i < B; i++)
            for (int j = 0; j < D; j++)
                xd[i][j] = 1.0;
        if (ys != NULL && yd != NULL) {
            for (int i = ycnt; i < B; i++)
                for (int j = 0; j < N; j++)
                    yd[i][j] = 0;
        }
    }
    return cnt;
}

/* Returns 1 if the most recent batch_copy() call exhausted a sequence
 * (i.e. a sequence boundary was just crossed), 0 otherwise.
 * Always returns 0 when data is not multi-sequence.
 */
int batch_eos(BATCH* restrict b)
{
    if (b->shufSeq == NULL)
        return 0;
    /* curVec == 0 means a sequence boundary was just crossed (curSeq was 
     * incremented and curVec reset). curSeq == 0 is excluded because that
     * would mean no data has been copied yet.
     */
    return (b->curVec == 0 && b->curSeq > 0) ? 1 : 0;
}

