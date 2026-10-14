/* Copyright (c) 2023-2024 Gilad Odinak */
/* Data preparation for model training  */
#ifndef BATCH_H
#define BATCH_H
#include "arraytypes.h"

typedef struct batch_s {
    fArr2D x;       /* Array of input vectors                       */
    fArr2D y;       /* Array of output vectors (optional)           */
    int  D;         /* Dimension of x vectors (may include bias)    */
    int  N;         /* Dimension of y vectors                       */
    int  B;         /* Number of vectors returned by batch_next()   */
    int  shuffle;   /* if set, batch_shuffle() shuffles, else only resets */
    int  num;       /* Number of sequences, or number of vectors    */
    int* shufSeq;   /* Offsets of shuffled training sequences       */
    int* shufLen;   /* Lengths of shuffled training sequences       */
    int* shufVec;   /* Offsets of shuffled training vectors         */
    int  curSeq;    /* Next vector from this sequence               */
    int  curVec;    /* Next vector in the sequence                  */
    fArr2D xbuf;    /* batch_copy staging output array              */
    fArr2D ybuf;    /* batch_copy staging output array (optional)   */
} BATCH;

/* Constructs an iterator that returns batches of input vectors, 
 * and optionally their expected output vectors.
 */
BATCH* batch_create(const fArr2D x, int D, const fArr2D y, int N, int B,
                    const int* len, int num, int shuffle);

/* Frees mmemory allocated by batch_create() */
void batch_free(BATCH* b);

/* Shuffles sequences, or samples within the single sequence, and resets 
 * iterator cursor.
 */
void batch_shuffle(BATCH* restrict b);

/* Copies a batch of input samples, and optionally their labels.
 * x is an array thet receives batch_size samples 
 * y is an array that receives batch_size corresponding lables, if it is
 * not NULL and labels were passed to batch_create.
 * 
 * Returns number of actual samples returned. If number of returned samples
 * is less than batch_size pads the returned data with zeros. Returns 0
 * past end of data.
 */
int batch_copy(BATCH* restrict b, fArr2D restrict x, fArr2D restrict y);

/* Returns 1 if the most recent batch_copy() call exhausted a sequence
 * (i.e. a sequence boundary was just crossed), 0 otherwise.
 * Always returns 0 when data is not multi-sequence.
 */
int batch_eos(BATCH* restrict b);

#endif
