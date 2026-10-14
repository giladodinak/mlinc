/* Copyright (c) 2023-2024 Gilad Odinak */
/* Scaling and normalization functions */

#ifndef SCALER_H
#define SCALER_H
#include "arraytypes.h"

typedef struct scaler_s {
    int batch;          /* 1 if batch normalization                        */
    int count;          /* Total number of samples                         */
    int dim;            /* Dimension of sample vectors (number of samples) */
    int exc_last;       /* Do not scale the last dimension                 */
    fVec mean/*[dim]*/; /* (moving) average of inputs                      */
    fVec var/*[dim]*/;  /* (moving) sum of squares of diff from mean       */
} SCALER;


/* Constructs a processor that normalizes features by scaling them to have
 * a mean of 0 and standard deviation of 1. It accepts an array of data 
 * consiting of m vectors, each having n elements, or features, and normalizes
 * the data in place. It operates in one of two modes:
 *
 *   In standard mode, it either first calculates mean and variance for
 *   the data, stores it, then normalizes the data using the stored values; 
 *   or it uses previously stored mean and variance to normalize the data.
 *
 *   In batch mode, it either first calculates mean and variance for
 *   the data, updates its internally stored moving averages of these values,
 *   then normalizes the data using the stored value; or it uses previously
 *   stored mean and variance to normalize the data.
 *
 * - if batch is not zero, operate in batch mode.
 * - dim is the dimension of the samples, that is, the number of elements
 *   in each sample vector, the number of columns in the data array.
 * - if exc_last is not zero, the last dimension, that is, the last element
 *   of the sample vectors, the last column of the data array, is not scaled:
 *   mean and variance are not calculated for it, and it is not normalized.
 *
 * Returns a SCALER structure that keeps the scaler's state and is passed
 * to other scaler functions.
 */ 
SCALER* scaler_init(int batch, int dim, int exc_last);

/* Frees mmemory allocated by scaler_init() */
void scaler_free(SCALER* s);

/* Normalizes the passed in data of num samples (vectors). 
 * If calc is not zero, first calculates and updates mean and variance for 
 * the data according to the mode (standard, or batch).
 */
void scaler_normalize(SCALER* s, fArr2D data/*[num][dim]*/, int num, int calc);

#endif
