/* Copyright (c) 2023-2024 Gilad Odinak */
/* Scaling and normalization functions */
#include <math.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "scaler.h"

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
SCALER* scaler_init(int batch, int dim, int exc_last)
{
    SCALER* s = allocmem(1,1,SCALER);
    s->batch = batch; 
    s->dim = dim; 
    s->exc_last = (exc_last) ? 1 : 0 ;
    s->count = 0;
    s->mean = allocmem(1,dim,float);
    s->var = allocmem(1,dim,float);
    fltclr(s->mean,dim);
    fltclr(s->var,dim);
    return s;
}

/* Frees mmemory allocated by scaler_init() */
void scaler_free(SCALER* s)
{
    freemem(s->mean);
    freemem(s->var);
    freemem(s);
}

static void calculate_mean_var(SCALER* restrict s, fArr2D data_, int num);
static void calculate_batch_mean_var(SCALER* restrict s, fArr2D data_, int num);
static void normalize(SCALER* restrict s, fArr2D data_, int num);
static void normalize_batch(SCALER* restrict s, fArr2D data_, int num);

/* Normalizes the passed in data of num samples (vectors). 
 * If calc is not zero, first calculates and updates mean and variance for 
 * the data according to the mode (standard, or batch).
 */
void scaler_normalize(SCALER* s, fArr2D data/*[num][dim]*/, int num, int calc)
{
    if (calc) {
        if (s->batch)
            calculate_batch_mean_var(s,data,num);
        else
            calculate_mean_var(s,data,num);
    }
    if (s->batch)
        normalize_batch(s,data,num);
    else
        normalize(s,data,num);
}
    
/* Calculates mean and variance of the input vectors in data
 * and updates the corresponding fields.
 * data - input vectors [num][dim]
 * num - number of vectors
 */
static void calculate_mean_var(SCALER* restrict s, fArr2D data_, int num)
{
    typedef float (*ArrBD)[s->dim];
    ArrBD data = (ArrBD) data_;
    if (s->count > 0) {
        fltclr(s->mean,s->dim);
        fltclr(s->var,s->dim);
    }
    s->count = num;
    for (int j = 0; j < s->dim - s->exc_last; j++)
        for (int i = 0; i < num; i++)
            s->mean[j] +=  data[i][j];
    for (int j = 0; j < s->dim - s->exc_last; j++)
        s->mean[j] /= num;
    for (int j = 0; j < s->dim - s->exc_last; j++)
        for (int i = 0; i < num; i++)
            s->var[j] += (data[i][j] - s->mean[j]) * (data[i][j] - s->mean[j]);
}

/* Normalizes the input vectors in data, using mean and variance information.
 * data - input vectors [num][dim]
 * num - number of vectors
 */
static void normalize(SCALER* restrict s, fArr2D data_, int num)
{
    typedef float (*ArrBD)[s->dim];
    ArrBD data = (ArrBD) data_;
    float stddev[s->dim];
    int cnt = s->count;
    if (cnt < 2 || num == 0 || (s->dim - s->exc_last) < 1)
        return;
    for (int j = 0; j < s->dim - s->exc_last; j++) {
        stddev[j] = sqrt(s->var[j] / cnt);
        if (stddev[j] == 0.0)
            stddev[j] = 1.0;
    }
    for (int i = 0; i < num; i++)
        for (int j = 0; j < s->dim - s->exc_last; j++)
            data[i][j] = (data[i][j] - s->mean[j]) / stddev[j];
}

/* Calculates online mean and variance of the input vectors in x
 * and updates the corresponding fields.
 * data - input vectors [num][dim]
 * num - number of vectors
 */
static void calculate_batch_mean_var(SCALER* restrict s, fArr2D data_, int num)
{
    typedef float (*ArrBD)[s->dim];
    ArrBD data = (ArrBD) data_;
    for (int i = 0; i < num; i++) {
        s->count++;
        for (int j = 0; j < s->dim - s->exc_last; j++) {
            float d = data[i][j] - s->mean[j];
            s->mean[j] += d / s->count;
            float d2 = data[i][j] - s->mean[j];
            s->var[j] += d * d2;
        }
    }
}

/* Normalizes the input vectors in data, using online mean and variance 
 * information. Notice that if the online variance for a column is less 
 * than 1, that column is not scaled.
 * data - input vectors [num][dim]
 * num - number of vectors
 */
static void normalize_batch(SCALER* restrict s, fArr2D data_, int num)
{
    typedef float (*ArrBD)[s->dim];
    ArrBD data = (ArrBD) data_;
    float stddev[s->dim];
    int cnt = s->count;
    if (cnt < 2 || num == 0 || (s->dim - s->exc_last) < 1)
        return;
    for (int j = 0; j < s->dim - s->exc_last; j++) {
        stddev[j] = sqrt(s->var[j] / (cnt - 1));
        if (stddev[j] < 1.0)
            stddev[j] = 1.0;
    }
    for (int i = 0; i < num; i++)
        for (int j = 0; j < s->dim - s->exc_last; j++)
            data[i][j] = (data[i][j] - s->mean[j]) / stddev[j];
}
