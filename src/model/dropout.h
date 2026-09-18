/* Copyright (c) 2026 Gilad Odinak            */
/* Dropout layer data structure and functions 
 *
 * Reference:
 *   Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks
 *   from Overfitting", JMLR 2014.
 */
#ifndef DROPOUT_H
#define DROPOUT_H
#include "random.h"
#include "array.h"

typedef struct dropout_s {
    int B;              /* Number of rows (input vectors)            */
    int D;              /* Number of columns (input dimension)       */
    float rate;         /* Dropout rate (0.0 - 1.0)                  */
    fArr2D mask;        /* Masks (and scales) the input array [B][D] */
    void* rng_state;    /* Internal RNG buffer (NULL if not used)    */
} DROPOUT;

/* Creates a dropout layer.
 *
 * Parameters:
 *   rate - Dropout rate (0.0 to 1.0)
 *
 * Returns:
 *   Pointer to a dropout
 *
 * Notes:
 *   - The layer needs to be further initialized using dropout_init()
 *     before it can be used.
 */
DROPOUT* dropout_create(float rate);

/* Initializes a dropout layer created by dropout_create().
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *   input_dim  - Size of input vectors
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(D).
 */
void dropout_init(DROPOUT* l, int input_dim, int batch_size);

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dropout_init(), it does nothing.
 */
void dropout_set_batch_size(DROPOUT* l, int batch_size);

/* Frees the memory allocated by dropout_create() / dropout_init().
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void dropout_free(DROPOUT* l);

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 */
void dropout_reset(DROPOUT* l);

/* Applies dropout to a 2D array in-place (training only).
 *
 * Parameters:
 *   mx   : Pointer to the 2D array to be processed
 */
static inline void dropout_forward(DROPOUT* restrict l, fArr2D mx_/*[B][D])*/)
{
   float rate = l->rate;
    if (rate == 1.0) {
        fltclr(mx_,l->B * l->D);
        return;
    }
    typedef float (*ArrBD)[l->D];
    ArrBD mx = (ArrBD) mx_;
    ArrBD mk = (ArrBD) l->mask;
    float scale = 1.0 / (1.0 - rate);
    for (int i = 0; i < l->B; i++) {
        for (int j = 0; j < l->D; j++) {
            float value = urand(0.0,1.0) >= rate ? scale : 0;
            mx[i][j] *= value;
            mk[i][j] = value;
        }
    }
}

/* Applies a precomputed dropout mask to an input array, writing the
 * result to a separate output array without modifying the input.
 *
 * Parameters:
 *   in  : Pointer to the 2D input array to be masked
 *   out : Pointer to the 2D output array to receive the masked result
 */
static inline void dropout_backward(DROPOUT* restrict l,
                                    const fArr2D in_/*[B][D]*/,
                                    fArr2D out_/*[B][D]*/)
{
    typedef float (*ArrBD)[l->D];
    const ArrBD in = (const ArrBD) in_;
    const ArrBD mk = (const ArrBD) l->mask;
    ArrBD out = (ArrBD) out_;
    for (int i = 0; i < l->B; i++)
        for (int j = 0; j < l->D; j++)
            out[i][j] = in[i][j] * mk[i][j];
}
#endif
