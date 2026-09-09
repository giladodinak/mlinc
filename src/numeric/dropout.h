/* Copyright (c) 2026 Gilad Odinak */
/* Dropout regularization          */
#ifndef DROPOUT_H
#define DROPOUT_H
#include "random.h"
#include "array.h"

/* Applies dropout to a 2D array in-place (training only).
 *
 * Parameters:
 *   mx   : Pointer to the 2D array to be processed
 *   mk   : Pointer to an optional 2D dropout mask array
 *   M    : Number of rows in the matrix
 *   N    : Number of columns in the matrix
 *   rate : Fraction of elements to zero out (e.g. 0.1 for 10%)
 *
 * Notes:
 *   - Each element is zeroed independently with probability `rate`.
 *   - Surviving elements are scaled by 1/(1-rate) to preserve
 *     expected values (inverted dropout).
 *   - Should only be called during training, not inference.
 *   - Requires rate to be in the range [0.0, 1.0).
 *
 * Reference:
 *   Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks
 *   from Overfitting", JMLR 2014.
 */
static inline void dropout(fArr2D mx_/*[M][N]*/, 
                           fArr2D mk_/*[M][N]*/,
                           int M, int N, float rate)
{
    typedef float (*ArrMN)[N];
    ArrMN mx = (ArrMN) mx_;
    ArrMN mk = (ArrMN) mk_;
    float scale = 1.0 / (1.0 - rate);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float value = urand(0.0,1.0) >= rate ? scale : 0;
            mx[i][j] *= value;
            if (mk) mk[i][j] = value;
        }
    }
}

/* Applies a precomputed dropout mask to an input array, writing the
 * result to a separate output array without modifying the input.
 *
 * Parameters:
 *   in_  : Pointer to the 2D input array to be masked
 *   mk_  : Pointer to the 2D dropout mask array (as produced by dropout())
 *   out_ : Pointer to the 2D output array to receive the masked result
 *   M    : Number of rows in the matrix
 *   N    : Number of columns in the matrix
 *
 * Notes:
 *   - Useful in backward passes where the same gradient value feeds
 *     both a dropout-masked branch and an unmasked residual branch;
 *     this keeps the residual gradient untouched while producing a
 *     separately masked copy for the sub-layer branch.
 */
static inline void apply_dropout_mask(const fArr2D in_/*[M][N]*/,
                                      const fArr2D mk_/*[M][N]*/,
                                      fArr2D out_/*[M][N]*/,
                                      int M, int N)
{
    typedef float (*ArrMN)[N];
    const ArrMN in = (ArrMN) in_;
    const ArrMN mk = (ArrMN) mk_;
    ArrMN out = (ArrMN) out_;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            out[i][j] = in[i][j] * mk[i][j];
}

#endif
