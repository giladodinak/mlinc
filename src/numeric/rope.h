/* Copyright (c) 2026 Gilad Odinak  */
/* Rotary Position Embedding (RoPE) */
#ifndef ROPE_H
#define ROPE_H
#include <math.h>
#include "array.h"

/* Precomputes the RoPE theta table for a given head dimension.
 *
 * Parameters:
 *   theta : Output array of length Dh/2; theta[i] = 10000^(-2i/Dh)
 *   Dh    : Head dimension (must be even)
 *
 * Reference:
 *   Su et al., "RoFormer: Enhanced Transformer with Rotary Position
 *   Embedding", 2021. https://arxiv.org/abs/2104.09864
 */
static inline void rope_init(float* theta, int Dh)
{
    for (int i = 0; i < Dh / 2; i++)
        theta[i] = powf(10000.0, -2.0 * i / (float) Dh);
}

/* Applies RoPE in-place to a single head slice of Q or K.
 *
 * Each token at position offset+t is rotated by angle
 * (pos_offset+t) * theta[i] for each consecutive pair of
 * dimensions (2i, 2i+1).
 *
 * Parameters:
 *   x       : Pointer to the [T][Dh] head slice to be rotated
 *   theta   : Precomputed frequency table of length Dh/2 (from rope_init())
 *   inverse : If non-zero, applies the inverse rotation (for backward pass)
 *   offset  : Position of the first token in x
 *   T       : Sequence length (number of rows)
 *   Dh      : Head dimension (number of columns, must be even)
 *
 */
static inline void rope_apply(fArr2D x_/*[T][Dh]*/,
                              const float* theta,
                              int inverse,
                              int offset,
                              int T, int Dh)
{
    typedef float (*ArrTDh)[Dh];
    ArrTDh x = (ArrTDh) x_;
    float sign = inverse ? -1.0 : 1.0;
    
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < Dh / 2; i++) {
            /* Calculate angle in double precision */
            double angle = ((double)offset + t) * (double)theta[i];
            float cos_a = (float)cos(angle);
            float sin_a = sign * (float)sin(angle);
            float x0 = x[t][2 * i];
            float x1 = x[t][2 * i + 1];
            x[t][2 * i]     = x0 * cos_a - x1 * sin_a;
            x[t][2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

#endif
