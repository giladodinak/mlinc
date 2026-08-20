/* Copyright (c) 2026 Gilad Odinak                  */
/* Group normalization data structure and functions */
/* References:
 *   Wu, Yuxin, and Kaiming He. "Group normalization" (2018)
 *     https://arxiv.org/pdf/1803.08494
 *   Ulyanov, Vedaldi & Lempitsky. "Instance Normalization" (2017)
 *     https://arxiv.org/pdf/1607.08022
 *   Ba, Kiros & Hinton. "Layer Normalization" (2016)
 *     https://arxiv.org/pdf/1607.06450
 */
#ifndef GRPNORM_H
#define GRPNORM_H
#include <math.h>
#include "float.h"
#include "array.h"

/* Group normalization over a batch of [T][C] sequences.
 *
 * Channels are partitioned into G contiguous groups of Cg = C / G channels.
 * For each batch item and group, statistics are pooled over that group's
 * Cg channels across all T time steps. A per-channel affine transform is
 * then applied using gamma[c] and beta[c].
 *
 * With G == C, each group contains one channel, so each channel is normalized
 * over the T time steps of that batch item. This is equivalent to Instance
 * Normalization for [T][C] sequences, assuming T > 1.
 *
 * With G == 1, each batch item is normalized over the full [T][C] sequence.
 * If T == 1, this is equivalent to row-wise LayerNorm over C channels.
 *
 * Input and output are laid out as [B*T][C]: element [b*T + t][c] is
 * channel c at time step t of batch item b.
 */
typedef struct grpnorm_s {
  int B;           /* Number of batch items         */
  int T;           /* Number of time steps          */
  int C;           /* Number of channels            */
  int G;           /* Number of groups              */
  fVec mean;       /* Calculated from input [B*G]   */
  fVec sdev;       /* Calculated from input [B*G]   */
  fArr2D xn;       /* Normalized values [B*T][C]    */
  fVec beta;       /* Learnable parameter [C]       */
  fVec gamma;      /* Learnable parameter [C]       */
} GRPNORM;

/* Creates a normalization layer.
 *
 * Parameters:
 *   groups - number of channel groups.
 *
 * Returns:
 *   Pointer to a normalization layer.
 *
 * Note:
 *   The normalization layer need to be further intialized using 
 *   grpnorm_init() before it can be used.
 *
 */
GRPNORM* grpnorm_create(int groups);

/* Initializes a normalization layer.
 *
 * Parameters:
 *   channels   - C, number of channels per time step
 *   steps      - T, number of time steps per batch item
 *   batch_size - B, number of batch items
 *
 * Notes:
 *   The number of channels must be a multiple of the group count
 *   Initialises gamma=1, beta=0 (identity transform).
 */
void grpnorm_init(GRPNORM* l, int channels, int steps, int batch_size);

/* Frees the memory allocated by grpnorm_create() / grpnorm_init()
 *
 * Parameters:
 *   l - pointer to the normalization layer layer to free
 */
void grpnorm_free(GRPNORM* l);

/* Normalizes each group of x, applies learnable gamma/beta,
 * and stores result in y.
 *
 * Parameters:
 *   l - Pointer to the GRPNORM layer
 *   x - Input array [B*T][C]
 *   y - Output array [B*T][C]
 *
 */
static inline void grpnorm_forward(GRPNORM* l, const fArr2D x_, fArr2D y_)
{
    const int B = l->B;
    const int T = l->T;
    const int C = l->C;
    const int G = l->G;
    const int Cg = C / G;
    typedef float (*ArrRC)[l->C];
    ArrRC x = (ArrRC) x_;
    ArrRC y = (ArrRC) y_;
    ArrRC xn = (ArrRC) l->xn;

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < G; g++) {
            int idx = b * G + g;
            int c0 = g * Cg;
            int c1 = c0 + Cg;

            float sum = 0;
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                for (int c = c0; c < c1; c++)
                    sum += x[r][c];
            }
            float mean = sum / (T * Cg);
            l->mean[idx] = mean;

            float var = 0;
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                for (int c = c0; c < c1; c++) {
                    float diff = x[r][c] - mean;
                    var += diff * diff;
                }
            }
            float sdev = sqrtf(var / (T * Cg) + 1e-9);
            l->sdev[idx] = sdev;

            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                for (int c = c0; c < c1; c++) {
                    xn[r][c] = (x[r][c] - mean) / sdev;
                    y[r][c] = l->gamma[c] * xn[r][c] + l->beta[c];
                }
            }
        }
    }
}

/* Computes gradient of group normalization.
 *
 * Parameters:
 *   l  - Pointer to the GRPNORM layer
 *   dy - Gradient of output [B*T][C]
 *   dx - Gradient w.r.t input x [B*T][C] (output)
 *   dg - Gradient w.r.t gamma [C] (output)
 *   db - Gradient w.r.t beta [C] (output)
 */
static inline void grpnorm_backward(GRPNORM* l, const fArr2D dy_,
                                    fArr2D dx_, fVec dg, fVec db)
{
    const int B = l->B;
    const int T = l->T;
    const int C = l->C;
    const int G = l->G;
    const int Cg = C / G;
    typedef float (*ArrRC)[C];
    const ArrRC dy = (const ArrRC) dy_;
    ArrRC dx = (ArrRC) dx_;
    const ArrRC xn = (const ArrRC) l->xn;

    fltclr(dg,C);
    fltclr(db,C);

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < G; g++) {
            int idx = b * G + g;
            int c0 = g * Cg;
            int c1 = c0 + Cg;
            float sdev = l->sdev[idx];
            int M = T * Cg;

            /* Accumulate dgamma and dbeta */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                for (int c = c0; c < c1; c++) {
                    dg[c] += dy[r][c] * xn[r][c];
                    db[c] += dy[r][c];
                }
            }

            /* Compute dx using group-wise derivative */
            float sum_dy = 0;
            float sum_dy_xn = 0;
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                for (int c = c0; c < c1; c++) {
                    float dyg = dy[r][c] * l->gamma[c];
                    sum_dy += dyg;
                    sum_dy_xn += dyg * xn[r][c];
                }
            }
            sum_dy /= M;
            sum_dy_xn /= M;

            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                for (int c = c0; c < c1; c++) {
                    float dyg = dy[r][c] * l->gamma[c];
                    dx[r][c] = (dyg - sum_dy - xn[r][c] * sum_dy_xn) / sdev;
                }
            }
        }
    }
}

#endif
