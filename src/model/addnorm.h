/* Copyright (c) 2026 Gilad Odinak */
/* Layer normalization data structure and functions */
#ifndef ADDNORM_H
#define ADDNORM_H
#include "float.h"
#include "arraytypes.h"
#include "array.h"

typedef struct lnorm_s {
  int B;           /* Number of input vectors   */
  int D;           /* Input vector dimension    */
  fVec mean;       /* Calculated from input [B] */
  fVec sdev;       /* Calculated from input [B] */
  fArr2D xn;       /* Normalized values [B][D]  */
  fVec beta;       /* Learnable parameter [D]   */
  fVec gamma;      /* Learnable parameter [D]   */
} ADDNORM;

/* Creates a normalization layer.
 *
 * Returns:
 *   Pointer to a normalization layer.
 *
 * Note:
 *   The normalization layer need to be further intialized using 
 *   addnorm_init() before it can be used.
 *
 */
ADDNORM* addnorm_create();

/* Initializes a normalization layer.
 *
 * Parameters:
 *   input_dim  - D, dimension of each input vector
 *   batch_size - B, number of input vectors
 *
 * Note:
 *   Initialises gamma=1, beta=0 (identity transform).
 */
void addnorm_init(ADDNORM* l, int input_dim, int batch_size);

/* Frees the memory allocated by addnorm_create() / addnorm_init()
 *
 * Parameters:
 *   l - pointer to the normalization layer layer to free
 */
void addnorm_free(ADDNORM* l);

/* Adds and normalizes each row of x1 and x2, applies learnable gamma/beta,
 * and stores result in y.
 *
 * Parameters:
 *   l - Pointer to the ADDNORM layer
 *   x1 - Input array [B][D]
 *   x2 - Input array [B][D]
 *   y  - Output array [B][D]
 *
 * Reference:
 * Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." (2016)
 * https://arxiv.org/pdf/1607.06450
 */
static inline void addnorm_forward(ADDNORM* l, 
                                   const fArr2D x1_,
                                   const fArr2D x2_,
                                   fArr2D y_)
{
    const int B = l->B;
    const int D = l->D;
    typedef float (*ArrBD)[l->D];
    ArrBD x1 = (ArrBD) x1_;
    ArrBD x2 = (ArrBD) x2_;
    ArrBD y = (ArrBD) y_;
    ArrBD xn = (ArrBD) l->xn;

    for (int i = 0; i < B; i++) {
        float sum = 0;
        for (int j = 0; j < D; j++)
            sum += x1[i][j] + x2[i][j];
        float mean = sum / D;
        l->mean[i] = mean;

        float var = 0;
        for (int j = 0; j < D; j++) {
            float diff = x1[i][j] + x2[i][j] - mean;
            var += diff * diff;
        }
        float sdev = sqrtf(var / D + 1e-9);
        l->sdev[i] = sdev;

        for (int j = 0; j < D; j++) {
            xn[i][j] = (x1[i][j] + x2[i][j] - mean) / sdev;
            y[i][j] = l->gamma[j] * xn[i][j] + l->beta[j];
        }
    }
}

/* Computes gradient of layer normalization.
 *
 * Parameters:
 *   l  - Pointer to the ADDNORM layer
 *   dy - Gradient of output [B][D]
 *   dx - Gradient w.r.t inputs x1 and x2 [B][D] (output)
 *   dg - Gradient w.r.t gamma [D] (output)
 *   db - Gradient w.r.t beta [D] (output)
 */
static inline void addnorm_backward(ADDNORM* l,
                                    const fArr2D dy_,
                                    fArr2D dx_,
                                    fVec dg,
                                    fVec db)
{
    const int B = l->B;
    const int D = l->D;
    typedef float (*ArrBD)[D];
    const ArrBD dy = (const ArrBD) dy_;
    ArrBD dx = (ArrBD) dx_;
    const ArrBD xn = (const ArrBD) l->xn;

    fltclr(dg,D);
    fltclr(db,D);

    for (int i = 0; i < B; i++) {
        float sdev = l->sdev[i];

        /* Accumulate dgamma and dbeta */
        for (int j = 0; j < D; j++) {
            dg[j] += dy[i][j] * xn[i][j];
            db[j] += dy[i][j];
        }

        /* Compute dx using row-wise derivative */
        float sum_dy = 0;
        float sum_dy_xn = 0;
        for (int j = 0; j < D; j++) {
            float dyg = dy[i][j] * l->gamma[j];
            sum_dy += dyg;
            sum_dy_xn += dyg * xn[i][j];
        }
        sum_dy /= D;
        sum_dy_xn /= D;

        for (int j = 0; j < D; j++) {
            float dyg = dy[i][j] * l->gamma[j];
            dx[i][j] = (dyg - sum_dy - xn[i][j] * sum_dy_xn) / sdev;
        }
    }
}

#endif
