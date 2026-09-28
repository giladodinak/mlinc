/* Copyright (c) 2023-2024 Gilad Odinak */
/* QR decomposistion */
#ifndef QR_H
#define QR_H
#include <math.h>
#include "arraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* QR - Performs QR decomposition of matrix M to an orthogonal matrix Q,
 *      and an upper right triangular matrix R, using Householder
 *      transformation.
 *
 *      If R is NULL, only returns Q matrix. If both Q and R are NULL,
 *      converts M to orthogonal matrix in place. In that case, M must
 *      be a square matrix m == n.
 *
 *      Note that if m <= n, then Q is an orthogonal squared matrix. If
 *      m > n, Q is neither square nor orthogonal, however, each column
 *      of Q has a length (norm) of 1.
 *
 *      Reference:
 *      https://en.wikipedia.org/wiki/QR_decomposition
 *
 * Parameters:
 *   M   - Pointer to the matrix M to be decomposed.
 *   Q   - Pointer to the orthogonal matrix Q (output).
 *   R   - Pointer to the upper right triangular matrix R (output).
 *   m   - Number of rows in matrix M.
 *   n   - Number of columns in matrix M.
 *
 * Returns:
 *   Q   - Orthogonal matrix.
 *   R   - Upper right triangular matrix.
 */
void QR(fArr2D M_/*[m][n]*/, 
        fArr2D Q_/*[m][min(n,m)]*/,
        fArr2D R_/*[min(m,n)][n]*/,
        int m, int n);

#ifdef __cplusplus
}
#endif

#endif
