/* Copyright (c) 2023-2024 Gilad Odinak */
/* SVD decomposistion */
#include <stdio.h>
#include <stdint.h>
#include "mem.h"
#include "array.h"
#include "svd.h"

#define SVD_TOL  3e-23 /* float tol is 1.401298e-45, but 3.0e-23 works better */
#define SVD_EPS  3e-13 /* float eps is 2.220446e-16, but 3.0e-13 works better */
#define MAX_ITER   100

static void svd_tall(int m, int n, fArr2D a_/*[m][n]*/,
                    fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D vt_/*[n][n]*/);

static void reorder_tall(int m,int n,
                    fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D vt_/*[n][n]*/);

static void svd_wide(int n, int m, fArr2D a_/*[m][n]*/,
                    fVec q_/*[m]*/, fArr2D vt_/*[m][n]*/, fArr2D u_/*[m][m]*/);

static void reorder_wide(int n,int m,
                    fVec q_/*[m]*/, fArr2D vt_/*[m][n]*/, fArr2D u_/*[m][m]*/);

/* SVD - Performs SVD decomposition of matrix A to obtain 
 *   a left orthogonal matrix U, a vector of non-negative singular values S,
 *   and a right orthogonal matrix V, such that A = U @ Sigma @ Vt.
 *
 *   The singular values in vector S represent the diagonal of the diagonal
 *   matrix Sigma, arranged in descending order.
 *
 *   Reference:
 *   https://en.wikipedia.org/wiki/Singular_value_decomposition
 *
 * Parameters:
 *   A   - Pointer to the matrix A to be decomposed.
 *   U   - Pointer to the left orthogonal matrix U (output).
 *   S   - Pointer to the vector of non-negative singular values S (output).
 *   Vt  - Pointer to the transpose of the right orthogonal matrix V (output).
 *   m   - Number of rows in matrix A.
 *   n   - Number of columns in matrix A.
 *
 * Returns:
 *   U   - Left orthogonal matrix U.
 *   S   - Vector of non-negative singular values.
 *   Vt  - Transpose of the right orthogonal matrix V.
 *
 * Notes:
 *   If m >= n output matrices dimensions are U[m][n] S[n] Vt[n][n]
 *   If m >= n then Vt may be NULL, in which case only U and S are returned;
 *   if both U and Vt are NULL, A is updated in place with the value of U.
 *
 *   If m < n output matrices dimensions are U[m][m] S[m] Vt[m][n]
 *   If m < n then U may be NULL, in which case only Vt and S are returned;
 *   if both U and Vt are NULL, A is updated in place with the value of Vt.
 *
 * This implementation follows the algorithm contributed by Golub and Reinsch
 * to Handbook Series Linear Algebra Vol. 14, pg 403-420 (1970)
 */
void SVD(fArr2D A_, fArr2D U_, fVec S_, fArr2D Vt_, int m, int n)
{
    if (A_ == NULL || m <= 1 || n <= 1) return;

    /* k is the larger dimension of A */
    int k = (m < n) ? m : n;
    /* Size of temporary S in bytes, if used, otherwise 0 */
    int bS = (S_ == NULL) * k * sizeof(float);
    /* Size of temporary A in bytes, if used, otherwise 0 */
    int bA = ((m >= n && U_ == NULL && Vt_ != NULL) || 
              (m < n && Vt_ == NULL && U_ != NULL)) * m * n * sizeof(float);
    /* Use stack based arrays if total size less than threshold */
    int useStack = (bS + bA) <= 1000000;
    /* Allocate stack based arrays if as needed, or dummy ones */
    float vS[(useStack && bS > 0) ? k : 1];
    float vA[(useStack && bA > 0) ? m * n : 1];

    fVec S = S_;
    if (bS) S = (fVec) ((useStack) ? vS : allocmem(k,1,float));

    fArr2D A = A_;
    if (bA) {
      A = (fArr2D) ((useStack) ? vA : allocmem(m,n,float));
      fltcpy(A,A_,m * n);
    }

    if (m >= n) {
        svd_tall(m,n,A,S,U_,Vt_);
        reorder_tall(m,n,S,U_,Vt_);
    }
    else {
        svd_wide(n,m,A,S,Vt_,U_);
        reorder_wide(n,m,S,Vt_,U_);
    }
    if (bS && !useStack) freemem(S);
    if (bA && !useStack) freemem(A);
}

/* Computes the singular values and complete orthogonal decomposition
 * of a real rectangular matrix A, A = U diag(q) V.T, U.T @ U = V.T @ V = I,
 * where the arrays a[m,n], u[m,n], vt[n,n], q[n]
 * represent A, U, V.T, q respectively.
 * assume m >= n.
 *
 * If vt is NULL only u and s are calculated. If both u and vt are NULL,
 * a is updated with the value of u in place.
 */
static void svd_tall(int m, int n, fArr2D a_/*[m][n]*/,
                     fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D vt_/*[n][n]*/)
{
    typedef float (*ArrMN)[n];
    typedef float (*ArrNN)[n];
    typedef float (*VecN);
    ArrMN a = (ArrMN) a_;
    VecN  q = (VecN)  q_;
    ArrMN u = (ArrMN) ((u_ != NULL) ? u_ : a_);
    ArrNN vt = (ArrNN) vt_;

    float tol = SVD_TOL;
    float eps = SVD_EPS;
    int max_iter = MAX_ITER;

    int i, j, k;
    float c, f, g, h, s, x, y, z;

    float e[n];

    if (u != a)
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            u[i][j] = a[i][j];

    /* Householder reduction to bidiagonal form */

    g = x = 0.0;
    for (i = 0; i < n; i++) {
        e[i] = g;

        s = 0.0;
        for (j = i; j < m; j++)
            s += u[j][i] * u[j][i];

        if (s < tol)
            g = 0.0;
        else {
            f = u[i][i];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            u[i][i] = f - g;

            for (j = i + 1; j < n; j++) {
                s = 0.0;
                for (k = i; k < m; k++)
                    s += u[k][i] * u[k][j];

                f = s / h;
                for (k = i; k < m; k++)
                    u[k][j] += f * u[k][i];
            }
        }

        q[i] = g;
        s = 0.0;
        for (j = i + 1; j < n; j++)
            s += u[i][j] * u[i][j];

        if (s < tol)
            g = 0.0;
        else if (i + 1 < n) {
            f = u[i][i + 1];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;

            u[i][i + 1] = f - g;
            for (j = i + 1; j < n; j++)
                e[j] = u[i][j] / h;

            for (j = i + 1; j < m; j++) {
                s = 0.0;
                for (k = i + 1; k < n; k++)
                    s += u[j][k] * u[i][k];

                for (k = i + 1; k < n; k++)
                    u[j][k] += s * e[k];
            }
        }

        y = fabsf(q[i]) + fabsf(e[i]);
        if (y > x)
            x = y;
    }

    /* accumulation of right-hand transformations */
    if (vt_ != NULL) {
        fltclr(vt,n * n);
        for (i = n - 1; i >= 0; i--) {
            if (g != 0.0 && i + 1 < n) {
                h = u[i][i + 1] * g;
                for (j = i + 1; j < n; j++)
                    vt[i][j] = u[i][j] / h;

                for (j = i + 1; j < n; j++) {
                    s = 0.0;
                    for (k = i + 1; k < n; k++)
                        s += u[i][k] * vt[j][k];

                    for (k = i + 1; k < n; k++)
                        vt[j][k] += s * vt[i][k];
                }
            }
            for (j = i + 1; j < n; j++)
                vt[j][i] = vt[i][j] = 0.0;

            vt[i][i] = 1.0;
            g = e[i];
        }
    }

    /* accumulation of left-hand transformations */
    if (u_ != NULL || vt_ == NULL)
    for (i = n - 1; i >= 0; i--) {
        g = q[i];
        for (j = + i + 1; j < n; j++)
            u[i][j] = 0.0;

        if (g != 0.0) {
            h = u[i][i] * g;
            for (j = i + 1; j < n; j++) {
                s = 0.0;
                for (k = i + 1; k < m; k++)
                    s += u[k][i] * u[k][j];

                f = s / h;
                for (k = i; k < m; k++)
                    u[k][j] += f * u[k][i];
            }
            for (j = i; j < m; j++)
                u[j][i] /= g;
        }
        else {
            for (j = i; j < m; j++)
                u[j][i] = 0.0;
        }

        u[i][i] += 1.0;
    }

    /* diagonalization of the bidiagonal form */
    eps *= x;
    for (k = n - 1; k >= 0; k--) {
        /* test for splitting */
        for (int iter = 0; iter < max_iter; iter++) {
            int l = k;
            /* convergence test */
            for (int once = 1; once > 0; once--) {
                /* Note: in the paper, in this loop, l reaches 0 => bug */
                for (l = k; l > 0; l--) {
                    if (fabsf(e[l]) <= eps)
                        break; /* test for convergence */
                    if (fabsf(q[l - 1]) <= eps)
                        break; /* cancellation */
                }
                if (fabsf(e[l]) <= eps || l == 0)
                    break; /* test for convergence */

                /* cancellation (l > 0) */
                c = 0.0;
                s = 1.0;
                for (i = l; i < k; i++) {
                    f = s * e[i];
                    e[i] = c * e[i];
                    if (fabsf(f) <= eps)
                        break; /* test for convergence */

                    g = q[i];
                    q[i] = sqrt(f * f + g * g);
                    h = q[i];
                    c = g / h;
                    s = -f / h;

                    if (u_ != NULL || vt_ == NULL)
                    for (j = 0; j < m; j++) {
                        y = u[j][l - 1];
                        z = u[j][i];
                        u[j][l - 1] = y * c + z * s;
                        u[j][i] = -y * s + z * c;
                    }
                }
            }

            /* test for convergence */
            z = q[k];
            if (l == k)
                break; /* convergence */

            /* shift from bottom 2 X 2 minor */

            x = q[l];
            y = q[k - 1];
            g = e[k - 1];
            h = e[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
            g = sqrt(f * f + 1.0);
            float t = (f < 0.0) ? f - g : f + g;
            f = ((x - z) * (x + z) + h * (y / t - h)) / x;

            /* Next Q R transformation */

            c = s = 1.0;
            for (i = l + 1; i <= k; i++) {
                g = e[i];
                y = q[i];
                h = s * g;
                g = c * g;
                z = sqrt(f * f + h * h);
                e[i - 1] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y = y * c;

                if (vt_ != NULL) 
                for (j = 0; j < n; j++) {
                    x = vt[i - 1][j];
                    z = vt[i][j];
                    vt[i - 1][j] = x * c + z * s;
                    vt[i][j] = -x * s + z * c;
                }
                z = sqrt(f * f + h * h);
                q[i - 1] = z;
                c = f / z;
                s = h / z;
                f = c * g + s * y;
                x = -s * g + c * y;

                if (u_ != NULL || vt_ == NULL)
                for (j = 0; j < m; j++) {
                    y = u[j][i - 1];
                    z = u[j][ i];
                    u[j][i - 1] = y * c + z * s;
                    u[j][i] = -y * s + z * c;
                }
            }
            e[l] = 0;
            e[k] = f;
            q[k] = x;
        }

        /* convergence */
        if (q[k] < 0.0) { /* ensure q values are positive */
            q[k] = -z;
            if (vt_ != NULL)
            for (j = 0; j < n; j++)
                vt[k][j] = -vt[k][j];
        }
    }
}

/* Reorders the values of q[n] in descending order,
 * and the columns  of u[m,n] and the rows of vt[n,n]
 * such that a = u diag(q) vt is preserved.
 * assume m >= n
 */
static void reorder_tall(int m,int n,
                         fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D vt_/*[n][n]*/)
{
    typedef float (*VecN);
    typedef float (*ArrMN)[n];
    typedef float (*ArrNN)[n];
    VecN  q =  (VecN)  q_;
    ArrMN u =  (ArrMN) u_;
    ArrNN vt = (ArrNN) vt_;

    int i, j, k;
    float q_t;
    float u_t[m];
    float vt_t[n];

    /* Check whether q already is ordered */
    for (i = 1; i < n - 1; i++)
        if (q[i - 1] < q[i])
            break;
    if (i == n)
        return;

    for(i = 1; i < n; i++) {
        q_t = q[i];

        if (u_ != NULL)
        for (k = 0; k < m; k++)
            u_t[k] = u[k][i];
        if (vt_ != NULL)
        for (k = 0; k < n; k++)
            vt_t[k] = vt[i][k];

        for (j = i - 1; j >= 0 && q[j] < q_t; j--) {
            q[j + 1] = q[j];
            if (u_ != NULL)
            for (k = 0; k < m; k++)
                u[k][j + 1] = u[k][j];
            if (vt_ != NULL)
            for (k = 0; k <  n; k++)
                vt[j + 1][k] = vt[j][k];
        }
        q[j + 1] = q_t;

        if (u_ != NULL)
        for (k = 0; k < m; k++)
            u[k][j + 1] = u_t[k];

        if (vt_ != NULL)
        for (k = 0; k < n; k++)
            vt[j + 1][k] = vt_t[k];
    }
}

/* Computes the singular values and complete orthogonal decomposition
 * of a real rectangular matrix A, A = U diag(q) V.T, U.T @ U = V.T @ V = I,
 * where the arrays a[m,n], u[m,m], vt[m,n], q[m]
 * represent A, U, V.T, q respectively.
 * assume n > m.
 *
 * If u is NULL only vt and s are calculated. If both u and vt are NULL,
 * a is updated with the value of vt in place.
 */
static void svd_wide(int n, int m, fArr2D a_/*[m][n]*/,
                     fVec q_/*[m]*/, fArr2D vt_/*[m][n]*/, fArr2D u_/*[m][m]*/)
{
    typedef float (*ArrMN)[n];
    typedef float (*ArrMM)[m];
    typedef float (*VecM);
    ArrMN a = (ArrMN) a_;
    VecM  q = (VecM)  q_;
    ArrMN vt = (ArrMN) ((vt_ != NULL) ? vt_ : a_);
    ArrMM u = (ArrMM) u_;

    float tol = SVD_TOL;
    float eps = SVD_EPS;
    int max_iter = MAX_ITER;

    int i, j, k, l;
    float c, f, g, h, s, x, y, z;

    float e[m];

    if (vt != a)
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            vt[j][i] = a[j][i];

    /* Householder reduction to bidiagonal form */

    g = x = 0.0;
    for (i = 0; i < m; i++) {
        e[i] = g;
        l = i + 1;

        s = 0.0;
        for (j = i; j < n; j++)
            s += vt[i][j] * vt[i][j];

        if (s < tol)
            g = 0.0;
        else {
            f = vt[i][i];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            vt[i][i] = f - g;

            for (j = l; j < m; j++) {
                s = 0.0;
                for (k = i; k < n; k++)
                    s += vt[i][k] * vt[j][k];

                f = s / h;
                for (k = i; k < n; k++)
                    vt[j][k] += f * vt[i][k];
            }
        }

        q[i] = g;
        s = 0.0;
        for (j = l; j < m; j++)
            s += vt[j][i] * vt[j][i];

        if (s < tol)
            g = 0.0;
        else {
            f = vt[i + 1][i];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;

            vt[i + 1][i] = f - g;
            for (j = l; j < m; j++)
                e[j] = vt[j][i] / h;

            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = l; k < m; k++)
                    s += vt[k][j] * vt[k][i];

                for (k = l; k < m; k++)
                    vt[k][j] += s * e[k];
            }
        }

        y = fabsf(q[i]) + fabsf(e[i]);
        if (y > x)
            x = y;
    }

    /* accumulation of right-hand transformations */
    l = m;
    if (u_ != NULL)
    for (i = m - 1; i >= 0; i--) {
        if (g != 0.0) {
            h = vt[i + 1][i] * g;
            for (j = l; j < m; j++)
                u[j][i] = vt[j][i] / h;

            for (j = l; j < m; j++) {
                s = 0.0;
                for (k = l; k < m; k++)
                    s += vt[k][i] * u[k][j];

                for (k = l; k < m; k++)
                    u[k][j] += s * u[k][i];
            }
        }
        for (j = l; j < m; j++)
            u[i][j] = u[j][i] = 0.0;

        u[i][i] = 1.0;
        g = e[i];
        l = i;
    }

    /* accumulation of left-hand transformations */
    if (vt_ != NULL || u_ == NULL)
    for (i = m - 1; i >= 0; i--) {
        g = q[i];
        l = i + 1;
        for (j = l; j < m; j++)
            vt[j][i] = 0.0;

        if (g != 0.0) {
            h = vt[i][i] * g;
            for (j = l; j < m; j++) {
                s = 0.0;
                for (k = l; k < n; k++)
                    s += vt[i][k] * vt[j][k];

                f = s / h;
                for (k = i; k < n; k++)
                    vt[j][k] += f * vt[i][k];
            }
            for (j = i; j < n; j++)
                vt[i][j] /= g;
        }
        else {
            for (j = i; j < n; j++)
                vt[i][j] = 0.0;
        }

        vt[i][i] += 1.0;
    }

    /* diagonalization of the bidiagonal form */
    eps *= x;
    for (k = m - 1; k >= 0; k--) {
        /* test for splitting */
        for (int iter = 0; iter < max_iter; iter++) {
            /* convergence test */
            for (int once = 1; once > 0; once--) {

                /* Note: in the paper, in this loop, l reaches 0 => bug */
                for (l = k; l > 0; l--) {
                    if (fabsf(e[l]) <= eps)
                        break; /* test for convergence */
                    if (fabsf(q[l - 1]) <= eps)
                        break; /* cancellation */
                }
                if (fabsf(e[l]) <= eps || l == 0)
                    break; /* test for convergence */

                /* cancellation (l > 0) */
                c = 0.0;
                s = 1.0;
                for (i = l; i < k; i++) {
                    f = s * e[i];
                    e[i] = c * e[i];
                    if (fabsf(f) <= eps)
                        break; /* test for convergence */

                    g = q[i];
                    q[i] = sqrt(f * f + g * g);
                    h = q[i];
                    c = g / h;
                    s = -f / h;

                    if (vt_ != NULL || u_ == NULL)
                    for (j = 0; j < n; j++) {
                        y = vt[l - 1][j];
                        z = vt[i][j];
                        vt[l - 1][j] = y * c + z * s;
                        vt[i][j] = -y * s + z * c;
                    }
                }
            }

            /* test for convergence */
            z = q[k];
            if (l == k)
                break; /* convergence */

            /* shift from bottom 2 X 2 minor */

            x = q[l];
            y = q[k - 1];
            g = e[k - 1];
            h = e[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
            g = sqrt(f * f + 1.0);
            float t = (f < 0.0) ? f - g : f + g;
            f = ((x - z) * (x + z) + h * (y / t - h)) / x;

            /* Next Q R transformation */

            c = s = 1.0;
            for (i = l + 1; i <= k; i++) {
                g = e[i];
                y = q[i];
                h = s * g;
                g = c * g;
                z = sqrt(f * f + h * h);
                e[i - 1] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y = y * c;

                if (u_ != NULL) 
                for (j = 0; j < m; j++) {
                    x = u[j][i - 1];
                    z = u[j][i];
                    u[j][i - 1] = x * c + z * s;
                    u[j][i] = -x * s + z * c;
                }
                z = sqrt(f * f + h * h);
                q[i - 1] = z;
                c = f / z;
                s = h / z;
                f = c * g + s * y;
                x = -s * g + c * y;

                if (vt_ != NULL || u_ == NULL)
                for (j = 0; j < n; j++) {
                    y = vt[i - 1][j];
                    z = vt[i][j];
                    vt[i - 1][j] = y * c + z * s;
                    vt[i][j] = -y * s + z * c;
                }
            }
            e[l] = 0;
            e[k] = f;
            q[k] = x;
        }

        /* convergence */
        if (q[k] < 0.0) { /* ensure q values are positive */
            q[k] = -z;
            if (u_ != NULL)
            for (j = 0; j < m; j++)
                u[j][k] = -u[j][k];
        }
    }
}

/* Reorders the values of q[m] in descending order,
 * and the columns  of u[m,m] and the rows of vt[m,n]
 * such that a = u diag(q) vt is preserved.
 * assume n > m
 */
static void reorder_wide(int n, int m,
                     fVec q_/*[m]*/, fArr2D vt_/*[m][n]*/, fArr2D u_/*[m][m]*/)
{
    typedef float (*VecM);
    typedef float (*ArrMM)[m];
    typedef float (*ArrMN)[n];
    VecM  q =  (VecM)  q_;
    ArrMM u =  (ArrMM) u_;
    ArrMN vt = (ArrMN) vt_;

    int i, j, k;
    float q_t;
    float u_t[m];
    float vt_t[n];

    /* Check whether q already is ordered */
    for (i = 1; i < m - 1; i++)
        if (q[i - 1] < q[i])
            break;
    if (i == m)
        return;

    for(i = 1; i < m; i++) {
        q_t = q[i];

        if (u_ != NULL)
        for (k = 0; k < m; k++)
            u_t[k] = u[k][i];
        if (vt_ != NULL)
        for (k = 0; k < n; k++)
            vt_t[k] = vt[i][k];

        for (j = i - 1; j >= 0 && q[j] < q_t; j--) {
            q[j + 1] = q[j];
            if (u_ != NULL)
            for (k = 0; k < m; k++)
                u[k][j + 1] = u[k][j];
            if (vt_ != NULL)
            for (k = 0; k <  n; k++)
                vt[j + 1][k] = vt[j][k];
        }
        q[j + 1] = q_t;

        if (u_ != NULL)
        for (k = 0; k < m; k++)
            u[k][j + 1] = u_t[k];

        if (vt_ != NULL)
        for (k = 0; k < n; k++)
            vt[j + 1][k] = vt_t[k];
    }
}
