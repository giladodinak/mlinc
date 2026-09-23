/* Copyright (c) 2023-2024 Gilad Odinak */
/* Loss functions and their derivatives */
#ifndef LOSS_H
#define LOSS_H
#include "arraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Calculates cross-entropy loss for a batch of probability vectors
 * and correspoding class labels.
 * yp[T][K] - array of T vectors, each having K class probabilites
 * yt[T][K] - array of T vectors, each having K one-hot encoded class
 * Returns the sum of cross-entropy loss value of all T vector pairs
 */
float cross_entropy_loss(const fArr2D yp_, const fArr2D yt_, int T, int K);

/* Calculates cross-entropy loss for a batch of probability vectors
 * and correspoding class label indices.
 * yp[T][K] - array of T vectors, each having K class probabilites
 * yt[T][1] - array of class indices
 * Returns the sum of cross-entropy loss value of all pairs
 */
float sparse_cross_entropy_loss(const fArr2D yp_,
                                const fArr2D yt_, int T, int K);

/* Calculates mean square error for a batch of prediction vectors
 * and correspoding true vectors.
 * y_batch[M][N] - array of M predicted vectors, each having N elements
 * y_true[M][N]  - array of M true vectors, each having N elements
 * Returns the mean-square error of all M vector pairs
 */
float mean_square_error(const fArr2D y_batch_,
                        const fArr2D y_true_, int M, int N);

/* Calculates the gradient of the combined softmax and cross-entropy loss
 * with respect to the input scores (dL/dy), for a batch of predictions
 * and their true labels.
 *
 * This function assumes yp is the output of a softmax activation.
 * The gradient simplifies to (yp - yt) / T, which is NOT the gradient
 * of cross-entropy alone (-yt / yp), but the combined gradient of
 * softmax + cross-entropy.
 *
 * Reference:
 *   Dahal, Paras. (Jun 2017). Softmax and Cross Entropy Loss. 
 *   https://parasdahal.com/softmax-crossentropy.
 *
 * yp - predicted vector
 * yt - corresponding true vector
 * dy - gradient of the loss
 * T  - batch size
 * K  - vectors dimension
 */
void dLdy_cross_entropy_loss(const fArr2D yp_/*[T][K]*/,
                             const fArr2D yt_/*[T][K]*/,
                             fArr2D dy_/*[T][K]*/,
                             int T, int K);

/* Calculates the gradient of the combined softmax and cross-entropy loss
 * with respect to the input scores (dL/dy), for sparse class labels.
 *
 * This function assumes yp is the output of a softmax activation.
 * The gradient simplifies to (yp - yt) / T, which is NOT the gradient
 * of cross-entropy alone (-yt / yp), but the combined gradient of
 * softmax + cross-entropy.
 *
 * yp - predicted vector
 * yt - corresponding class indexes of the true vector
 * dy - gradient of the loss
 * T  - batch size
 * K  - vectors dimension
 */
void dLdy_sparse_cross_entropy_loss(const fArr2D yp_/*[T][K]*/,
                                    const fArr2D yt_/*[T][1]*/,
                                    fArr2D dy_/*[T][K]*/,
                                    int T, int K);

/* Calculates the gradient of the mean square error loss with respect to 
 * prediction (dL/dy) for a batch of predicted vector and one true vector.
 *
 * yp - predicted vector
 * yt - corresponding true vector
 * dy - gradient of the error
 * M  - batch size
 * N  - vectors dimension
 */
void dLdy_mean_square_error(const fArr2D yp_/*[M][N]*/,
                            const fArr2D yt_/*[M][N]*/,
                            fArr2D dy_/*[M][N]*/,
                            int M, int N);

#ifdef __cplusplus
}
#endif

#endif
