/* Copyright (c) 2023-2024 Gilad Odinak */
/* Loss functions and their derivatives */
#include <math.h>
#include "arraytypes.h"
#include "loss.h"

/* Calculates cross-entropy loss for a batch of probability vectors
 * and correspoding class labels.
 * yp[T][K] - array of T vectors, each having K class probabilites
 * yt[T][K] - array of T vectors, each having K one-hot encoded class
 * Returns the sum of cross-entropy loss value of all T vector pairs
 */
float cross_entropy_loss(const fArr2D yp_, const fArr2D yt_, int T, int K)
{
    typedef float (*ArrTK)[K];
    const ArrTK yp = (const ArrTK) yp_;
    const ArrTK yt = (const ArrTK) yt_;
    float loss = 0.0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < K; j++)
            loss += -yt[i][j] * log(fmax(yp[i][j],1e-16));
    return loss;
}

/* Calculates cross-entropy loss for a batch of probability vectors
 * and correspoding class label indices.
 * yp[T][K] - array of T vectors, each having K class probabilites
 * yt[T][1] - array of class indices
 * Returns the sum of cross-entropy loss value of all pairs
 */
float sparse_cross_entropy_loss(const fArr2D yp_,
                                const fArr2D yt_, int T, int K)
{
    typedef float (*ArrTK)[K];
    typedef float (*ArrT1)[1];
    const ArrTK yp = (const ArrTK) yp_;
    const ArrT1 yt = (const ArrT1) yt_;
    float loss = 0.0;
    for (int i = 0; i < T; i++)
        loss += -log(fmax(yp[i][(int) yt[i][0]],1e-16));
    return loss;
}

/* Calculates mean square error for a batch of prediction vectors
 * and correspoding true vectors.
 * y_batch[M][N] - array of M predicted vectors, each having N elements
 * y_true[M][N]  - array of M true vectors, each having N elements
 * Returns the mean-square error of all M vector pairs
 */
float mean_square_error(const fArr2D y_batch_,
                        const fArr2D y_true_, int M, int N)
{
    
    typedef float (*ArrMN)[N];
    const ArrMN y_batch = (const ArrMN) y_batch_;
    const ArrMN y_true = (const ArrMN) y_true_;
    float error = 0.0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            error += pow(y_batch[i][j] - y_true[i][j],2);
    return error / M / N;
}

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
                             int T, int K)
{
    typedef float (*ArrTK)[K];
    const ArrTK yp = (const ArrTK) yp_;
    const ArrTK yt = (const ArrTK) yt_;
    ArrTK dy = (ArrTK) dy_;

    for (int i = 0; i < T; i++)
        for (int j = 0; j < K; j++)
            dy[i][j] = (yp[i][j] - yt[i][j]) / T;
}

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
                                    int T, int K)
{
    typedef float (*ArrTK)[K];
    typedef float (*ArrT1)[1];
    const ArrTK yp = (const ArrTK) yp_;
    const ArrT1 yt = (const ArrT1) yt_;
    ArrTK dy = (ArrTK) dy_;

    for (int i = 0; i < T; i++)
        for (int j = 0; j < K; j++)
            dy[i][j] = (yp[i][j] - ((j == (int) yt[i][0]) ? 1 : 0)) / T;
}

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
                            int M, int N)
{
    typedef float (*ArrMN)[N];
    const ArrMN yp = (const ArrMN) yp_;
    const ArrMN yt = (const ArrMN) yt_;
    ArrMN dy = (ArrMN) dy_;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            dy[i][j] = 2 * (yp[i][j] - yt[i][j]) / N / M;
}

