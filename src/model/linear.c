/* Copyright (c) 2023-2024 Gilad Odinak */
/* Function that updates neural network weights */
#include "arraytypes.h"
#include "linear.h"

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using linear update.
 * The rate of update is controlled by learning_rate, weight_decay.
 */
void linear_update(fArr2D w_/*[M][N]*/, fArr2D g_/*[M][N]*/,
                   int M, int N, float learning_rate, float weight_decay)
{
    typedef float (*ArrMN)[N];
    ArrMN w = (ArrMN) w_;
    ArrMN g = (ArrMN) g_;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            w[i][j] -= learning_rate * (g[i][j] + weight_decay * w[i][j]);
}
