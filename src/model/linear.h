/* Copyright (c) 2023-2024 Gilad Odinak */
/* Function that updates neural network weights */
#ifndef LINEAR_H
#define LINEAR_H
#include "arraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using linear update.
 * The rate of update is controlled by learning_rate, weight_decay.
 */
void linear_update(fArr2D w_/*[M][N]*/, fArr2D g_/*[M][N]*/,
                   int M, int N, float learning_rate, float weight_decay);


#ifdef __cplusplus
}
#endif

#endif
