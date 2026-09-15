/* Copyright (c) 2023-2024 Gilad Odinak */
/* ADAM optimizer with weight decay data structures and functions    */
#ifndef ADAMW_H
#define ADAMW_H
#include "arraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Updates all weights in array w[M][N], according to the corresponding 
 * gradients in g[M][N], using the ADAM optimizer algorithm.  The arrays 
 * m[M][N] and v[M][N] stores coefficients used by the algorithm.
 * The rate of update is controlled by learning_rate, weight_decay.
 */
void adamw_update(fArr2D w_/*[M][N]*/,fArr2D g_/*[M][N]*/,
                  fArr2D m_/*[M][N]*/,fArr2D v_/*[M][N]*/,
                  int M, int N, 
                  float learning_rate, float weight_decay, int update_step);

#ifdef __cplusplus
}
#endif

#endif
