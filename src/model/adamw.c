/* Copyright (c) 2023-2024 Gilad Odinak */
/* ADAM optimizer with weight decay data structures and functions  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "array.h"
#include "clip.h"
#include "adamw.h"

/* Adaptive Moment Estimation with weight decay
 * https://arxiv.org/pdf/1711.05101.pdf
 * "Decoupled Weight Decay Regularization" - Algorithm 2 - AdamW
 */

static const float beta1 = 0.9;
static const float beta2 = 0.999;
static const float epsilon = 1.0e-7;

static inline void adamw(
    float* restrict w, // w: weight, for example Wf[][]
    float* restrict g, // g: gradient of the weight, for example gWf[][]
    float* restrict m, // m: moment1 of the gradient, for example mWf[][]
    float* restrict v, // v: moment2 of the gradient, for example vWf[][]
    const float bc1,   // precomputed 1 / (1 - beta1^update_num)
    const float bc2,   // precomputed 1 / (1 - beta2^update_num)
    float learning_rate,
    float weight_decay
)
{
    if (*v < 0) { /* weight, gradient explosion */
        fflush(stdout);
        fprintf(stderr,"adamw: weight or gradient explosion\n");
        exit(-1);
    }

    *m = beta1 * (*m) + (1.0f - beta1) * (*g);
    *v = beta2 * (*v) + (1.0f - beta2) * (*g) * (*g);

    float mh = (*m) * bc1;
    float vh = (*v) * bc2;
    float ag = mh / (sqrtf(vh) + epsilon);

    /* Weight update with weight decay */
    *w -= (learning_rate * (ag + weight_decay * (*w)));
}

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using the ADAM optimizer algorithm.  The arrays
 * m[M][N] and v[M][N] stores coefficients used by the algorithm.
 * The rate of update is controlled by learning_rate, weight_decay.
 */
void adamw_update(fArr2D w_/*[M][N]*/,fArr2D g_/*[M][N]*/,
                  fArr2D m_/*[M][N]*/,fArr2D v_/*[M][N]*/,
                  int M, int N,
                  float learning_rate, float weight_decay, int update_step)
{
    typedef float (*ArrMN)[N];
    ArrMN w = (ArrMN) w_;
    ArrMN g = (ArrMN) g_;
    ArrMN m = (ArrMN) m_;
    ArrMN v = (ArrMN) v_;

    clip_gradients(g,M,N,1.0e-16,10.0);

    /* Compute the bias-correction denominators once per update, not once
     * per weight. beta1^t and beta2^t depend only on the step number. */
    float bc1 = 1.0f / (1.0f - powf(beta1,update_step));
    float bc2 = 1.0f / (1.0f - powf(beta2,update_step));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            adamw(&(w[i][j]),&(g[i][j]),&(m[i][j]),&(v[i][j]),
                  bc1,bc2,learning_rate,weight_decay);
        }
    }
}
