/* Copyright (c) 2023-2024 Gilad Odinak */
/* Dense (feed forward) neural network layer data structures and functions */
#ifndef DENSE_H
#define DENSE_H
#include "array.h"
#include "activation.h"

typedef struct dense_s {
  int D;           /* Input vector dimension                   */
  int S;           /* Number of units, size of hidden state    */
  int B;           /* Number of input vectors in a batch       */
  char activation; /* n,s,r,g,S (see below)                    */
  char use_bias;   /* 1 - add bias, 0 - do not add bias        */
  char training;   /* 1 - training mode, 0 - inference only    */
  fArr2D h;        /* Hidden State matrix [B][S]               */
  fArr2D z;        /* Pre-activation values of h (gelu only)   */
  fArr2D Wx;       /* Weights matrix [D][S]                    */
  fArr2D gWx;      /* Weight gradients (only if training == 1) */
  fVec b;          /* Bias vector [S] (only if use_bias == 1)  */
  fVec gb;         /* Bias gradients [S] (use_bias && training)*/
} DENSE;

/* Creates a feed forward neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   activation - Can be one of "none", "sigmoid", "relu", "gelu", or "Softmax"
 *   use_bias   - If set add bias
 *
 * Returns:
 *   Pointer to a dense neural network.
 *
 * Notes:
 *   - The neural network needs to be further intialized using dense_init()
 *     before it can be used.
 *   - Softmax is only supported with cross-entropy loss; the backward pass
 *     expects dy = y_pred - y_true. See d_softmax_xe() in activation.h
 */
DENSE* dense_create(int units, char* activation, int use_bias);

/* Initializes a feed forward neural network created by dense_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors
 *   batch_size - Number of input vectors processed simultaneously
 *   training   - 1: allocate backward/gradient buffers, 0 for inference-only
 *
 * Notes:
 *   The network's weights are initialized using glorot normal distribution 
 */
void dense_init(DENSE* l, int input_dim, int batch_size, int training);

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dense_init(), it does nothing.
 *   Otherwise, the network's hidden state is resized and re-initialized
 */
void dense_set_batch_size(DENSE* l, int batch_size);

/* Frees the memory allocated by dense_create() / dense_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void dense_free(DENSE* l);

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the DENSE neural network layer to be reset
 */
void dense_reset(DENSE* l);


static inline void add_bias(fArr2D h_, const fVec b_, int B, int S)
{
    typedef float (*ArrBS)[S];
    ArrBS h = (ArrBS) h_;
    const float* b = (const float*) b_;

    for (int i = 0; i < B; i++)
        for (int j = 0; j < S; j++)
            h[i][j] += b[j];
}

static inline void bias_gradient(fVec gb_, const fArr2D dy_, int B, int S)
{
    typedef float (*ArrBS)[S];
    const ArrBS dy = (ArrBS) dy_;
    float* gb = (float*) gb_;

    fltclr(gb,S);    
    for (int j = 0; j < S; j++)
        for (int i = 0; i < B; i++)
            gb[j] += dy[i][j];
}

/* Performs dense layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the dense layer's data
 *   X   - An array of input vectors (BxD dimensions, which includes bias)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the predicted values.
 * 
 * Note that in a multi-layered neural network, after the first layer
 * X is the (activated) output of a previous layer.
 */
static inline fArr2D dense_forward(DENSE* restrict l, 
                                   const fArr2D restrict X/*[B][D]*/, int lyr)
{
    (void) lyr;
    matmul(l->h,X,l->Wx,l->B,l->D,l->S); /* h = X @ Wx */
    if (l->use_bias)
        add_bias(l->h,l->b,l->B,l->S);
    switch (l->activation) {
        case 's' : sigmoid(l->h,l->B,l->S); break;
        case 'r' : relu(l->h,l->B,l->S); break;
        case 'g' : 
            if (l->training)
                fltcpy(l->z,l->h,l->B * l->S);
             gelu(l->h,l->B,l->S);
        break;
        case 'S' : softmax(l->h,l->B,l->S); break;
    }
    return l->h;
}

/* Performs dense layer training's backward pass.
 *
 * Parameters:
 *   l   - Pointer to the dense layer's data
 *   dy  - The output vector gradient of dense_create's units dimension
 *   X   - An input vector (D dimensions, which includes bias)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Calculates the weight matrix gradients with respect to the weights 
 * and update the matrix gWx
 *
 * Calculates the input vector gradient and returns it in dx, if dx is not NULL
 *
 * Note that in a multi-layered neural network, except the last layer,
 * dy is the gradient of the previous layer's input (dx), thus, the dimension
 * of dx (this layer's D) must equal the dimension of the previous layer's
 * dy (previous layer's S).
 */
static inline void dense_backward(DENSE* restrict l, 
                                  fArr2D restrict dy/*[B][S]*/, 
                                  const fArr2D restrict X/*[B][D]*/,
                                  fArr2D restrict dx/*[B][D]*/,
                                  int lyr)
{
    (void) lyr;

    switch (l->activation) {
        case 's':
            d_sigmoid(dy,l->h,l->B,l->S);
            break;
        case 'r':
            d_relu(dy,l->h,l->B,l->S);
            break;
        case 'g':
            d_gelu(dy,l->z,l->B,l->S);
            break;
        /* Softmax intentionally excluded:
         * dy must already be (y_pred - y_true)
         */
    }

    /* Gradient with respect to weights: gWx = X.T @ dy */
    Tmatmul(l->gWx,X,dy,l->D,l->B,l->S);
    if (l->use_bias)
        bias_gradient(l->gb,dy,l->B,l->S);

    if (dx != NULL) {
        /* dx = (dy @ Wx.T) */
        matmulT(dx,dy,l->Wx,l->B,l->S,l->D);
    }
}
#endif
