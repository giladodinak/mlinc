/* Copyright (c) 2023-2024 Gilad Odinak   */
/* LSTM (reccurrent) neural network layer */
#ifndef LSTM_H
#define LSTM_H
#include "arraytypes.h"

typedef struct lstm_s {
  int D;           /* Input vector dimension                        */
  int S;           /* Number of units, size of hidden state         */
  int B;           /* Number of input vectors in a batch            */
  /* Note that B also is the sequence length (number of time steps) */
  int stateful;    /* 1: maintain state between batches             */
  /* Stateful mode preserves the final hidden and cell state across
   * forward passes only. Gradients do NOT propagate across sequences
   * (i.e., truncated BPTT with truncation at sequence boundaries).
   */
  char use_bias;   /* 1 - add bias, 0 - do not add bias             */
  char training;   /* 1 - training mode, 0 - inference only         */
  fArr2D Wf;       /* Weights matrix [D][S]                         */
  fArr2D Wi;       /* Weights matrix [D][S]                         */
  fArr2D Wc;       /* Weights matrix [D][S]                         */
  fArr2D Wo;       /* Weights matrix [D][S]                         */
  fArr2D Uf;       /* Recurrent Weights matrix [S][S]               */
  fArr2D Ui;       /* Recurrent Weights matrix [S][S]               */
  fArr2D Uc;       /* Recurrent Weights matrix [S][S]               */
  fArr2D Uo;       /* Recurrent Weights matrix [S][S]               */
  fVec bf;         /* Bias vector [S] (only if use_bias == 1)       */
  fVec bi;         /* Bias vector [S] (only if use_bias == 1)       */
  fVec bc;         /* Bias vector [S] (only if use_bias == 1)       */
  fVec bo;         /* Bias vector [S] (only if use_bias == 1)       */
  fArr2D f;        /* Forget gate matrix [B][S]                     */
  fArr2D i;        /* Input  gate matrix [B][S]                     */
  fArr2D o;        /* Output gate matrix [B][S]                     */
  fArr2D cc;       /* Cell candidate matrix [B+1][S]                */
  fArr2D c;        /* Cell matrix [B+1][S]                          */
  fArr2D h;        /* Hidden state matrix [B+1][S]                  */
  fVec ph;         /* Previous batch last hidden state vector [S]   */
  fVec pc;         /* Previous batch last cell state vector [S]     */
  /* Gradients (only if training == 1) */
  fArr2D gWf;      /* Weight gradients [D][S]                       */
  fArr2D gWi;      /* Weight gradients [D][S]                       */
  fArr2D gWc;      /* Weight gradients [D][S]                       */
  fArr2D gWo;      /* Weight gradients [D][S]                       */
  fArr2D gUf;      /* Recurrent Weight gradients [S][S]             */
  fArr2D gUi;      /* Recurrent Weight gradients [S][S]             */
  fArr2D gUc;      /* Recurrent Weight gradients [S][S]             */
  fArr2D gUo;      /* Recurrent Weight gradients [S][S]             */
  fVec gbf;        /* Bias gradients [S] (use_bias && training)     */
  fVec gbi;        /* Bias gradients [S] (use_bias && training)     */
  fVec gbc;        /* Bias gradients [S] (use_bias && training)     */
  fVec gbo;        /* Bias gradients [S] (use_bias && training)     */
} LSTM;

#ifdef __cplusplus
extern "C" {
#endif

/* Creates a long short term memory (LSTM) neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   stateful   - If not zero, maintain state across batches.
 *   use_bias   - If not zero, add a per-gate bias.
 * 
 * Returns:
 *   Pointer to an LSTM neural network layer.
 *
 * Notes:
 *   - The neural network needs to be further initialized using lstm_init()
 *     before it can be used.
 */
LSTM* lstm_create(int units, int stateful, int use_bias);

/* Initializes an LSTM neural network created by lstm_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors
 *   batch_size - Number of input vectors processed simultaneously
 *   training   - 1: allocate gradient buffers, 0 for inference-only
 *
 * Notes:
 *   - Kernel weights (Wx) are initialized using Glorot normal distribution.
 *   - Recurrent weights (Ux) are initialized using orthogonal uniform 
 *     distribution.
 */
void lstm_init(LSTM* l, int input_dim, int batch_size, int training);

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before ilstm_init(), it does nothing.
 *   Otherwise, the network's hidden states are resized and re-initialized
 */
void lstm_set_batch_size(LSTM* l, int batch_size);

/* Frees the memory allocated by lstm_create() / lstm_init().
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be freed
 */
void lstm_free(LSTM* l);

/* Resets the LSTM internal state.
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be reset
 */
void lstm_reset(LSTM* l);

/* Performs LSTM layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the LSTM layer's data
 *   X   - Array of input vectors BxD, where B is the number of 
 *         input vectors, and D is the number of features in each vector
 *   lyr - Ordinal number of this layer in a model (not used)
 * 
 * Returns:
 *   Pointer to the predicted values.
 * 
 * Note:
 * - In a multi-layered neural network, after the first layer,
 *   X is the (activated) output of a previous layer.
 */
fArr2D lstm_forward(LSTM* restrict l,
                    const fArr2D restrict X /*[B][D]*/, int lyr);

/* Performs LSTM layer training's backward pass.
 * 
 * Parameters:
 *   l    - Pointer to the LSTM layer's data
 *   dY   - Output vector gradient of lstm_create's units dimension
 *   X    - Array of input vectors BxD, where B is the number of input
 *          vectors, and D is the number of features in each vector
 *   dX   - Output parameter for the input vector gradient (if not NULL)
 *   lyr  - Ordinal number of this layer in a model (not used)
 * 
 * Returns:
 *   None
 * 
 * Note:
 *   - Calculates the weight and bias gradients and stores them in the
 *     layer's own gradient buffers (gWf..gUo, and gbf..gbo if use_bias).
 *   - Calculates the input vector gradient and returns it in dx, if dx is 
 *     not NULL
 *   - When stateful is enabled, forward state is carried across calls,
 *     but gradients do not propagate across sequence boundaries.
 *   - In a multi-layered neural network, except the last layer, dy is the 
 *     gradient of the previous layer's input (dx), thus, the dimension of 
 *     dx (this layer's D) must equal the dimension of the previous layer's
 *     dy (previous layer's S)
 */
void lstm_backward(LSTM* restrict l,
                   const fArr2D restrict dY/*[B][S]*/,
                   const fArr2D restrict X/*[B][D]*/,
                   fArr2D restrict dX/*[B][D]*/,
                   int lyr);


#ifdef __cplusplus
}
#endif

#endif
