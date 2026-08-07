/* Copyright (c) 2023-2024 Gilad Odinak */
/* Embedding neural network layer data structures and functions */
#ifndef EMBEDDING_H
#define EMBEDDING_H
#include "array.h"

typedef struct embedding_s {
  int D;       /* Vocabulary size            */
  int B;       /* Batch size                 */
  int M;       /* Context length             */
  int E;       /* Embedding dimension        */
  int padinx;  /* Pad index, -1 if not used  */
  fArr2D h;    /* Hidden State matrix [B][E] */
  fArr2D Wx;   /* Weights matrix [D][E]      */
} EMBEDDING;

/* Creates an embedding layer.
 *
 * Parameters:
 *   embedding_dim - Dimension of token embedding vectors
 *   context_len   - Number of token indices in a context
 *   padinx        - Index value of pad token, -1 if not used
 *
 *
 * Returns:
 *   Pointer to an EMBEDDING layer.
 *
 * Notes:
 *   - Contexts shorter than context_length are padded with blank (pad index)
 *   - The embedding needs to be further intialized using embedding_init()
 *     before it can be used.
 */
EMBEDDING* embedding_create(int embedding_dim, int context_len, int padinx);

/* Initializes an embedding layer created by embedding_create().
 *
 * Parameters:
 *   vocab_size - Number of vocabulary tokens (including blank, if any)
 *   batch_size - Number of input contexts processed simultaneously
 *   training   - 1: used for training, 0: container for embedding and weights
 *
 * Notes:
 *   If training, the weights are initialized using uniform distribution
 */
void embedding_init(EMBEDDING* l, int vocab_size, int batch_size, int training);

/* Frees the memory allocated by embedding_create() / embedding_init()
 *
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void embedding_free(EMBEDDING* l);

/* Resets the network hidden state.
 *
 * Parameters:
 *   l - Pointer to the EMBEDDING neural network layer to be reset
 */
void embedding_reset(EMBEDDING* l);

/* Prepares gradients row for update.
 *
 * Parameters:
 *   gWx  - gradient array [M][N]
 *   N    - number of elemens in each gradient row
 *   inx  - index of the row to be updated 0 <= 0 < M
 *   rows - array that stores the indices of the affected rows
 *   cnt  - points to the count of affected rows (updated in place)
 *   cnt  - points to a count of updated rows (output)
 *
 * On first update initializes the row zero, and records its index in rows.
 * On subsequent updates, skips rows that already have bee initialized.
 */
static inline void prep_row(fArr2D gWx_, int N, int inx, int* rows, int* cnt)
{
    for (int r = 0; r < *cnt; r++)
        if (rows[r] == inx)
            return;
    typedef float (*ArrMN)[N];
    ArrMN gWx = (ArrMN) gWx_;
    fltclr(gWx[inx],N);
    rows[(*cnt)++] = inx;
}

/* Performs embedding layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   X   - An array of input indices representing one hot encoded vectors
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the predicted values, h[B][E]
 *
 * Note that in a multi-layered neural network, after the first layer
 * X is the output of a previous layer.
 */
static inline fArr2D embedding_forward(EMBEDDING* restrict l,
                                       fArr2D restrict X_/*[B][M]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBE)[l->E];
    typedef float (*ArrDE)[l->E];
    ArrBM X = (ArrBM) X_;
    ArrBE h = (ArrBE) l->h;
    ArrDE Wx = (ArrDE) l->Wx;
    /* X[B][M] => x[B][M][D] where x[B][M] is one-hot encoded */
    /* h = x @ Wx  => sum context vectors */
    fltclr(h,l->B * l->E);
    for (int i = 0; i < l->B; i++)
        for (int j = 0; j < l->M; j++)
            for (int k = 0; k <l->E; k++)
                h[i][k] += Wx[(int)X[i][j]][k];
    return l->h;
}

/* Performs embedding layer training's backward pass.
 *
 * Parameters:
 *   l     - Pointer to the embedding layer's data
 *   dy    - The output vector gradient of embedding_create's units dimension
 *   X     - An array of input indices representing one hot encoded vectors
 *   gWx   - Gradient array (with respect to X)
 *   grows - If not NULL, points to an integer array of size B*M
 *   grcnt - If not NULL, points to an integer count of entries in grows
 *   lyr   - The ordinal number of this layer in a model (not used)
 *
 * Calculates the gradients with respect to the weights and adds them gWx.
 *
 * If grows is NULL, the whole gWx matrix is cleared.
 * If grows is not NULL, gWx is not fully cleared; instead each context row
 * is zeroed on first use and its index stored in grows[] with the count of
 * such rows returned in *grcnt. The caller then updates only those rows.
 */
static inline void embedding_backward(EMBEDDING* restrict l,
                                  const fArr2D restrict dy_/*[B][E]*/,
                                  const fArr2D restrict X_/*[B][M]*/,
                                  fArr2D restrict gWx_/*[D][E]*/,
                                  int* grows, int* grcnt, int lyr)
{
    (void) lyr;
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBE)[l->E];
    typedef float (*ArrDE)[l->E];
    ArrBE dy = (ArrBE) dy_;
    ArrBM X = (ArrBM) X_;
    ArrDE gWx = (ArrDE) gWx_;

    /* X[B][M] => x[B][M][D] where x[B][M] is one-hot encoded */
    /* Gradient with respect to weights: gWx = x.T @ dy                     */
    /* Only context rows receive gradients. If grows != NULL, zero each     */
    /* such row on first use and record it (sparse); else clear all of gWx. */
    if (grows != NULL && grcnt != NULL)
        *grcnt = 0;
    else
        fltclr(gWx,l->D * l->E);
    for (int i = 0; i < l->B; i++) {
        for (int j = 0; j < l->M; j++) {
            int xij = (int) X[i][j];
            if (xij != l->padinx) {
                if (grows != NULL && grcnt != NULL)
                    prep_row(gWx_,l->E,xij,grows,grcnt);
                for (int k = 0; k < l->E; k++)
                    gWx[xij][k] += dy[i][k];
            }
        }
    }
}
#endif
