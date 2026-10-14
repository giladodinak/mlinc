/* Copyright (c) 2023-2024 Gilad Odinak */
/* Embedding neural network layer data structures and functions */
#ifndef EMBEDDING_H
#define EMBEDDING_H
#include "arraytypes.h"

typedef struct embedding_s {
  int D;       /* Vocabulary size            */
  int B;       /* Batch size                 */
  int M;       /* Context length             */
  int E;       /* Embedding dimension        */
  int padinx;  /* Pad index, -1 if not used  */
  fArr2D h;    /* Hidden State matrix [B][E] */
  fArr2D Wx;   /* Weights matrix [D][E]      */
  void* devstat; /* Additional device state (NULL if not used) */
} EMBEDDING;

#ifdef __cplusplus
extern "C" {
#endif

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

/* Performs embedding layer training/prediction's forward pass.
 *
 * X is an array of input token indices [B][M].
 * Returns h[B][E].
 */
fArr2D embedding_forward(EMBEDDING* restrict l,
                         fArr2D restrict X/*[B][M]*/, int lyr);

/* Performs embedding layer training's backward pass.
 *
 * dy    - Output gradient [B][E]
 * X     - Input token indices [B][M]
 * gWx   - Weight gradients [D][E]
 * grows - If not NULL, receives distinct touched rows [B*M]
 * grcnt - If not NULL, receives number of touched rows
 */
void embedding_backward(EMBEDDING* restrict l,
                        const fArr2D restrict dy/*[B][E]*/,
                        const fArr2D restrict X/*[B][M]*/,
                        fArr2D restrict gWx/*[D][E]*/,
                        int* grows, int* grcnt, int lyr);

/* Updates only the embedding rows listed in grows[]. */
void embedding_update(EMBEDDING* restrict l,
                      const fArr2D restrict gWx/*[D][E]*/,
                      const int* grows, int grcnt, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif
