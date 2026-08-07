/* Copyright (c) 2023-2024 Gilad Odinak */
/* Embedding neural network layer functions */
#include <math.h>
#include "mem.h"
#include "array.h"
#include "random.h"
#include "embedding.h"

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
EMBEDDING* embedding_create(int embedding_dim, int context_len, int padinx)
{
    EMBEDDING* l = allocmem(1,1,EMBEDDING);
    l->E = embedding_dim;
    l->M = context_len;
    l->padinx = padinx;
    l->h = NULL;
    l->Wx = NULL;
    return l;
}

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
void embedding_init(EMBEDDING* l, int vocab_size, int batch_size, int training)
{
    l->D = vocab_size;
    l->B = batch_size;
    l->Wx = allocmem(l->D,l->E,float);
    if (!training)
        return;
    l->h = allocmem(l->B,l->E,float);
    typedef float (*ArrDE)[l->E];
    ArrDE Wx = (ArrDE) l->Wx;

    float scale = 1.0 / l->E;
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->E; j++)
            Wx[i][j] = urand(-scale,scale);
    if (l->padinx >= 0 && l->padinx < vocab_size)
        fltclr(Wx[l->padinx],l->E);
}

/* Frees the memory allocated by embedding_create() / embedding_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void embedding_free(EMBEDDING* l)
{
    freemem(l->Wx);
    freemem(l->h);
    freemem(l);
}

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the EMBEDDING neural network layer to be reset
 */
void embedding_reset(EMBEDDING* l)
{
    (void) l; /* Do nothing */
}
