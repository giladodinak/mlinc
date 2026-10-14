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
    l->devstat = NULL;
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

fArr2D embedding_forward(EMBEDDING* restrict l,
                         fArr2D restrict X_, int lyr)
{
    (void) lyr;
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBE)[l->E];
    typedef float (*ArrDE)[l->E];
    ArrBM X = (ArrBM) X_;
    ArrBE h = (ArrBE) l->h;
    ArrDE Wx = (ArrDE) l->Wx;

    fltclr(h,l->B * l->E);
    for (int i = 0; i < l->B; i++)
        for (int j = 0; j < l->M; j++)
            for (int k = 0; k < l->E; k++)
                h[i][k] += Wx[(int)X[i][j]][k];
    return l->h;
}

void embedding_backward(EMBEDDING* restrict l,
                        const fArr2D restrict dy_,
                        const fArr2D restrict X_,
                        fArr2D restrict gWx_,
                        int* grows, int* grcnt, int lyr)
{
    (void) lyr;
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBE)[l->E];
    typedef float (*ArrDE)[l->E];
    ArrBE dy = (ArrBE) dy_;
    ArrBM X = (ArrBM) X_;
    ArrDE gWx = (ArrDE) gWx_;

    if (grows != NULL && grcnt != NULL)
        *grcnt = 0;
    else
        fltclr(gWx,l->D * l->E);

    for (int i = 0; i < l->B; i++) {
        for (int j = 0; j < l->M; j++) {
            int xij = (int)X[i][j];
            if (xij != l->padinx) {
                if (grows != NULL && grcnt != NULL)
                    prep_row(gWx_,l->E,xij,grows,grcnt);
                for (int k = 0; k < l->E; k++)
                    gWx[xij][k] += dy[i][k];
            }
        }
    }
}

void embedding_update(EMBEDDING* restrict l,
                      const fArr2D restrict gWx_,
                      const int* grows, int grcnt, float learning_rate)
{
    typedef float (*ArrDE)[l->E];
    ArrDE Wx = (ArrDE)l->Wx;
    ArrDE gWx = (ArrDE)gWx_;

    for (int r = 0; r < grcnt; r++) {
        int i = grows[r];
        for (int j = 0; j < l->E; j++)
            Wx[i][j] -= learning_rate * gWx[i][j];
    }
}

