/* Copyright (c) 2026 Gilad Odinak */
/* Language-model token embedding layer functions */
#include <math.h>
#include "mem.h"
#include "array.h"
#include "random.h"
#include "lmemb.h"

/* Creates a language-model token embedding layer.
 *
 * Parameters:
 *   embedding_dim - Dimension of token embedding vectors (E)
 *   steps         - Sequence length T (number of tokens per sequence)
 *   padinx        - Index value of pad token, -1 if not used
 *
 * Returns:
 *   Pointer to an LMEMB layer.
 *
 * Notes:
 *   - The layer needs to be further initialized using lmemb_init() before
 *     it can be used.
 */
LMEMB* lmemb_create(int embedding_dim, int steps, int padinx)
{
    LMEMB* l = allocmem(1,1,LMEMB);
    l->E = embedding_dim;
    l->T = steps;
    l->padinx = padinx;
    l->h = NULL;
    l->Wx = NULL;
    l->gWx = NULL;
    l->mWx = NULL;
    l->vWx = NULL;
    l->seen = NULL;
    l->touched = NULL;
    l->ids = NULL;
    l->devstat = NULL;
    return l;
}

/* Initializes a language-model token embedding layer created by lmemb_create().
 *
 * Parameters:
 *   vocab_size - Number of vocabulary tokens V (including pad, if any)
 *   batch_size - Number of sequences processed simultaneously, B.
 *   training   - 1: allocate backward/optimizer buffers; 0: inference only
 *   tied       - 1: Wx/gWx/moments are borrowed from an output projection and
 *                   are set later with lmemb_set_tied_weights(); 0: owned here
 *
 * Notes:
 *   - When not tied, Wx is allocated here. If training, it is initialized with
 *     a normal distribution scaled by 1/sqrt(E), and the pad row is zeroed;
 *     if not training, Wx is left for the caller to load.
 *   - When tied, no weight/gradient/moment buffers are allocated here.
 *   - The output buffer h and, when training, the sparse touched-row buffers
 *     are always allocated (they are owned regardless of tying).
 */
void lmemb_init(LMEMB* l, int vocab_size, int batch_size, int training, int tied)
{
    l->V = vocab_size;
    l->B = batch_size;
    l->training = training;
    l->tied = tied;
    l->stamp = 0;
    l->ntouched = 0;
    l->BT = l->B * l->T;

    l->h = allocmem(l->BT,l->E,float);

    if (!tied) {
        l->Wx = allocmem(l->V,l->E,float);
        if (training) {
            typedef float (*ArrVE)[l->E];
            ArrVE Wx = (ArrVE) l->Wx;
            float scale = 1.0 / sqrtf((float) l->E);
            for (int i = 0; i < l->V; i++)
                for (int j = 0; j < l->E; j++)
                    Wx[i][j] = nrand(0.0,scale);
            if (l->padinx >= 0 && l->padinx < vocab_size)
                fltclr(Wx[l->padinx],l->E);
        }
    }
    /* When tied, Wx/gWx are set later via lmemb_set_tied_weights() */

    if (!training)
        return;

    /* Backward / optimizer buffers */
    if (!tied) {
        l->gWx = allocmem(l->V,l->E,float);
        l->mWx = allocmem(l->V,l->E,float);
        l->vWx = allocmem(l->V,l->E,float);
    }
    /* Sparse touched-row tracking is owned regardless of tying */
    l->seen = allocmem(l->V,1,int);
    for (int i = 0; i < l->V; i++)
        l->seen[i] = -1;
    l->touched = allocmem(l->BT,1,int);
}

/* Points a tied layer's Wx/gWx at the output projection's shared buffers.
 * Only valid when the layer was initialized with tied = 1.
 *
 * Parameters:
 *   Wx  - Shared weight matrix   [V][E]
 *   gWx - Shared weight gradient  [V][E]
 */
void lmemb_set_tied_weights(LMEMB* l, fArr2D Wx, fArr2D gWx)
{
    if (!l->tied) return;
    l->Wx = Wx;
    l->gWx = gWx;
    /* mWx/vWx remain NULL: the output layer owns and steps the moments */
}

/* Frees the memory allocated by lmemb_create() / lmemb_init().
 * When tied, the borrowed Wx/gWx/moments are not freed.
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void lmemb_free(LMEMB* l)
{
    if (l == NULL) return;
    freemem(l->h);
    if (!l->tied) {
        freemem(l->Wx);
        freemem(l->gWx);
        freemem(l->mWx);
        freemem(l->vWx);
    }
    freemem(l->seen);
    freemem(l->touched);
    freemem(l);
}

/* Resets any per-batch state the layer carries.
 *
 * Parameters:
 *   l - Pointer to the LMEMB layer to be reset
 */
void lmemb_reset(LMEMB* l)
{
    l->ntouched = 0;
    /* stamp continues to advance; seen[] stays valid across resets */
}

/* Performs the token embedding forward pass: a per-token gather.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   ids - An array of B*T token indices [B*T]
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the output values, h[B*T][E]. Row t is Wx[ids[t]].
 *
 * The ids pointer is retained for the matching backward pass.
 */
fArr2D lmemb_forward(LMEMB* restrict l,
                     const int* restrict ids/*[B*T]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBTE)[l->E];
    typedef float (*ArrVE)[l->E];
    ArrBTE h = (ArrBTE) l->h;
    ArrVE Wx = (ArrVE) l->Wx;

    l->ids = ids;
    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        for (int k = 0; k < l->E; k++)
            h[i][k] = Wx[id][k];
    }
    return l->h;
}

/* Performs the token embedding backward pass: a per-token scatter-add.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   dy  - The output gradient [B*T][E], from the layer above (the stack input)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Accumulates each token's gradient into its Wx row of gWx, recording the
 * distinct touched rows in l->touched / l->ntouched for a sparse update.
 * The token indices from the last lmemb_forward() are used. There is no
 * input gradient: this is the first layer of the model.
 *
 * Rows are zeroed on first touch only when this layer owns gWx (untied);
 * when tied, the shared gWx is prepared by the output layer and this pass
 * only accumulates. A token index appearing at several positions accumulates.
 */
void lmemb_backward(LMEMB* restrict l,
                    const fArr2D restrict dy_/*[B*T][E]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBTE)[l->E];
    typedef float (*ArrVE)[l->E];
    ArrBTE dy = (ArrBTE) dy_;
    ArrVE gWx = (ArrVE) l->gWx;
    const int* ids = l->ids;

    l->stamp++;
    l->ntouched = 0;

    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        if (id == l->padinx)
            continue;
        if (l->seen[id] == l->stamp)
            continue;
        l->seen[id] = l->stamp;
        l->touched[l->ntouched++] = id;
    }

    if (l->ntouched > 0) {
        if (!l->tied) {
            for (int r = 0; r < l->ntouched; r++)
                fltclr(gWx[l->touched[r]],l->E);
        }
    }

    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        if (id == l->padinx)
            continue;
        for (int k = 0; k < l->E; k++)
            gWx[id][k] += dy[i][k];
    }
}

/* Applies one sparse SGD step to rows touched by lmemb_backward().
 * Weight decay is intentionally omitted for embeddings.
 */
void lmemb_update(LMEMB* restrict l, float learning_rate)
{
    if (l->tied)
        return;

    typedef float (*ArrVE)[l->E];
    ArrVE Wx = (ArrVE) l->Wx;
    ArrVE gWx = (ArrVE) l->gWx;

    for (int r = 0; r < l->ntouched; r++) {
        int row = l->touched[r];
        for (int j = 0; j < l->E; j++)
            Wx[row][j] -= learning_rate * gWx[row][j];
    }
}

