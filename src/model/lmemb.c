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
