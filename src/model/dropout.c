/* Copyright (c) 2026 Gilad Odinak */
/* Dropout layer data functions    */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "float.h"
#include "mem.h"
#include "array.h"
#include "random.h"
#include "dropout.h"

/* Creates a dropout layer.
 *
 * Parameters:
 *   rate - Dropout rate (0.0 to 1.0)
 *
 * Returns:
 *   Pointer to a dropout
 *
 * Notes:
 *   - The layer needs to be further initialized using dropout_init()
 *     before it can be used.
 */
DROPOUT* dropout_create(float rate)
{
    DROPOUT *l = allocmem(1,1,DROPOUT);
    if (rate < 0) rate = 0;
    else if (rate > 1) rate = 1;
    l->rate = rate;
    return l;
}

/* Initializes a dropout layer created by dropout_create().
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *   input_dim  - Size of input vectors
 *
 * Notes:
 *   The output weights are initialized using a normal distribution
 *   scaled by 1/sqrt(D).
 */
void dropout_init(DROPOUT* l, int input_dim, int batch_size)
{
    l->B = batch_size;
    l->D = input_dim;
    l->mask = allocmem(l->B,l->D,float);
}

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dropout_init(), it does nothing.
 */
void dropout_set_batch_size(DROPOUT* l, int batch_size)
{
    if (l->B == 0 || l->B == batch_size) return;
    l->B = batch_size;
    freemem(l->mask);
    l->mask = allocmem(l->B,l->D,float);
}

/* Frees the memory allocated by dropout_create() / dropout_init().
 *
 * Parameters:
 *   l - Pointer to the layer to be freed
 */
void dropout_free(DROPOUT* l)
{
    if (l == NULL) return;
    freemem(l->mask);
    freemem(l);
}

/* Resets any state the layer carries across batches.
 *
 * Parameters:
 *   l - Pointer to the layer to be reset
 */
void dropout_reset(DROPOUT* l)
{
    (void) l; /* do nothing */
}
