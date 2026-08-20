/* Copyright (c) 2026 Gilad Odinak */
/* Group normalization data structure and functions */
#include <stdio.h>
#include <stdlib.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "grpnorm.h"

/* Creates a normalization layer.
 *
 * Parameters:
 *   groups - number of channel groups.
 *
 * Returns:
 *   Pointer to a normalization layer.
 *
 * Note:
 *   The normalization layer need to be further intialized using 
 *   grpnorm_init() before it can be used.
 *
 */
GRPNORM* grpnorm_create(int groups)
{
    GRPNORM* l = allocmem(1,1,GRPNORM);
    l->G = groups;
    return l;
}

/* Initializes a normalization layer.
 *
 * Parameters:
 *   channels   - C, number of channels per time step
 *   steps      - T, number of time steps per batch item
 *   batch_size - B, number of batch items
 *
 * Notes:
 *   The number of channels must be a multiple of the group count
 *   Initialises gamma=1, beta=0 (identity transform).
 */
void grpnorm_init(GRPNORM* l, int channels, int steps, int batch_size)
{
    if (channels % l->G != 0) {
        fflush(stdout);
        fprintf(stderr,"grpnorm_init: channels %d not an integral multiple of groups %d\n",channels,l->G);
        freemem(l);
        exit(-1);
    }
    l->B = batch_size;
    l->T = steps;
    l->C = channels;
    l->mean = allocmem(batch_size * l->G,1,float);
    l->sdev = allocmem(batch_size * l->G,1,float);
    l->xn = allocmem(batch_size * steps,channels,float);
    l->gamma = allocmem(channels,1,float);
    l->beta = allocmem(channels,1,float);
    for (int j = 0; j < channels; j++)
        l->gamma[j] = 1;
}

/* Frees the memory allocated by grpnorm_create() / grpnorm_init()
 *
 * Parameters:
 *   l - pointer to the normalization layer layer to free
 */
void grpnorm_free(GRPNORM* l)
{
    if (l == NULL) return;
    freemem(l->mean);
    freemem(l->sdev);
    freemem(l->xn);
    freemem(l->gamma);
    freemem(l->beta);
    freemem(l);
}
