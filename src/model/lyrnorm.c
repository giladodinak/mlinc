/* Copyright (c) 2026 Gilad Odinak */
/* Layer normalization data structure and functions */
#include "mem.h"
#include "float.h"
#include "array.h"
#include "lyrnorm.h"

/* Creates a normalization layer.
 *
 * Returns:
 *   Pointer to a normalization layer.
 *
 * Note:
 *   The normalization layer need to be further intialized using 
 *   lyrnorm_init() before it can be used.
 *
 */
LYRNORM* lyrnorm_create(void)
{
    return allocmem(1,1,LYRNORM);
}

/* Initializes a normalization layer.
 *
 * Parameters:
 *   input_dim  - D, dimension of each input vector
 *   batch_size - B, number of input vectors
 *
 * Note:
 *   Initialises gamma=1, beta=0 (identity transform).
 */
void lyrnorm_init(LYRNORM* l, int input_dim, int batch_size)
{
    l->B = batch_size;
    l->D = input_dim;
    l->mean = allocmem(batch_size,1,float);
    l->sdev = allocmem(batch_size,1,float);
    l->xn = allocmem(batch_size,input_dim,float);
    l->gamma = allocmem(input_dim, 1,float);
    l->beta = allocmem(input_dim,1,float);
    for (int j = 0; j < input_dim; j++)
        l->gamma[j] = 1;
}

/* Frees the memory allocated by lyrnorm_create() / lyrnorm_init()
 *
 * Parameters:
 *   l - pointer to the normalization layer layer to free
 */
void lyrnorm_free(LYRNORM* l)
{
    if (l == NULL) return;
    freemem(l->mean);
    freemem(l->sdev);
    freemem(l->xn);
    freemem(l->gamma);
    freemem(l->beta);
    freemem(l);
}
