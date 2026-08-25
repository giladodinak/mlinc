/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN sampled-softmax layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "smsftmax.h"
#include "smsftmaxio.h"

/* read_smsftmax - Read a sampled-softmax layer from a file
 *
 * Reads a sampled-softmax layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read layer if successful, NULL otherwise
 *
 * Notes:
 *   - Only the output weights (Wo) are persisted; the passthrough, touched,
 *     and seen buffers are scratch and are (re)allocated here.
 *   - The unigram sampling table (dist) is not owned by the layer and is not
 *     persisted; the caller must re-attach it via smsftmax_set_dist() after
 *     loading, before calling smsftmax_loss(). smsftmax_set_dist() also fills
 *     the logq[] table, which is allocated (uninitialised) here.
 */
SMSFTMAX* read_smsftmax(FILE* fp)
{
    int K, E, B, n_neg;
    int cnt = fscanf(fp," SMSFTMAX K %d E %d B %d n_neg %d\n",
                     &K,&E,&B,&n_neg);
    if (cnt < 4 || cnt == EOF) {
        fprintf(stderr,"In read_smsftmax: failed to read the header\n");
        return NULL;
    }
    SMSFTMAX* l = allocmem(1,1,SMSFTMAX);
    l->K = K;
    l->E = E;
    l->B = B;
    l->n_neg = n_neg;
    l->Wo = allocmem(l->K,l->E,float);
    l->h = allocmem(l->B,l->E,float);
    l->touched = allocmem(l->B * (l->n_neg + 1),1,int);
    l->ntouched = 0;
    l->seen = allocmem(l->K,1,int);
    for (int i = 0; i < l->K; i++)
        l->seen[i] = -1;
    l->stamp = 0;
    l->dist = NULL;
    l->dist_size = 0;
    l->logq = allocmem(l->K,1,float);   /* filled by smsftmax_set_dist() */

    int ok = read_array(l->Wo,l->K,l->E,fp,0);
    if (ok)
        return l;
    /* error exit */
    fprintf(stderr,"In read_smsftmax: failed to read weights\n");
    smsftmax_free(l);
    return NULL;
}

/* write_smsftmax - Write a sampled-softmax layer to a file
 *
 * Writes the sampled-softmax layer pointed to by l to the file pointed
 * to by fp.
 *
 * Parameters:
 *   l  - Pointer to the layer to be written
 *   fp - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_smsftmax(const SMSFTMAX* l, FILE* fp)
{
    int cnt = fprintf(fp,"SMSFTMAX K %d E %d B %d n_neg %d\n",
                      l->K,l->E,l->B,l->n_neg);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_smsftmax: failed to write the header\n");
        return 0;
    }
    int ok = write_array(l->Wo,l->K,l->E,fp,NULL,0);
    if (ok)
        return 1;
    /* error exit */
    fprintf(stderr,"In write_smsftmax: failed to write the weights\n");
    return 0;
}

/* load_smsftmax - Load a sampled-softmax layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads a sampled-softmax layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the layer from
 *
 * Returns:
 *   Pointer to the loaded layer if successful, NULL otherwise
 */
SMSFTMAX* load_smsftmax(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_smsftmax: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    SMSFTMAX* l = read_smsftmax(fp);
    fclose(fp);
    return l;
}

/* store_smsftmax - Store a sampled-softmax layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the sampled-softmax layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the layer to be stored
 *   filename - Name of the file to store the layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_smsftmax(const SMSFTMAX* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_smsftmax: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_smsftmax(l,fp);
    fclose(fp);
    return ok;
}
