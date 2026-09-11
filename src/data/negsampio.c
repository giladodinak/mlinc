/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN negative-sampling layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "negsample.h"
#include "negsampio.h"

/* read_negsample - Read a negative-sampling layer from a file
 *
 * Reads a negative-sampling layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read layer if successful, NULL otherwise
 *
 * Notes:
 *   - Only the output weights (Wo) and layer configuration are persisted.
 *     Training-only buffers (gWo, touched, seen) are allocated only when
 *     training is enabled.
 *   - The unigram sampling table (dist) is not owned by the layer and is not
 *     persisted; the caller must re-attach it via negsample_set_dist() after
 *     loading, before calling negsample_loss().
 */
NEGSAMPLE* read_negsample(FILE* fp)
{
    int K, E, B, n_neg, training;
    int cnt = fscanf(fp," NEGSAMPLE K %d E %d B %d n_neg %d training %d\n",
                     &K,&E,&B,&n_neg,&training);
    if (cnt < 5 || cnt == EOF) {
        fprintf(stderr,"In read_negsample: failed to read the header\n");
        return NULL;
    }
    NEGSAMPLE* l = allocmem(1,1,NEGSAMPLE);
    l->K = K;
    l->E = E;
    l->B = B;
    l->n_neg = n_neg;
    l->training = training ? 1 : 0;
    l->Wo = allocmem(l->K,l->E,float);
    l->h = allocmem(l->B,l->E,float);
    if (l->training) {
        l->gWo = allocmem(l->K,l->E,float);
        l->touched = allocmem(l->B * (l->n_neg + 1),1,int);
        l->ntouched = 0;
        l->seen = allocmem(l->K,1,int);
        for (int i = 0; i < l->K; i++)
            l->seen[i] = -1;
    }
    l->stamp = 0;
    l->dist = NULL;
    l->dist_size = 0;

    int ok = read_array(l->Wo,l->K,l->E,fp,0);
    if (ok)
        return l;
    /* error exit */
    fprintf(stderr,"In read_negsample: failed to read weights\n");
    negsample_free(l);
    return NULL;
}

/* write_negsample - Write a negative-sampling layer to a file
 *
 * Writes the negative-sampling layer pointed to by l to the file pointed
 * to by fp.
 *
 * Parameters:
 *   l     - Pointer to the layer to be written
 *   final - If not zero, record the layer as inference-only
 *   fp    - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_negsample(const NEGSAMPLE* l, int final, FILE* fp)
{
    int training = final ? 0 : l->training;
    int cnt = fprintf(fp,"NEGSAMPLE K %d E %d B %d n_neg %d training %d\n",
                      l->K,l->E,l->B,l->n_neg,training);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_negsample: failed to write the header\n");
        return 0;
    }
    int ok = write_array(l->Wo,l->K,l->E,fp,NULL,0);
    if (ok)
        return 1;
    /* error exit */
    fprintf(stderr,"In write_negsample: failed to write the weights\n");
    return 0;
}

/* load_negsample - Load a negative-sampling layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads a negative-sampling layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the layer from
 *
 * Returns:
 *   Pointer to the loaded layer if successful, NULL otherwise
 */
NEGSAMPLE* load_negsample(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_negsample: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    NEGSAMPLE* l = read_negsample(fp);
    fclose(fp);
    return l;
}

/* store_negsample - Store a negative-sampling layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the negative-sampling layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the layer to be stored
 *   filename - Name of the file to store the layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_negsample(const NEGSAMPLE* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_negsample: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_negsample(l,0,fp);
    fclose(fp);
    return ok;
}
