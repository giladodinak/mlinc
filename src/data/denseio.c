/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store NN dense layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "dense.h"
#include "denseio.h"

/* read_dense - Read a dense layer from a file
 * 
 * Reads a dense layer from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read dense layer if successful, NULL otherwise
 */
DENSE* read_dense(FILE* fp)
{
    int D, S, B, use_bias, training;
    char c;
    int cnt = fscanf(fp," DENSE D %d S %d B %d activation '%c' "
                        "use_bias %d training %d\n",
                        &D,&S,&B,&c,&use_bias,&training);
    if (cnt < 6 || cnt == EOF) {
        fprintf(stderr,"In read_dense: failed to read the header\n");
        return NULL;
    }
    if (c != 'n' && c != 'r' && c != 'g' && c != 's' && c != 'S') {
        fprintf(stderr,"In read_dense: invalid activation code\n");
        return NULL;
    }
    use_bias = (use_bias) ? 1 : 0;
    training = (training) ? 1 : 0;

    DENSE* d = allocmem(1,1,DENSE);
    d->S = S;
    d->D = D;
    d->B = B;
    d->activation = c;
    d->use_bias = use_bias;
    d->training = training;

    d->h = allocmem(d->B,d->S,float);
    if (training && c == 'g')
        d->z = allocmem(d->B,d->S,float);

    d->Wx = allocmem(d->D,d->S,float);
    if (!read_array(d->Wx,d->D,d->S,fp,0)) {
        fprintf(stderr,"In read_dense: failed to read weights\n");
        goto err;
    }

    if (d->use_bias) {
        d->b = allocmem(1,d->S,float);
        if (!read_array((fArr2D) d->b,1,d->S,fp,0)) {
            fprintf(stderr,"In read_dense: failed to read bias\n");
            goto err;
        }
    }

    if (d->training) {
        d->gWx = allocmem(d->D,d->S,float);
        if (d->use_bias)
            d->gb = allocmem(1,d->S,float);
    }

    return d;

err:
    dense_free(d);
    return NULL;
}

/* write_dense - Write a dense layer to a file
 * 
 * Writes the dense layer pointed to by d to the file pointed to by fp. 
 * 
 * Parameters:
 *   d     - Pointer to the dense layer to be written
 *   final - If not zero, record the layer as inference-only
 *   fp    - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_dense(const DENSE* d, int final, FILE* fp)
{
    int training = final ? 0 : d->training;
    int cnt = fprintf(fp,"DENSE D %d S %d B %d activation '%c' "
                        "use_bias %d training %d\n",
                        d->D,d->S,d->B,d->activation,
                        d->use_bias,training);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_dense: failed to write the header\n");
        return 0;
    }

    if (!write_array(d->Wx,d->D,d->S,fp,NULL,0)) {
        fprintf(stderr,"In write_dense: failed to write weights\n");
        return 0;
    }

    if (d->use_bias &&
        !write_array((fArr2D) d->b,1,d->S,fp,NULL,0)) {
        fprintf(stderr,"In write_dense: failed to write bias\n");
        return 0;
    }

    return 1;
}

/* load_dense - Load a dense layer from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads a dense layer from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the dense layer from
 * 
 * Returns:
 *   Pointer to the loaded dense layer if successful, NULL otherwise
 */
DENSE* load_dense(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_dense: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    DENSE* l = read_dense(fp);
    fclose(fp);
    return l;
}

/* store_dense - Store a dense layer into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the dense layer pointed to by d into it.
 * 
 * Parameters:
 *   d        - Pointer to the dense layer to be stored
 *   filename - Name of the file to store the dense layer in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_dense(const DENSE* d, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_dense: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_dense(d,0,fp);
    fclose(fp);
    return ok;
}
