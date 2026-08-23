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
    int D, S, B;
    char c;
    int cnt = fscanf(fp," DENSE D %d S %d B %d activation '%c'\n",&D,&S,&B,&c);
    if (cnt < 4 || cnt == EOF) {
        fprintf(stderr,"In read_dense: failed to read the header\n");
        return NULL;
    }
    if (c != 'n' && c != 'r' && c != 'g' && c != 's' && c != 'S') {
        fprintf(stderr,"In read_dense: invalid activation code\n");
        return NULL;
    }
    DENSE* d = allocmem(1,1,DENSE);
    d->S = S;
    d->D = D;
    d->B = B;
    d->activation = c;
    if (c == 'g') d->z = allocmem(d->B,d->S,float);
    d->h = allocmem(d->B,d->S,float);
    d->Wx = allocmem(d->D,d->S,float);
    int ok = read_array(d->Wx,d->D,d->S,fp,0);
    if (ok)
        return d;
    /* error exit */
    fprintf(stderr,"In read_dense: failed to read weights\n");
    freemem(d->h);
    freemem(d->z);
    freemem(d->Wx);
    freemem(d);
    return NULL;
}

/* write_dense - Write a dense layer to a file
 * 
 * Writes the dense layer pointed to by d to the file pointed to by fp. 
 * 
 * Parameters:
 *   d  - Pointer to the dense layer to be written
 *   fp - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_dense(const DENSE* d, FILE* fp)
{
    int cnt = fprintf(fp,"DENSE D %d S %d B %d activation '%c'\n",d->D,d->S,d->B,d->activation);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_dense: failed to write the header\n");
        return 0;
    }
    int ok = write_array(d->Wx,d->D,d->S,fp,NULL,0);
    if (ok)
        return 1;
    /* error exit */
    fprintf(stderr,"In write_dense: failed to write the weights\n");
    return 0;
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
    int ok = write_dense(d,fp);
    fclose(fp);
    return ok;
}

