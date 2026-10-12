/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store NN dense layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "arrayio.h"
#include "embedding.h"
#include "embedio.h"

/* read_embedding - Read an embedding layer from a file
 * 
 * Reads an embedding layer from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read embedding layer if successful, NULL otherwise
 */
EMBEDDING* read_embedding(FILE* fp)
{
    int D, B, M, E;
    int pad;
    int cnt = fscanf(fp," EMBEDDING D %d B %d M %d E %d pad %d\n",
                                                          &D,&B,&M,&E,&pad);
    if (cnt < 5 || cnt == EOF) {
        fprintf(stderr,"In read_embedding: failed to read the header\n");
        return NULL;
    }
    EMBEDDING* e = embedding_create(E,M,pad);
    embedding_init(e,D,B,0);
    int ok = read_array(e->Wx,e->D,e->E,fp,0);
    if (ok)
        return e;
    /* error exit */
    fprintf(stderr,"In read_embedding: failed to read weights\n");
    freemem(e->h);
    freemem(e->Wx);
    freemem(e);
    return NULL;
}


/* write_embedding - Write an embedding layer to a file
 * 
 * Writes the embedding layer pointed to by d to the file pointed to by fp. 
 * 
 * Parameters:
 *   e  - Pointer to the embedding layer to be written
 *   fp - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_embedding(const EMBEDDING* e, FILE* fp)
{
    int cnt = fprintf(fp,"EMBEDDING D %d B %d M %d E %d pad %d\n",
                                           e->D,e->B,e->M,e->E,e->padinx);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_embedding: failed to write the header\n");
        return 0;
    }
    int ok = write_array(e->Wx,e->D,e->E,fp,NULL,0);
    if (ok)
        return 1;
    /* error exit */
    fprintf(stderr,"In write_embedding: failed to write the weights\n");
    return 0;
}


/* load_embedding - Load an embedding layer from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads an embedding layer from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the embedding layer from
 * 
 * Returns:
 *   Pointer to the loaded embedding layer if successful, NULL otherwise
 */
EMBEDDING* load_embedding(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_embedding: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    EMBEDDING* e = read_embedding(fp);
    fclose(fp);
    return e;
}

/* store_embedding - Store an embedding layer into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the embedding layer pointed to by d into it.
 * 
 * Parameters:
 *   e        - Pointer to the embedding layer to be stored
 *   filename - Name of the file to store the embedding layer in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_embedding(const EMBEDDING* e, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_embedding: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_embedding(e,fp);
    fclose(fp);
    return ok;
}

