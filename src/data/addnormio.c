/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN ADDNORM (layer normalization) layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "arrayio.h"
#include "addnorm.h"
#include "addnormio.h"

/* read_addnorm - Read an ADDNORM layer from a file
 *
 * Reads an ADDNORM layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read ADDNORM layer if successful, NULL otherwise
 *
 * Notes:
 *   - Only the learnable parameters (gamma, beta) are persisted; the
 *     mean, sdev, and xn buffers are scratch and are (re)allocated here.
 */
ADDNORM* read_addnorm(FILE* fp)
{
    int D, B;
    int cnt = fscanf(fp," ADDNORM D %d B %d\n",&D,&B);
    if (cnt < 2 || cnt == EOF) {
        fprintf(stderr,"In read_addnorm: failed to read the header\n");
        return NULL;
    }
    ADDNORM* l = addnorm_create();
    addnorm_init(l,D,B);

    int ok = read_array((fArr2D)l->gamma,1,l->D,fp,0);
    if (ok)
        ok = read_array((fArr2D)l->beta,1,l->D,fp,0);
    if (ok)
        return l;
    /* error exit */
    fprintf(stderr,"In read_addnorm: failed to read gamma, beta data\n");
    addnorm_free(l);
    return NULL;
}

/* write_addnorm - Write an ADDNORM layer to a file
 *
 * Writes the ADDNORM layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l  - Pointer to the ADDNORM layer to be written
 *   fp - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_addnorm(const ADDNORM* l, FILE* fp)
{
    int cnt = fprintf(fp,"ADDNORM D %d B %d\n",l->D,l->B);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_addnorm: failed to write the header\n");
        return 0;
    }
    int ok = write_array((fArr2D)l->gamma,1,l->D,fp,NULL,0);
    if (ok)
        ok = write_array((fArr2D)l->beta,1,l->D,fp,NULL,0);
    if (ok)
        return 1;
    /* error exit */
    fprintf(stderr,"In write_addnorm: failed to write gamma, beta data\n");
    return 0;
}

/* load_addnorm - Load an ADDNORM layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads an ADDNORM layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the ADDNORM layer from
 *
 * Returns:
 *   Pointer to the loaded ADDNORM layer if successful, NULL otherwise
 */
ADDNORM* load_addnorm(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_addnorm: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    ADDNORM* l = read_addnorm(fp);
    fclose(fp);
    return l;
}

/* store_addnorm - Store an ADDNORM layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the ADDNORM layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the ADDNORM layer to be stored
 *   filename - Name of the file to store the ADDNORM layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_addnorm(const ADDNORM* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_addnorm: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_addnorm(l,fp);
    fclose(fp);
    return ok;
}
