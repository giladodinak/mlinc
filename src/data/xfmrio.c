/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN decoder-only TRANSFORMER layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "arrayio.h"
#include "mha.h"
#include "mhaio.h"
#include "dense.h"
#include "denseio.h"
#include "addnorm.h"
#include "addnormio.h"
#include "transformer.h"
#include "xfmrio.h"

/* read_transformer - Read a TRANSFORMER layer from a file
 *
 * Reads a TRANSFORMER layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read TRANSFORMER layer if successful, NULL otherwise
 *
 * Notes:
 *   - The TRANSFORMER owns no learnable weights of its own; all persisted
 *     parameters live in its sub-layers, read via read_mha() (Wq/Wk/Wv/Wo),
 *     read_dense() (ffn1, ffn2 weights), and read_addnorm() (norm1, norm2
 *     gamma/beta). The heads count and lookahead are recovered from the MHA
 *     sub-layer's own header.
 *   - The transformer's forward/backward scratch buffers are (re)allocated
 *     here, mirroring transformer_init(): backward/gradient buffers only when
 *     training is non-zero, and dropout layers only when dropout_rate > 0.
 */
TRANSFORMER* read_transformer(FILE* fp)
{
    int B, T, D, Dff, training;
    float dropout_rate;
    int cnt = fscanf(fp," TRANSFORMER B %d T %d D %d Dff %d"
                        " training %d dropout %g\n",
                     &B,&T,&D,&Dff,&training,&dropout_rate);
    if (cnt < 6 || cnt == EOF) {
        fprintf(stderr,"In read_transformer: failed to read the header\n");
        return NULL;
    }

    MHA* mha = read_mha(fp);
    if (mha == NULL) {
        fprintf(stderr,"In read_transformer: failed to read mha\n");
        return NULL;
    }

    TRANSFORMER* l = transformer_create(mha->H,T,D,Dff,mha->lookahead);
    transformer_init(l,B,training,dropout_rate);
    mha_free(l->mha);
    l->mha = mha;

    /* Sub-layers (order must match write_transformer) */
    DENSE* ffn1 = read_dense(fp);
    if (ffn1 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read ffn1\n");
        goto err;
    }
    dense_free(l->ffn1);
    l->ffn1 = ffn1;
    DENSE* ffn2 = read_dense(fp);
    if (ffn2 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read ffn2\n");
        goto err;
    }
    dense_free(l->ffn2);
    l->ffn2 = ffn2;
    ADDNORM* norm1 = read_addnorm(fp);
    if (norm1 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read norm1\n");
        goto err;
    }
    addnorm_free(l->norm1);
    l->norm1 = norm1;
    ADDNORM* norm2 = read_addnorm(fp);
    if (norm2 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read norm2\n");
        goto err;
    }
    addnorm_free(l->norm2);
    l->norm2 = norm2;

    /* Transformer's own scratch buffers (mirror transformer_init) */
    return l;

err: /* error exit - free only the sub-layers already created, then self */
    transformer_free(l);
    return NULL;
}

/* write_transformer - Write a TRANSFORMER layer to a file
 *
 * Writes the TRANSFORMER layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l     - Pointer to the TRANSFORMER layer to be written
 *   final - If not zero, record the layer as inference-only
 *   fp    - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_transformer(const TRANSFORMER* l, int final, FILE* fp)
{
    int training = final ? 0 : l->training;
    float dropout_rate = final || l->dropout1 == NULL ? 0.0 : l->dropout1->rate;
    int cnt = fprintf(fp,"TRANSFORMER B %d T %d D %d Dff %d"
                         " training %d dropout %.9g\n",
                      l->B,l->T,l->D,l->Dff,training,dropout_rate);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_transformer: failed to write the header\n");
        return 0;
    }

    /* Sub-layers (order must match read_transformer) */
    if (!write_mha(l->mha,final,fp)) {
        fprintf(stderr,"In write_transformer: failed to write mha\n");
        return 0;
    }
    if (!write_dense(l->ffn1,final,fp)) {
        fprintf(stderr,"In write_transformer: failed to write ffn1\n");
        return 0;
    }
    if (!write_dense(l->ffn2,final,fp)) {
        fprintf(stderr,"In write_transformer: failed to write ffn2\n");
        return 0;
    }
    if (!write_addnorm(l->norm1,fp)) {
        fprintf(stderr,"In write_transformer: failed to write norm1\n");
        return 0;
    }
    if (!write_addnorm(l->norm2,fp)) {
        fprintf(stderr,"In write_transformer: failed to write norm2\n");
        return 0;
    }
    return 1;
}

/* load_transformer - Load a TRANSFORMER layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads a TRANSFORMER layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the TRANSFORMER layer from
 *
 * Returns:
 *   Pointer to the loaded TRANSFORMER layer if successful, NULL otherwise
 */
TRANSFORMER* load_transformer(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_transformer: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    TRANSFORMER* l = read_transformer(fp);
    fclose(fp);
    return l;
}

/* store_transformer - Store a TRANSFORMER layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the TRANSFORMER layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the TRANSFORMER layer to be stored
 *   filename - Name of the file to store the TRANSFORMER layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_transformer(const TRANSFORMER* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_transformer: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_transformer(l,0,fp);
    fclose(fp);
    return ok;
}
