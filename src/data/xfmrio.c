/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN decoder-only TRANSFORMER layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
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
 *     training is non-zero, and dropout masks only when dropout_rate > 0.
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

    TRANSFORMER* l = allocmem(1,1,TRANSFORMER);
    l->B = B;
    l->T = T;
    l->D = D;
    l->Dff = Dff;
    l->BT = B * T;
    l->training = (training) ? 1 : 0;
    l->dropout_rate = dropout_rate;

    /* Sub-layers (order must match write_transformer) */
    l->mha = read_mha(fp);
    if (l->mha == NULL) {
        fprintf(stderr,"In read_transformer: failed to read mha\n");
        goto err;
    }
    l->ffn1 = read_dense(fp);
    if (l->ffn1 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read ffn1\n");
        goto err;
    }
    l->ffn2 = read_dense(fp);
    if (l->ffn2 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read ffn2\n");
        goto err;
    }
    l->norm1 = read_addnorm(fp);
    if (l->norm1 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read norm1\n");
        goto err;
    }
    l->norm2 = read_addnorm(fp);
    if (l->norm2 == NULL) {
        fprintf(stderr,"In read_transformer: failed to read norm2\n");
        goto err;
    }

    /* Transformer's own scratch buffers (mirror transformer_init) */
    l->mha_out = allocmem(l->BT,l->D,float);
    l->norm1_out = allocmem(l->BT,l->D,float);

    if (l->training) {
        l->d_norm2_in = allocmem(l->BT,l->D,float);
        l->d_ffn1_in = allocmem(l->BT,l->Dff,float);
        l->d_norm1_in = allocmem(l->BT,l->D,float);
        l->d_mha_out = allocmem(l->BT,l->D,float);
        l->d_ffn2_in = allocmem(l->BT,l->D,float);
        l->d_mha_masked = allocmem(l->BT,l->D,float);

        l->dg1 = allocmem(l->D,1,float);
        l->db1 = allocmem(l->D,1,float);
        l->dg2 = allocmem(l->D,1,float);
        l->db2 = allocmem(l->D,1,float);

        if (l->dropout_rate > 0) {
            l->drop_mask1 = allocmem(l->BT,l->D,float);
            l->drop_mask2 = allocmem(l->BT,l->D,float);
        }
    }
    return l;

err: /* error exit - free only the sub-layers already created, then self */
    if (l->mha != NULL) mha_free(l->mha);
    if (l->ffn1 != NULL) dense_free(l->ffn1);
    if (l->ffn2 != NULL) dense_free(l->ffn2);
    if (l->norm1 != NULL) addnorm_free(l->norm1);
    if (l->norm2 != NULL) addnorm_free(l->norm2);
    freemem(l);
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
    float dropout_rate = final ? 0.0f : l->dropout_rate;
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
