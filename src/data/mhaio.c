/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN MHA (Multi-Head Attention) layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "arrayio.h"
#include "rope.h"
#include "mha.h"
#include "mhaio.h"

/* read_mha - Read an MHA layer from a file
 *
 * Reads an MHA layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read MHA layer if successful, NULL otherwise
 *
 * Notes:
 *   - Only the projection weights (Wq, Wk, Wv, Wo) are persisted; all
 *     other buffers are scratch and are (re)allocated here. The RoPE
 *     frequency table (theta) is derived and rebuilt via rope_init().
 *   - Backward/gradient buffers are allocated only when the stored
 *     training flag is non-zero.
 */
MHA* read_mha(FILE* fp)
{
    int H, T, D, B, lookahead, training;
    float dropout_rate;
    int cnt = fscanf(fp," MHA H %d T %d D %d B %d lookahead %d"
                        " training %d dropout %g\n",
                     &H,&T,&D,&B,&lookahead,&training,&dropout_rate);
    if (cnt < 7 || cnt == EOF) {
        fprintf(stderr,"In read_mha: failed to read header\n");
        return NULL;
    }
    if (H <= 0 || D % H != 0) {
        fprintf(stderr,"In read_mha: D %d not an integral multiple of H %d\n",
                       D,H);
        return NULL;
    }

    MHA* l = mha_create(H,T,lookahead);

    /* Persistent projection weights */

    /* Forward scratch buffers */

    /* Backward / gradient buffers (only when training) */
    mha_init(l,D,B,training,dropout_rate);

    fArr2D Wx[4] = {l->Wq,l->Wk,l->Wv,l->Wo};
    char* sWx[4] = {"Wq","Wk","Wv","Wo"};
    for (int i = 0; i < 4; i++) {
        int ok = read_array(Wx[i],l->D,l->D,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_mha: failed to read %s weights\n",sWx[i]);
            goto err;
        }
    }
    return l;

err: /* error exit */
    mha_free(l);
    return NULL;
}

/* write_mha - Write an MHA layer to a file
 *
 * Writes the MHA layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l     - Pointer to the MHA layer to be written
 *   final - If not zero, record the layer as inference-only.
 *   fp    - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_mha(const MHA* l, int final, FILE* fp)
{
    int training = final ? 0 : l->training;
    float dropout_rate = final ? 0.0f : (l->dropout && l->dropout[0]->rate);
    int cnt = fprintf(fp,"MHA H %d T %d D %d B %d lookahead %d"
                         " training %d dropout %.9g\n",
                      l->H,l->T,l->D,l->B,l->lookahead,
                      training,dropout_rate);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_mha: failed to write the header\n");
        return 0;
    }

    fArr2D Wx[4] = {l->Wq,l->Wk,l->Wv,l->Wo};
    char* sWx[4] = {"Wq","Wk","Wv","Wo"};
    for (int i = 0; i < 4; i++) {
        int ok = write_array(Wx[i],l->D,l->D,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,
                    "In write_mha: failed to write %s weights\n",sWx[i]);
            return 0;
        }
    }
    return 1;
}

/* load_mha - Load an MHA layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads an MHA layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the MHA layer from
 *
 * Returns:
 *   Pointer to the loaded MHA layer if successful, NULL otherwise
 */
MHA* load_mha(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_mha: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    MHA* l = read_mha(fp);
    fclose(fp);
    return l;
}

/* store_mha - Store an MHA layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the MHA layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the MHA layer to be stored
 *   filename - Name of the file to store the MHA layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_mha(const MHA* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_mha: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_mha(l,0,fp);
    fclose(fp);
    return ok;
}
