/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store the language-model token embedding layer */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "lmemb.h"
#include "lmembio.h"

/* read_lmemb - Read a token embedding layer from a file */
LMEMB* read_lmemb(FILE* fp)
{
    int V, E, T, padinx, B, training, tied;
    int cnt = fscanf(fp," LMEMB V %d E %d T %d padinx %d B %d"
                        " training %d tied %d\n",
                     &V,&E,&T,&padinx,&B,&training,&tied);
    if (cnt < 7 || cnt == EOF) {
        fprintf(stderr,"In read_lmemb: failed to read the header\n");
        return NULL;
    }

    /* Rebuild the layer exactly as lmemb_create()/lmemb_init() would, so its
     * owned and scratch buffers match a fresh layer; then overwrite Wx. */
    LMEMB* l = lmemb_create(E,T,padinx);
    lmemb_init(l,V,B,training,tied);

    if (!tied) {
        int ok = read_array(l->Wx,l->V,l->E,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_lmemb: failed to read the weights\n");
            lmemb_free(l);
            return NULL;
        }
    }
    /* When tied, Wx is borrowed and absent from the file; the caller wires
     * it up with lmemb_set_tied_weights(). */
    return l;
}

/* write_lmemb - Write a token embedding layer to a file */
int write_lmemb(const LMEMB* l, int final, FILE* fp)
{
    int training = final ? 0 : l->training;
    int cnt = fprintf(fp,"LMEMB V %d E %d T %d padinx %d B %d"
                         " training %d tied %d\n",
                      l->V,l->E,l->T,l->padinx,l->B,training,l->tied);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_lmemb: failed to write the header\n");
        return 0;
    }
    /* Borrowed (tied) weights belong to the output projection: not written. */
    if (!l->tied) {
        int ok = write_array(l->Wx,l->V,l->E,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,"In write_lmemb: failed to write the weights\n");
            return 0;
        }
    }
    return 1;
}

/* load_lmemb - Load a token embedding layer from a file */
LMEMB* load_lmemb(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_lmemb: failed to open file '%s' for read\n",
                filename);
        return NULL;
    }
    LMEMB* l = read_lmemb(fp);
    fclose(fp);
    return l;
}

/* store_lmemb - Store a token embedding layer into a file */
int store_lmemb(const LMEMB* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_lmemb: failed to open file '%s' for write\n",
                filename);
        return 0;
    }
    int ok = write_lmemb(l,0,fp);
    fclose(fp);
    return ok;
}
