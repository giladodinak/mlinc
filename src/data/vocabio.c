/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store a vocabulary */
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "mem.h"
#include "hash.h"
#include "vocab.h"
#include "vocabio.h"

/* read_vocab - Read a vocabulary from a file
 *
 * Reads a vocabulary from the file pointed to by fp. Frequency and negative-
 * sampling distribution tables are loaded only when present in the file.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read vocabulary if successful, NULL otherwise
 */
VOCAB* read_vocab(FILE* fp)
{
    int size, mem_size, has_freq, has_dist, dist_size;
    float coverage;
    int cnt = fscanf(fp," VOCAB size %d coverage %g mem %d freq %d"
                        " dist %d dist_size %d",
                     &size,&coverage,&mem_size,&has_freq,&has_dist,&dist_size);
    if (cnt < 6 || cnt == EOF) {
        fprintf(stderr,"In read_vocab: failed to read header\n");
        return NULL;
    }
    int c = fgetc(fp);
    if (c == '\r') c = fgetc(fp);
    if (c != '\n') {
        fprintf(stderr,"In read_vocab: invalid header\n");
        return NULL;
    }
    if (size <= 0 || size > INT_MAX / 3 || mem_size <= 0 ||
        (has_freq != 0 && has_freq != 1) ||
        (has_dist != 0 && has_dist != 1) ||
        dist_size < 0 || (!has_dist && dist_size != 0)) {
        fprintf(stderr,"In read_vocab: invalid dimensions\n");
        return NULL;
    }

    VOCAB* v = allocmem(1,1,VOCAB);
    v->hmap = NULL;
    v->size = size;
    v->coverage = coverage;
    v->freq = NULL;
    v->dist = NULL;
    v->dist_size = 0;

    v->hmap = hashmap_create(size * 3,mem_size);
    for (int i = 0; i < size; i++) {
        uint32_t len;
        if (fread(&len,sizeof(len),1,fp) != 1 || len >= (uint32_t) mem_size) {
            fprintf(stderr,"In read_vocab: failed to read word %d\n",i);
            goto err;
        }
        char* word = allocmem((size_t) len + 1,1,char);
        if (len > 0 && fread(word,1,len,fp) != len) {
            fprintf(stderr,"In read_vocab: failed to read word %d\n",i);
            freemem(word);
            goto err;
        }
        word[len] = '\0';
        if (i == 0 && len != 0) {
            fprintf(stderr,"In read_vocab: word 0 is not PAD\n");
            freemem(word);
            goto err;
        }
        int id = hashmap_str2inx(v->hmap,word,1);
        freemem(word);
        if (id != i) {
            fprintf(stderr,"In read_vocab: invalid or duplicate word %d\n",i);
            goto err;
        }
    }

    if (has_freq) {
        v->freq = allocmem(size,1,float);
        if (fread(v->freq,sizeof(float),size,fp) != (size_t) size) {
            fprintf(stderr,"In read_vocab: failed to read frequencies\n");
            goto err;
        }
    }
    if (has_dist) {
        v->dist_size = dist_size;
        if (dist_size > 0) {
            v->dist = allocmem(dist_size,1,int);
            if (fread(v->dist,sizeof(int),dist_size,fp) != (size_t) dist_size) {
                fprintf(stderr,"In read_vocab: failed to read distribution\n");
                goto err;
            }
            for (int i = 0; i < dist_size; i++) {
                if (v->dist[i] <= 0 || v->dist[i] >= size) {
                    fprintf(stderr,"In read_vocab: invalid distribution entry\n");
                    goto err;
                }
            }
        }
    }
    return v;

err: /* error exit */
    vocab_free(v);
    return NULL;
}

/* write_vocab - Write a vocabulary to a file
 *
 * Writes the vocabulary pointed to by v to the file pointed to by fp.
 * Frequency and negative-sampling distribution tables are written only when
 * they exist.
 *
 * Parameters:
 *   v  - Pointer to the vocabulary to be written
 *   fp - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_vocab(const VOCAB* v, FILE* fp)
{
    if (v == NULL || v->hmap == NULL || v->size <= 0 ||
        v->hmap->mem_used <= 0 || v->dist_size < 0 ||
        (v->dist == NULL && v->dist_size != 0)) {
        fprintf(stderr,"In write_vocab: invalid vocabulary\n");
        return 0;
    }
    int has_freq = (v->freq != NULL);
    int has_dist = (v->dist != NULL);
    int dist_size = has_dist ? v->dist_size : 0;
    int cnt = fprintf(fp,"VOCAB size %d coverage %.9g mem %d freq %d"
                         " dist %d dist_size %d\n",
                      v->size,v->coverage,v->hmap->mem_used,
                      has_freq,has_dist,dist_size);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_vocab: failed to write the header\n");
        return 0;
    }

    for (int i = 0; i < v->size; i++) {
        const char* word = hashmap_inx2str(v->hmap,i);
        if (word == NULL || strlen(word) > UINT32_MAX) {
            fprintf(stderr,"In write_vocab: invalid word %d\n",i);
            return 0;
        }
        uint32_t len = (uint32_t) strlen(word);
        if (fwrite(&len,sizeof(len),1,fp) != 1 ||
            (len > 0 && fwrite(word,1,len,fp) != len)) {
            fprintf(stderr,"In write_vocab: failed to write word %d\n",i);
            return 0;
        }
    }
    if (has_freq &&
        fwrite(v->freq,sizeof(float),v->size,fp) != (size_t) v->size) {
        fprintf(stderr,"In write_vocab: failed to write frequencies\n");
        return 0;
    }
    if (has_dist && dist_size > 0 &&
        fwrite(v->dist,sizeof(int),dist_size,fp) != (size_t) dist_size) {
        fprintf(stderr,"In write_vocab: failed to write distribution\n");
        return 0;
    }
    return 1;
}

/* load_vocab - Load a vocabulary from a file
 *
 * Opens the file specified by the filename parameter for reading and loads a
 * vocabulary from it.
 *
 * Parameters:
 *   filename - Name of the file to load the vocabulary from
 *
 * Returns:
 *   Pointer to the loaded vocabulary if successful, NULL otherwise
 */
VOCAB* load_vocab(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_vocab: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    VOCAB* v = read_vocab(fp);
    fclose(fp);
    return v;
}

/* store_vocab - Store a vocabulary into a file
 *
 * Opens the file specified by the filename parameter for writing and stores
 * the vocabulary pointed to by v into it.
 *
 * Parameters:
 *   v        - Pointer to the vocabulary to be stored
 *   filename - Name of the file to store the vocabulary in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_vocab(const VOCAB* v, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_vocab: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_vocab(v,fp);
    fclose(fp);
    return ok;
}
