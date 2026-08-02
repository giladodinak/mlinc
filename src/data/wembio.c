/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store word embeddings                             */
/*                                                                         */
/* File format:                                                            */
/*   Line 1: #,vocab_size,<n>,embedding_dim,<n>,learning_rate,<f>,         */
/*           learning_rate_decay,<f>,epochs,<n>                            */
/*   Line 2 and on, one line per word:                                     */
/*           <index>,<word>,<v1>,<v2>, ... ,<vD>                           */
#include <stdio.h>
#include <string.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "hash.h"
#include "wembio.h"

/* Maximum word length, including the terminating null. The scan width
 * in the word format below must be kept equal to MAX_WORD - 1.
 */
#define MAX_WORD 256

/* read_word_embeddings - Read word embeddings from an open file
 *
 * Reads word embeddings from the file pointed to by fp and returns them
 * through the output parameters. See load_word_embeddings() for the
 * ownership and NULL rules, which this function follows.
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
static int read_word_embeddings(FILE* fp,
                                int* vocab_size_, int* embedding_dim_,
                                float* learning_rate_,
                                float* learning_rate_decay_,
                                int* epochs_, HASHMAP** hmap_,
                                fArr2D* embeddings_)
{
    int vocab_size, embedding_dim, epochs;
    float learning_rate, learning_rate_decay;
    int cnt = fscanf(fp," #,vocab_size,%d,embedding_dim,%d,"
                        "learning_rate,%f,learning_rate_decay,%f,epochs,%d",
                     &vocab_size,&embedding_dim,
                     &learning_rate,&learning_rate_decay,&epochs);
    if (cnt < 5 || cnt == EOF) {
        fprintf(stderr,"In read_word_embeddings: failed to read the header\n");
        return 0;
    }
    if (vocab_size < 1 || embedding_dim < 1) {
        fprintf(stderr,"In read_word_embeddings: invalid vocab_size %d "
                       "or embedding_dim %d\n",vocab_size,embedding_dim);
        return 0;
    }

    HASHMAP* hmap = hashmap_create(vocab_size * 3,vocab_size * 15);
    fArr2D embeddings = allocmem(vocab_size,embedding_dim,float);

    typedef float (*ArrVE)[embedding_dim];
    ArrVE E = (ArrVE) embeddings;

    /* Values of words outside the declared vocabulary are parsed into
     * this row and discarded, so that parsing stays in step with the file.
     */
    fVec skip = allocmem(1,embedding_dim,float);

    int nwords = 0;  /* Number of words actually stored  */
    int ignored = 0; /* Number of words beyond the vocab */
    /* Note that the file starts at line 1, and the first word is on line 2 */
    for (int lineno = 2; ; lineno++) {
        int wrdinx;
        char word[MAX_WORD];
        cnt = fscanf(fp,"%d,%255[^,],",&wrdinx,word);
        if (cnt == EOF) /* End of file */
            break;
        if (cnt < 2) {
            fprintf(stderr,"In read_word_embeddings: invalid index, word "
                           "format at line %d\n",lineno);
            goto err;
        }
        /* The row index of a word is the index assigned by the hashmap;
         * the index recorded in the file is not used.
         */
        wrdinx = hashmap_str2inx(hmap,word,1);
        float* row;
        if (wrdinx < 0 || wrdinx >= vocab_size) {
            row = skip;
            ignored++;
        }
        else {
            row = E[wrdinx];
            if (wrdinx + 1 > nwords)
                nwords = wrdinx + 1;
        }
        for (int j = 0; j < embedding_dim; j++) {
            cnt = fscanf(fp,"%f%*[,]",&row[j]);
            if (cnt != 1) {
                fprintf(stderr,"In read_word_embeddings: invalid embedding "
                               "value at line %d, value #%d\n",lineno,j + 1);
                goto err;
            }
        }
    }
    if (ignored > 0)
        fprintf(stderr,"In read_word_embeddings: ignored %d word(s) beyond "
                       "the declared vocabulary size of %d\n",
                       ignored,vocab_size);
    if (nwords < vocab_size)
        vocab_size = nwords;
    freemem(skip);

    *vocab_size_ = vocab_size;
    *embedding_dim_ = embedding_dim;
    if (learning_rate_ != NULL) *learning_rate_ = learning_rate;
    if (learning_rate_decay_ != NULL) *learning_rate_decay_ = learning_rate_decay;
    if (epochs_ != NULL) *epochs_ = epochs;
    *hmap_ = hmap;
    *embeddings_ = embeddings;
    return 1;

err: /* error exit */
    freemem(skip);
    hashmap_free(hmap);
    freemem(embeddings);
    return 0;
}

/* write_word_embeddings - Write word embeddings to an open file
 *
 * Writes the given word embeddings to the file pointed to by fp. See
 * store_word_embeddings() for the (non-)ownership rules.
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
static int write_word_embeddings(FILE* fp,
                                 int vocab_size, int embedding_dim,
                                 float learning_rate,
                                 float learning_rate_decay,
                                 int epochs, HASHMAP* hmap,
                                 fArr2D embeddings)
{
    int cnt = fprintf(fp,"#,vocab_size,%d,embedding_dim,%d,"
                         "learning_rate,%f,learning_rate_decay,%f,epochs,%d\n",
                      vocab_size,embedding_dim,
                      learning_rate,learning_rate_decay,epochs);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_word_embeddings: failed to write the header\n");
        return 0;
    }

    typedef float (*ArrVE)[embedding_dim];
    ArrVE E = (ArrVE) embeddings;
    for (int wrdinx = 0; wrdinx < vocab_size; wrdinx++) {
        const char* word = hashmap_inx2str(hmap,wrdinx);
        if (word == NULL || strlen(word) == 0)
            word = "<unk>";
        cnt = fprintf(fp,"%d,%s",wrdinx,word);
        for (int j = 0; j < embedding_dim && cnt > 0; j++)
            cnt = fprintf(fp,",%10.8f",E[wrdinx][j]);
        if (cnt > 0)
            cnt = fprintf(fp,"\n");
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,"In write_word_embeddings: failed to write "
                           "word %d\n",wrdinx);
            return 0;
        }
    }
    return 1;
}

/* load_word_embeddings - Load word embeddings from a file */
int load_word_embeddings(const char* filename,
                         int* vocab_size, int* embedding_dim,
                         float* learning_rate, float* learning_rate_decay,
                         int* epochs, HASHMAP** hmap, fArr2D* embeddings)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_word_embeddings: failed to open file '%s' "
                       "for read\n",filename);
        return 0;
    }
    int ok = read_word_embeddings(fp,vocab_size,embedding_dim,learning_rate,
                                  learning_rate_decay,epochs,hmap,embeddings);
    fclose(fp);
    return ok;
}

/* store_word_embeddings - Store word embeddings into a file */
int store_word_embeddings(const char* filename,
                          int vocab_size, int embedding_dim,
                          float learning_rate, float learning_rate_decay,
                          int epochs, HASHMAP* hmap, fArr2D embeddings)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_word_embeddings: failed to open file '%s' "
                       "for write\n",filename);
        return 0;
    }
    int ok = write_word_embeddings(fp,vocab_size,embedding_dim,learning_rate,
                                   learning_rate_decay,epochs,hmap,embeddings);
    fclose(fp);
    return ok;
}
