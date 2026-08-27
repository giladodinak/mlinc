/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store a full decoder-only language model */
#ifndef LMIO_H
#define LMIO_H
#include <stdio.h>
#include "hash.h"
#include "lm.h"

/* Resumable training schedule / optimizer state carried alongside the model */
typedef struct {
    char  optimizer;      /* Optimizer code driving the transformer stack   */
    int   update_cnt;     /* AdamW step counter (for bias correction)       */
    int   epoch;          /* Last completed epoch (resume at epoch + 1)     */
    int   num_epochs;     /* Total epochs requested                         */
    int   final;          /* If not zero, store inference-only              */
    float sample_frac;    /* Fraction of dataset sampled each epoch         */
    float learning_rate;  /* Current learning rate (after prior decays)     */
    float lr_decay;       /* Per-epoch learning-rate decay                  */
    float weight_decay;   /* AdamW weight decay                             */
    int   lrng_seed;      /* RNG state (random.h) for exact continuation    */
} LMTRAIN;

/* write_lm / read_lm - Write / read a model to / from an open file.
 *
 * The file holds, in order: an LM header, the vocabulary hashmap, the unigram
 * sampling table (omitted when final), the embedding (lmembio), then the
 * transformer stack (MODEL via modelio) then the sampled-softmax head.
 *
 * write_lm parameters:
 *   m         - The model to write
 *   final     - If not zero, store inference-only: gradients/AdamW moments,
 *               dropout and the sampling table are omitted; the file can be
 *               loaded to run but not to resume training. m is not modified.
 *   hmap      - Vocabulary (word <-> index), stored in full
 *   dist      - Unigram sampling table (stored when not final)
 *   dist_size - Number of entries in dist
 *   st        - Training schedule / optimizer state (required)
 *
 * read_lm parameters:
 *   hmap      - Receives the allocated vocabulary; free with hashmap_free()
 *   st        - Receives the training schedule / optimizer state (or NULL)
 *
 * Returns:
 *   write_lm: 1 on success, 0 otherwise.
 *   read_lm:  the allocated model on success, NULL otherwise.
 *
 * Ownership notes for read_lm / load_lm:
 *   - The returned model is freed with lm_free().
 *   - The sampling table is allocated and attached to the head via
 *     smsftmax_set_dist(); it is NOT freed by lm_free()/smsftmax_free(),
 *     so the caller frees m->head->dist when done.
 *   - When the file was written final, no table is present and the head's
 *     dist is left NULL; the caller must rebuild/attach one before training.
 */
int write_lm(const LM* m, int final, HASHMAP* hmap,
             const int* dist, int dist_size, const LMTRAIN* st, FILE* fp);
LM*  read_lm(FILE* fp, HASHMAP** hmap, LMTRAIN* st);

/* load_lm / store_lm - Open the named file and read / write a model. */
LM*  load_lm(const char* filename, HASHMAP** hmap, LMTRAIN* st);
int  store_lm(const char* filename, const LM* m, int final, HASHMAP* hmap,
              const int* dist, int dist_size, const LMTRAIN* st);

#endif
