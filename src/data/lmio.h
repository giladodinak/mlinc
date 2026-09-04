/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store a full decoder-only language model */
#ifndef LMIO_H
#define LMIO_H
#include <stdio.h>
#include "vocab.h"
#include "lm.h"

/* Laguage Model training schedule / optimizer state */
typedef struct {
    char  optimizer;     /* Optimizer code ('a' -> AdamW 'a')          */
    int   update_cnt;    /* AdamW step counter (for bias correction)   */
    int   epoch;         /* Last completed epoch (resume at epoch + 1) */
    int   num_epochs;    /* Total epochs requested                     */
    int   final;         /* If not zero, store inference data only     */
    float sample_frac;   /* Fraction of dataset sampled each epoch     */
    float learning_rate; /* Current learning rate (after prior decays) */
    float lr_decay;      /* Per-epoch learning-rate decay              */
    float weight_decay;  /* AdamW weight decay                         */
    int   lrng_seed;     /* RNG state for exact continuation           */
} LMPARAM;

/* write_lm / read_lm - Write / read a model to / from an open file.
 *
 * The file holds, in order: an LM header, the vocabulary (including optional
 * frequency and unigram sampling tables), the embedding (lmembio), the
 * transformer stack (MODEL via modelio), then the sampled-softmax head.
 *
 * write_lm parameters:
 *   m         - The model to write
 *   final     - If not zero, store inference-only: gradients/AdamW moments,
 *               dropout and the sampling table are omitted; the file can be
 *               loaded to run but not to resume training. m is not modified.
 *   vocab     - Vocabulary; its optional tables are stored when present,
 *               except that both are omitted when final
 *   st        - Training schedule / optimizer state (required)
 *
 * read_lm parameters:
 *   vocab     - Receives the allocated vocabulary; free with vocab_free()
 *   st        - Receives the training schedule / optimizer state (or NULL)
 *
 * Returns:
 *   write_lm: 1 on success, 0 otherwise.
 *   read_lm:  the allocated model on success, NULL otherwise.
 *
 * Ownership notes for read_lm / load_lm:
 *   - The returned model is freed with lm_free().
 *   - The vocabulary owns its frequency and sampling tables. The sampling
 *     table is also attached to the head but is not owned by the head. Keep
 *     the vocabulary alive while using the model and free it with vocab_free().
 *   - When the file was written final, the sampling table is absent and the
 *     head's dist is left NULL; rebuild/attach one before training.
 */
int write_lm(const LM* m, int final, const VOCAB* vocab,
             const LMPARAM* st, FILE* fp);
LM*  read_lm(FILE* fp, VOCAB** vocab, LMPARAM* st);

/* load_lm / store_lm - Open the named file and read / write a model. */
LM*  load_lm(const char* filename, VOCAB** vocab, LMPARAM* st);
int  store_lm(const char* filename, const LM* m, int final,
              const VOCAB* vocab, const LMPARAM* st);

#endif
