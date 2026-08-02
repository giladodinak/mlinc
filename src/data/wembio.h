/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store word embeddings */
#ifndef WEMBIO_H
#define WEMBIO_H
#include "array.h"
#include "hash.h"

/* load_word_embeddings - Load word embeddings from a file
 *
 * Opens the file named by filename, reads the word embeddings stored in
 * it, and returns them through the output parameters. The vocabulary
 * hashmap and the embeddings array are allocated by this function and
 * ownership is transferred to the caller, which must free them with
 * hashmap_free() and freemem() respectively.
 *
 * Parameters:
 *   filename            - Name of the file to load from
 *   vocab_size          - Receives the number of words (rows)
 *   embedding_dim       - Receives the embedding dimension (cols)
 *   learning_rate       - Receives the training learning rate, or NULL
 *   learning_rate_decay - Receives the training decay, or NULL
 *   epochs              - Receives the number of training epochs, or NULL
 *   hmap                - Receives the allocated word <=> index map
 *   embeddings          - Receives the allocated [vocab_size][embedding_dim]
 *                         embeddings array
 *
 * Returns:
 *   1 if successful, 0 otherwise. On failure no memory is returned and
 *   the output parameters are left unchanged.
 *
 * Notes:
 *   - vocab_size, embedding_dim, hmap and embeddings are required; the
 *     three training-parameter outputs may be NULL if not wanted.
 *   - A word's row index is the index assigned by the hashmap, not the
 *     index recorded in the file. Words beyond the declared vocabulary
 *     size are parsed and ignored.
 */
int load_word_embeddings(const char* filename,
                         int* vocab_size, int* embedding_dim,
                         float* learning_rate, float* learning_rate_decay,
                         int* epochs, HASHMAP** hmap, fArr2D* embeddings);

/* store_word_embeddings - Store word embeddings into a file
 *
 * Opens the file named by filename and writes the given word embeddings
 * to it. Nothing is taken ownership of; hmap and embeddings are only read
 * and are left as is.
 *
 * Parameters:
 *   filename            - Name of the file to store into
 *   vocab_size          - Number of words (rows in embeddings)
 *   embedding_dim       - Embedding dimension (cols)
 *   learning_rate       - Training learning rate, for reference
 *   learning_rate_decay - Training decay, for reference
 *   epochs              - Number of training epochs, for reference
 *   hmap                - Word <=> index map
 *   embeddings          - [vocab_size][embedding_dim] embeddings array
 *
 * Returns:
 *   1 if successful, 0 otherwise
 *
 * Notes:
 *   - Words that the hashmap maps to an empty string are written as <unk>.
 */
int store_word_embeddings(const char* filename,
                          int vocab_size, int embedding_dim,
                          float learning_rate, float learning_rate_decay,
                          int epochs, HASHMAP* hmap, fArr2D embeddings);

#endif
