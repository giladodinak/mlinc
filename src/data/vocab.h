/* Copyright (c) 2023-2026 Gilad Odinak */
#ifndef VOCAB_H
#define VOCAB_H

#include <ctype.h>
#include "hash.h"   /* HASHMAP */

typedef struct VOCAB {
    HASHMAP* hmap;      /* Map word <-> index; index 0 = PAD       */
    int      size;      /* Number of entries including PAD         */
    float    coverage;  /* Fraction of corpus covered              */
    float*   freq;      /* [size] Unigram freq; NULL if unused     */
    int*     dist;      /* Negative-sampling table; NULL if unused */
    int      dist_size; /* Entries in dist, 0 if dist is NULL      */
} VOCAB;

/* vocab_build - Builds a vocabulary by scanning a corpus
 *
 * Parameters:
 *   file_list  - list of text files
 *   num_files  - number of files in file_list
 *   data_dir   - location of files
 *   vocab_size - Number of vocabulary words; 0 - derive from coverage
 *   coverage   - Target corpus-coverage fraction when vocab_size == 0
 *   max_vocab  - Maximum number of distinct words (typ. 3 x vocab_size)
 *   calc_freq  - If set, calculate and store word frequencies
 *   calc_dist  - If set, calculate and store negative sampling distribution
 *   verbose    - If set, print progress to stdout
 *
 * Returns:
 *   A VOCAB; returns NULL on failure.
 */
VOCAB* vocab_build(char** file_list, int num_files, const char* data_dir,
                   int vocab_size, float coverage, int max_vocab,
                   int calc_freq, int calc_dist, int verbose);

/* vocab_free - Releases all memory owned by a VOCAB
 *
 * Frees the hashmap, and the frequency and distribution tables if they
 * exists; then free the VOCAB itself.
 *
 * Parameters:
 *   v - Pointer to the VOCAB
 */
void vocab_free(VOCAB* v);

/* vocab_lookup - Looks up the token id of a single word
 *
 * Parameters:
 *   v    - Pointer to the VOCAB
 *   word - A string representing the word
 *
 * Returns:
 *   The token id (integer) of the word, if it exists in the vocabulary;
 *   otherwise returns 0. The lookup is case insensitive.
 */
static inline int vocab_lookup(const VOCAB* v, const char* word)
{
    char buf[256];
    size_t i = 0;
    for (; word[i] != '\0' && i < sizeof(buf) - 1; i++)
        buf[i] = (char) tolower((unsigned char) word[i]);
    buf[i] = '\0';
    return hashmap_str2inx(v->hmap,buf,0);
}

/* vocab_word - Returns the word string of a token id
 *
 * Parameters:
 *   v  - Pointer to the VOCAB
 *   id - Token id of a word
 *
 * Returns:
 *   A pointer to a string representing the word.
 *   Returns "" if the value of id is not a valid token id.
 */
static inline const char* vocab_word(const VOCAB* v, int id)
{
    if (id < 0 || id >= v->size) id = 0;
    return hashmap_inx2str(v->hmap,id);
}

/* vocab_tokenize - Split text on whitespace and map words to token ids
 *
 * Split text on whitespace, lowercase each word, map to indices, and store up
 * to max ids in out[]. Out-of-vocabulary words are reported on stderr and
 * skipped (there is no <unk> token).
 *
 * Parameters:
 *   v      - Pointer to the VOCAB
 *   text   - The input text
 *   ids    - Buffer receiving up to max token ids
 *   maxids - Size of ids[] buffer
 *
 * Returns:
 *   The number of ids written.
 */
int vocab_tokenize(const VOCAB* v, const char* text, int* ids, int maxids);

/* vocab_freq - Unigram frequency of a token id
 *
 * Parameters:
 *   v  - Pointer to the VOCAB
 *   id - Token id of a word
 *
 * Returns:
 *   The unigram frequency of the token id, 0 when frequencies are unavailable
 *   (wrapped/loaded vocab) or the id is out of range. Used for subsampling.
 */
static inline float vocab_freq(const VOCAB* v, int id)
{
    if (v->freq == NULL || id < 0 || id >= v->size) return 0.0;
    return v->freq[id];
}

/* vocab_dist - Return the distribution table
 *
 * Parameters:
 *   v    - Pointer to the VOCAB
 *   size - Receives the length of the table
 *
 * Returns:
 *   The distribution table and, via *size, its length. Returns NULL (and
 *   sets *size to 0) when there is no table.
 */
static inline const int* vocab_dist(const VOCAB* v, int* size)
{
    if (size != NULL) *size = v->dist_size;
    return v->dist;
}

#endif /* VOCAB_H */
