/* Copyright (c) 2023-2026 Gilad Odinak */
#ifndef VOCAB_H
#define VOCAB_H

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "hash.h"   /* HASHMAP */
#include "tokenizer.h"

typedef struct VOCAB {
    HASHMAP* hmap;      /* Map word <-> index; index 0 = PAD       */
    int      size;      /* Number of entries including PAD         */
    float    coverage;  /* Fraction of corpus covered              */
    int      unk_id;    /* id of <unk>, PAD if not used            */
    int      num_id;    /* id of <num>, PAD if not used            */
    int max_word_len;   /* words longer then this are skipped      */
    float*   freq;      /* [size] Unigram freq; NULL if unused     */
    int*     dist;      /* Negative-sampling table; NULL if unused */
    int      dist_size; /* Entries in dist, 0 if dist is NULL      */
    int      token_flags; /* Tokenizer flags (TOK_*) used by build */
} VOCAB;

/* vocab_build - Builds a vocabulary by scanning a corpus
 *
 * Parameters:
 *   file_list    - list of text files
 *   num_files    - number of files in file_list
 *   data_dir     - location of files
 *   vocab_size   - Number of vocabulary words; 0 - derive from coverage
 *   coverage     - Target corpus-coverage fraction when vocab_size == 0
 *   max_vocab    - Maximum number of distinct words (typ. 3 x vocab_size)
 *   max_word_len - words longer then this are skipped
 *   flags        - Bitwise OR of optional flags (see below)
 *   excpfx       - Tokens starting with this prefix (e.g. input-only tokens
 *                  such as "<_" for <_aa>) are excluded from the negative
 *                  sampling distribution; NULL or "" - none excluded
 *
 * Flags:
 *   VOCAB_UNK     - unknown words mapped to <unk>; otherwise ignored
 *   VOCAB_NUM     - numbers mapped to <num>; otherwise unknown words
 *   VOCAB_SPECIAL - recognize <name> special tokens
 *   VOCAB_FREQ    - calculate and store word frequencies
 *   VOCAB_DIST    - calculate and store negative sampling distribution
 *   VOCAB_VERBOSE - print progress to stdout
 *
 *
 * Notes:
 *   - With VOCAB_SPECIAL, special tokens (<name>, e.g. <bos> <eos> <sep>)
 *     found in the corpus are always retained, regardless of frequency.
 *   - With VOCAB_UNK, <unk> is added; its count is the number of corpus
 *     tokens dropped from the vocabulary
 *   - With VOCAB_NUM numbers are counted as <num>; otherwise they are
 *     unknown words.
 *   - vocab_size counts PAD, special tokens and words; <unk> and <num> are
 *     added on top of it. The final size is in VOCAB.size.
 *
 * Returns:
 *   A VOCAB; returns NULL on failure.
 */
VOCAB* vocab_build(char** file_list, int num_files, const char* data_dir,
                   int vocab_size, float coverage, int max_vocab, int flags,
                   int max_word_len, const char* excpfx);

#define VOCAB_UNK      1
#define VOCAB_NUM      2
#define VOCAB_SPECIAL  4
#define VOCAB_FREQ     8
#define VOCAB_DIST    16
#define VOCAB_VERBOSE 32

/* vocab_free - Releases all memory owned by a VOCAB
 *
 * Frees the hashmap, and the frequency and distribution tables if they
 * exist; then free the VOCAB itself.
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
 *   The token id (integer) of the word, if it exists in the vocabulary.
 *   Otherwise: the <num> id for a number if numbers are mapped; else the
 *   <unk> id if unknown words are mapped; else 0 (PAD).
 *   The lookup is case insensitive.
 */
static inline int vocab_lookup(const VOCAB* v, const char* word)
{
    char buf[256];
    size_t i = 0;
    for (; word[i] != '\0' && i < sizeof(buf) - 1; i++)
        buf[i] = (char) tolower((unsigned char) word[i]);
    buf[i] = '\0';
    int id = hashmap_str2inx(v->hmap,buf,0);
    if (id > 0)
        return id;
    if (v->num_id && token_is_number(buf))
        return v->num_id;
    return v->unk_id;
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

/* vocab_tokenize - Split text into words and map them to token ids
 *
 * Split text into words with the same rules used to read the training
 * corpus (tokenizer.h), map them to indices, and store up to max ids
 * in out[]. Words are mapped with vocab_lookup. Words that map to
 * 0 (PAD), i.e. unknown words without <unk>, are reported on stderr and
 * skipped.
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
