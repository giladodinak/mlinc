/* Copyright (c) 2023-2026 Gilad Odinak */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "mem.h"
#include "hash.h"
#include "tokenizer.h"
#include "textfile.h"
#include "vocab.h"

/* Order WRDCNT by count, descending (most frequent first) */
static int qsort_compare_word_cnt(const void* a, const void* b)
{   /* WRDCNT declared in textfile.h */
    const WRDCNT* wa = (const WRDCNT*) a;
    const WRDCNT* wb = (const WRDCNT*) b;
    if (wb->cnt > wa->cnt) return  1;
    if (wb->cnt < wa->cnt) return -1;
    return 0;
}

/* True if the token with id i starts with prefix pfx (NULL/"": never) */
static int has_prefix(HASHMAP* h, int i, const char* pfx)
{
    return pfx != NULL && *pfx != '\0' &&
           strncmp(hashmap_inx2str(h,i),pfx,strlen(pfx)) == 0;
}

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
                   int max_word_len, const char* excpfx)
{
    if (file_list == NULL || num_files <= 0 || max_vocab < 1)
        return NULL;

    int has_unk = flags & VOCAB_UNK;
    int has_num = flags & VOCAB_NUM;
    int calc_freq = flags & VOCAB_FREQ;
    int calc_dist = flags & VOCAB_DIST;
    int verbose = flags & VOCAB_VERBOSE;

    int token_flags = 0;
    if (flags & VOCAB_NUM) token_flags |= TOK_NUMBER;
    if (flags & VOCAB_SPECIAL) token_flags |= TOK_SPECIAL;

    int stdout_is_tty = isatty(fileno(stdout));

    /* Count word frequencies over the whole corpus */
    HASHMAP* hmap = hashmap_create(max_vocab,4 * max_vocab);
    hashmap_str2inx(hmap,"",1);   /* Index 0 = PAD   */
    WRDCNT* word_cnt = allocmem(max_vocab,1,WRDCNT);
    word_cnt[0].inx = 0; /* PAD */
    word_cnt[0].cnt = 0;

    long long tot_word_cnt = 0;
    for (int i = 0; i < num_files; i++) {
        tot_word_cnt += process_text_file(file_list[i],data_dir,hmap,1,
                                          max_vocab,word_cnt,NULL,0,
                                          max_word_len,token_flags);
        if (verbose && stdout_is_tty) {
            printf("Processed file %d of %d, %lld words\r",
                   i + 1,num_files,tot_word_cnt);
            fflush(stdout);
        }
    }
    if (verbose)
        printf("\nDataset: %d files, %lld words, %d unique\n",
               num_files,tot_word_cnt,hmap->map_used);

    /* Remove numbers from word_cnt[1..n-1]: they never get their own entry
     * and are counted as <num>, or as unknown (<unk>) without VOCAB_NUM.
     */
    int used = hmap->map_used < max_vocab ? hmap->map_used : max_vocab;
    long long num_cnt = 0;
    int n = 1;
    for (int i = 1; i < used; i++) {
        if (token_is_number(hashmap_inx2str(hmap,word_cnt[i].inx)))
            num_cnt += word_cnt[i].cnt;
        else
            word_cnt[n++] = word_cnt[i];
    }

    /* Move special tokens to word_cnt[1..nspec-1], always retained */
    int nspec = 1;
    for (int i = 1; i < n; i++) {
        if (token_is_special(hashmap_inx2str(hmap,word_cnt[i].inx))) {
            WRDCNT t = word_cnt[i];
            word_cnt[i] = word_cnt[nspec];
            word_cnt[nspec++] = t;
        }
    }

    /* Rank ordinary words only */
    qsort(word_cnt + nspec,n - nspec,sizeof(WRDCNT),qsort_compare_word_cnt);

    /* Retained tokens: specials, numbers (if <num>), and top-ranked words */
    long long wcnt = has_num ? num_cnt : 0;
    for (int i = 1; i < nspec; i++)
        wcnt += word_cnt[i].cnt;
    int keep = nspec; /* Number of word_cnt entries retained */
    if (vocab_size <= 0) {
        long long target = (long long)(coverage * (double) tot_word_cnt);
        while (keep < n && wcnt < target)
            wcnt += word_cnt[keep++].cnt;
    }
    else {
        int lim = vocab_size < n ? vocab_size : n;
        while (keep < lim)
            wcnt += word_cnt[keep++].cnt;
    }
    coverage = tot_word_cnt ? (double) wcnt / tot_word_cnt : 0.0;

    /* Re-index retained words into a compact hashmap; cnt[] is by new id */
    HASHMAP* hmap2 = hashmap_create(3 * (keep + 2),hmap->mem_used);
    long long* cnt = allocmem(keep + 2,1,long long);
    memset(cnt,0,(keep + 2) * sizeof(long long));
    for (int i = 0; i < keep; i++) { /* i == 0 is PAD ("") -> id 0 */
        const char* w = hashmap_inx2str(hmap,word_cnt[i].inx);
        cnt[hashmap_str2inx(hmap2,w,1)] = word_cnt[i].cnt;
    }
    hashmap_free(hmap);
    freemem(word_cnt);

    /* <unk>/<num> reuse the corpus entry if the corpus already has one */
    int unk_id = 0;
    if (has_unk) {
        unk_id = hashmap_str2inx(hmap2,"<unk>",1);
        cnt[unk_id] += tot_word_cnt - wcnt; /* Dropped tokens */
    }
    int num_id = 0;
    if (has_num) {
        num_id = hashmap_str2inx(hmap2,"<num>",1);
        cnt[num_id] += num_cnt;
    }

    /* Use the retained vocabulary */
    hmap = hmap2;
    vocab_size = hmap->map_used;
    if (verbose)
        printf("Vocabulary limited to %d words, covering %2.0f%% of corpus\n",
               vocab_size, 100.0 * coverage);

    long long total = 0; /* Tokens as seen by the model (PAD excluded) */
    for (int i = 1; i < vocab_size; i++)
        total += cnt[i];

    float* freq = NULL;
    if (calc_freq) {/* Per-word frequencies, indexed by token id */
        freq = allocmem(vocab_size,1,float);
        freq[0] = 0.0; /* PAD */
        for (int i = 1; i < vocab_size; i++)
            freq[i] = total ? (double) cnt[i] / total : 0.0;
    }

    int* dist = NULL;
    int dist_size = 0;
    if (calc_dist) { /* Negative-sampling distribution table (PAD excluded) */
        /* Unigram distribution table (freq^0.75), PAD excluded
         * Reference:
         * Distributed Representations of Words and Phrases and their
         *    Compositionality, Mikolov et al., 2013,
         *    https://arxiv.org/pdf/1310.4546
         */
        const float dpow = 0.75;
        const int dscale = 10;
        for (int i = 1; i < vocab_size; i++)
            if (!has_prefix(hmap,i,excpfx))
                dist_size += (int) (powf(cnt[i],dpow) / dscale) + 1;
        dist = allocmem(dist_size,1,int);
        for (int i = 1, j = 0; i < vocab_size; i++) {
            if (has_prefix(hmap,i,excpfx))
                continue;
            int rpt = (int)(powf(cnt[i],dpow) / dscale) + 1;
            for (int k = 0; j < dist_size && k < rpt; k++)
                dist[j++] = i;
        }
        if (verbose)
            printf("Distribution table size %d\n", dist_size);
    }
    freemem(cnt);

    VOCAB* v = allocmem(1,1,VOCAB);
    v->hmap = hmap;
    v->size = vocab_size;
    v->coverage = coverage;
    v->unk_id = unk_id;
    v->num_id = num_id;
    v->token_flags = token_flags;
    v->max_word_len = max_word_len;
    v->freq = freq;
    v->dist = dist;
    v->dist_size = dist_size;
    return v;
}

void vocab_free(VOCAB* v)
{
    if (v == NULL) return;
    hashmap_free(v->hmap);
    freemem(v->dist);
    freemem(v->freq);
    freemem(v);
}

int vocab_tokenize(const VOCAB* v, const char* text, int* out, int maxids)
{
    int maxlen = v->max_word_len;
    char token[maxlen + 1];
    int cnt = 0;
    for (const char* p = text_next_token(text,v->token_flags,token,maxlen + 1);
         p != NULL; p = text_next_token(p,v->token_flags,token,maxlen + 1)) {
        int id = vocab_lookup(v,token); /* <unk>/<num> fallbacks */
        if (id > 0) {
            if (cnt < maxids) {
                out[cnt++] = id;
            }
            else {
                fprintf(stderr, "(prompt truncated to %d tokens)\n",maxids);
                break;
            }
        }
        else {
            fprintf(stderr, "'%s' not in vocabulary - skipping\n",token);
        }
    }
    return cnt;
}
