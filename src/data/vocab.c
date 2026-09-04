/* Copyright (c) 2023-2026 Gilad Odinak */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "mem.h"
#include "hash.h"
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
                   int calc_freq, int calc_dist, int verbose)
{
    if (file_list == NULL || num_files <= 0 || max_vocab < 1)
        return NULL;

    int stdout_is_tty = isatty(fileno(stdout));

    /* Count word frequencies over the whole corpus */
    HASHMAP* hmap = hashmap_create(max_vocab,4 * max_vocab);
    hashmap_str2inx(hmap,"",1);   /* Index 0 = PAD   */
    WRDCNT* word_cnt = allocmem(max_vocab,1,WRDCNT);
    word_cnt[0].inx = 0; /* PAD */
    word_cnt[0].cnt = 0;

    long long tot_word_cnt = 0;
    for (int i = 0; i < num_files; i++) {
        tot_word_cnt += process_text_file(file_list[i],data_dir,
                                          hmap,1,max_vocab,word_cnt,NULL,0);
        if (verbose && stdout_is_tty) {
            printf("Processed file %d of %d, %lld words\r",
                   i + 1,num_files,tot_word_cnt);
            fflush(stdout);
        }
    }
    if (verbose)
        printf("\nDataset: %d files, %lld words, %d unique\n",
               num_files,tot_word_cnt,hmap->map_used);

    qsort(word_cnt + 1,hmap->map_used - 1,
          sizeof(WRDCNT),qsort_compare_word_cnt);

    long long wcnt = 0;
    if (vocab_size <= 0) {
        long long target = (long long)(coverage * (float) tot_word_cnt);
        vocab_size = 1;
        for (int i = 1; i < hmap->map_used && wcnt < target; i++) {
            wcnt += word_cnt[i].cnt;
            vocab_size = i + 1;
        }
    }
    else {
        if (vocab_size > hmap->map_used)
            vocab_size = hmap->map_used;
        for (int i = 1; i < vocab_size; i++)
            wcnt += word_cnt[i].cnt;
    }
    coverage = tot_word_cnt ? (double) wcnt / tot_word_cnt : 0.0;
    if (verbose)
        printf("Vocabulary limited to %d words, covering %2.0f%% of corpus\n",
               vocab_size, 100.0 * coverage);

    /* Re-index retained words into a compact hashmap */
    HASHMAP* hmap2 = hashmap_create(vocab_size * 3,hmap->mem_used);
    hashmap_str2inx(hmap2, "", 1); /* PAD at 0 */
    for (int i = 1; i < vocab_size; i++) {
        const char* w = hashmap_inx2str(hmap,word_cnt[i].inx);
        if (strlen(w) == 0)
            continue;
        word_cnt[i].inx = hashmap_str2inx(hmap2,w,1);
    }
    hashmap_free(hmap);
    
    /* Use the retained vocabulary */
    hmap = hmap2;
    tot_word_cnt = wcnt;

    float* freq = NULL;
    if (calc_freq) {/* Per-word frequencies */
        freq = allocmem(vocab_size,1,float);
        freq[0] = 0.0; /* PAD */
        for (int i = 1; i < vocab_size; i++)
            freq[i] = tot_word_cnt ? (double) word_cnt[i].cnt / tot_word_cnt : 0.0;
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
            dist_size += (int) (powf(word_cnt[i].cnt,dpow) / dscale) + 1;
        dist = allocmem(dist_size,1,int);
        for (int i = 1, j = 0; i < vocab_size; i++) {
            int rpt = (int)(powf(word_cnt[i].cnt,dpow) / dscale) + 1;
            for (int k = 0; j < dist_size && k < rpt; k++)
                dist[j++] = word_cnt[i].inx;
        }
        if (verbose)
            printf("Distribution table size %d\n", dist_size);
    }
    freemem(word_cnt);

    /* freq[i] is the word frequency of word stored at index i in hmap */
    VOCAB* v = allocmem(1,1,VOCAB);
    v->hmap = hmap;
    v->size = vocab_size;
    v->coverage = coverage;
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

int vocab_tokenize(const VOCAB* v, const char* text, int* out, int max)
{
    char buf[65536];
    snprintf(buf, sizeof(buf), "%s", text);

    int cnt = 0;
    char* save = NULL;
    char* tok = strtok_r(buf, " \t\r\n", &save);
    while (tok != NULL) {
        for (char* q = tok; *q != '\0'; q++)
            *q = (char) tolower((unsigned char) *q);
        int id = hashmap_str2inx(v->hmap, tok, 0);
        if (id > 0) {
            if (cnt < max) {
                out[cnt++] = id;
            }
            else {
                fprintf(stderr, "(prompt truncated to %d tokens)\n", max);
                break;
            }
        }
        else {
            fprintf(stderr, "'%s' not in vocabulary - skipping\n",tok);
        }
        tok = strtok_r(NULL, " \t\r\n", &save);
    }
    return cnt;
}
