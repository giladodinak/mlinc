/* Copyright (c) 2026 Gilad Odinak */
/* Evaluates a word-embedding file against SimLex-999.
 *
 * Expected simlex.csv format:
 *   A SimLex-999 TSV file whose header row includes the columns
 *   word1, word2, and SimLex999.
 *
 * For every pair whose two words are both in the vocabulary, computes
 * the cosine similarity of their vectors and correlates it with the 
 * SimLex999 human-similarity rating using Spearman's rho.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "hash.h"
#include "wembio.h"

#ifndef LINE_MAX_LEN
#define LINE_MAX_LEN 65536
#endif

typedef struct {
    fArr2D emb;       /* [size][dim] matrix owned by load_word_embeddings */
    float* norm;      /* per-vector L2 norm, length size                  */
    int size;         /* vocabulary size                                  */
    int dim;          /* embedding dimension                              */
    HASHMAP* map;     /* word -> vocab index (== row in emb)              */
} Embeddings;

typedef struct {
    float model;      /* model cosine similarity */
    float rating;     /* SimLex999 rating        */
} PairScore;

typedef struct {
    float value;
    int index;
} SortItem;

static void chomp(char* s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r'))
        s[--n] = '\0';
}

/* Counts the number of lines in an open file, then rewinds it. */
static int count_lines(FILE* fp)
{
    char line[LINE_MAX_LEN];
    int n = 0;
    while (fgets(line,sizeof(line),fp))
        n++;
    rewind(fp);
    return n;
}

/* Loads the embedding file and precomputes each vector's L2 norm (needed for
 * cosine similarity and reused across every pair). */
static int load_embeddings(const char* path, Embeddings* e)
{
    if (!load_word_embeddings(path,&e->size,&e->dim,
                              NULL,NULL,NULL,&e->map,&e->emb))
        return -1;

    typedef float (*ArrDE)[e->dim];
    ArrDE E = (ArrDE) e->emb;
    e->norm = allocmem(e->size,1,float);
    for (int i = 0; i < e->size; i++) {
        double ss = 0.0;
        for (int j = 0; j < e->dim; j++)
            ss += (double)E[i][j] * E[i][j];
        e->norm[i] = (float)sqrt(ss);
    }
    return 0;
}

static void free_embeddings(Embeddings* e)
{
    if (e == NULL)
        return;
    freemem(e->norm);
    if (e->map != NULL)
        hashmap_free(e->map);
    freemem(e->emb);
    memset(e,0,sizeof(*e));
}

/* Cosine similarity between vocabulary rows i and j. */
static float cosine(const Embeddings* e, int i, int j)
{
    float ni = e->norm[i], nj = e->norm[j];
    if (ni == 0.0f || nj == 0.0f)
        return 0.0f;

    typedef float (*ArrDE)[e->dim];
    ArrDE E = (ArrDE) e->emb;
    const float* a = E[i];
    const float* b = E[j];

    double dot = 0.0;
    for (int k = 0; k < e->dim; k++)
        dot += (double)a[k] * b[k];
    return (float)(dot / ((double)ni * nj));
}

static int cmp_sort_item(const void* pa, const void* pb)
{
    const SortItem* a = (const SortItem*)pa;
    const SortItem* b = (const SortItem*)pb;
    if (a->value < b->value) return -1;
    if (a->value > b->value) return 1;
    return a->index - b->index;
}

/* Converts values to 1-based ranks, averaging ranks within ties. */
static void make_ranks(const float* x, float* rank, int n)
{
    SortItem* items = allocmem(n,1,SortItem);
    for (int i = 0; i < n; i++) {
        items[i].value = x[i];
        items[i].index = i;
    }
    qsort(items,(size_t)n,sizeof(SortItem),cmp_sort_item);

    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && items[j].value == items[i].value)
            j++;
        float avg_rank = ((float)(i + 1) + (float)j) / 2.0;
        for (int k = i; k < j; k++)
            rank[items[k].index] = avg_rank;
        i = j;
    }
    freemem(items);
}

static float pearson(const float* x, const float* y, int n)
{
    float sx = 0.0, sy = 0.0;
    for (int i = 0; i < n; i++) {
        sx += x[i];
        sy += y[i];
    }
    float mx = sx / n;
    float my = sy / n;

    float num = 0.0, dx2 = 0.0, dy2 = 0.0;
    for (int i = 0; i < n; i++) {
        float dx = x[i] - mx;
        float dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if (dx2 == 0.0 || dy2 == 0.0)
        return 0.0;
    return num / sqrt(dx2 * dy2);
}

/* Spearman's rho: Pearson correlation of the rank vectors. */
static float spearman(const PairScore* pairs, int n)
{
    float* model = allocmem(n,1,float);
    float* rating = allocmem(n,1,float);
    float* rm = allocmem(n,1,float);
    float* rg = allocmem(n,1,float);

    for (int i = 0; i < n; i++) {
        model[i] = pairs[i].model;
        rating[i] = pairs[i].rating;
    }
    make_ranks(model,rm,n);
    make_ranks(rating,rg,n);
    float rho = pearson(rm,rg,n);

    freemem(model);
    freemem(rating);
    freemem(rm);
    freemem(rg);
    return rho;
}

/* Splits a line in place on tabs, storing up to max_fields field pointers. */
static int split_tsv(char* line, char** fields, int max_fields)
{
    int n = 0;
    char* p = line;
    while (n < max_fields) {
        fields[n++] = p;
        char* tab = strchr(p,'\t');
        if (tab == NULL)
            break;
        *tab = '\0';
        p = tab + 1;
    }
    return n;
}

static int find_field(char** fields, int n, const char* name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(fields[i],name) == 0)
            return i;
    return -1;
}

static int eval_simlex(const char* path, const Embeddings* e)
{
    FILE* fp = fopen(path,"r");
    if (fp == NULL) {
        perror(path);
        return -1;
    }

    int nlines = count_lines(fp);
    if (nlines < 1)
        nlines = 1;

    char line[LINE_MAX_LEN];
    char* fields[64];
    if (!fgets(line,sizeof(line),fp)) {
        fprintf(stderr,"empty SimLex file: %s\n",path);
        fclose(fp);
        return -1;
    }
    chomp(line);
    int nf = split_tsv(line,fields,64);
    int w1_col = find_field(fields,nf,"word1");
    int w2_col = find_field(fields,nf,"word2");
    int score_col = find_field(fields,nf,"SimLex999");
    if (w1_col < 0 || w2_col < 0 || score_col < 0) {
        fprintf(stderr,"SimLex header must include word1, word2, SimLex999\n");
        fclose(fp);
        return -1;
    }

    PairScore* pairs = allocmem(nlines,1,PairScore);
    int used = 0, total = 0, oov = 0;

    while (fgets(line,sizeof(line),fp)) {
        chomp(line);
        if (line[0] == '\0')
            continue;
        nf = split_tsv(line,fields,64);
        if (nf <= score_col || nf <= w1_col || nf <= w2_col)
            continue;
        total++;

        int i1 = hashmap_str2inx(e->map,fields[w1_col],0);
        int i2 = hashmap_str2inx(e->map,fields[w2_col],0);
        if (i1 < 0 || i1 >= e->size || i2 < 0 || i2 >= e->size) {
            oov++;
            continue;
        }

        pairs[used].model = cosine(e,i1,i2);
        pairs[used].rating = atof(fields[score_col]);
        used++;
    }
    fclose(fp);

    printf("vocab_size: %d\n",e->size);
    printf("embedding_dim: %d\n",e->dim);
    printf("simlex_total_pairs: %d\n",total);
    printf("covered_pairs: %d\n",used);
    printf("oov_pairs: %d\n",oov);
    if (total > 0)
        printf("coverage: %.2f%%\n",100.0 * (double)used / (double)total);

    if (used < 2) {
        printf("spearman: N/A, fewer than 2 covered pairs\n");
        freemem(pairs);
        return 0;
    }

    printf("spearman: %.6f\n",spearman(pairs,used));
    freemem(pairs);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr,"Usage: %s word-embedding-file data/simlex/simlex.csv\n",argv[0]);
        return 1;
    }

    Embeddings e;
    memset(&e,0,sizeof(e));
    if (load_embeddings(argv[1],&e) != 0) {
        free_embeddings(&e);
        return 1;
    }
    if (e.size == 0 || e.dim == 0) {
        fprintf(stderr,"no embeddings loaded from %s\n",argv[1]);
        free_embeddings(&e);
        return 1;
    }

    int rc = eval_simlex(argv[2],&e);
    free_embeddings(&e);
    return rc == 0 ? 0 : 1;
}
