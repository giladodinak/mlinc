/* Copyright (c) 2026 Gilad Odinak */
/* Evaluates a word-embedding file against the Bigger Analogy Test Set (BATS)
 *
 * References: 
 *   - Anna Gladkova, Aleksandr Drozd, and Satoshi Matsuoka. 2016. 
 *     Analogy-based detection of morphological and semantic relations with
 *     word embeddings. https://aclanthology.org/N16-2002.pdf
 *   - Omer Levy and Yoav Goldberg. 2014. Linguistic Regularities in Sparse
 *     and Explicit Word Representations. https://aclanthology.org/W14-1618.pdf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "hash.h"
#include "wembio.h"

#ifndef LINE_MAX_LEN
#define LINE_MAX_LEN 65536
#endif

#define PATH_BUF 4096

typedef struct {
    fArr2D emb;       /* [size][dim] matrix owned by load_word_embeddings */
    float* norm;      /* per-vector L2 norm, length size                  */
    int size;         /* vocabulary size                                  */
    int dim;          /* embedding dimension                              */
    HASHMAP* map;     /* word -> vocab index (== row in emb)              */
} Embeddings;

typedef struct {
    int src;      /* vocab index of the source word     */
    int* tgts;    /* vocab indices of in-vocab targets  */
    int ntgts;
} BatsPair;

static char* xstrdup(const char* s)
{
    size_t n = strlen(s) + 1;
    char* p = allocmem(n,1,char);
    memcpy(p,s,n);
    return p;
}

static void chomp(char* s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r'))
        s[--n] = '\0';
}

static char* trim(char* s)
{
    while (*s && isspace((unsigned char)*s))
        s++;
    if (*s) {
        char* end = s + strlen(s) - 1;
        while (end > s && isspace((unsigned char)*end))
            *end-- = '\0';
    }
    return s;
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

/* Loads the embedding file and precomputes each vector's L2 norm 
 * (needed for cosine scoring and reused across every analogy question).
 */
static int load_embeddings(const char* path, Embeddings* e)
{
    if (!load_word_embeddings(path,&e->size,&e->dim,
                              NULL,NULL,NULL,&e->map,&e->emb))
        return -1;

    typedef float (*ArrDE)[e->dim];
    ArrDE E = (ArrDE) e->emb;
    e->norm = allocmem(e->size,1,float);
    for (int i = 0; i < e->size; i++) {
        float ss = 0.0;
        for (int j = 0; j < e->dim; j++)
            ss += E[i][j] * E[i][j];
        e->norm[i] = sqrtf(ss);
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

/* 3CosAdd: returns the vocab index of the best word for a:b :: c:?,
 * excluding a, b, c.
 * Returns -1 if the question cannot be scored.
 */
static int predict_analogy(const Embeddings* e, int ai, int bi, int ci, float* query)
{
    float na = e->norm[ai], nb = e->norm[bi], nc = e->norm[ci];
    if (na == 0.0f || nb == 0.0f || nc == 0.0f)
        return -1;

    typedef float (*ArrDE)[e->dim];
    ArrDE E = (ArrDE) e->emb;
    const float* A = E[ai];
    const float* B = E[bi];
    const float* C = E[ci];

    for (int i = 0; i < e->dim; i++)
        query[i] = B[i] / nb - A[i] / na + C[i] / nc;

    float best_score = -INFINITY;
    int best = -1;
    for (int w = 0; w < e->size; w++) {
        if (w == ai || w == bi || w == ci)
            continue;
        float nw = e->norm[w];
        if (nw == 0.0f)
            continue;
        const float* W = E[w];
        float dot = 0.0;
        for (int i = 0; i < e->dim; i++)
            dot += W[i] * query[i];
        float score = dot / nw;   /* |query| is constant across w */
        if (score > best_score) {
            best_score = score;
            best = w;
        }
    }
    return best;
}

/* Reads one subcategory file, keeping only pairs whose source and at least
 * one target are in the vocabulary.
 * Returns the count, or -1 on open error.
 */
static int parse_bats_file(const char* path, const Embeddings* e, BatsPair** out)
{
    FILE* fp = fopen(path,"r");
    if (fp == NULL) {
        perror(path);
        return -1;
    }

    int nlines = count_lines(fp);
    if (nlines < 1)
        nlines = 1;
    BatsPair* pairs = allocmem(nlines,1,BatsPair);
    int n = 0;

    char line[LINE_MAX_LEN];
    while (fgets(line,sizeof(line),fp)) {
        chomp(line);
        if (line[0] == '\0')
            continue;

        char* sep = strpbrk(line,"\t ");   /* source / target separator */
        if (sep == NULL)
            continue;
        *sep = '\0';
        char* src = trim(line);
        char* tgtstr = sep + 1;
        int si = hashmap_str2inx(e->map,src,0);
        if (si < 0)
            continue;

        /* Target count = number of '/'-separated fields. */
        int tcap = 1;
        for (char* p = tgtstr; *p != '\0'; p++)
            if (*p == '/')
                tcap++;
        int* tgts = allocmem(tcap,1,int);
        int tn = 0;
        char* save = NULL;
        for (char* tok = strtok_r(tgtstr,"/",&save); tok;
             tok = strtok_r(NULL,"/",&save)) {
            char* w = trim(tok);
            if (*w == '\0')
                continue;
            int ti = hashmap_str2inx(e->map,w,0);
            if (ti < 0)
                continue;
            tgts[tn++] = ti;
        }

        if (tn == 0) {
            freemem(tgts);
            continue;
        }
        pairs[n].src = si;
        pairs[n].tgts = tgts;
        pairs[n].ntgts = tn;
        n++;
    }
    fclose(fp);

    *out = pairs;
    return n;
}

/* Evaluates every ordered pair of pairs in one subcategory file. */
static void eval_file(const char* path, const Embeddings* e, float* query,
                      long* out_correct, long* out_total)
{
    BatsPair* pairs = NULL;
    int n = parse_bats_file(path,e,&pairs);
    long correct = 0, total = 0;

    for (int i = 0; i < n; i++) {
        int ai = pairs[i].src;
        int bi = pairs[i].tgts[0];
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            int ci = pairs[j].src;
            int best = predict_analogy(e,ai,bi,ci,query);
            if (best < 0)
                continue;
            total++;
            for (int t = 0; t < pairs[j].ntgts; t++) {
                if (best == pairs[j].tgts[t]) {
                    correct++;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < n; i++)
        freemem(pairs[i].tgts);
    freemem(pairs);

    *out_correct = correct;
    *out_total = total;
}

static int cmp_str(const void* a, const void* b)
{
    const char* const* pa = (const char* const*)a;
    const char* const* pb = (const char* const*)b;
    return strcmp(*pa,*pb);
}

/* Lists non-hidden entries of a directory into a sorted array of names. */
static int list_dir(const char* path, char*** out_names)
{
    DIR* d = opendir(path);
    if (d == NULL)
        return -1;

    int n = 0;
    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.')
            continue;
        n++;
    }
    rewinddir(d);

    int cap = n < 1 ? 1 : n;
    char** names = allocmem(cap,1,char*);
    n = 0;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.')
            continue;
        names[n++] = xstrdup(de->d_name);
    }
    closedir(d);

    qsort(names,(size_t)n,sizeof(char*),cmp_str);
    *out_names = names;
    return n;
}

static int has_suffix(const char* s, const char* suf)
{
    size_t ls = strlen(s);
    size_t lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf,suf) == 0;
}

static int is_dir(const char* path)
{
    struct stat st;
    return stat(path,&st) == 0 && S_ISDIR(st.st_mode);
}

static int eval_bats(const char* root, const Embeddings* e)
{
    char** cats = NULL;
    int ncats = list_dir(root,&cats);
    if (ncats < 0) {
        perror(root);
        return -1;
    }

    float* query = allocmem(e->dim,1,float);

    printf("vocab_size: %d\n",e->size);
    printf("embedding_dim: %d\n\n",e->dim);

    long grand_c = 0, grand_t = 0;
    for (int ci = 0; ci < ncats; ci++) {
        char catpath[PATH_BUF / 2 - 2];
        snprintf(catpath,sizeof(catpath),"%s/%s",root,cats[ci]);
        if (!is_dir(catpath))
            continue;

        char** files = NULL;
        int nf = list_dir(catpath,&files);
        long cat_c = 0, cat_t = 0;
        for (int fi = 0; fi < nf; fi++) {
            if (!has_suffix(files[fi],".txt"))
                continue;
            char fpath[PATH_BUF];
            snprintf(fpath,sizeof(fpath),"%s/%s",catpath,files[fi]);
            long fc = 0, ft = 0;
            eval_file(fpath,e,query,&fc,&ft);
            float acc = ft ? 100.0 * (float)fc / (float)ft : 0.0;
            printf("  %-45s %6.2f%%  (%ld/%ld)\n",files[fi],acc,fc,ft);
            cat_c += fc;
            cat_t += ft;
        }
        for (int fi = 0; fi < nf; fi++)
            freemem(files[fi]);
        freemem(files);

        float cacc = cat_t ? 100.0 * (float)cat_c / (float)cat_t : 0.0;
        printf("%-35s %6.2f%%  (%ld/%ld)\n\n",cats[ci],cacc,cat_c,cat_t);
        grand_c += cat_c;
        grand_t += cat_t;
    }

    for (int ci = 0; ci < ncats; ci++)
        freemem(cats[ci]);
    freemem(cats);
    freemem(query);

    float gacc = grand_t ? 100.0 * (float)grand_c / (float)grand_t : 0.0;
    printf("OVERALL %6.2f%%  (%ld/%ld)\n",gacc,grand_c,grand_t);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr,"Usage: %s word-embedding-file data/bats/BATS_3.0\n",argv[0]);
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

    int rc = eval_bats(argv[2],&e);
    free_embeddings(&e);
    return rc == 0 ? 0 : 1;
}
