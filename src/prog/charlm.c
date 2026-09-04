/* Copyright (c) 2026 Gilad Odinak */
/* Character-level language model on the TinyShakespeare corpus (or any text).
 *
 * References: Andrej Karpathy
 *   The Unreasonable Effectiveness of Recurrent Neural Networks" (2015)
 *   nanoGPT: https://karpathy-nanogpt.mintlify.app/
 *
 * A single program that trains and samples from either a decoder-only
 * Transformer or a stacked LSTM, selectable at run time with -m.
 *
 * Two ways to supply data:
 *
 *   1. Single file
 *        -d path/to/input.txt
 *      The corpus is cut into length-T windows and split into train/val
 *      internally according to -F.
 *
 *   2. File lists:
 *        -t train.txt [-v val.txt] -D corpus/
 *      Each list file names one data file per line (blank lines and lines
 *      beginning with '#' are ignored). File names are resolved relative to
 *      -D (an absolute name overrides -D). All files in a list are
 *      concatenated, separated by a newline, and cut into windows; a window
 *      never spans two source files. The vocabulary is built from BOTH lists
 *      so no validation character is unknown. -v is optional.
 *
 * Pipeline (shared head, only the middle stack changes):
 *
 *   one-hot[K] --> dense(D,"none")            (learned char embedding)
 *              --> N x transformer(H,T,D,Dff) (causal; RoPE inside MHA)
 *      or      --> N x lstm(D)
 *              --> dense(K,"Softmax")          (vocab logits -> probs)
 *
 * Design notes:
 *   - The Transformer's MHA applies rotary position embeddings internally,
 *     so a plain one-hot per character is enough; no positional features.
 *   - The MODEL container fixes batch size == sequence length T, so every
 *     training example is exactly T characters and the target is the input
 *     shifted left by one.
 *   - Dropout is 0, so training and inference forward passes are identical.
 *   - Autoregressive sampling uses a fixed-width sliding window of length T,
 *     re-run each step; the next character is drawn from position T-1. RoPE
 *     makes this correct (positions are relative within the window). The LSTM
 *     uses the same windowed loop (stateful = 0), so one generate() serves both.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "loss.h"
#include "model.h"
#include "dense.h"
#include "transformer.h"
#include "lstm.h"

typedef struct config_s {
    const char* model;      /* "transformer" or "lstm"                 */
    const char* data;       /* single-file corpus path (mode 1)        */
    const char* train_list; /* list of training files (mode 2)         */
    const char* val_list;   /* list of validation files (mode 2, opt)  */
    const char* dir;        /* directory the listed files live in      */
    const char* prompt;     /* seed text for generation                */
    int   block;            /* sequence length T                       */
    int   dim;              /* model dimension D                       */
    int   ffn;              /* FFN hidden dimension Dff (transformer)  */
    int   heads;            /* attention heads (transformer)           */
    int   layers;           /* number of transformer/lstm layers       */
    int   epochs;           /* training epochs                         */
    float lr;               /* learning rate                           */
    float wd;               /* weight decay                            */
    int   gen;              /* number of characters to generate        */
    float temp;             /* sampling temperature (<=0 => greedy)    */
    int   max_seqs;         /* cap on #sequences per list, 0 = all     */
    float val_frac;         /* held-out fraction (single-file mode)    */
    unsigned int seed;      /* RNG seed (0 => time based)              */
} CONFIG;

static void config_defaults(CONFIG* c)
{
    c->model      = "transformer";
    c->data       = NULL;
    c->train_list = NULL;
    c->val_list   = NULL;
    c->dir        = ".";
    c->prompt     = "\n";
    c->block      = 64;
    c->dim        = 64;
    c->ffn        = 256;
    c->heads      = 4;
    c->layers     = 2;
    c->epochs     = 20;
    c->lr         = 5e-4f;
    c->wd         = 0.01f;
    c->gen        = 500;
    c->temp       = 0.8f;
    c->max_seqs   = 4000;
    c->val_frac   = 0.05f;
    c->seed       = 0;
}

static const char* USAGE =
 "Usage: charlm [options]\n"
 "  Data (choose one mode):\n"
 "    -d PATH        single corpus file; split by -F\n"
 "    -t FILE        file listing training files, one per line\n"
 "    -v FILE        file listing validation files (optional)\n"
 "    -D DIR         directory holding the listed files (default .)\n"
 "  Model:\n"
 "    -m M           transformer | lstm | hybrid | pattern (default transformer)\n"
 "                   pattern is T/L chars, e.g. TLTL (T=transformer L=lstm)\n"
 "    -b T           sequence length               (default 64)\n"
 "    -n D           model dimension               (default 64)\n"
 "    -f F           FFN hidden dim (transformer)  (default 256)\n"
 "    -H H           attention heads (transformer) (default 4)\n"
 "    -L N           number of layers              (default 2)\n"
 "  Training:\n"
 "    -e E           training epochs               (default 20)\n"
 "    -r R           learning rate                 (default 0.0005)\n"
 "    -w W           weight decay                  (default 0.01)\n"
 "    -M N           cap on #sequences per list, 0 = all (default 4000)\n"
 "    -F F           validation fraction, single-file mode (default 0.05)\n"
 "    -R N           RNG seed, 0 = time based      (default 0)\n"
 "  Sampling:\n"
 "    -g G           chars to generate after train (default 500)\n"
 "    -T X           temperature, <=0 greedy       (default 0.8)\n"
 "    -p S           seed text                     (default newline)\n"
 "  -h             print this message\n";

static int parse_args(int argc, char** argv, CONFIG* c)
{
    int opt;
    while ((opt = getopt(argc,argv,"m:d:t:v:D:b:n:f:H:L:e:r:w:g:T:p:M:F:R:h")) != -1) {
        switch (opt) {
            case 'm': c->model      = optarg;                  break;
            case 'd': c->data       = optarg;                  break;
            case 't': c->train_list = optarg;                  break;
            case 'v': c->val_list   = optarg;                  break;
            case 'D': c->dir        = optarg;                  break;
            case 'b': c->block      = atoi(optarg);            break;
            case 'n': c->dim        = atoi(optarg);            break;
            case 'f': c->ffn        = atoi(optarg);            break;
            case 'H': c->heads      = atoi(optarg);            break;
            case 'L': c->layers     = atoi(optarg);            break;
            case 'e': c->epochs     = atoi(optarg);            break;
            case 'r': c->lr         = atof(optarg);            break;
            case 'w': c->wd         = atof(optarg);            break;
            case 'g': c->gen        = atoi(optarg);            break;
            case 'T': c->temp       = atof(optarg);            break;
            case 'p': c->prompt     = optarg;                  break;
            case 'M': c->max_seqs   = atoi(optarg);            break;
            case 'F': c->val_frac   = atof(optarg);            break;
            case 'R': c->seed       = (unsigned) atoi(optarg); break;
            case 'h': printf("%s",USAGE); exit(0);
            case '?':
            default:  fprintf(stderr,"%s",USAGE); return 0;
        }
    }
    if (c->train_list == NULL && c->data == NULL) {
        fprintf(stderr,"Provide either -d (data) or -t (train list).\n%s",USAGE);
        return 0;
    }
    return 1;
}

typedef struct buffer_s {
    unsigned char* text;
    long n;
} BUFFER;

typedef struct vocab_s {
    int K;                      /* vocabulary size                   */
    int ch2idx[256];            /* byte -> class index, -1 if unused */
    unsigned char idx2ch[256];  /* class index -> byte               */
} VOCAB;

/* Reads an entire file into a buffer */
static void buffer_from_file(BUFFER* b, const char* path)
{
    FILE* f = fopen(path,"rb");
    if (f == NULL) {
        fprintf(stderr,"Cannot open '%s'.\n",path);
        exit(1);
    }
    fseek(f,0,SEEK_END);
    long n = ftell(f);
    fseek(f,0,SEEK_SET);
    if (n < 1) {
        fprintf(stderr,"'%s' is empty.\n",path);
        fclose(f);
        exit(1);
    }
    b->text = allocmem(n,1,unsigned char);
    long got = (long) fread(b->text,1,n,f);
    fclose(f);
    if (got != n) {
        fprintf(stderr,"Short read on '%s'.\n",path);
        exit(1);
    }
    b->n = n;
}

/* Resolves 'name' against 'dir' (absolute names ignore dir). */
static void join_path(char* out, size_t outsz, const char* dir, const char* name)
{
    if (name[0] == '/')
        snprintf(out,outsz,"%s",name);
    else
    if (dir == NULL || !dir[0])
        snprintf(out,outsz,"%s",name);
    else {
        size_t dl = strlen(dir);
        if (dir[dl-1] == '/')
            snprintf(out,outsz,"%s%s",dir,name);
        else
            snprintf(out,outsz,"%s/%s",dir,name);
    }
}

/* Reads a list file into an array of trimmed, malloc'd file names.
 * Blank lines and lines beginning with '#' are skipped. Caller frees each
 * string and the array (helper below). Exits on failure.
 */
static char** read_list(const char* path, int* count)
{
    BUFFER lb;
    buffer_from_file(&lb,path);

    int cap = 16, n = 0;
    char** names = (char**) allocmem(cap,1,char*);

    long i = 0;
    while (i < lb.n) {
        long j = i;
        while (j < lb.n && lb.text[j] != '\n') j++;      /* end of line */
        long s = i, e = j;
        while (s < e && (lb.text[s]==' '||lb.text[s]=='\t'||lb.text[s]=='\r')) s++;
        while (e > s && (lb.text[e-1]==' '||lb.text[e-1]=='\t'||lb.text[e-1]=='\r')) e--;
        if (e > s && lb.text[s] != '#') {                /* keep non-blank */
            int len = (int)(e - s);
            char* nm = (char*) allocmem(1,len + 1,char);
            memcpy(nm,lb.text + s,len);
            nm[len] = '\0';
            if (n == cap) {
                cap *= 2;
                names = (char**) realloc(names,cap * sizeof(char*));
            }
            names[n++] = nm;
        }
        i = j + 1;
    }
    freemem(lb.text);
    *count = n;
    return names;
}

static void free_list(char** names, int count)
{
    for (int i = 0; i < count; i++)
        freemem(names[i]);
    freemem(names);
}

/* Concatenates every file named in 'listpath' (resolved against 'dir') into
 * one buffer, inserting a newline between files so a window can't run words
 * from two files together. Exits on failure.
 */
static void buffer_from_list(BUFFER* b, const char* listpath, const char* dir)
{
    int nfiles = 0;
    char** names = read_list(listpath,&nfiles);
    if (nfiles == 0) { fprintf(stderr,"'%s' lists no files.\n",listpath); exit(1); }

    long* sizes = allocmem(nfiles,1,long);
    long total = 0;
    char path[4096];

    for (int i = 0; i < nfiles; i++) {                   /* pass 1: sizes */
        join_path(path,sizeof(path),dir,names[i]);
        FILE* f = fopen(path,"rb");
        if (f == NULL) { fprintf(stderr,"Cannot open '%s'.\n",path); exit(1); }
        fseek(f,0,SEEK_END);
        sizes[i] = ftell(f);
        fclose(f);
        if (sizes[i] < 0) sizes[i] = 0;
        total += sizes[i] + 1;                           /* +1 separator  */
    }
    if (total < 2) {
        fprintf(stderr,"'%s' files are empty.\n",listpath);
        exit(1);
    }

    b->text = allocmem(total,1,unsigned char);
    long off = 0;
    for (int i = 0; i < nfiles; i++) {                   /* pass 2: read  */
        join_path(path,sizeof(path),dir,names[i]);
        FILE* f = fopen(path,"rb");
        if (f == NULL) {
            fprintf(stderr,"Cannot open '%s'.\n",path);
            exit(1);
        }
        long cnt = (long) fread(b->text + off,1,sizes[i],f);
        fclose(f);
        off += cnt;
        b->text[off++] = '\n';                           /* file separator */
    }
    b->n = off;

    freemem(sizes);
    free_list(names,nfiles);
}

/* Builds a shared vocabulary by scanning one or more buffers */
static void vocab_build(VOCAB* v, BUFFER** bufs, int nbufs)
{
    for (int i = 0; i < 256; i++)
        v->ch2idx[i] = -1;
    int K = 0;
    for (int i = 0; i < nbufs; i++) {
        BUFFER* b = bufs[i];
        for (long p = 0; p < b->n; p++) {
            int by = b->text[p];
            if (v->ch2idx[by] < 0) {
                v->ch2idx[by] = K;
                v->idx2ch[K] = (unsigned char) by;
                K++;
            }
        }
    }
    v->K = K;
}

/* Each sequence is T consecutive characters; its per-position target 
 * is the next character. Windows are cut with a stride chosen so that,
 * if a buffer yields more than max_seqs windows, they are spread evenly
 * across the whole buffer. Stored as dense one-hot [num*T][K].
 */
typedef struct {
    fArr2D x;   /* [num*T][K] one-hot inputs      */
    fArr2D y;   /* [num*T][K] one-hot targets     */
    int*   len; /* [num] sequence lengths (all T) */
    int    num; /* number of sequences            */
    int    T;   /* sequence length                */
    int    K;   /* vocabulary size                */
} DATASET;

static void dataset_build(DATASET* d, const BUFFER* b, const VOCAB* v,
                          int T, int max_seqs)
{
    long usable = b->n - 1;                              /* need next char  */
    long max_windows = usable / T;
    if (max_windows < 1) { fprintf(stderr,"Data shorter than block.\n"); exit(1); }

    long stride = T;
    long num = max_windows;
    if (max_seqs > 0 && (long) max_seqs < max_windows) {
        num = max_seqs;
        stride = usable / num;
        if (stride < 1) stride = 1;
    }

    int K = v->K;
    d->T = T; d->K = K; d->num = (int) num;
    d->x   = allocmem(num * T, K, float);
    d->y   = allocmem(num * T, K, float);
    d->len = allocmem(num, 1, int);

    typedef float (*ArrK)[K];
    ArrK X = (ArrK) d->x;
    ArrK Y = (ArrK) d->y;

    for (long s = 0; s < num; s++) {
        long start = s * stride;
        if (start + T + 1 > b->n) start = b->n - T - 1;  /* clamp tail */
        for (int t = 0; t < T; t++) {
            long r  = s * T + t;
            int in  = v->ch2idx[b->text[start + t]];
            int tgt = v->ch2idx[b->text[start + t + 1]];
            X[r][in]  = 1.0f;
            Y[r][tgt] = 1.0f;
        }
        d->len[s] = T;
    }
}

static void dataset_free(DATASET* d)
{
    freemem(d->x);
    freemem(d->y);
    freemem(d->len);
    d->x = NULL;
    d->y = NULL;
    d->len = NULL;
}

/* Resolves -m into an explicit middle-layer pattern of 'T' (transformer)
 * and 'L' (lstm) characters. Keyword forms expand using -L:
 *   "transformer" -> "TTT..."   "lstm" -> "LLL..."   "hybrid" -> "TLTL..."
 * Anything else is taken as a literal pattern (e.g. "TLTL", "llt", "TLLTLL").
 * Returns a malloc'd, upper-cased string; sets *nmid to its length. Exits on
 * an empty or invalid pattern. Caller frees.
 */
static char* resolve_pattern(const CONFIG* c, int* nmid)
{
    char* p;
    if (!strcmp(c->model,"transformer") ||
        !strcmp(c->model,"lstm") || !strcmp(c->model,"hybrid")) {

        int n = c->layers;
        if (n < 1) {
            fprintf(stderr,"-L (layers) must be >= 1\n");
            exit(1);
        }
        p = (char*) allocmem(1,n + 1,char);
        for (int i = 0; i < n; i++) {
            if (!strcmp(c->model,"transformer"))
                p[i] = 'T';
            else
            if (!strcmp(c->model,"lstm"))
                p[i] = 'L';
            else
                p[i] = (i % 2 == 0) ? 'T' : 'L';
        }
        p[n] = '\0';
    }
    else { /* Literal pattern string */
        int n = (int) strlen(c->model);
        if (n < 1) {
            fprintf(stderr,"-m pattern is empty\n");
            exit(1);
        }
        p = (char*) allocmem(1,n + 1,char);
        for (int i = 0; i < n; i++) {
            char ch = c->model[i];
            if (ch=='T'||ch=='t')
                p[i] = 'T';
            else
            if (ch=='L'||ch=='l')
                p[i] = 'L';
            else {
                fprintf(stderr,"-m pattern may contain only T/L, got '%c'\n",ch);
                exit(1);
            }
        }
        p[n] = '\0';
    }
    *nmid = (int) strlen(p);
    return p;
}

/* Builds the model from a resolved T/L pattern.
 *
 * A leading dense(D,"none") always projects the one-hot input to the model
 * dimension, so the residual stream is D-dimensional from the start and any
 * mix of transformer (D->D) and lstm (D->D). The output is dense(K,"Softmax").
 * Layer count = pattern length + 2.
 */
static MODEL* build_model(const CONFIG* c, const char* pattern, int nmid, int K)
{
    const int T = c->block;
    const int D = c->dim;

    int L = nmid + 2;                   /* embedding + middle + output   */
    MODEL* m = model_create(L,T,K,1,0); /* batch==T, one-hot K, add bias */

    model_add(m,dense_create(D,"none"),"dense");    /* embedding    */
    for (int i = 0; i < nmid; i++) {
        if (pattern[i] == 'T')
            model_add(m,transformer_create(c->heads,T,D,c->ffn,0),"transformer");
        else
            model_add(m,lstm_create(D,0),"lstm");   /* stateful=0   */
    }
    model_add(m,dense_create(K,"Softmax"),"dense"); /* vocab output */

    model_compile(m,"cross-entropy","adamw");
    return m;
}

/* Draws an index from a probability row with temperature. Because the head
 * is softmax, temperature on logits equals raising probabilities to 1/temp
 * and renormalising (softmax(log p / temp) is proportional to p^(1/temp)).
 */
static int sample_row(const float* p, int K, float temp)
{
    if (temp <= 0.0) {
        int best = 0;
        for (int k = 1; k < K; k++)
            if (p[k] > p[best])
                best = k;
        return best;
    }
    float w[256];
    float sum = 0.0;
    float inv = 1.0 / temp;
    for (int k = 0; k < K; k++) {
        float pk = p[k] > 1e-16f ? p[k] : 1e-16f;
        w[k] = powf(pk,inv);
        sum += w[k];
    }
    float r = urand(0,1) * sum;
    float acc = 0;
    for (int k = 0; k < K; k++) {
        acc += w[k];
        if (r <= acc)
            return k;
    }
    return K - 1;
}

static void generate(MODEL* m, const VOCAB* v, const CONFIG* cfg)
{
    const int T = cfg->block;
    const int K = v->K;

    int fill = (v->ch2idx['\n'] >= 0) ? v->ch2idx['\n']
             : (v->ch2idx[' ']  >= 0) ? v->ch2idx[' '] : 0;

    int* window = allocmem(T,1,int);
    for (int t = 0; t < T; t++)
        window[t] = fill;

    const char* pr = cfg->prompt;
    int plen = (int) strlen(pr);
    printf("---- sample (%s, temp %.2f) ----\n",cfg->model,cfg->temp);
    for (int i = 0; i < plen; i++) {
        int idx = v->ch2idx[(unsigned char) pr[i]];
        if (idx < 0)
            continue;
        memmove(window,window + 1,(T - 1) * sizeof(int));
        window[T - 1] = idx;
        putchar(pr[i]);
    }

    fArr2D X = allocmem(T,K,float);
    fArr2D Y = allocmem(T,K,float);
    typedef float (*ArrK)[K];
    ArrK x = (ArrK) X;
    ArrK y = (ArrK) Y;

    for (int step = 0; step < cfg->gen; step++) {
        fltclr(X,T * K);
        for (int t = 0; t < T; t++)
            x[t][window[t]] = 1;

        model_predict(m,X,Y,T);

        int next = sample_row(y[T - 1],K,cfg->temp);
        putchar(v->idx2ch[next]);

        memmove(window,window + 1,(T - 1) * sizeof(int));
        window[T - 1] = next;
    }
    printf("\n----------------------------------\n");

    freemem(X); freemem(Y); freemem(window);
}

int main(int argc, char** argv)
{
    CONFIG cfg;
    config_defaults(&cfg);
    if (!parse_args(argc,argv,&cfg))
        return 1;

    unsigned int seed = cfg.seed;
    if (seed == 0) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts);
        seed = (unsigned int)(ts.tv_sec ^ ts.tv_nsec);
    }
    init_lrng(seed);
    printf("seed %u\n",seed);

    int list_mode = (cfg.train_list != NULL);
    const int T = cfg.block;

    VOCAB voc;
    DATASET dtr, dval;         /* List mode: two datasets      */
    DATASET ds;                /* Single-file mode: one, split */
    memset(&dtr,0,sizeof dtr);
    memset(&dval,0,sizeof dval);
    memset(&ds,0,sizeof ds);

    fArr2D xTr, yTr, xVd = NULL, yVd = NULL;
    int   *sTr, *sVd = NULL;
    int    num_tr, num_val = 0;

    if (list_mode) {
        int have_val = (cfg.val_list != NULL);
        BUFFER btr, bval;
        buffer_from_list(&btr,cfg.train_list,cfg.dir);
        if (have_val)
            buffer_from_list(&bval,cfg.val_list,cfg.dir);

        BUFFER* bufs[2]; int nb = 0;
        bufs[nb++] = &btr;
        if (have_val) bufs[nb++] = &bval;
        vocab_build(&voc,bufs,nb);
        printf("vocab %d (from %d list%s)\n",voc.K,nb,nb>1?"s":"");

        dataset_build(&dtr,&btr,&voc,T,cfg.max_seqs);
        xTr = dtr.x; yTr = dtr.y; sTr = dtr.len; num_tr = dtr.num;
        freemem(btr.text);

        if (have_val) {
            dataset_build(&dval,&bval,&voc,T,cfg.max_seqs);
            xVd = dval.x; yVd = dval.y; sVd = dval.len; num_val = dval.num;
            freemem(bval.text);
        }
    }
    else {
        BUFFER b;
        buffer_from_file(&b,cfg.data);
        BUFFER* one = &b;
        vocab_build(&voc,&one,1);
        printf("corpus: %ld chars, vocab %d\n",b.n,voc.K);

        dataset_build(&ds,&b,&voc,T,cfg.max_seqs);
        freemem(b.text);

        num_val = (int)(ds.num * cfg.val_frac);
        if (num_val < 0) num_val = 0;
        num_tr  = ds.num - num_val;
        if (num_tr < 1) { num_tr = ds.num; num_val = 0; }

        xTr = ds.x; yTr = ds.y; sTr = ds.len;
        if (num_val) {
            typedef float (*ArrK)[ds.K];
            xVd = (fArr2D)(((ArrK) ds.x) + (size_t) num_tr * ds.T);
            yVd = (fArr2D)(((ArrK) ds.y) + (size_t) num_tr * ds.T);
            sVd = ds.len + num_tr;
        }
    }

    int nmid = 0;
    char* pattern = resolve_pattern(&cfg,&nmid);
    int has_xfmr = (strchr(pattern,'T') != NULL);
    if (has_xfmr && (cfg.dim % cfg.heads) != 0) {
        fprintf(stderr,"-n (dim %d) must be a multiple of -H (heads %d) "
                "for transformer layers\n",cfg.dim,cfg.heads);
        free(pattern); return 1;
    }

    if (has_xfmr)
        printf("model %s -> stack %s (%d layers), "
               "D %d, ffn %d, heads %d, block %d\n",
               cfg.model,pattern,nmid,cfg.dim,cfg.ffn,cfg.heads,cfg.block);
    else
        printf("model %s -> stack %s (%d layers), D %d, block %d\n",
               cfg.model,pattern,nmid,cfg.dim,cfg.block);
    printf("sequences: %d train, %d val\n\n",num_tr,num_val);
    printf("%d epochs, learning rate %g weight decay %g\n",
           cfg.epochs,cfg.lr,cfg.wd);

    MODEL* m = build_model(&cfg,pattern,nmid,voc.K);

    float* losses = allocmem(cfg.epochs,1,float);
    float* accs   = allocmem(cfg.epochs,1,float);
    float* vloss  = num_val ? allocmem(cfg.epochs,1,float) : NULL;
    float* vaccs  = num_val ? allocmem(cfg.epochs,1,float) : NULL;

    model_fit(m,xTr,yTr,sTr,num_tr,
              xVd,yVd,sVd,num_val,
              cfg.epochs,cfg.lr,cfg.wd,
              losses,accs,vloss,vaccs,
              "final=1 verbose=2");
    printf("\n");

    generate(m,&voc,&cfg);

    freemem(losses);
    freemem(accs);
    freemem(vloss);
    freemem(vaccs);
    freemem(pattern);
    model_free(m);
    if (list_mode) {
        dataset_free(&dtr);
        dataset_free(&dval);
    }
    else
        dataset_free(&ds);
    return 0;
}
