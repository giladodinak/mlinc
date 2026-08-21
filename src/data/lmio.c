/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store a full decoder-only language model.          */
/*                                                                          */
/* File layout:                                                             */
/*   LM <header: dims + resumable training schedule + final flag>           */
/*   HASHMAP <vocabulary: index/word lines>                                 */
/*   DIST <unigram sampling table>            (omitted when final)          */
/*   LMEMB <embedding weights>                (via lmembio)                 */
/*   MODEL <N transformer layers + head>      (via modelio)                 */
/*                                                                          */
/* The transformer stack and the negative-sampling head are serialized      */
/* through modelio's write_model()/read_model() by wrapping them in a        */
/* borrow-shell MODEL (N transformer LAYERs of type 't' followed by one      */
/* 'n' head LAYER). This reuses modelio's transformer AdamW-moment layout,   */
/* so a resumed run restores the exact optimizer state of the stack.         */
#include <stdio.h>
#include <string.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "hash.h"
#include "lmemb.h"
#include "lmembio.h"
#include "negsample.h"
#include "transformer.h"
#include "layer.h"
#include "model.h"
#include "modelio.h"
#include "lm.h"
#include "lmio.h"

/* ---- vocabulary (hashmap) ---------------------------------------------- */

/* Index 0 is PAD (the empty word) and is reconstructed on read, so only
 * indices 1..map_used-1 (all non-empty tokens) are written. Tokens are
 * whitespace-free in this corpus, so "%d %s" round-trips them. */
static int write_hashmap(HASHMAP* h, FILE* fp)
{
    int cnt = fprintf(fp,"HASHMAP map_used %d mem_used %d\n",
                      h->map_used,h->mem_used);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_hashmap: failed to write the header\n");
        return 0;
    }
    for (int i = 1; i < h->map_used; i++) {
        const char* w = hashmap_inx2str(h,i);
        if (w == NULL) w = "";
        cnt = fprintf(fp,"%d %s\n",i,w);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,"In write_hashmap: failed to write word %d\n",i);
            return 0;
        }
    }
    return 1;
}

static HASHMAP* read_hashmap(FILE* fp)
{
    int map_used, mem_used;
    int cnt = fscanf(fp," HASHMAP map_used %d mem_used %d\n",
                     &map_used,&mem_used);
    if (cnt < 2 || cnt == EOF) {
        fprintf(stderr,"In read_hashmap: failed to read the header\n");
        return NULL;
    }
    HASHMAP* h = hashmap_create(map_used * 3,mem_used);
    hashmap_str2inx(h,"",1);            /* index 0 = PAD */
    char word[256];
    for (int i = 1; i < map_used; i++) {
        int idx;
        cnt = fscanf(fp,"%d %255s",&idx,word);
        if (cnt < 2 || cnt == EOF) {
            fprintf(stderr,"In read_hashmap: failed to read word %d\n",i);
            hashmap_free(h);
            return NULL;
        }
        int a = hashmap_str2inx(h,word,1);
        if (a != i)
            fprintf(stderr,"In read_hashmap: warning: '%s' got index %d, "
                           "expected %d\n",word,a,i);
    }
    return h;
}

/* ---- unigram sampling table -------------------------------------------- */

static int write_dist(const int* dist, int dist_size, FILE* fp)
{
    int cnt = fprintf(fp,"DIST dist_size %d\n",dist_size);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_dist: failed to write the header\n");
        return 0;
    }
    for (int j = 0; j < dist_size; j++) {
        char sep = ((j % 20) == 19 || j == dist_size - 1) ? '\n' : ' ';
        cnt = fprintf(fp,"%d%c",dist[j],sep);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,"In write_dist: failed to write entry %d\n",j);
            return 0;
        }
    }
    if (dist_size == 0)
        fprintf(fp,"\n");
    return 1;
}

static int read_dist(FILE* fp, int** dist_out, int* size_out)
{
    int dist_size;
    int cnt = fscanf(fp," DIST dist_size %d",&dist_size);
    if (cnt < 1 || cnt == EOF || dist_size < 0) {
        fprintf(stderr,"In read_dist: failed to read the header\n");
        return 0;
    }
    int* dist = (dist_size > 0) ? allocmem(dist_size,1,int) : NULL;
    for (int j = 0; j < dist_size; j++) {
        cnt = fscanf(fp," %d",&dist[j]);
        if (cnt < 1 || cnt == EOF) {
            fprintf(stderr,"In read_dist: failed to read entry %d\n",j);
            freemem(dist);
            return 0;
        }
    }
    *dist_out = dist;
    *size_out = dist_size;
    return 1;
}

/* ---- transformer stack + head (via modelio) ---------------------------- */

/* Wrap the N transformer LAYERs and the head in a borrow-shell MODEL and
 * hand it to write_model(). The shell borrows m's pointers; write_model
 * frees nothing, so only the temporary LAYER array is freed here. */
static int write_lm_stack(const LM* m, int fin, const LMTRAIN* st, FILE* fp)
{
    LAYER* layers = allocmem(1,m->N + 1,LAYER);
    for (int i = 0; i < m->N; i++)
        layers[i] = m->tr[i];                 /* borrow transformer + grads */
    layers[m->N].type = 'n';
    layers[m->N].negsample = m->head;
    layers[m->N].grads = NULL;
    layers[m->N].num_grads = 0;               /* head uses sparse SGD        */
    layers[m->N].out = NULL;

    MODEL shell;
    memset(&shell,0,sizeof shell);
    shell.num_layers = m->N + 1;
    shell.batch_size = m->B;
    shell.input_dim  = m->E;
    shell.output_dim = m->E;
    shell.target_dim = 1;
    shell.add_bias   = 0;
    shell.normalize  = 0;
    shell.loss_func  = 'n';                    /* anything but 'C' (no ctc)   */
    shell.optimizer  = st->optimizer;
    shell.update_cnt = st->update_cnt;
    shell.final      = fin;
    shell.ctc        = NULL;
    shell.mean       = NULL;
    shell.sdev       = NULL;
    shell.compiled   = 1;
    shell.layer      = layers;

    int ok = write_model(&shell,fin,fp);
    freemem(layers);
    if (!ok)
        fprintf(stderr,"In write_lm_stack: failed to write the stack\n");
    return ok;
}

/* Read the stack MODEL and harvest its layers into an LM-owned tr[] array
 * plus the head. The harvested LAYER contents (transformer + grads) are moved
 * out of the shell before it is freed, so model_free() does not touch them. */
static int read_lm_stack(FILE* fp, LAYER** tr_out, int* N_out,
                         NEGSAMPLE** head_out)
{
    MODEL* shell = read_model(fp);
    if (shell == NULL) {
        fprintf(stderr,"In read_lm_stack: failed to read the stack\n");
        return 0;
    }
    int nl = shell->num_layers;
    if (nl < 1 || shell->layer[nl - 1].type != 'n') {
        fprintf(stderr,"In read_lm_stack: unexpected stack layout\n");
        /* fall through to free the shell fully */
        model_free(shell);
        return 0;
    }
    int N = nl - 1;
    LAYER* tr = allocmem(N,1,LAYER);
    for (int i = 0; i < N; i++)
        tr[i] = shell->layer[i];              /* move transformer + grads    */
    NEGSAMPLE* head = shell->layer[N].negsample;

    /* Detach so model_free() frees only the shell, not the harvested layers. */
    freemem(shell->layer);
    shell->layer = NULL;
    shell->num_layers = 0;
    shell->ctc = NULL;
    model_free(shell);

    *tr_out = tr;
    *N_out = N;
    *head_out = head;
    return 1;
}

/* ---- whole model ------------------------------------------------------- */

int write_lm(const LM* m, int final, HASHMAP* hmap,
             const int* dist, int dist_size, const LMTRAIN* st, FILE* fp)
{
    int fin = final ? 1 : 0;

    int cnt = fprintf(fp,"LM V %d E %d T %d B %d N %d n_neg %d "
                         "optimizer '%c' weight_decay %.9g learning_rate %.9g "
                         "lr_decay %.9g num_epochs %d epoch %d update_cnt %d "
                         "lrng_seed %d final %d\n",
                      m->V,m->E,m->T,m->B,m->N,m->n_neg,
                      st->optimizer,st->weight_decay,st->learning_rate,
                      st->lr_decay,st->num_epochs,st->epoch,st->update_cnt,
                      st->lrng_seed,fin);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_lm: failed to write the header\n");
        return 0;
    }

    if (!write_hashmap(hmap,fp))
        return 0;

    if (!fin) {                                /* sampling table: training    */
        if (!write_dist(dist,dist_size,fp))
            return 0;
    }

    if (!write_lmemb(m->emb,fin,fp)) {
        fprintf(stderr,"In write_lm: failed to write the embedding\n");
        return 0;
    }

    if (!write_lm_stack(m,fin,st,fp))
        return 0;

    return 1;
}

LM* read_lm(FILE* fp, HASHMAP** hmap_out, LMTRAIN* st)
{
    int V, E, T, B, N, n_neg, num_epochs, epoch, update_cnt, lrng_seed, final;
    char optimizer;
    float weight_decay, learning_rate, lr_decay;
    int cnt = fscanf(fp," LM V %d E %d T %d B %d N %d n_neg %d "
                        "optimizer '%c' weight_decay %g learning_rate %g "
                        "lr_decay %g num_epochs %d epoch %d update_cnt %d "
                        "lrng_seed %d final %d\n",
                     &V,&E,&T,&B,&N,&n_neg,&optimizer,&weight_decay,
                     &learning_rate,&lr_decay,&num_epochs,&epoch,
                     &update_cnt,&lrng_seed,&final);
    if (cnt < 15 || cnt == EOF) {
        fprintf(stderr,"In read_lm: failed to read the header\n");
        return NULL;
    }

    HASHMAP* hmap = read_hashmap(fp);
    if (hmap == NULL)
        return NULL;

    int* dist = NULL;
    int dist_size = 0;
    if (!final) {
        if (!read_dist(fp,&dist,&dist_size)) {
            hashmap_free(hmap);
            return NULL;
        }
    }

    LMEMB* emb = read_lmemb(fp);
    if (emb == NULL) {
        fprintf(stderr,"In read_lm: failed to read the embedding\n");
        freemem(dist);
        hashmap_free(hmap);
        return NULL;
    }

    LAYER* tr = NULL;
    NEGSAMPLE* head = NULL;
    int Nread = 0;
    if (!read_lm_stack(fp,&tr,&Nread,&head)) {
        lmemb_free(emb);
        freemem(dist);
        hashmap_free(hmap);
        return NULL;
    }
    if (Nread != N)
        fprintf(stderr,"In read_lm: warning: header N %d != stack N %d\n",
                N,Nread);
    N = Nread;

    /* Assemble the model and (re)allocate the scratch buffers exactly as
     * lm_create() does, so lm_free() stays balanced. */
    LM* m = allocmem(1,1,LM);
    m->V = V; m->E = E; m->T = T; m->B = B; m->N = N;
    m->BT = B * T; m->n_neg = n_neg;
    m->emb = emb; m->tr = tr; m->head = head;

    m->gHead = allocmem(1,1,fArr2D*);
    m->gHead[0] = allocmem(V,E,float);

    m->acts = allocmem(N + 1,1,fArr2D*);
    for (int i = 0; i <= N; i++)
        m->acts[i] = allocmem(m->BT,E,float);
    m->dtop  = allocmem(m->BT,E,float);
    m->dcur  = allocmem(m->BT,E,float);
    m->dnext = allocmem(m->BT,E,float);

    m->pad_mask = allocmem(m->BT,1,int);
    m->labels   = allocmem(m->BT,1,float);
    m->ids      = allocmem(m->BT,1,int);

    /* Re-attach the sampling table to the head (borrowed, caller frees it). */
    if (dist != NULL)
        negsample_set_dist(head,dist,dist_size);

    *hmap_out = hmap;
    if (st != NULL) {
        st->optimizer     = optimizer;
        st->update_cnt    = update_cnt;
        st->epoch         = epoch;
        st->num_epochs    = num_epochs;
        st->learning_rate = learning_rate;
        st->lr_decay      = lr_decay;
        st->weight_decay  = weight_decay;
        st->lrng_seed     = lrng_seed;
    }
    return m;
}

LM* load_lm(const char* filename, HASHMAP** hmap, LMTRAIN* st)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_lm: failed to open file '%s' for read\n",
                filename);
        return NULL;
    }
    LM* m = read_lm(fp,hmap,st);
    fclose(fp);
    return m;
}

int store_lm(const char* filename, const LM* m, int final, HASHMAP* hmap,
             const int* dist, int dist_size, const LMTRAIN* st)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_lm: failed to open file '%s' for write\n",
                filename);
        return 0;
    }
    int ok = write_lm(m,final,hmap,dist,dist_size,st,fp);
    fclose(fp);
    return ok;
}
