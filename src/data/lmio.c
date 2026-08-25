/* Copyright (c) 2026 Gilad Odinak */

/* Functions to load and store a full decoder-only language model.
 *
 * File layout:
 *   LM <header: dims + resumable training schedule + final flag>
 *   HASHMAP <vocabulary: index/word lines>
 *   DIST <sampling table, run-length encoded> (omitted if final)
 *   LMEMB <embedding weights>                (via lmembio)
 *   MODEL <N transformer layers + output>    (via modelio)
 *
 * The transformer stack and the negative-sampling output are serialized
 * through modelio's write_model()/read_model() by wrapping them in a MODEL
 * (N transformer LAYERs of type 't' followed by one 'n' output LAYER).
 */
#include <stdio.h>
#include <string.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "hash.h"
#include "lmemb.h"
#include "lmembio.h"
#include "smsftmax.h"
#include "smsftmaxio.h"
#include "transformer.h"
#include "layer.h"
#include "model.h"
#include "modelio.h"
#include "lm.h"
#include "lmio.h"

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

static int write_dist(const int* dist, int dist_size, FILE* fp)
{
    int cnt = fprintf(fp,"DIST dist_size %d\n",dist_size);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_dist: failed to write the header\n");
        return 0;
    }
    /* Run-length encode: the table is built as contiguous runs 
     * of each vocab index, so store (value, run_length) pairs,
     * one per run, instead of every entry.
     */
    for (int a = 0, c = 0; a < dist_size; a += c) {
        int v = dist[a];
        for (c = 1; a + c < dist_size && dist[a + c] == v; c++);
        cnt = fprintf(fp,"%d %d\n",v,c);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,"In write_dist: failed to write run at %d\n",a);
            return 0;
        }
    }
    return 1;
}

static int read_dist(FILE* fp, int** dist_out, int* size_out)
{
    int dist_size;
    int cnt = fscanf(fp," DIST dist_size %d",&dist_size);
    if (cnt < 1 || cnt == EOF || dist_size <= 0) {
        fprintf(stderr,"In read_dist: failed to read the header\n");
        return 0;
    }
    int* dist = allocmem(dist_size,1,int);
    /* Expand (value, run_length) pairs back into the flat table. The runs
     * must sum exactly to dist_size, which doubles as the integrity check.
     */
    int j = 0;
    while (j < dist_size) {
        int v, c;
        cnt = fscanf(fp," %d %d",&v,&c);
        if (cnt < 2 || cnt == EOF || c <= 0 || j + c > dist_size) {
            fprintf(stderr,"In read_dist: bad run at %d (v=%d run=%d)\n",j,v,c);
            freemem(dist);
            return 0;
        }
        for (int k = 0; k < c; k++)
            dist[j++] = v;
    }
    if (j != dist_size) {
        fprintf(stderr,"In read_dist: total %d != dist_size %d\n",j,dist_size);
        freemem(dist);
        return 0;
    }
    *dist_out = dist;
    *size_out = dist_size;
    return 1;
}

/* Wrap the N transformer LAYERs and the output layer in a MODEL 
 * and pass it to write_model().
 */
static int write_lm_stack(const LM* m, int fin, const LMTRAIN* st, FILE* fp)
{
    LAYER* layers = allocmem(1,m->N,LAYER);
    for (int i = 0; i < m->N; i++)
        layers[i] = m->tr[i];

    MODEL model;
    memset(&model,0,sizeof(model));
    model.num_layers = m->N;
    model.batch_size = m->B;
    model.input_dim  = m->E;
    model.output_dim = m->E;
    model.target_dim = 1;
    model.add_bias   = 0;
    model.normalize  = 0;
    model.loss_func  = 'n';
    model.optimizer  = st->optimizer;
    model.update_cnt = st->update_cnt;
    model.final      = fin;
    model.ctc        = NULL;
    model.mean       = NULL;
    model.sdev       = NULL;
    model.compiled   = 1;
    model.layer      = layers;

    int ok = write_model(&model,fin,fp);
    freemem(layers);
    if (!ok)
        fprintf(stderr,"In write_lm_stack: failed to write the stack\n");
    return ok;
}

/* Read the stack MODEL and harvest its layers into an LM-owned tr[] array
 * and the output. The LAYER contents (transformer + grads) are moved out
 * of the model before it is freed, so model_free() does not touch them.
 */
static int read_lm_stack(FILE* fp, LAYER** ptr, int* pN)
{
    MODEL* model = read_model(fp);
    if (model == NULL) {
        fprintf(stderr,"In read_lm_stack: failed to read the stack\n");
        return 0;
    }
    int N = model->num_layers;
    if (N < 1) {
        fprintf(stderr,"In read_lm_stack: unexpected stack layout\n");
        model_free(model);
        return 0;
    }
    LAYER* tr = allocmem(N,1,LAYER);
    for (int i = 0; i < N; i++)
        tr[i] = model->layer[i];

    /* Detach so model_free() frees only the shell, not the harvested layers. */
    freemem(model->layer);
    model->layer = NULL;
    model->num_layers = 0;
    model->ctc = NULL;
    model_free(model);

    *ptr = tr;
    *pN = N;
    return 1;
}

int write_lm(const LM* m, int final, HASHMAP* hmap,
             const int* dist, int dist_size, const LMTRAIN* st, FILE* fp)
{
    final = final ? 1 : 0;

    int cnt = fprintf(fp,"LM V %d E %d T %d B %d N %d n_neg %d "
                         "optimizer '%c' weight_decay %.6g learning_rate %.6g "
                         "lr_decay %.6g num_epochs %d epoch %d update_cnt %d "
                         "lrng_seed %d sample_frac %.6g final %d\n",
                      m->V,m->E,m->T,m->B,m->N,m->n_neg,
                      st->optimizer,st->weight_decay,st->learning_rate,
                      st->lr_decay,st->num_epochs,st->epoch,st->update_cnt,
                      st->lrng_seed,st->sample_frac,final);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_lm: failed to write the header\n");
        return 0;
    }

    if (!write_hashmap(hmap,fp))
        return 0;

    if (!final) {
        if (!write_dist(dist,dist_size,fp))
            return 0;
    }

    if (!write_lmemb(m->emb,final,fp))
        return 0;

    if (!write_lm_stack(m,final,st,fp))
        return 0;

    if (!write_smsftmax(m->head,fp)) {
        fprintf(stderr,"In write_lm: failed to write the head\n");
        return 0;
    }

    return 1;
}

LM* read_lm(FILE* fp, HASHMAP** phmap, LMTRAIN* st)
{
    int V, E, T, B, N, n_neg, num_epochs, epoch, update_cnt, lrng_seed, final;
    char optimizer;
    float weight_decay, learning_rate, lr_decay, sample_frac;
    int cnt = fscanf(fp," LM V %d E %d T %d B %d N %d n_neg %d "
                        "optimizer '%c' weight_decay %g learning_rate %g "
                        "lr_decay %g num_epochs %d epoch %d update_cnt %d "
                        "lrng_seed %d sample_frac %g final %d\n",
                     &V,&E,&T,&B,&N,&n_neg,&optimizer,&weight_decay,
                     &learning_rate,&lr_decay,&num_epochs,&epoch,
                     &update_cnt,&lrng_seed,&sample_frac,&final);
    if (cnt < 16 || cnt == EOF) {
        fprintf(stderr,"In read_lm: failed to read the header\n");
        return NULL;
    }

    HASHMAP* hmap = read_hashmap(fp);
    if (hmap == NULL) {
        fprintf(stderr,"In read_lm: failed to read the hashmap\n");
        return NULL;
    }

    int* dist = NULL;
    int dist_size = 0;
    if (!final) {
        if (!read_dist(fp,&dist,&dist_size)) {
            fprintf(stderr,"In read_lm: failed to read the distribution table\n");
            hashmap_free(hmap);
            return NULL;
        }
    }

    LMEMB* emb = read_lmemb(fp);
    if (emb == NULL) {
        fprintf(stderr,"In read_lm: failed to read the embedding layer\n");
        freemem(dist);
        hashmap_free(hmap);
        return NULL;
    }

    LAYER* tr = NULL;
    int Nread = 0;
    if (!read_lm_stack(fp,&tr,&Nread)) {
        fprintf(stderr,"In read_lm: failed to read the transformer stack\n");
        lmemb_free(emb);
        freemem(dist);
        hashmap_free(hmap);
        return NULL;
    }

    SMSFTMAX* head = read_smsftmax(fp);
    if (head == NULL) {
        fprintf(stderr,"In read_lm: failed to read the head\n");
        /* free harvested transformer layers */
        for (int i = 0; i < Nread; i++)
            layer_free(&tr[i]);
        freemem(tr);
        lmemb_free(emb);
        freemem(dist);
        hashmap_free(hmap);
        return NULL;
    }
    if (Nread != N)
        fprintf(stderr,"In read_lm: warning: header N %d != stack N %d\n",
                N,Nread);

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

    /* Re-attach the sampling table to the output */
    if (dist != NULL)
        smsftmax_set_dist(head,dist,dist_size);

    *phmap = hmap;
    if (st != NULL) {
        st->optimizer     = optimizer;
        st->update_cnt    = update_cnt;
        st->epoch         = epoch;
        st->num_epochs    = num_epochs;
        st->sample_frac   = sample_frac;
        st->final         = final;
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

int store_lm(const char* filename, const LM* m, int final, 
             HASHMAP* hmap, const int* dist, int dist_size, const LMTRAIN* st)
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
