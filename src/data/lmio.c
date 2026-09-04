/* Copyright (c) 2026 Gilad Odinak */

/* Functions to load and store a full decoder-only language model.
 *
 * File layout:
 *   LM <header: dims + resumable training schedule + final flag>
 *   VOCAB <vocabulary + optional frequency and sampling tables>
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
#include "vocab.h"
#include "vocabio.h"
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

/* Wrap the N transformer LAYERs and the output layer in a MODEL 
 * and pass it to write_model().
 */
static int write_lm_stack(const LM* m, int fin, const LMPARAM* st, FILE* fp)
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

int write_lm(const LM* m, int final, const VOCAB* vocab,
             const LMPARAM* st, FILE* fp)
{
    if (m == NULL || vocab == NULL || vocab->size != m->V ||
        st == NULL || fp == NULL) {
        fprintf(stderr,"In write_lm: invalid parameter\n");
        return 0;
    }
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

    /* A final model does not need the negative-sampling table. */
    VOCAB stored_vocab = *vocab;
    if (final) {
        stored_vocab.freq = NULL;
        stored_vocab.dist = NULL;
        stored_vocab.dist_size = 0;
    }
    if (!write_vocab(&stored_vocab,fp))
        return 0;

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

LM* read_lm(FILE* fp, VOCAB** pvocab, LMPARAM* st)
{
    if (fp == NULL || pvocab == NULL) {
        fprintf(stderr,"In read_lm: invalid parameter\n");
        return NULL;
    }
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

    VOCAB* vocab = read_vocab(fp);
    if (vocab == NULL) {
        fprintf(stderr,"In read_lm: failed to read the vocabulary\n");
        return NULL;
    }
    if (vocab->size != V) {
        fprintf(stderr,"In read_lm: vocabulary size %d != V %d\n",
                vocab->size,V);
        vocab_free(vocab);
        return NULL;
    }

    LMEMB* emb = read_lmemb(fp);
    if (emb == NULL) {
        fprintf(stderr,"In read_lm: failed to read the embedding layer\n");
        vocab_free(vocab);
        return NULL;
    }

    LAYER* tr = NULL;
    int Nread = 0;
    if (!read_lm_stack(fp,&tr,&Nread)) {
        fprintf(stderr,"In read_lm: failed to read the transformer stack\n");
        lmemb_free(emb);
        vocab_free(vocab);
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
        vocab_free(vocab);
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
    if (vocab->dist != NULL)
        smsftmax_set_dist(head,vocab->dist,vocab->dist_size);

    *pvocab = vocab;
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

LM* load_lm(const char* filename, VOCAB** vocab, LMPARAM* st)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_lm: failed to open file '%s' for read\n",
                filename);
        return NULL;
    }
    LM* m = read_lm(fp,vocab,st);
    fclose(fp);
    return m;
}

int store_lm(const char* filename, const LM* m, int final,
             const VOCAB* vocab, const LMPARAM* st)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_lm: failed to open file '%s' for write\n",
                filename);
        return 0;
    }
    int ok = write_lm(m,final,vocab,st,fp);
    fclose(fp);
    return ok;
}
