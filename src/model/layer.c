/* Copyright (c) 2023-2024 Gilad Odinak   */
/* Model layer abstraction implementation */
#include <stdio.h>
#include <stdlib.h>
#include "mem.h"
#include "array.h"
#include "clip.h"
#include "adamw.h"
#include "dense.h"
#include "lstm.h"
#include "transformer.h"
#include "negsample.h"
#include "layer.h"

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using linear update.
 * The rate of update is controlled by learning_rate, weight_decay.
 */
static inline void linear_update(fArr2D w_/*[M][N]*/, fArr2D g_/*[M][N]*/,
                                 int M, int N,
                                 float learning_rate, float weight_decay)
{
    typedef float (*ArrMN)[N];
    ArrMN w = (ArrMN) w_;
    ArrMN g = (ArrMN) g_;

    clip_gradients(g,M,N,1.0e-12,10.0);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            w[i][j] -= learning_rate * (g[i][j] + weight_decay * w[i][j]);
}

int layer_init(LAYER* l, int input_dim, int batch_size)
{
    switch (l->type) {
        case 'd':
            dense_init(l->dense,input_dim,batch_size,1);
            return l->dense->S;
        case 'l':
            lstm_init(l->lstm,input_dim,batch_size,1);
            return l->lstm->S;
        case 't': {
            TRANSFORMER* tr = l->transformer;
            if (input_dim != tr->D) {
                fflush(stdout);
                fprintf(stderr,"layer_init: transformer input_dim %d "
                        "!= model_dim %d\n",input_dim,tr->D);
                exit(-1);
            }
            if (tr->T <= 0 || batch_size % tr->T != 0) {
                fflush(stdout);
                fprintf(stderr,"layer_init: batch_size %d not a multiple "
                        "of transformer T %d\n",batch_size,tr->T);
                exit(-1);
            }
            transformer_init(tr,batch_size / tr->T,1,0.0);
            l->out = allocmem(tr->BT,tr->D,float);
            return tr->D;
        }
        case 'n':
            negsample_init(l->negsample,input_dim,batch_size,1);
            return l->negsample->E;
    }
    layer_unsupported("layer_init",l->type);
    return 0; /* not reached */
}

void layer_reset(LAYER* l)
{
    switch (l->type) {
        case 'd': dense_reset(l->dense); break;
        case 'l': lstm_reset(l->lstm); break;
        case 't': /* transformer carries no cross-batch state */ break;
        case 'n': negsample_reset(l->negsample); break;
    }
}

void layer_free(LAYER* l)
{
    switch (l->type) {
        case 'd': dense_free(l->dense); break;
        case 'l': lstm_free(l->lstm); break;
        case 't': transformer_free(l->transformer); freemem(l->out); break;
        case 'n': negsample_free(l->negsample); break;
    }
}

void layer_set_batch_size(LAYER* l, int batch_size)
{
    switch (l->type) {
        case 'd': dense_set_batch_size(l->dense,batch_size); break;
        case 'l': lstm_set_batch_size(l->lstm,batch_size); break;
        case 't':
            fflush(stdout);
            fprintf(stderr,
                "layer_set_batch_size: not supprted by transformer layer\n");
            exit(-1);
        break;
        case 'n': negsample_set_batch_size(l->negsample,batch_size); break;
    }
}

void layer_alloc_opt_state(LAYER* l, char optimizer)
{
    switch (l->type) {
        case 'd': {
            DENSE* ld = l->dense;
            if (optimizer == 'l') {
                l->opt_state = NULL;
                l->num_opt_state = 0;
            } else { /* 'a' adamw */
                int ng = ld->use_bias ? 4 : 2;
                fArr2D* g = allocmem(1,ng,fArr2D*);
                g[0] = allocmem(ld->D,ld->S,float); /* mWx */
                g[1] = allocmem(ld->D,ld->S,float); /* vWx */
                if (ld->use_bias) {
                    g[2] = allocmem(1,ld->S,float); /* mb */
                    g[3] = allocmem(1,ld->S,float); /* vb */
                }
                l->opt_state = g;
                l->num_opt_state = ng;
            }
        }
        break;
        case 'l': {
            LSTM* ll = l->lstm;
            if (optimizer == 'l') {
                l->opt_state = NULL;
                l->num_opt_state = 0;
            } else { /* 'a' adamw */
                int ng = ll->use_bias ? 24 : 16;
                fArr2D* g = allocmem(1,ng,fArr2D*);
                for (int j = 0; j < 8; j++) {
                    int M = (j < 4) ? ll->D : ll->S;
                    g[j]     = allocmem(M,ll->S,float); /* mX  */
                    g[j + 8] = allocmem(M,ll->S,float); /* vXx */
                }
                if (ll->use_bias) {
                    for (int j = 0; j < 4; j++) {
                        g[j + 16] = allocmem(1,ll->S,float); /* mb */
                        g[j + 20] = allocmem(1,ll->S,float); /* vb */
                    }
                }
                l->opt_state = g;
                l->num_opt_state = ng;
            }
        }
        break;
        case 't': {
            TRANSFORMER* tr = l->transformer;
            int D = tr->D;
            int Dff = tr->Dff;
            if (optimizer == 'l') {
                l->opt_state = NULL;
                l->num_opt_state = 0;
            } else { /* 'a' adamw */
                int rows[10] = { D, D, D, D, D,   Dff, D, D, D, D };
                int cols[10] = { D, D, D, D, Dff, D,   1, 1, 1, 1 };
                int ng = 20;
                fArr2D* g = allocmem(1,ng,fArr2D*);
                for (int j = 0; j < 10; j++) {
                    g[j]      = allocmem(rows[j],cols[j],float); /* m */
                    g[j + 10] = allocmem(rows[j],cols[j],float); /* v */
                }
                l->opt_state = g;
                l->num_opt_state = ng;
            }
        }
        break;
        case 'n': {
            /* Uses sparse SGD regardless of optimizer */
            l->opt_state = NULL;
            l->num_opt_state = 0;
        }
        break;
    }
}

void layer_update(LAYER* l, char optimizer,
                  float learning_rate, float weight_decay, int update_cnt)
{
    float lr = learning_rate;
    float wd = weight_decay;
    int uc = update_cnt;
    fArr2D* g = l->opt_state;
    switch (l->type) {
        case 'd': { /* dense */
            DENSE* ld = l->dense;
            int D = ld->D;
            int S = ld->S;
            switch (optimizer) {
                case 'l': /* linear */
                    linear_update(ld->Wx,ld->gWx,D,S,lr,wd);
                    if (ld->use_bias)
                        linear_update((fArr2D) ld->b,(fArr2D) ld->gb,1,S,lr,wd);
                break;
                case 'a': /* adamw */
                    adamw_update(ld->Wx,ld->gWx,g[0],g[1],D,S,lr,wd,uc);
                    if (ld->use_bias)
                        adamw_update((fArr2D) ld->b,(fArr2D) ld->gb,
                                     g[2],g[3],1,S,lr,wd,uc);
                break;
            }
        }
        break;
        case 'l': { /* lstm */
            LSTM* ll = l->lstm;
            int D = ll->D;
            int S = ll->S;
            switch (optimizer) {
                case 'l': /* linear */
                    linear_update(ll->Wf,ll->gWf,D,S,lr,wd);
                    linear_update(ll->Wi,ll->gWi,D,S,lr,wd);
                    linear_update(ll->Wc,ll->gWc,D,S,lr,wd);
                    linear_update(ll->Wo,ll->gWo,D,S,lr,wd);
                    linear_update(ll->Uf,ll->gUf,S,S,lr,wd);
                    linear_update(ll->Ui,ll->gUi,S,S,lr,wd);
                    linear_update(ll->Uc,ll->gUc,S,S,lr,wd);
                    linear_update(ll->Uo,ll->gUo,S,S,lr,wd);
                    if (ll->use_bias) {
                        linear_update((fArr2D) ll->bf,(fArr2D) ll->gbf,1,S,lr,wd);
                        linear_update((fArr2D) ll->bi,(fArr2D) ll->gbi,1,S,lr,wd);
                        linear_update((fArr2D) ll->bc,(fArr2D) ll->gbc,1,S,lr,wd);
                        linear_update((fArr2D) ll->bo,(fArr2D) ll->gbo,1,S,lr,wd);
                    }
                break;
                case 'a': /* adamw */
                    adamw_update(ll->Wf,ll->gWf,g[0],g[8],D,S,lr,wd,uc);
                    adamw_update(ll->Wi,ll->gWi,g[1],g[9],D,S,lr,wd,uc);
                    adamw_update(ll->Wc,ll->gWc,g[2],g[10],D,S,lr,wd,uc);
                    adamw_update(ll->Wo,ll->gWo,g[3],g[11],D,S,lr,wd,uc);
                    adamw_update(ll->Uf,ll->gUf,g[4],g[12],S,S,lr,wd,uc);
                    adamw_update(ll->Ui,ll->gUi,g[5],g[13],S,S,lr,wd,uc);
                    adamw_update(ll->Uc,ll->gUc,g[6],g[14],S,S,lr,wd,uc);
                    adamw_update(ll->Uo,ll->gUo,g[7],g[15],S,S,lr,wd,uc);
                    if (ll->use_bias) {
                        adamw_update((fArr2D) ll->bf,(fArr2D) ll->gbf,g[16],g[20],1,S,lr,wd,uc);
                        adamw_update((fArr2D) ll->bi,(fArr2D) ll->gbi,g[17],g[21],1,S,lr,wd,uc);
                        adamw_update((fArr2D) ll->bc,(fArr2D) ll->gbc,g[18],g[22],1,S,lr,wd,uc);
                        adamw_update((fArr2D) ll->bo,(fArr2D) ll->gbo,g[19],g[23],1,S,lr,wd,uc);
                    }
                break;
            }
        }
        break;
        case 't': { /* transformer */
            TRANSFORMER* tr = l->transformer;
            MHA* mha = tr->mha;
            int D = tr->D;
            int Dff = tr->Dff;
            /* Gradients are read straight from the transformer's internal
             * buffers. For AdamW the moments live in g[0..9] (m) and
             * g[10..19] (v), in the same parameter order as the updates
             * below. Note: weight decay is applied uniformly, including to
             * the norm gamma/beta; set weight_decay to 0 to disable. 
             */
            switch (optimizer) {
                case 'l': /* linear */
                    linear_update(mha->Wq,mha->gWq,D,D,lr,wd);
                    linear_update(mha->Wk,mha->gWk,D,D,lr,wd);
                    linear_update(mha->Wv,mha->gWv,D,D,lr,wd);
                    linear_update(mha->Wo,mha->gWo,D,D,lr,wd);
                    linear_update(tr->ffn1->Wx,tr->ffn1->gWx,D,Dff,lr,wd);
                    linear_update(tr->ffn2->Wx,tr->ffn2->gWx,Dff,D,lr,wd);
                    linear_update((fArr2D) tr->norm1->gamma,(fArr2D) tr->dg1,D,1,lr,wd);
                    linear_update((fArr2D) tr->norm1->beta,(fArr2D) tr->db1,D,1,lr,wd);
                    linear_update((fArr2D) tr->norm2->gamma,(fArr2D) tr->dg2,D,1,lr,wd);
                    linear_update((fArr2D) tr->norm2->beta,(fArr2D) tr->db2,D,1,lr,wd);
                break;
                case 'a': /* adamw */
                    adamw_update(mha->Wq,mha->gWq,g[0],g[10],D,D,lr,wd,uc);
                    adamw_update(mha->Wk,mha->gWk,g[1],g[11],D,D,lr,wd,uc);
                    adamw_update(mha->Wv,mha->gWv,g[2],g[12],D,D,lr,wd,uc);
                    adamw_update(mha->Wo,mha->gWo,g[3],g[13],D,D,lr,wd,uc);
                    adamw_update(tr->ffn1->Wx,tr->ffn1->gWx,g[4],g[14],D,Dff,lr,wd,uc);
                    adamw_update(tr->ffn2->Wx,tr->ffn2->gWx,g[5],g[15],Dff,D,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm1->gamma,(fArr2D) tr->dg1,g[6],g[16],D,1,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm1->beta,(fArr2D) tr->db1,g[7],g[17],D,1,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm2->gamma,(fArr2D) tr->dg2,g[8],g[18],D,1,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm2->beta,(fArr2D) tr->db2,g[9],g[19],D,1,lr,wd,uc);
                break;
            }
        }
        break;
        case 'n': /* Sparse SGD over touched rows, any optimizer */
            (void) uc;
            negsample_update(l->negsample,lr,wd);
        break;
    }
}
