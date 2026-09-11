/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store multi-layer neural network model */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "arrayio.h"
#include "dense.h"
#include "denseio.h"
#include "lstm.h"
#include "lstmio.h"
#include "transformer.h"
#include "xfmrio.h"
#include "negsample.h"
#include "negsampio.h"
#include "model.h"
#include "modelio.h"
#include "ctc.h"

/* read_model - Read a model from a file
 * 
 * Reads a model from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read model if successful, NULL otherwise
 */
MODEL* read_model(FILE* fp)
{
    MODEL* m = allocmem(1,1,MODEL);
    int ok;
    int cnt = fscanf(fp," MODEL num_layers %d batch_size %d input_dim %d "
                     "output_dim %d target_dim %d normalize %d "
                     "loss_func '%c' optimizer '%c' update_cnt %d final %d\n",
                     &m->num_layers,&m->batch_size,&m->input_dim,
                     &m->output_dim,&m->target_dim,&m->normalize,
                     &m->loss_func,&m->optimizer,&m->update_cnt,&m->final);
    if (cnt < 10 || cnt == EOF) {
        fprintf(stderr,"In read_model: failed to read the header\n");
        goto err;
    }
    m->layer = allocmem(1,m->num_layers,LAYER);
    if (m->normalize) {
        int D = m->input_dim;
        m->mean = allocmem(1,D,float);
        m->sdev = allocmem(1,D,float);
        ok = read_array((fArr2D)m->mean,1,D,fp,0);
        if (ok)
            ok = read_array((fArr2D)m->sdev,1,D,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_model: failed to read mean, sdev data\n");
            goto err;
        }
    }
    if (m->loss_func == 'C') { /* ctc */
        int T, L, blank;
        cnt = fscanf(fp," CTC T %d L %d blank %d\n",&T,&L,&blank);
        if (cnt < 3 || cnt == EOF) {
            fprintf(stderr,"In read_model: failed to read the ctc header\n");
            goto err;
        }
        m->ctc = ctc_create(T,L,blank);
    }
    for (int i = 0; i < m->num_layers; i++) {
        LAYER* l = &m->layer[i];
        cnt = fscanf(fp," LAYER type '%c' num_opt_state %d\n",
                                             &l->type,&l->num_opt_state);
        if (cnt < 2 || cnt == EOF) {
            fprintf(stderr,
                    "In read_model: failed to read layer %d header\n",i);
            goto err;
        }
        switch (l->type) {
            case 'd': 
                l->dense = read_dense(fp); 
                ok = (l->dense != NULL);
            break;
            case 'l': 
                l->lstm = read_lstm(fp); 
                ok = (l->lstm != NULL);
            break;
            case 't':
                l->transformer = read_transformer(fp);
                ok = (l->transformer != NULL);
            break;
            case 'n':
                l->negsample = read_negsample(fp);
                ok = (l->negsample != NULL);
            break;
        }
        if (!ok) {
            fprintf(stderr,
                    "In read_model: failed to read layer %d data\n",i);
            goto err;
        }
        if (l->num_opt_state > 0) {
            /* opt_state is an array of pointers to arrays 
             * see layer_alloc_opt_state() for layout
             */
            l->opt_state  = allocmem(1,l->num_opt_state,fArr2D*);
            ok = 1; /* assume success */
            switch (l->type) {
                case 'd': /* dense layer optimizer moments */
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        int rows = (j < 2) ? l->dense->D : 1;
                        l->opt_state[j] = allocmem(rows,l->dense->S,float);
                        ok = read_array(l->opt_state[j],rows,l->dense->S,fp,0);
                    }
                break;
                case 'l': /* lstm layer optimizer moments */
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        int rows;
                        if (j < 16) {
                            int k = j % 8;
                            rows = (k < 4) ? l->lstm->D : l->lstm->S;
                        }
                        else
                            rows = 1;
                        l->opt_state[j] = allocmem(rows,l->lstm->S,float);
                        ok = read_array(l->opt_state[j],rows,l->lstm->S,fp,0);
                    }
                break;
                case 't': /* transformer layer gradients (adamw m/v moments) */
                {
                    int D = l->transformer->D;
                    int Dff = l->transformer->Dff;
                    int gr[10] = { D, D, D, D, D,   Dff, D, D, D, D };
                    int gc[10] = { D, D, D, D, Dff, D,   1, 1, 1, 1 };
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        int k = j % 10;
                        l->opt_state[j] = allocmem(gr[k],gc[k],float);
                        ok = read_array(l->opt_state[j],gr[k],gc[k],fp,0);
                    }
                }
                break;
                case 'n': /* negsample layer gradient */
                {
                    int K = l->negsample->K;
                    int E = l->negsample->E;
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        l->opt_state[j] = allocmem(K,E,float);
                        ok = read_array(l->opt_state[j],K,E,fp,0);
                    }
                }
                break;
            }
            if (!ok) {
                fprintf(stderr,"In read_model: "
                        "failed to read layer %d optimizer state data\n",i);
                goto err;
            }
        }
    }
    m->compiled = 1;
    return m;
err: /* error return */
    fflush(stderr);
    model_free(m);
    return NULL;
}

/* write_model - Write a model to a file
 * 
 * Writes the model pointed to by m to the file pointed to by fp. 
 * 
 * Parameters:
 *   m     - Pointer to the model to be written
 *   final - If not zero, store the model as final: gradient/optimizer
 *           state is omitted so the model can be used for inference
 *           but not further trained.
 *   fp    - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_model(const MODEL* m, int final, FILE* fp)
{
    int ok;
    int fin = (final || m->final) ? 1 : 0;
    int cnt = fprintf(fp,"MODEL num_layers %d batch_size %d input_dim %d "
                 "output_dim %d target_dim %d normalize %d " 
                 "loss_func '%c' optimizer '%c' update_cnt %d final %d\n",
                 m->num_layers,m->batch_size,m->input_dim,
                 m->output_dim,m->target_dim,m->normalize,
                 m->loss_func,m->optimizer,m->update_cnt,fin);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_model: failed to write the header\n");
        return 0;
    }
    if (m->normalize) {
        int D = m->input_dim;
        int ok = write_array((fArr2D)m->mean,1,D,fp,NULL,0);
        if (ok)
            ok = write_array((fArr2D)m->sdev,1,D,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,"In write_model: failed to write mean, sdev data\n");
            return 0;
        }
    }
    if (m->ctc != NULL) {
        cnt = fprintf(fp,"CTC T %d L %d blank %d\n",
                                            m->ctc->T,m->ctc->L,m->ctc->blank);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,"In write_model: failed to write the header\n");
            return 0;
        }
    }
    for (int i = 0; i < m->num_layers; i++) {
        LAYER* l = &m->layer[i];
        int num_opt_state = fin ? 0 : l->num_opt_state;
        cnt = fprintf(fp,"LAYER type '%c' num_opt_state %d\n",
                                                    l->type,num_opt_state);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,
                    "In write_model: failed to write layer %d header\n",i);
            return 0;
        }
        switch (l->type) {
            case 'd': ok = write_dense(l->dense,fin,fp); break;
            case 'l': ok = write_lstm(l->lstm,fin,fp); break;
            case 't': ok = write_transformer(l->transformer,fin,fp); break;
            case 'n': ok = write_negsample(l->negsample,fin,fp); break;
        }
        if (!ok) {
            fprintf(stderr,
                    "In write_model: failed to write layer %d data\n",i);
            return 0;
        }
        if (num_opt_state > 0 && l->opt_state != NULL) {
            /* opt_state is an array of pointers to arrays 
             * see layer_alloc_opt_state() for layout
             */
            ok = 1; /* assume success */
            switch (l->type) {
                case 'd': /* dense layer optimizer moments */
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        int rows = (j < 2) ? l->dense->D : 1;
                        ok = write_array(
                                   l->opt_state[j],rows,l->dense->S,fp,NULL,0);
                    }
                break;
                case 'l': /* lstm layer optimizer moments */
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        int rows;
                        if (j < 16) {
                            int k = j % 8;
                            rows = (k < 4) ? l->lstm->D : l->lstm->S;
                        }
                        else
                            rows = 1;
                        ok = write_array(
                                   l->opt_state[j],rows,l->lstm->S,fp,NULL,0);
                    }
                break;
                case 't': /* transformer layer gradients (adamw m/v moments) */
                {
                    int D = l->transformer->D;
                    int Dff = l->transformer->Dff;
                    int gr[10] = { D, D, D, D, D,   Dff, D, D, D, D };
                    int gc[10] = { D, D, D, D, Dff, D,   1, 1, 1, 1 };
                    for (int j = 0; j < l->num_opt_state && ok; j++) {
                        int k = j % 10;
                        ok = write_array(l->opt_state[j],gr[k],gc[k],fp,NULL,0);
                    }
                }
                break;
                case 'n': /* negsample layer gradient */
                {
                    int K = l->negsample->K;
                    int E = l->negsample->E;
                    for (int j = 0; j < l->num_opt_state && ok; j++)
                        ok = write_array(l->opt_state[j],K,E,fp,NULL,0);
                }
                break;
            }
            if (!ok) {
                fprintf(stderr,"In write_model: "
                        "failed to write layer %d optimizer state data\n",i);
                return 0;
            }
        }
    }
    return 1;
}

/* load_model - Load a model from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads a model from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the model from
 * 
 * Returns:
 *   Pointer to the loaded model if successful, NULL otherwise
 */
MODEL* load_model(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_model: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    MODEL* m = read_model(fp);
    fclose(fp);
    return m;
}

/* store_model - Store a model into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the model pointed to by m into it.
 * 
 * Parameters:
 *   m        - Pointer to the model to be stored
 *   filename - Name of the file to store the model in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_model(const MODEL* m, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_model: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_model(m,m->final,fp);
    fclose(fp);
    return ok;
}

