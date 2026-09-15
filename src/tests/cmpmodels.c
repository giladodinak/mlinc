/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to compare models and their components */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "array.h"
#include "dense.h"
#include "lstm.h"
#include "model.h"

int compare_arrays(fArr2D a1_, fArr2D a2_, int rows, int cols)
{
    typedef float (*ArrMN)[cols];
    ArrMN a1 = (ArrMN) a1_;
    ArrMN a2 = (ArrMN) a2_;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (fabsf(a1[i][j] - a2[i][j]) > 1e-6)
                return 0;
    return 1;
}

void compare_dense(DENSE* l1, DENSE* l2, int lyr)
{
    if (l1->D != l2->D)
        printf("layer %d l1->D %d l2->D %d\n",lyr,l1->D,l2->D);
    if (l1->S != l2->S)
        printf("layer %d l1->S %d l2->S %d\n",lyr,l1->S,l2->S);
    if (l1->B != l2->B)
        printf("layer %d l1->B %d l2->B %d\n",lyr,l1->B,l2->B);
    if (l1->activation != l2->activation)
        printf("layer %d l1->activation '%c' l2->activation '%c'\n",
               lyr,l1->activation,l2->activation);
    if (l1->use_bias != l2->use_bias)
        printf("layer %d l1->use_bias %d l2->use_bias %d\n",
               lyr,l1->use_bias,l2->use_bias);
    if (l1->D != l2->D || l1->S != l2->S || l1->B != l2->B ||
        l1->activation != l2->activation || l1->use_bias != l2->use_bias)
        exit(-1);

    if (!compare_arrays(l1->h,l2->h,l1->B,l1->S))
        printf("layer %d h arrays differ\n",lyr);
    if (!compare_arrays(l1->Wx,l2->Wx,l1->D,l1->S))
        printf("layer %d Wx arrays differ\n",lyr);
    if (l1->use_bias)
        if (!compare_arrays((fArr2D) l1->b,(fArr2D) l2->b,1,l1->S))
            printf("layer %d b arrays differ\n",lyr);

    /* Gradient buffers exist only in training mode */
    if (l1->training && l2->training) {
        if (!compare_arrays(l1->gWx,l2->gWx,l1->D,l1->S))
            printf("layer %d gWx arrays differ\n",lyr);
        if (l1->use_bias)
            if (!compare_arrays((fArr2D) l1->gb,(fArr2D) l2->gb,1,l1->S))
                printf("layer %d gb arrays differ\n",lyr);
    }
}

void compare_lstm(LSTM* l1, LSTM* l2, int lyr)
{
    if (l1->D != l2->D)
        printf("layer %d l1->D %d l2->D %d\n",lyr,l1->D,l2->D);
    if (l1->S != l2->S)
        printf("layer %d l1->S %d l2->S %d\n",lyr,l1->S,l2->S);
    if (l1->B != l2->B)
        printf("layer %d l1->B %d l2->B %d\n",lyr,l1->B,l2->B);
    if (l1->stateful != l2->stateful)
        printf("layer %d l1->stateful %d l2->stateful %d\n",
               lyr,l1->stateful,l2->stateful);
    if (l1->use_bias != l2->use_bias)
        printf("layer %d l1->use_bias %d l2->use_bias %d\n",
               lyr,l1->use_bias,l2->use_bias);
    if (l1->D != l2->D || l1->S != l2->S || l1->B != l2->B ||
        l1->stateful != l2->stateful || l1->use_bias != l2->use_bias)
        exit(-1);

    /* Kernel weights [D][S] */
    if (!compare_arrays(l1->Wf,l2->Wf,l1->D,l1->S))
        printf("layer %d Wf arrays differ\n",lyr);
    if (!compare_arrays(l1->Wi,l2->Wi,l1->D,l1->S))
        printf("layer %d Wi arrays differ\n",lyr);
    if (!compare_arrays(l1->Wc,l2->Wc,l1->D,l1->S))
        printf("layer %d Wc arrays differ\n",lyr);
    if (!compare_arrays(l1->Wo,l2->Wo,l1->D,l1->S))
        printf("layer %d Wo arrays differ\n",lyr);
    /* Recurrent weights [S][S] */
    if (!compare_arrays(l1->Uf,l2->Uf,l1->S,l1->S))
        printf("layer %d Uf arrays differ\n",lyr);
    if (!compare_arrays(l1->Ui,l2->Ui,l1->S,l1->S))
        printf("layer %d Ui arrays differ\n",lyr);
    if (!compare_arrays(l1->Uc,l2->Uc,l1->S,l1->S))
        printf("layer %d Uc arrays differ\n",lyr);
    if (!compare_arrays(l1->Uo,l2->Uo,l1->S,l1->S))
        printf("layer %d Uo arrays differ\n",lyr);
    /* Carried state [1][S] */
    if (!compare_arrays((fArr2D) l1->pc,(fArr2D) l2->pc,1,l1->S))
        printf("layer %d pc arrays differ\n",lyr);
    if (!compare_arrays((fArr2D) l1->ph,(fArr2D) l2->ph,1,l1->S))
        printf("layer %d ph arrays differ\n",lyr);
    /* Biases [1][S] */
    if (l1->use_bias) {
        if (!compare_arrays((fArr2D) l1->bf,(fArr2D) l2->bf,1,l1->S))
            printf("layer %d bf arrays differ\n",lyr);
        if (!compare_arrays((fArr2D) l1->bi,(fArr2D) l2->bi,1,l1->S))
            printf("layer %d bi arrays differ\n",lyr);
        if (!compare_arrays((fArr2D) l1->bc,(fArr2D) l2->bc,1,l1->S))
            printf("layer %d bc arrays differ\n",lyr);
        if (!compare_arrays((fArr2D) l1->bo,(fArr2D) l2->bo,1,l1->S))
            printf("layer %d bo arrays differ\n",lyr);
    }

    /* Gradient buffers exist only in training mode */
    if (l1->training && l2->training) {
        if (!compare_arrays(l1->gWf,l2->gWf,l1->D,l1->S))
            printf("layer %d gWf arrays differ\n",lyr);
        if (!compare_arrays(l1->gWi,l2->gWi,l1->D,l1->S))
            printf("layer %d gWi arrays differ\n",lyr);
        if (!compare_arrays(l1->gWc,l2->gWc,l1->D,l1->S))
            printf("layer %d gWc arrays differ\n",lyr);
        if (!compare_arrays(l1->gWo,l2->gWo,l1->D,l1->S))
            printf("layer %d gWo arrays differ\n",lyr);
        if (!compare_arrays(l1->gUf,l2->gUf,l1->S,l1->S))
            printf("layer %d gUf arrays differ\n",lyr);
        if (!compare_arrays(l1->gUi,l2->gUi,l1->S,l1->S))
            printf("layer %d gUi arrays differ\n",lyr);
        if (!compare_arrays(l1->gUc,l2->gUc,l1->S,l1->S))
            printf("layer %d gUc arrays differ\n",lyr);
        if (!compare_arrays(l1->gUo,l2->gUo,l1->S,l1->S))
            printf("layer %d gUo arrays differ\n",lyr);
        if (l1->use_bias) {
            if (!compare_arrays((fArr2D) l1->gbf,(fArr2D) l2->gbf,1,l1->S))
                printf("layer %d gbf arrays differ\n",lyr);
            if (!compare_arrays((fArr2D) l1->gbi,(fArr2D) l2->gbi,1,l1->S))
                printf("layer %d gbi arrays differ\n",lyr);
            if (!compare_arrays((fArr2D) l1->gbc,(fArr2D) l2->gbc,1,l1->S))
                printf("layer %d gbc arrays differ\n",lyr);
            if (!compare_arrays((fArr2D) l1->gbo,(fArr2D) l2->gbo,1,l1->S))
                printf("layer %d gbo arrays differ\n",lyr);
        }
    }
}

void compare_layers(LAYER* l1, LAYER* l2, int lyr)
{
    if (l1->type != l2->type)
        printf("m1->layer[%d] type '%c' m2->layer[%d] type '%c'\n",
               lyr,l1->type,lyr,l2->type);
    if (l1->num_opt_state != l2->num_opt_state)
        printf("m1->layer[%d] num_opt_state %d m2->layer[%d] num_opt_state %d\n",
               lyr,l1->num_opt_state,lyr,l2->num_opt_state);
    if (l1->type != l2->type || l1->num_opt_state != l2->num_opt_state)
        exit(-1);
    switch (l1->type) {
        case 'd':
            compare_dense(l1->dense,l2->dense,lyr);
        break;
        case 'l':
            compare_lstm(l1->lstm,l2->lstm,lyr);
        break;
        default:
            printf("compare_layers: unsupported layer type '%c'\n",l1->type);
            exit(-1);
    }
}

void compare_models(MODEL* m1, MODEL* m2)
{
    if (m1->num_layers != m2->num_layers)
        printf("m1->num_layers %d m2->num_layers %d\n",m1->num_layers,m2->num_layers);
    if (m1->batch_size != m2->batch_size)
        printf("m1->batch_size %d m2->batch_size %d\n",m1->batch_size,m2->batch_size);
    if (m1->input_dim != m2->input_dim)
        printf("m1->input_dim %d m2->input_dim %d\n",m1->input_dim,m2->input_dim);
    if (m1->output_dim != m2->output_dim)
        printf("m1->output_dim %d m2->output_dim %d\n",m1->output_dim,m2->output_dim);
    if (m1->loss_func != m2->loss_func)
        printf("m1->loss_func '%c' m2->loss_func '%c'\n",m1->loss_func,m2->loss_func);
    if (m1->ctc != m2->ctc)
        printf("m1->ctc %p m2->ctc %p\n",(void*) m1->ctc,(void*) m2->ctc);
    if (m1->optimizer != m2->optimizer)
        printf("m1->optimizer '%c' m2->optimizer '%c'\n",m1->optimizer,m2->optimizer);
    if (m1->update_cnt != m2->update_cnt)
        printf("m1->update_cnt %d m2->update_cnt %d\n",m1->update_cnt,m2->update_cnt);
    if (m1->normalize != m2->normalize)
        printf("m1->normalize %d m2->normalize %d\n",m1->normalize,m2->normalize);
    if (m1->final != m2->final)
        printf("m1->final %d m2->final %d\n",m1->final,m2->final);
    if (m1->num_layers != m2->num_layers)
        exit(-1);
    for (int i = 0; i < m1->num_layers; i++)
        compare_layers(&m1->layer[i],&m2->layer[i],i);
}
