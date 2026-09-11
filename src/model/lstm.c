/* Copyright (c) 2023-2024 Gilad Odinak */
/* LSTM (reccurent) neural networkfunctions */
/* References:
 * https://en.wikipedia.org/wiki/Long_short-term_memory
 * https://www.bioinf.jku.at/publications/older/2604.pdf
 */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "qr.h"
#include "lstm.h"

/* Creates a long short term memory (LSTM) neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   stateful   - If not zero, maintain state across batches.
 *   use_bias   - If not zero, add a per-gate bias.
 * 
 * Returns:
 *   Pointer to an LSTM neural network layer.
 *
 * Notes:
 *   - The neural network needs to be further intialized using lstm_init()
 *     before it can be used.
 */
LSTM* lstm_create(int units, int stateful, int use_bias)
{
    LSTM* l = allocmem(1,1,LSTM);
    l->S = units;
    l->stateful = stateful ? 1 : 0;
    l->use_bias = use_bias ? 1 : 0;
    return l;    
}

/* Initializes an LSTM neural network created by lstm_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors
 *   batch_size - Number of input vectors processed simultaneously
 *   training   - 1: allocate gradient buffers, 0 for inference-only
 *
 * Notes:
 *   - Kernel weights (Wx) are initialized using Glorot normal distribution.
 *   - Recurrent weights (Ux) are initialized using orthogonal uniform 
 *     distribution.
 */
void lstm_init(LSTM* l, int input_dim, int batch_size, int training)
{
    l->D = input_dim;
    l->B = batch_size;
    l->training = training ? 1 : 0;
    l->f = allocmem(l->B,l->S,float);
    l->i = allocmem(l->B,l->S,float);
    l->o = allocmem(l->B,l->S,float);
    l->cc = allocmem(l->B + 1,l->S,float);
    l->h = allocmem(l->B + 1,l->S,float);
    l->c = allocmem(l->B + 1,l->S,float);
    l->Wf = allocmem(l->D,l->S,float);
    l->Wi = allocmem(l->D,l->S,float);
    l->Wc = allocmem(l->D,l->S,float);
    l->Wo = allocmem(l->D,l->S,float);
    l->Uf = allocmem(l->S,l->S,float);
    l->Ui = allocmem(l->S,l->S,float);
    l->Uc = allocmem(l->S,l->S,float);
    l->Uo = allocmem(l->S,l->S,float);
    l->ph = allocmem(1,l->S,float);
    l->pc = allocmem(1,l->S,float);
    if (l->use_bias) {
        l->bf = allocmem(1,l->S,float);
        l->bi = allocmem(1,l->S,float);
        l->bc = allocmem(1,l->S,float);
        l->bo = allocmem(1,l->S,float);
    }

    typedef float (*ArrDS)[l->S];
    ArrDS Wf = (ArrDS) l->Wf;
    ArrDS Wi = (ArrDS) l->Wi;
    ArrDS Wc = (ArrDS) l->Wc;
    ArrDS Wo = (ArrDS) l->Wo;
    float scale = sqrt(2.0 / (l->D + l->S));
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wf[i][j] = nrand(0.0,scale);
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wi[i][j] = nrand(0.0,scale);
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wc[i][j] = nrand(0.0,scale);
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wo[i][j] = nrand(0.0,scale);
    typedef float (*ArrS2)[l->S];
    ArrS2 Uf = (ArrS2) l->Uf;
    ArrS2 Ui = (ArrS2) l->Ui;
    ArrS2 Uc = (ArrS2) l->Uc;
    ArrS2 Uo = (ArrS2) l->Uo;
    scale = sqrt(6.0 / ((float) (l->S * 2)));
    for (int i = 0; i < l->S; i++)
        for (int j = 0; j < l->S; j++)
            Uf[i][j] = urand(-scale,scale);
    for (int i = 0; i < l->S; i++)
        for (int j = 0; j < l->S; j++)
            Ui[i][j] = urand(-scale,scale);
    for (int i = 0; i < l->S; i++)
        for (int j = 0; j < l->S; j++)
            Uc[i][j] = urand(-scale,scale);
    for (int i = 0; i < l->S; i++)
        for (int j = 0; j < l->S; j++)
            Uo[i][j] = urand(-scale,scale);
    
    QR(Uf,NULL,NULL,l->S,l->S);
    QR(Ui,NULL,NULL,l->S,l->S);
    QR(Uc,NULL,NULL,l->S,l->S);
    QR(Uo,NULL,NULL,l->S,l->S);

    if (l->training) {
        l->gWf = allocmem(l->D,l->S,float);
        l->gWi = allocmem(l->D,l->S,float);
        l->gWc = allocmem(l->D,l->S,float);
        l->gWo = allocmem(l->D,l->S,float);
        l->gUf = allocmem(l->S,l->S,float);
        l->gUi = allocmem(l->S,l->S,float);
        l->gUc = allocmem(l->S,l->S,float);
        l->gUo = allocmem(l->S,l->S,float);
        if (l->use_bias) {
            l->gbf = allocmem(1,l->S,float);
            l->gbi = allocmem(1,l->S,float);
            l->gbc = allocmem(1,l->S,float);
            l->gbo = allocmem(1,l->S,float);
        }
    }
}

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before ilstm_init(), it does nothing.
 *   Otherwise, the network's hidden states are resized and re-initialized
 */
void lstm_set_batch_size(LSTM* l, int batch_size)
{
    if (l->B == 0)
        return;
    if (batch_size != l->B) {
        freemem(l->f);
        freemem(l->i);
        freemem(l->o);
        freemem(l->cc);
        freemem(l->h);
        freemem(l->c);
        l->B = batch_size;
        l->f = allocmem(l->B,l->S,float);
        l->i = allocmem(l->B,l->S,float);
        l->o = allocmem(l->B,l->S,float);
        l->cc = allocmem(l->B + 1,l->S,float);
        l->h = allocmem(l->B + 1,l->S,float);
        l->c = allocmem(l->B + 1,l->S,float);
    }
    else {
        fltclr(l->f,l->B * l->S);
        fltclr(l->i,l->B * l->S);
        fltclr(l->o,l->B * l->S);
        fltclr(l->cc,(l->B + 1) * l->S);
        fltclr(l->h,(l->B + 1) * l->S);
        fltclr(l->c,(l->B + 1) * l->S);
    }
}


/* Frees the memory allocated by lstm_create().
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be freed
 */
void lstm_free(LSTM* l)
{
    freemem(l->f);
    freemem(l->i);
    freemem(l->o);
    freemem(l->cc);
    freemem(l->h);
    freemem(l->c);
    freemem(l->Wf);
    freemem(l->Wi);
    freemem(l->Wc);
    freemem(l->Wo);
    freemem(l->Uf);
    freemem(l->Ui);
    freemem(l->Uc);
    freemem(l->Uo);
    freemem(l->ph);
    freemem(l->pc);
    freemem(l->bf);
    freemem(l->bi);
    freemem(l->bc);
    freemem(l->bo);
    freemem(l->gWf);
    freemem(l->gWi);
    freemem(l->gWc);
    freemem(l->gWo);
    freemem(l->gUf);
    freemem(l->gUi);
    freemem(l->gUc);
    freemem(l->gUo);
    freemem(l->gbf);
    freemem(l->gbi);
    freemem(l->gbc);
    freemem(l->gbo);
    freemem(l);
}

/* Resets the LSTM internal state.
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be reset
 */
void lstm_reset(LSTM* l)
{
    fltclr(l->ph,l->S);
    fltclr(l->pc,l->S);
}
