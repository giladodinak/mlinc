/* Copyright (c) 2026 Gilad Odinak           */
/* LSTM (reccurent) neural network functions */
/* References:
 * https://en.wikipedia.org/wiki/Long_short-term_memory
 * https://www.bioinf.jku.at/publications/older/2604.pdf
 */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <cuda_runtime.h>
#include "float.h"
#include "mem.h"
#include "random.h"
#include "arraytypes.h"
#include "array.h"
#include "activation.h"
#include "qr.h"
#include "lstm.h"
#include "graph.h"

#define THCNT 256


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
extern "C" LSTM* lstm_create(int units, int stateful, int use_bias)
{
    LSTM* l = (LSTM*) allocmem(1,1,LSTM);
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
extern "C" void lstm_init(LSTM* l, int input_dim, int batch_size, int training)
{
    l->D = input_dim;
    l->B = batch_size;
    l->training = training ? 1 : 0;
    /* Fused gate storage: [f | i | cc | o] for each time step */
    l->f = (fArr2D) allocmem(l->B,4 * l->S,float);
    l->i = (fArr2D) ((float*) l->f + l->S);
    l->cc = (fArr2D) ((float*) l->f + 2 * l->S);
    l->o = (fArr2D) ((float*) l->f + 3 * l->S);
    l->h = (fArr2D) allocmem(l->B + 1,l->S,float);
    l->c = (fArr2D) allocmem(l->B + 1,l->S,float);

    /* Fused transposed weights: [Wf.T | Wi.T | Wc.T | Wo.T] */
    l->Wf = (fArr2D) allocmem(4 * l->S,l->D,float);
    l->Wi = (fArr2D) ((float*) l->Wf + l->S * l->D);
    l->Wc = (fArr2D) ((float*) l->Wf + 2 * l->S * l->D);
    l->Wo = (fArr2D) ((float*) l->Wf + 3 * l->S * l->D);

    /* Fused transposed recurrent weights: [Uf.T | Ui.T | Uc.T | Uo.T] */
    l->Uf = (fArr2D) allocmem(4 * l->S,l->S,float);
    l->Ui = (fArr2D) ((float*) l->Uf + l->S * l->S);
    l->Uc = (fArr2D) ((float*) l->Uf + 2 * l->S * l->S);
    l->Uo = (fArr2D) ((float*) l->Uf + 3 * l->S * l->S);
    l->ph = (fVec) allocmem(1,l->S,float);
    l->pc = (fVec) allocmem(1,l->S,float);
    if (l->use_bias) {
        l->bf = (fVec) allocmem(1,4 * l->S,float);
        l->bi = (fVec) l->bf + l->S;
        l->bc = (fVec) l->bf + 2 * l->S;
        l->bo = (fVec) l->bf + 3 * l->S;
    }

    typedef float (*ArrSD)[l->D];
    ArrSD Wf = (ArrSD) l->Wf;
    ArrSD Wi = (ArrSD) l->Wi;
    ArrSD Wc = (ArrSD) l->Wc;
    ArrSD Wo = (ArrSD) l->Wo;
    float scale = sqrt(2.0 / (l->D + l->S));
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wf[j][i] = nrand(0.0,scale);
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wi[j][i] = nrand(0.0,scale);
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wc[j][i] = nrand(0.0,scale);
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wo[j][i] = nrand(0.0,scale);
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
        l->gWf = (fArr2D) allocmem(4 * l->S,l->D,float);
        l->gWi = (fArr2D) ((float*) l->gWf + l->S * l->D);
        l->gWc = (fArr2D) ((float*) l->gWf + 2 * l->S * l->D);
        l->gWo = (fArr2D) ((float*) l->gWf + 3 * l->S * l->D);
        l->gUf = (fArr2D) allocmem(4 * l->S,l->S,float);
        l->gUi = (fArr2D) ((float*) l->gUf + l->S * l->S);
        l->gUc = (fArr2D) ((float*) l->gUf + 2 * l->S * l->S);
        l->gUo = (fArr2D) ((float*) l->gUf + 3 * l->S * l->S);
        if (l->use_bias) {
            l->gbf = (fVec) allocmem(1,4 * l->S,float);
            l->gbi = (fVec) l->gbf + l->S;
            l->gbc = (fVec) l->gbf + 2 * l->S;
            l->gbo = (fVec) l->gbf + 3 * l->S;
        }
    }

    /* Initialization above used CPU memory. Move each allocation base to
     * CUDA memory, then restore the aliases into the fused allocations,
     * and free the CPU memory.
     */
    fArr2D f = (fArr2D) cuallocmem(l->B,4 * l->S,float);
    fArr2D h = (fArr2D) cuallocmem(l->B + 1,l->S,float);
    fArr2D c = (fArr2D) cuallocmem(l->B + 1,l->S,float);
    fArr2D Wf2 = (fArr2D) cuallocmem(4 * l->S,l->D,float);
    fArr2D Uf2 = (fArr2D) cuallocmem(4 * l->S,l->S,float);
    fVec ph = (fVec) cuallocmem(1,l->S,float);
    fVec pc = (fVec) cuallocmem(1,l->S,float);

    cufltcpy(f,l->f,l->B * 4 * l->S);
    cufltcpy(h,l->h,(l->B + 1) * l->S);
    cufltcpy(c,l->c,(l->B + 1) * l->S);
    cufltcpy(Wf2,l->Wf,4 * l->S * l->D);
    cufltcpy(Uf2,l->Uf,4 * l->S * l->S);
    cufltcpy(ph,l->ph,l->S);
    cufltcpy(pc,l->pc,l->S);

    freemem(l->f);
    freemem(l->h);
    freemem(l->c);
    freemem(l->Wf);
    freemem(l->Uf);
    freemem(l->ph);
    freemem(l->pc);

    l->f = f;
    l->i = (fArr2D) ((float*) l->f + l->S);
    l->cc = (fArr2D) ((float*) l->f + 2 * l->S);
    l->o = (fArr2D) ((float*) l->f + 3 * l->S);
    l->h = h;
    l->c = c;

    l->Wf = Wf2;
    l->Wi = (fArr2D) ((float*) l->Wf + l->S * l->D);
    l->Wc = (fArr2D) ((float*) l->Wf + 2 * l->S * l->D);
    l->Wo = (fArr2D) ((float*) l->Wf + 3 * l->S * l->D);

    l->Uf = Uf2;
    l->Ui = (fArr2D) ((float*) l->Uf + l->S * l->S);
    l->Uc = (fArr2D) ((float*) l->Uf + 2 * l->S * l->S);
    l->Uo = (fArr2D) ((float*) l->Uf + 3 * l->S * l->S);

    l->ph = ph;
    l->pc = pc;

    if (l->use_bias) {
        fVec bf = (fVec) cuallocmem(1,4 * l->S,float);
        cufltcpy(bf,l->bf,4 * l->S);
        freemem(l->bf);
        l->bf = bf;
        l->bi = l->bf + l->S;
        l->bc = l->bf + 2 * l->S;
        l->bo = l->bf + 3 * l->S;
    }

    if (l->training) {
        fArr2D gWf = (fArr2D) cuallocmem(4 * l->S,l->D,float);
        fArr2D gUf = (fArr2D) cuallocmem(4 * l->S,l->S,float);

        cufltcpy(gWf,l->gWf,4 * l->S * l->D);
        cufltcpy(gUf,l->gUf,4 * l->S * l->S);

        freemem(l->gWf);
        freemem(l->gUf);

        l->gWf = gWf;
        l->gWi = (fArr2D) ((float*) l->gWf + l->S * l->D);
        l->gWc = (fArr2D) ((float*) l->gWf + 2 * l->S * l->D);
        l->gWo = (fArr2D) ((float*) l->gWf + 3 * l->S * l->D);

        l->gUf = gUf;
        l->gUi = (fArr2D) ((float*) l->gUf + l->S * l->S);
        l->gUc = (fArr2D) ((float*) l->gUf + 2 * l->S * l->S);
        l->gUo = (fArr2D) ((float*) l->gUf + 3 * l->S * l->S);

        if (l->use_bias) {
            fVec gbf = (fVec) cuallocmem(1,4 * l->S,float);
            cufltcpy(gbf,l->gbf,4 * l->S);
            freemem(l->gbf);
            l->gbf = gbf;
            l->gbi = l->gbf + l->S;
            l->gbc = l->gbf + 2 * l->S;
            l->gbo = l->gbf + 3 * l->S;
        }

        /* Backward workspace: dh_next[S], dc_next[S],
         * and fused [df | di | dcc | do][B][4S].
         */
        l->dh_next = (fVec) cuallocmem(2 + 4 * l->B,l->S,float);
        l->dc_next = l->dh_next + l->S;
        l->df = (fArr2D) (l->dh_next + 2 * l->S);
        l->di = (fArr2D) ((float*) l->df + l->S);
        l->dcc = (fArr2D) ((float*) l->df + 2 * l->S);
        l->do_ = (fArr2D) ((float*) l->df + 3 * l->S);
    }

    l->state = graph_create();
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
extern "C" void lstm_set_batch_size(LSTM* l, int batch_size)
{
    if (l->B == 0)
        return;
    if (batch_size != l->B) {
        graph_destroy((GRAPH*) l->state);
        cufreemem(l->f);
        cufreemem(l->h);
        cufreemem(l->c);
        if (l->training)
            cufreemem(l->dh_next);

        l->B = batch_size;
        l->f = (fArr2D) cuallocmem(l->B,4 * l->S,float);
        l->i = (fArr2D) ((float*) l->f + l->S);
        l->cc = (fArr2D) ((float*) l->f + 2 * l->S);
        l->o = (fArr2D) ((float*) l->f + 3 * l->S);
        l->h = (fArr2D) cuallocmem(l->B + 1,l->S,float);
        l->c = (fArr2D) cuallocmem(l->B + 1,l->S,float);

        if (l->training) {
            l->dh_next = (fVec) cuallocmem(2 + 4 * l->B,l->S,float);
            l->dc_next = l->dh_next + l->S;
            l->df = (fArr2D) (l->dh_next + 2 * l->S);
            l->di = (fArr2D) ((float*) l->df + l->S);
            l->dcc = (fArr2D) ((float*) l->df + 2 * l->S);
            l->do_ = (fArr2D) ((float*) l->df + 3 * l->S);
        }

        l->state = graph_create();
    }
    else {
        cufltclr(l->f,l->B * 4 * l->S);
        cufltclr(l->h,(l->B + 1) * l->S);
        cufltclr(l->c,(l->B + 1) * l->S);
    }
}


/* Frees the memory allocated by lstm_create().
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be freed
 */
extern "C" void lstm_free(LSTM* l)
{
    graph_destroy((GRAPH*) l->state);
    cufreemem(l->f);
    cufreemem(l->h);
    cufreemem(l->c);
    cufreemem(l->Wf);
    cufreemem(l->Uf);
    cufreemem(l->ph);
    cufreemem(l->pc);
    cufreemem(l->bf);
    cufreemem(l->gWf);
    cufreemem(l->gUf);
    cufreemem(l->gbf);
    cufreemem(l->dh_next);
    freemem(l);
}

/* Resets the LSTM internal state.
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be reset
 */
extern "C" void lstm_reset(LSTM* l)
{
    cufltclr(l->ph,l->S);
    cufltclr(l->pc,l->S);
}

static inline void lstm_activate(fVec v, int S)
{
   sigmoid((fArr2D) v,1,S);
}

/* Adds bias vector b to a gate pre-activation vector v (length S) */
static inline void lstm_add_bias(fVec v_, const fVec b_, int S)
{
    float* v = (float*) v_;
    const float* b = (const float*) b_;
    for (int j = 0; j < S; j++)
        v[j] += b[j];
}

/* Applies the gate activations and updates cell and hidden state.
 * g is the fused gate row [f | i | cc | o].
 */
static __global__ void lstm_forward_kernel(float* g, const float* b,
                                           const float* cprev,
                                           float* c, float* h,
                                           int S, int use_bias)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < S) {
        float f  = g[j];
        float i  = g[S + j];
        float cc = g[2 * S + j];
        float o  = g[3 * S + j];

        if (use_bias) {
            f  += b[j];
            i  += b[S + j];
            cc += b[2 * S + j];
            o  += b[3 * S + j];
        }

        f  = 1.0f / (1.0f + expf(-f));
        i  = 1.0f / (1.0f + expf(-i));
        cc = tanhf(cc);
        o  = 1.0f / (1.0f + expf(-o));

        g[j] = f;
        g[S + j] = i;
        g[2 * S + j] = cc;
        g[3 * S + j] = o;

        float cv = f * cprev[j] + i * cc;
        c[j] = cv;
        h[j] = o * tanhf(cv);
    }
}

/* Performs LSTM layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the LSTM layer's data
 *   X   - Array of input vectors BxD, where B is the number of 
 *         input vectors, and D is the number of features in each vector
 *   lyr - Ordinal number of this layer in a model (not used)
 * 
 * Returns:
 *   Pointer to the predicted values.
 * 
 * Note:
 * - In a multi-layered neural network, after the first layer,
 *   X is the (activated) output of a previous layer.
 */
extern "C" fArr2D lstm_forward(LSTM* restrict l,
                    const fArr2D restrict X /*[B][D]*/, int lyr)
{
    (void) lyr;
    const int D = l->D;
    const int S = l->S;
    const int B = l->B;
    const int blocks = (S + THCNT - 1) / THCNT;
    typedef float (*ArrTD)[D];
    ArrTD x = (ArrTD) X;

    cufltclr(l->f,B * 4 * S);
    cufltclr(l->c,(B+1)*S);
    cufltclr(l->h,(B+1)*S);

    typedef float (*ArrB4S)[4 * S];
    ArrB4S f = (ArrB4S) l->f;
    typedef float (*ArrBS)[S];
    typedef float (*ArrB1S)[S];
    ArrBS c = ((ArrB1S) l->c) + 1;
    ArrBS h = ((ArrB1S) l->h) + 1;

    if (l->stateful) {
        cufltcpy(h[-1],l->ph,S);
        cufltcpy(c[-1],l->pc,S);
    }

    /* f[B][4S] = X[B][D] @ Wf[4S][D].T */
    matmulT(l->f,X,l->Wf,B,D,4*S);

    GRAPH* graph = (GRAPH*) l->state;

    if (graph_record_exists(graph,GRAPH_FORWARD)) {
        graph_play(graph,GRAPH_FORWARD);
    }
    else {
        graph_record(graph,GRAPH_FORWARD,GRAPH_RECORD_ON);
        cudaStream_t stream = graph_stream(graph);

        for (int t = 0; t < B; t++) {
            /* [f i cc o] = X[t] @ W.T */
            //addinnermul(f[t],x[t],l->Wf,4 * S,D);
            /* [f i cc o] += h[t-1] @ U.T */
            addinnermul(f[t],h[t-1],l->Uf,4 * S,S);

            lstm_forward_kernel<<<blocks,THCNT,0,stream>>>(
                (float*)f[t],(const float*)l->bf,(const float*)c[t-1],
                (float*)c[t],(float*)h[t],S,l->use_bias);
        }

        graph_record(graph,GRAPH_FORWARD,GRAPH_RECORD_OFF);
        graph_play(graph,GRAPH_FORWARD);
    }

    cufltcpy(l->ph,h[B-1],S);
    cufltcpy(l->pc,c[B-1],S);
    return h;
}

static inline float lstm_d_activate(float x)
{
    return d_sigmoid1(x);
}

/* Computes dg = [df | di | dcc | do], updates dc_next, and accumulates
 * the fused bias gradient when bias is enabled.
 */
static __global__ void lstm_backward_kernel(const float* dy,
                                            const float* f,
                                            const float* i,
                                            const float* cc,
                                            const float* o,
                                            const float* cprev,
                                            const float* c,
                                            const float* dh_next,
                                            float* dc_next,
                                            float* dg,
                                            float* gb,
                                            int S, int use_bias)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < S) {
        float dh = dy[j] + dh_next[j];
        float tc = tanhf(c[j]);
        float dc = dh * o[j] * (1.0f - tc * tc) + dc_next[j];

        float df = dc * cprev[j] * (f[j] * (1.0f - f[j]));
        float di = dc * cc[j] * (i[j] * (1.0f - i[j]));
        float dcc = dc * i[j] * (1.0f - cc[j] * cc[j]);
        float do_ = dh * tc * (o[j] * (1.0f - o[j]));

        dg[j] = df;
        dg[S + j] = di;
        dg[2 * S + j] = dcc;
        dg[3 * S + j] = do_;

        if (use_bias) {
            gb[j] += df;
            gb[S + j] += di;
            gb[2 * S + j] += dcc;
            gb[3 * S + j] += do_;
        }

        dc_next[j] = f[j] * dc;
    }
}

/* Performs LSTM layer training's backward pass.
 * 
 * Parameters:
 *   l    - Pointer to the LSTM layer's data
 *   dY   - Output vector gradient of lstm_create's units dimension
 *   X    - Array of input vectors BxD, where B is the number of input
 *          vectors, and D is the number of features in each vector
 *   dX   - Output parameter for the input vector gradient (if not NULL)
 *   lyr  - Ordinal number of this layer in a model (not used)
 * 
 * Returns:
 *   None
 * 
 * Note:
 *   - Calculates the weight and bias gradients and stores them in the
 *     layer's own gradient buffers (gWf..gUo, and gbf..gbo if use_bias).
 *   - Calculates the input vector gradient and returns it in dx, if dx is 
 *     not NULL
 *   - When stateful is enabled, forward state is carried across calls,
 *     but gradients do not propagate across sequence boundaries.
 *   - In a multi-layered neural network, except the last layer, dy is the 
 *     gradient of the previous layer's input (dx), thus, the dimension of 
 *     dx (this layer's D) must equal the dimension of the previous layer's
 *     dy (previous layer's S)
 */
extern "C" void lstm_backward(LSTM* restrict l,
                   const fArr2D restrict dY/*[B][S]*/,
                   const fArr2D restrict X/*[B][D]*/,
                   fArr2D restrict dX/*[B][D]*/,
                   int lyr)
{
    (void) lyr;
    const int D = l->D;
    const int S = l->S;
    const int B = l->B;
    const int blocks = (S + THCNT - 1) / THCNT;
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) X;
    ArrBD dx = (ArrBD) dX;
    typedef float (*ArrBS)[S];
    ArrBS dy = (ArrBS) dY;

    typedef float (*ArrB4S)[4 * S];
    ArrB4S f = (ArrB4S) l->f;
    ArrB4S i = (ArrB4S) l->i;
    ArrB4S cc = (ArrB4S) l->cc;
    ArrB4S o = (ArrB4S) l->o;
    typedef float (*ArrB1S)[S];
    ArrBS c = ((ArrB1S) l->c) + 1;
    ArrBS h = ((ArrB1S) l->h) + 1;

    cufltclr(l->gWf,4 * S * D);
    cufltclr(l->gUf,4 * S * S);
    if (l->use_bias)
        cufltclr(l->gbf,4 * S);

    /* dh_next[S], dc_next[S], dg[B][4*S] */
    fVec dh_next = l->dh_next;
    fVec dc_next = l->dc_next;
    ArrB4S dg = (ArrB4S) l->df;
    cufltclr(dh_next,S);
    cufltclr(dc_next,S);

    GRAPH* graph = (GRAPH*) l->state;

    if (graph_record_exists(graph,GRAPH_BACKWARD)) {
        graph_play(graph,GRAPH_BACKWARD);
    }
    else {
        graph_record(graph,GRAPH_BACKWARD,GRAPH_RECORD_ON);
        cudaStream_t stream = graph_stream(graph);

        for (int t = B - 1; t >= 0; t--) {
            const float* ft = (const float*) f[t];
            const float* it = (const float*) i[t];
            const float* cct = (const float*) cc[t];
            const float* ot = (const float*) o[t];

            lstm_backward_kernel<<<blocks,THCNT,0,stream>>>(
                (const float*) dy[t],ft,it,cct,ot,
                (const float*) c[t-1],(const float*) c[t],
                dh_next,dc_next,dg[t],(float*) l->gbf,S,l->use_bias);

            vecmatmul(dh_next,(fVec) dg[t],l->Uf,4 * S,S);
        }

        graph_record(graph,GRAPH_BACKWARD,GRAPH_RECORD_OFF);
        graph_play(graph,GRAPH_BACKWARD);
    }

    /* Gradients of the stored transposed fused matrices */
    Tmatmul(l->gWf,dg,x,4 * S,B,D);
    Tmatmul(l->gUf,dg,l->h,4 * S,B,S); /* l->h is h[-1] */

    if (dx != NULL)
        matmul(dx,dg,l->Wf,B,4 * S,D);
}
