/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA LSTM (recurrent) neural network functions */
/* References:
 * https://en.wikipedia.org/wiki/Long_short-term_memory
 * https://www.bioinf.jku.at/publications/older/2604.pdf
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "mem.h"
#include "cuflt.h"
#include "array.h"
#include "activation.h"
#include "lstm.h"

#define THCNT 256

static __device__ unsigned int lstm_hash(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __global__ void init_normal_kernel(float* w, int n, float scale,
                                          unsigned int seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        unsigned int h1 = lstm_hash((unsigned int)k + seed);
        unsigned int h2 = lstm_hash((unsigned int)k + seed + 0x9e3779b9u);
        float u1 = ((float)h1 + 1.0f) / 4294967296.0f;
        float u2 = ((float)h2 + 1.0f) / 4294967296.0f;
        float z = sqrtf(-2.0f * logf(u1)) *
                  cosf(6.2831853071795864769f * u2);
        w[k] = z * scale;
    }
}

static __global__ void init_uniform_kernel(float* w, int n, float scale,
                                           unsigned int seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        unsigned int h = lstm_hash((unsigned int)k + seed);
        float u = ((float)h + 0.5f) / 4294967296.0f;
        w[k] = (2.0f * u - 1.0f) * scale;
    }
}

/* Initialization-only Gram-Schmidt orthogonalization of a square row-major
 * matrix. It intentionally runs serially: recurrent weights are initialized
 * once, while forward/backward remain fully device-resident.
 */
static __global__ void orthogonalize_kernel(float* u, int S)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    for (int j = 0; j < S; j++) {
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int r = 0; r < S; r++)
                dot += u[r * S + j] * u[r * S + k];
            for (int r = 0; r < S; r++)
                u[r * S + j] -= dot * u[r * S + k];
        }

        float norm = 0.0f;
        for (int r = 0; r < S; r++) {
            float v = u[r * S + j];
            norm += v * v;
        }
        norm = sqrtf(norm);

        if (norm > 1e-12f) {
            float inv = 1.0f / norm;
            for (int r = 0; r < S; r++)
                u[r * S + j] *= inv;
        }
    }
}

static __global__ void add_bias_kernel(float* v, const float* b, int S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < S)
        v[j] += b[j];
}

static __global__ void tanh_kernel(float* v, int S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < S)
        v[j] = tanhf(v[j]);
}

static __global__ void state_kernel(const float* f, const float* i,
                                    const float* o, const float* cc,
                                    const float* cprev,
                                    float* c, float* h, int S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < S) {
        float cv = f[j] * cprev[j] + i[j] * cc[j];
        c[j] = cv;
        h[j] = o[j] * tanhf(cv);
    }
}

static __global__ void backward_gate_kernel(const float* dy,
                                             const float* f,
                                             const float* i,
                                             const float* o,
                                             const float* cc,
                                             const float* cprev,
                                             const float* c,
                                             const float* dh_next,
                                             float* dc_next,
                                             float* do_,
                                             float* dc,
                                             float* dcc,
                                             float* di,
                                             float* df,
                                             int S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < S) {
        float dh = dy[j] + dh_next[j];
        float cv = c[j];
        float tanhc = tanhf(cv);
        float dcv = dh * o[j] * (1.0f - tanhc * tanhc) + dc_next[j];

        do_[j] = dh * tanhc * (o[j] * (1.0f - o[j]));
        dc[j] = dcv;
        dcc[j] = dcv * i[j] * (1.0f - cc[j] * cc[j]);
        di[j] = dcv * cc[j] * (i[j] * (1.0f - i[j]));
        df[j] = dcv * cprev[j] * (f[j] * (1.0f - f[j]));
        dc_next[j] = f[j] * dcv;
    }
}

static __global__ void add_vector_kernel(float* dst, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] += src[i];
}

extern "C" LSTM* lstm_create(int units, int stateful, int use_bias)
{
    LSTM* l = (LSTM*) allocmem(1,1,LSTM);
    l->S = units;
    l->stateful = stateful ? 1 : 0;
    l->use_bias = use_bias ? 1 : 0;
    return l;
}

extern "C" void lstm_init(LSTM* l, int input_dim, int batch_size, int training)
{
    l->D = input_dim;
    l->B = batch_size;
    l->training = training ? 1 : 0;

    l->f  = (fArr2D)cuallocmem(l->B,l->S,float);
    l->i  = (fArr2D)cuallocmem(l->B,l->S,float);
    l->o  = (fArr2D)cuallocmem(l->B,l->S,float);
    l->cc = (fArr2D)cuallocmem(l->B + 1,l->S,float);
    l->h  = (fArr2D)cuallocmem(l->B + 1,l->S,float);
    l->c  = (fArr2D)cuallocmem(l->B + 1,l->S,float);

    l->Wf = (fArr2D)cuallocmem(l->D,l->S,float);
    l->Wi = (fArr2D)cuallocmem(l->D,l->S,float);
    l->Wc = (fArr2D)cuallocmem(l->D,l->S,float);
    l->Wo = (fArr2D)cuallocmem(l->D,l->S,float);

    l->Uf = (fArr2D)cuallocmem(l->S,l->S,float);
    l->Ui = (fArr2D)cuallocmem(l->S,l->S,float);
    l->Uc = (fArr2D)cuallocmem(l->S,l->S,float);
    l->Uo = (fArr2D)cuallocmem(l->S,l->S,float);

    l->ph = (fVec)cuallocmem(1,l->S,float);
    l->pc = (fVec)cuallocmem(1,l->S,float);

    if (l->use_bias) {
        l->bf = (fVec)cuallocmem(1,l->S,float);
        l->bi = (fVec)cuallocmem(1,l->S,float);
        l->bc = (fVec)cuallocmem(1,l->S,float);
        l->bo = (fVec)cuallocmem(1,l->S,float);
    }

    int nW = l->D * l->S;
    float wscale = sqrtf(2.0f / (l->D + l->S));
    int wblocks = (nW + THCNT - 1) / THCNT;
    init_normal_kernel<<<wblocks,THCNT>>>((float*)l->Wf,nW,wscale,0x10001u);
    init_normal_kernel<<<wblocks,THCNT>>>((float*)l->Wi,nW,wscale,0x20003u);
    init_normal_kernel<<<wblocks,THCNT>>>((float*)l->Wc,nW,wscale,0x30007u);
    init_normal_kernel<<<wblocks,THCNT>>>((float*)l->Wo,nW,wscale,0x40009u);

    int nU = l->S * l->S;
    float uscale = sqrtf(6.0f / (2.0f * l->S));
    int ublocks = (nU + THCNT - 1) / THCNT;
    init_uniform_kernel<<<ublocks,THCNT>>>((float*)l->Uf,nU,uscale,0x50021u);
    init_uniform_kernel<<<ublocks,THCNT>>>((float*)l->Ui,nU,uscale,0x6002bu);
    init_uniform_kernel<<<ublocks,THCNT>>>((float*)l->Uc,nU,uscale,0x7002fu);
    init_uniform_kernel<<<ublocks,THCNT>>>((float*)l->Uo,nU,uscale,0x8003bu);

    orthogonalize_kernel<<<1,1>>>((float*)l->Uf,l->S);
    orthogonalize_kernel<<<1,1>>>((float*)l->Ui,l->S);
    orthogonalize_kernel<<<1,1>>>((float*)l->Uc,l->S);
    orthogonalize_kernel<<<1,1>>>((float*)l->Uo,l->S);

    if (l->training) {
        l->gWf = (fArr2D)cuallocmem(l->D,l->S,float);
        l->gWi = (fArr2D)cuallocmem(l->D,l->S,float);
        l->gWc = (fArr2D)cuallocmem(l->D,l->S,float);
        l->gWo = (fArr2D)cuallocmem(l->D,l->S,float);
        l->gUf = (fArr2D)cuallocmem(l->S,l->S,float);
        l->gUi = (fArr2D)cuallocmem(l->S,l->S,float);
        l->gUc = (fArr2D)cuallocmem(l->S,l->S,float);
        l->gUo = (fArr2D)cuallocmem(l->S,l->S,float);

        if (l->use_bias) {
            l->gbf = (fVec)cuallocmem(1,l->S,float);
            l->gbi = (fVec)cuallocmem(1,l->S,float);
            l->gbc = (fVec)cuallocmem(1,l->S,float);
            l->gbo = (fVec)cuallocmem(1,l->S,float);
        }
    }
}

extern "C" void lstm_set_batch_size(LSTM* l, int batch_size)
{
    if (l->B == 0)
        return;

    if (batch_size != l->B) {
        cufreemem(l->f);
        cufreemem(l->i);
        cufreemem(l->o);
        cufreemem(l->cc);
        cufreemem(l->h);
        cufreemem(l->c);

        l->B = batch_size;
        l->f  = (fArr2D)cuallocmem(l->B,l->S,float);
        l->i  = (fArr2D)cuallocmem(l->B,l->S,float);
        l->o  = (fArr2D)cuallocmem(l->B,l->S,float);
        l->cc = (fArr2D)cuallocmem(l->B + 1,l->S,float);
        l->h  = (fArr2D)cuallocmem(l->B + 1,l->S,float);
        l->c  = (fArr2D)cuallocmem(l->B + 1,l->S,float);
    }
    else {
        cufltclr(l->f,l->B * l->S);
        cufltclr(l->i,l->B * l->S);
        cufltclr(l->o,l->B * l->S);
        cufltclr(l->cc,(l->B + 1) * l->S);
        cufltclr(l->h,(l->B + 1) * l->S);
        cufltclr(l->c,(l->B + 1) * l->S);
    }
}

extern "C" void lstm_free(LSTM* l)
{
    if (l == NULL)
        return;

    cufreemem(l->f);
    cufreemem(l->i);
    cufreemem(l->o);
    cufreemem(l->cc);
    cufreemem(l->h);
    cufreemem(l->c);

    cufreemem(l->Wf);
    cufreemem(l->Wi);
    cufreemem(l->Wc);
    cufreemem(l->Wo);
    cufreemem(l->Uf);
    cufreemem(l->Ui);
    cufreemem(l->Uc);
    cufreemem(l->Uo);

    cufreemem(l->ph);
    cufreemem(l->pc);
    cufreemem(l->bf);
    cufreemem(l->bi);
    cufreemem(l->bc);
    cufreemem(l->bo);

    cufreemem(l->gWf);
    cufreemem(l->gWi);
    cufreemem(l->gWc);
    cufreemem(l->gWo);
    cufreemem(l->gUf);
    cufreemem(l->gUi);
    cufreemem(l->gUc);
    cufreemem(l->gUo);
    cufreemem(l->gbf);
    cufreemem(l->gbi);
    cufreemem(l->gbc);
    cufreemem(l->gbo);

    memfree(l);
}

extern "C" void lstm_reset(LSTM* l)
{
    cufltclr(l->ph,l->S);
    cufltclr(l->pc,l->S);
}

extern "C" fArr2D lstm_forward(LSTM* restrict l,
                                const fArr2D restrict X,
                                int lyr)
{
    (void)lyr;
    const int D = l->D;
    const int S = l->S;
    const int B = l->B;
    const int blocks = (S + THCNT - 1) / THCNT;

    typedef float (*ArrTD)[D];
    typedef float (*ArrBS)[S];
    typedef float (*ArrB1S)[S];

    ArrTD x = (ArrTD)X;
    ArrBS f = (ArrBS)l->f;
    ArrBS i = (ArrBS)l->i;
    ArrBS o = (ArrBS)l->o;
    ArrBS cc = ((ArrB1S)l->cc) + 1;
    ArrBS c = ((ArrB1S)l->c) + 1;
    ArrBS h = ((ArrB1S)l->h) + 1;

    cufltclr(l->f,B * S);
    cufltclr(l->i,B * S);
    cufltclr(l->o,B * S);
    cufltclr(l->cc,(B + 1) * S);
    cufltclr(l->c,(B + 1) * S);
    cufltclr(l->h,(B + 1) * S);

    if (l->stateful) {
        cufltcpy(h[-1],l->ph,S);
        cufltcpy(c[-1],l->pc,S);
    }

    for (int t = 0; t < B; t++) {
        addvecmatmul(f[t],x[t],l->Wf,D,S);
        addvecmatmul(f[t],h[t-1],l->Uf,S,S);
        if (l->use_bias)
            add_bias_kernel<<<blocks,THCNT>>>((float*)f[t],(const float*)l->bf,S);
        sigmoid((fArr2D)f[t],1,S);

        addvecmatmul(i[t],x[t],l->Wi,D,S);
        addvecmatmul(i[t],h[t-1],l->Ui,S,S);
        if (l->use_bias)
            add_bias_kernel<<<blocks,THCNT>>>((float*)i[t],(const float*)l->bi,S);
        sigmoid((fArr2D)i[t],1,S);

        addvecmatmul(o[t],x[t],l->Wo,D,S);
        addvecmatmul(o[t],h[t-1],l->Uo,S,S);
        if (l->use_bias)
            add_bias_kernel<<<blocks,THCNT>>>((float*)o[t],(const float*)l->bo,S);
        sigmoid((fArr2D)o[t],1,S);

        addvecmatmul(cc[t],x[t],l->Wc,D,S);
        addvecmatmul(cc[t],h[t-1],l->Uc,S,S);
        if (l->use_bias)
            add_bias_kernel<<<blocks,THCNT>>>((float*)cc[t],(const float*)l->bc,S);
        tanh_kernel<<<blocks,THCNT>>>((float*)cc[t],S);

        state_kernel<<<blocks,THCNT>>>((const float*)f[t],(const float*)i[t],
                                      (const float*)o[t],(const float*)cc[t],
                                      (const float*)c[t-1],(float*)c[t],
                                      (float*)h[t],S);
    }

    cufltcpy(l->ph,h[B-1],S);
    cufltcpy(l->pc,c[B-1],S);
    return h;
}

extern "C" void lstm_backward(LSTM* restrict l,
                               const fArr2D restrict dY,
                               const fArr2D restrict X,
                               fArr2D restrict dX,
                               int lyr)
{
    (void)lyr;
    const int D = l->D;
    const int S = l->S;
    const int B = l->B;
    const int blocks = (S + THCNT - 1) / THCNT;

    typedef float (*ArrBD)[D];
    typedef float (*ArrBS)[S];
    typedef float (*ArrB1S)[S];

    ArrBD x = (ArrBD)X;
    ArrBD dx = (ArrBD)dX;
    ArrBS dy = (ArrBS)dY;
    ArrBS f = (ArrBS)l->f;
    ArrBS i = (ArrBS)l->i;
    ArrBS o = (ArrBS)l->o;
    ArrBS cc = ((ArrB1S)l->cc) + 1;
    ArrBS c = ((ArrB1S)l->c) + 1;
    ArrBS h = ((ArrB1S)l->h) + 1;

    cufltclr(l->gWf,D * S);
    cufltclr(l->gWi,D * S);
    cufltclr(l->gWc,D * S);
    cufltclr(l->gWo,D * S);
    cufltclr(l->gUf,S * S);
    cufltclr(l->gUi,S * S);
    cufltclr(l->gUc,S * S);
    cufltclr(l->gUo,S * S);

    if (l->use_bias) {
        cufltclr(l->gbf,S);
        cufltclr(l->gbi,S);
        cufltclr(l->gbc,S);
        cufltclr(l->gbo,S);
    }

    float* tmp = (float*)cuallocmem(7,S,float);
    float* dh_next = tmp + 0 * S;
    float* dc_next = tmp + 1 * S;
    float* do_     = tmp + 2 * S;
    float* dc      = tmp + 3 * S;
    float* dcc     = tmp + 4 * S;
    float* di      = tmp + 5 * S;
    float* df      = tmp + 6 * S;

    for (int t = B - 1; t >= 0; t--) {
        backward_gate_kernel<<<blocks,THCNT>>>(
            (const float*)dy[t],(const float*)f[t],(const float*)i[t],
            (const float*)o[t],(const float*)cc[t],(const float*)c[t-1],
            (const float*)c[t],dh_next,dc_next,do_,dc,dcc,di,df,S);

        addoutermul(l->gWo,x[t],(fVec)do_,D,S);
        addoutermul(l->gUo,h[t-1],(fVec)do_,S,S);
        addoutermul(l->gWc,x[t],(fVec)dcc,D,S);
        addoutermul(l->gUc,h[t-1],(fVec)dcc,S,S);
        addoutermul(l->gWi,x[t],(fVec)di,D,S);
        addoutermul(l->gUi,h[t-1],(fVec)di,S,S);
        addoutermul(l->gWf,x[t],(fVec)df,D,S);
        addoutermul(l->gUf,h[t-1],(fVec)df,S,S);

        if (l->use_bias) {
            add_vector_kernel<<<blocks,THCNT>>>((float*)l->gbo,do_,S);
            add_vector_kernel<<<blocks,THCNT>>>((float*)l->gbc,dcc,S);
            add_vector_kernel<<<blocks,THCNT>>>((float*)l->gbi,di,S);
            add_vector_kernel<<<blocks,THCNT>>>((float*)l->gbf,df,S);
        }

        cufltclr(dh_next,S);
        addinnermul((fVec)dh_next,(fVec)df,l->Uf,S,S);
        addinnermul((fVec)dh_next,(fVec)di,l->Ui,S,S);
        addinnermul((fVec)dh_next,(fVec)dcc,l->Uc,S,S);
        addinnermul((fVec)dh_next,(fVec)do_,l->Uo,S,S);

        if (dX != NULL) {
            cufltclr(dx[t],D);
            addinnermul(dx[t],(fVec)df,l->Wf,D,S);
            addinnermul(dx[t],(fVec)di,l->Wi,D,S);
            addinnermul(dx[t],(fVec)dcc,l->Wc,D,S);
            addinnermul(dx[t],(fVec)do_,l->Wo,D,S);
        }
    }

    cufreemem(tmp);
}
