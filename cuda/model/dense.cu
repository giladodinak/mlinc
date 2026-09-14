/* Copyright (c) 2023-2024 Gilad Odinak */
/* CUDA Dense (feed forward) neural network functions */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cumem.h"
#include "cuflt.h"
#include "arraytypes.h"
#include "array.h"
#include "activation.h"
#include "dense.h"

#define THCNT 256

static __device__ unsigned int dense_hash(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __global__ void dense_init_weights_kernel(float* w, int n, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        unsigned int h1 = dense_hash((unsigned int)i + 0x9e3779b9u);
        unsigned int h2 = dense_hash((unsigned int)i + 0x243f6a88u);

        float u1 = ((float)h1 + 1.0f) / 4294967296.0f;
        float u2 = ((float)h2 + 1.0f) / 4294967296.0f;

        float z = sqrtf(-2.0f * logf(u1)) *
                  cosf(6.2831853071795864769f * u2);

        w[i] = z * scale;
    }
}

static __global__ void add_bias_kernel(float* h,
                                       const float* b,
                                       int B, int S)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * S;

    if (i < n)
        h[i] += b[i % S];
}

static __global__ void bias_gradient_kernel(float* gb,
                                            const float* dy,
                                            int B, int S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < S) {
        float g = 0.0f;
        for (int i = 0; i < B; i++)
            g += dy[i * S + j];
        gb[j] = g;
    }
}

extern "C" DENSE* dense_create(int units, char* activation, int use_bias)
{
    DENSE* l = (DENSE*)calloc(1,sizeof(DENSE));
    if (l == NULL) {
        fprintf(stderr,"dense_create: out of memory\n");
        exit(1);
    }

    l->S = units;
    l->use_bias = use_bias ? 1 : 0;

    if (!strcasecmp("none",activation))    l->activation = 'n';
    if (!strcasecmp("sigmoid",activation)) l->activation = 's';
    if (!strcasecmp("relu",activation))    l->activation = 'r';
    if (!strcasecmp("gelu",activation))    l->activation = 'g';
    if (!strcasecmp("softmax",activation)) l->activation = 'S';

    if (l->activation == 0) {
        free(l);
        fflush(stdout);
        fprintf(stderr,"dense_create: invalid activation '%s'\n",activation);
        exit(-1);
    }

    return l;
}

extern "C" void dense_init(DENSE* l, int input_dim, int batch_size, int training)
{
    l->D = input_dim;
    l->B = batch_size;
    l->training = training ? 1 : 0;

    l->Wx = (fArr2D)cuallocmem(l->D,l->S,float);
    l->h  = (fArr2D)cuallocmem(l->B,l->S,float);

    if (l->training && l->activation == 'g')
        l->z = (fArr2D)cuallocmem(l->B,l->S,float);

    if (l->use_bias)
        l->b = (fVec)cuallocmem(1,l->S,float);

    int n = l->D * l->S;
    float scale = sqrtf(2.0f / (l->D + l->S));
    dense_init_weights_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
        (float*)l->Wx,n,scale);

    if (l->training) {
        l->gWx = (fArr2D)cuallocmem(l->D,l->S,float);
        if (l->use_bias)
            l->gb = (fVec)cuallocmem(1,l->S,float);
    }
}

extern "C" void dense_set_batch_size(DENSE* l, int batch_size)
{
    if (l->B == 0)
        return;

    if (batch_size != l->B) {
        l->B = batch_size;

        cufreemem(l->h);
        l->h = (fArr2D)cuallocmem(l->B,l->S,float);

        if (l->training && l->activation == 'g') {
            cufreemem(l->z);
            l->z = (fArr2D)cuallocmem(l->B,l->S,float);
        }
    }
    else {
        cufltclr(l->h,l->B * l->S);
    }
}

extern "C" void dense_free(DENSE* l)
{
    if (l == NULL)
        return;

    cufreemem(l->h);
    cufreemem(l->z);
    cufreemem(l->Wx);
    cufreemem(l->gWx);
    cufreemem(l->b);
    cufreemem(l->gb);

    free(l);
}

extern "C" void dense_reset(DENSE* l)
{
    (void)l;
}

extern "C" fArr2D dense_forward(DENSE* restrict l,
                                const fArr2D restrict X,
                                int lyr)
{
    (void)lyr;

    matmul(l->h,X,l->Wx,l->B,l->D,l->S);

    if (l->use_bias) {
        int n = l->B * l->S;
        add_bias_kernel<<<(n + THCNT - 1) / THCNT,THCNT>>>(
            (float*)l->h,(const float*)l->b,l->B,l->S);
    }

    switch (l->activation) {
        case 's':
            sigmoid(l->h,l->B,l->S);
            break;
        case 'r':
            relu(l->h,l->B,l->S);
            break;
        case 'g':
            if (l->training)
                cufltcpy(l->z,l->h,l->B * l->S);
            gelu(l->h,l->B,l->S);
            break;
        case 'S':
            softmax(l->h,l->B,l->S);
            break;
    }

    return l->h;
}

extern "C" void dense_backward(DENSE* restrict l,
                               fArr2D restrict dy,
                               const fArr2D restrict X,
                               fArr2D restrict dx,
                               int lyr)
{
    (void)lyr;

    switch (l->activation) {
        case 's':
            d_sigmoid(dy,l->h,l->B,l->S);
            break;
        case 'r':
            d_relu(dy,l->h,l->B,l->S);
            break;
        case 'g':
            d_gelu(dy,l->z,l->B,l->S);
            break;
    }

    Tmatmul(l->gWx,X,dy,l->D,l->B,l->S);

    if (l->use_bias) {
        bias_gradient_kernel<<<(l->S + THCNT - 1) / THCNT,THCNT>>>(
            (float*)l->gb,(const float*)dy,l->B,l->S);
    }

    if (dx != NULL)
        matmulT(dx,dy,l->Wx,l->B,l->S,l->D);
}
