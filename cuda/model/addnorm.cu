/* Copyright (c) 2026 Gilad Odinak */
/* CUDA layer normalization implementation */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "cuctx.h"
#include "mem.h"
#include "addnorm.h"

static __global__ void addnorm_init_gamma_kernel(float* gamma, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D)
        gamma[j] = 1.0f;
}

extern "C" ADDNORM* addnorm_create()
{
    ADDNORM* l = (ADDNORM*) allocmem(1,1,ADDNORM);
    return l;
}

extern "C" void addnorm_init(ADDNORM* l, int input_dim, int batch_size)
{
    l->B = batch_size;
    l->D = input_dim;

    l->mean  = (fVec)cuallocmem(batch_size,1,float);
    l->sdev  = (fVec)cuallocmem(batch_size,1,float);
    l->xn    = (fArr2D)cuallocmem(batch_size,input_dim,float);
    l->gamma = (fVec)cuallocmem(input_dim,1,float);
    l->beta  = (fVec)cuallocmem(input_dim,1,float);

    int thds = THDCNT;
    int blks = (input_dim + thds - 1) / thds;
    float* gamma = (float*) l->gamma;
    addnorm_init_gamma_kernel<<<blks,thds,0,cuda_stream()>>>(gamma,input_dim);
}

extern "C" void addnorm_free(ADDNORM* l)
{
    if (l == NULL) return;

    cufreemem(l->mean);
    cufreemem(l->sdev);
    cufreemem(l->xn);
    cufreemem(l->gamma);
    cufreemem(l->beta);
    freemem(l);
}
