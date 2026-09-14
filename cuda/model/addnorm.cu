/* Copyright (c) 2026 Gilad Odinak */
/* CUDA layer normalization implementation */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "cumem.h"
#include "addnorm.h"

static __global__ void addnorm_init_gamma_kernel(float* gamma, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D)
        gamma[j] = 1.0f;
}

extern "C" ADDNORM* addnorm_create()
{
    ADDNORM* l = (ADDNORM*)calloc(1,sizeof(ADDNORM));
    if (l == NULL) {
        fprintf(stderr,"addnorm_create: out of memory\n");
        exit(1);
    }
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

    int threads = THCNT;
    addnorm_init_gamma_kernel<<<(input_dim + threads - 1) / threads,threads>>>(
        (float*)l->gamma,input_dim);
}

extern "C" void addnorm_free(ADDNORM* l)
{
    if (l == NULL)
        return;

    cufreemem(l->mean);
    cufreemem(l->sdev);
    cufreemem(l->xn);
    cufreemem(l->gamma);
    cufreemem(l->beta);
    free(l);
}
