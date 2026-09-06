/* Copyright (c) 2026 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include "float.h"
#include "etime.h"
#include "random.h"
#include "mem.h"

#define ASIZE 512

float a[ASIZE][ASIZE];
float b[ASIZE][ASIZE];
float c[ASIZE][ASIZE];


int main (int argc, char **argv) 
{
    int cuda = 0;
    int iter = 0;
    if (argc == 1) {
        fprintf(stderr,"syntax: cudatest [-cuda] [iterations]\n");
        return 1;
    }
    if (argc >= 2) {
        if (argc >= 3)
            iter = atoi(argv[2]);      
        if (strcmp(argv[1],"-cuda") == 0)
            cuda = 1;
        else
            iter = atoi(argv[1]);
    }
    printf("matrix size %d X %d, cuda %s iterations %d\n",ASIZE,ASIZE,(cuda)?"true":"false",iter);        

    for (int i = 0; i < ASIZE; i++) {
        for (int j = 0; j < ASIZE; j++) {
            a[i][j] = nrand(0.0,0.8);
            b[i][j] = nrand(0.0,0.4);
        }
    }    

#ifdef USE_CUDA
    float* da = NULL;
    float* db = NULL;
    float* dc = NULL;
    cublasHandle_t handle;

    if (cuda) {
        da = (float*) cuallocmem(ASIZE,ASIZE,float);
        db = (float*) cuallocmem(ASIZE,ASIZE,float);
        dc = (float*) cuallocmem(ASIZE,ASIZE,float);

        cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice);
        cudaMemcpy(db,b,sizeof(b),cudaMemcpyHostToDevice);

        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,"\ncuBLAS initialization failed\n");
            return 1;
        }
    }
#endif

    float start_time = current_time();
    for (int i = 0; i < iter; i++) {
        if (cuda) {
#ifdef USE_CUDA
            float alpha = 1.0;
            float beta = 1.0;

            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                        ASIZE,ASIZE,ASIZE,
                        &alpha,(const float*) db,ASIZE,
                        (const float*) da,ASIZE,
                        &beta,(float*) dc,ASIZE);
#else
            fprintf(stderr,"\nCUDA not supported\n");
            return 1;
#endif
        }
        else {    
            for (int i = 0; i < ASIZE; i++) {
                for (int j = 0; j < ASIZE; j++) {
                    for (int k = 0; k < ASIZE; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }

#ifdef USE_CUDA
    if (cuda) {
        cudaDeviceSynchronize();
        cublasDestroy(handle);
        cufreemem(da);
        cufreemem(db);
        cufreemem(dc);
    }
#endif

    float time_total = elapsed_time(start_time);
    float time_iter_us = time_total * 1000000 / iter;
    printf("Elapsed time %g seconds, time per iteration %.f microseconds\n",time_total,time_iter_us);
    return 0;
}
