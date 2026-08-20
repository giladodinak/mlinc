/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#ifdef USE_DOUBLE
#define GEMM cblas_dgemm
#else
#define GEMM cblas_sgemm
#endif
#endif

#include "float.h"
#include "etime.h"
#include "random.h"

#define ASIZE 512

float a[ASIZE][ASIZE];
float b[ASIZE][ASIZE];
float c[ASIZE][ASIZE];


int main (int argc, char **argv) 
{
    int blas = 0;
    int iter = 0;
    if (argc == 1) {
        fprintf(stderr,"syntax: cblastest [-blas] [iterations]\n");
        return 1;
    }
    if (argc >= 2) {
        if (argc >= 3)
            iter = atoi(argv[2]);      
        if (strcmp(argv[1],"-blas") == 0)
            blas = 1;
        else
            iter = atoi(argv[1]);
    }
    printf("matrix size %d X %d, blas %s iterations %d\n",ASIZE,ASIZE,(blas)?"true":"false",iter);        
            
    for (int i = 0; i < ASIZE; i++) {
        for (int j = 0; j < ASIZE; j++) {
            a[i][j] = nrand(0.0,0.8);
            b[i][j] = nrand(0.0,0.4);
        }
    }    

    float start_time = current_time();
    for (int i = 0; i < iter; i++) {
        if (blas) {
#ifdef USE_BLAS
            GEMM(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                ASIZE,ASIZE,ASIZE,
                1.0,(const float *) a,ASIZE,
                (const float *) b,ASIZE,
                1.0,(float *) c,ASIZE);
#else
            fprintf(stderr,"\nBLAS not supported\n");
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
    float time_total = elapsed_time(start_time);
    float time_iter_us = time_total * 1000000 / iter;
    printf("Elapsed time %g seconds, time per iteration %.f microseconds\n",time_total,time_iter_us);
    return 0;
}
