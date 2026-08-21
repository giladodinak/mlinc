/* Copyright (c) 2023-2024 Gilad Odinak */
#ifndef BLASCPU_H
#define BLASCPU_H

#include <sched.h>
#include <string.h>
#include <cblas.h>

/*Configures OpenBLAS to use the specified CPUs.
 *
 * Creates one OpenBLAS thread per CPU and pins
 * each thread to the corresponding CPU in cpu[].
 *
 * cpu - array of logical CPU numbers
 * n   - number of CPUs and OpenBLAS threads
 * 
 * Usage example (hyperthreadingi cores)
 *
 * int pcores[] = { 0, 2, 4, 6, 8, 10 };
 * openblas_use_cpus(pcores, sizeof(pcores) / sizeof(pcores[0]));
 *
 */
static void openblas_use_cpus(const int *cpu, int n)
{
#ifdef USE_BLAS
    openblas_set_num_threads(n);
#ifndef __APPLE__
    for (int i = 0; i < n; i++) {
        cpu_set_t set;
        memset(&set,0,sizeof(set));

        set.__bits[cpu[i] / (8 * sizeof(set.__bits[0]))] |=
            1UL << (cpu[i] % (8 * sizeof(set.__bits[0])));

        openblas_setaffinity(i,sizeof(set),&set);
    }
#endif
#else
    (void) cpu; (void) n;
#endif
}

#endif
