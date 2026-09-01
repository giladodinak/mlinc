/* Copyright (c) 2023-2024 Gilad Odinak */
/* Random number generation routines    */
#include "random.h"

int32_t lrng_seed = 96431; /* Prime number */
void init_lrng(int seed)
{
    int32_t m = 0x7FFFFFFF;
    seed = seed & m;
    if (seed == 0 || seed == m) seed = 1;
    lrng_seed = seed;
}

int get_lrng_seed(void)
{
    return lrng_seed;
}
