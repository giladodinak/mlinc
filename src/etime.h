/* Copyright (c) 2023-2024 Gilad Odinak */
/* Time measurment functions            */
#ifndef ETIME_H
#define ETIME_H
#include "float.h"

#ifdef __cplusplus
extern "C" { /* nvcc is a c++ compiler */
#endif

extern float current_time();

static inline float elapsed_time(float start_time)
{
    return current_time() - start_time;
}

char* date_time(char buffer[20]);

#ifdef __cplusplus
}
#endif

#endif
