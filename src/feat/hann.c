/* Copyright (c) 2023-2024 Gilad Odinak */
#include "hann.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Initializes a Hann window of winSize samples (even, 2..MAX_WINDOW_SIZE).
 *
 * If periodic is not zero, creates a periodic window, suited for spectral
 * analysis and overlap-add; with 50% overlap, consecutive windows sum to
 * exactly one. Otherwise, creates a symmetric window.
 *
 * Returns 0 on success, -1 if winSize is invalid.
 */
int hannWindowInit(HANNWIN* hannWin, int winSize, int periodic)
{
    if (winSize < 2 || winSize > MAX_WINDOW_SIZE)
        return -1;
    if (winSize % 2 != 0)
        return -1;
    hannWin->winSize = winSize;
    hannWin->periodic = (periodic) ? 1 : 0;
    int len = (periodic) ? winSize : winSize - 1; /* cosine period */
    int cnt = (periodic) ? winSize / 2 + 1 : winSize / 2;
    for (int n = 0; n < cnt; n++)
        hannWin->coeff[n] = 0.5 * (1 - cos((2 * M_PI * n) / len));
    return 0;
}

/* Multiplies winSize samples of inData by the window, into outData.
 */
void hannWindow(HANNWIN* hannWin, const float* inData, float* outData)
{
    int winSize = hannWin->winSize;
    const double *coeff = hannWin->coeff;
    int last = (hannWin->periodic) ? winSize : winSize - 1;
    for (int i = 0; i < winSize / 2; i++)
        outData[i] = (float) (inData[i] * coeff[i]);
    for (int i = winSize / 2; i < winSize; i++)
        outData[i] = (float) (inData[i] * coeff[last - i]);
}
