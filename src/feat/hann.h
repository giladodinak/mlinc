/* Copyright (c) 2023-2024 Gilad Odinak */
#ifndef HANN_H
#define HANN_H

#define MAX_WINDOW_SIZE 1024
typedef struct hannwin_s {
    int winSize;  /* Must be multiple of 2       */
    int periodic; /* 1 - periodic, 0 - symmetric */
    double coeff[MAX_WINDOW_SIZE/2 + 1];
} HANNWIN;

/* Initializes a Hann window of winSize samples (even, 2..MAX_WINDOW_SIZE).
 *
 * If periodic is not zero, creates a periodic window, suited for spectral
 * analysis and overlap-add; with 50% overlap, consecutive windows sum to
 * exactly one. Otherwise, creates a symmetric window.
 *
 * Returns 0 on success, -1 if winSize is invalid.
 */
int hannWindowInit(HANNWIN* hannWin, int winSize, int periodic);

/* Multiplies winSize samples of inData by the window, into outData.
 */
void hannWindow(HANNWIN* hannWin, const float* inData, float* outData);

#endif
