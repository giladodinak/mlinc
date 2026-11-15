/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"
#include "hann.h"
#include "lpc.h"
#include "filter.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,"Syntax: testlpc <lpcorder>  <infilename> <outfilename>\n");
        return -1;
    }
    int lpcOrder = atoi(argv[1]);
    if (lpcOrder < 6 || lpcOrder > 16) {
        fprintf(stderr,"LPC order must be between 6 and 16 (inclusive)\n");
        return -1; 
    }
    if (strcmp(argv[2],argv[3]) == 0) {
        fprintf(stderr,"Input and output file names must be different\n");
        return -1;
    }

    char* infilename = argv[2];
    char* outfilename = argv[3];
    WAVFILE wfin, wfout, *wfp;

    wfp = openWavFile(infilename,"r",&wfin);
    if (wfp == NULL) {
        fprintf(stderr,"Failed to open wav file '%s' for read\n",infilename);
        return -1;
    }

    // 20 millisecond window
    int winSize = 20 * wfp->sampleRate / 1000;
    HANNWIN hannWin;
    int rv = hannWindowInit(&hannWin,winSize,1);
    if (rv == -1) {
        fprintf(stderr,"Failed to initialize hann window of winSize %d\n",winSize);
        closeWavFile(&wfin);
        return -1; 
    }
    FILTER filter;
    int rv2 = initFilter(&filter,4,"h",wfp->sampleRate,240);
    if (rv2 == -1) {
        fprintf(stderr,"Failed to initialize filter\n");
        closeWavFile(&wfin);
        return -1;
    }
 
    wfout.audioFormat = 3; // float
    wfout.sampleRate = wfp->sampleRate; // same as input
    wfout.bitDepth = 32;
    wfout.numChannels = 1;  
    wfp = openWavFile(outfilename,"w",&wfout);
    if (wfp == NULL) {
        fprintf(stderr,"Failed to open wav file '%s' for write\n",outfilename);
        closeWavFile(&wfin);
        return -1;
    }
    float window[winSize];
    float window2[winSize];
    float rdBuf[winSize];
    float wrBuf[winSize];
    size_t wcnt = 0;

    memset(rdBuf,0,winSize * sizeof(rdBuf[0]));
    memset(wrBuf,0,winSize * sizeof(wrBuf[0]));

    for (size_t cnt = 0; cnt < wfin.numSamples; cnt += winSize / 2) {
        size_t samplesRead = 0;
        if (wfin.audioFormat != 3) {
            int16_t pcmBuf[winSize / 2];
            if (wfin.audioFormat == 7) {
                uint8_t ulawBuf[winSize / 2];
                samplesRead = readWavFile(&wfin,ulawBuf,winSize / 2);
                ulaw2pcm(ulawBuf,pcmBuf,samplesRead);
            }
            else
                samplesRead = readWavFile(&wfin,pcmBuf,winSize / 2);
            pcm2flt(pcmBuf,rdBuf + winSize / 2,samplesRead);
        }
        else
            samplesRead = readWavFile(&wfin,rdBuf + winSize / 2,winSize / 2);

        if (samplesRead < (size_t) winSize / 2) {
            memset(rdBuf + (winSize / 2) + samplesRead,0,
                   ((winSize / 2) - samplesRead) * sizeof(rdBuf[0]));
        }

        hannWindow(&hannWin,rdBuf,window);

        double lpcCoeff[lpcOrder + 1];      
        double error = computeLPC(window,winSize,lpcOrder,lpcCoeff);
        
        LPCsynthesis(lpcCoeff,lpcOrder,sqrt(error),winSize,window2);
        
        for (size_t i = 0; i < (size_t) winSize; i++)
            wrBuf[i] += window2[i];

        runFiler(&filter,wrBuf,wrBuf,winSize / 2);

        wcnt = writeWavFile(&wfout,wrBuf,winSize / 2);
        if (wcnt < (size_t) winSize / 2) {
            fprintf(stderr,"Failed to write to output file\n");
            break;
        }

        memmove(rdBuf,rdBuf + winSize / 2,(winSize / 2) * sizeof(rdBuf[0]));
        memset(rdBuf + winSize / 2,0,(winSize / 2) * sizeof(rdBuf[0]));
        memmove(wrBuf,wrBuf + winSize / 2,(winSize / 2) * sizeof(wrBuf[0]));
        memset(wrBuf + winSize / 2,0,(winSize / 2) * sizeof(wrBuf[0]));
    }
    /* Filter and write the second half of the last frame */
    runFiler(&filter,wrBuf,wrBuf,winSize / 2);
    wcnt = writeWavFile(&wfout,wrBuf,winSize / 2);
    if (wcnt < (size_t) winSize / 2)
        fprintf(stderr,"Failed to write to output file\n");
    closeWavFile(&wfin);    
    closeWavFile(&wfout);
    return 0;
}
