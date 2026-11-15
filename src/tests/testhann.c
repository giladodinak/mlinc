/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"
#include "hann.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,"Syntax: testhann <windowsize> <infilename> <outfilename>\n");
        return 0;
    }
    WAVFILE wfin, wfout, *wfp;
    char* infilename = argv[2];
    char* outfilename = argv[3];

    int winSize = atoi(argv[1]);
    if (winSize < 2 || winSize > MAX_WINDOW_SIZE || winSize % 2 != 0) {
        fprintf(stderr,"Window size must be even number between 2 and %d (inclusive)\n",MAX_WINDOW_SIZE);
        return 0; 
    }
    HANNWIN hannWin;
    int rv = hannWindowInit(&hannWin,winSize,1);
    if (rv == -1) {
         fprintf(stderr,"Failed to initialize hann window of winSize %d\n",winSize);
        return 0; 
    }

    if (strcmp(argv[2],argv[3]) == 0) {
        fprintf(stderr,"Input and output file names must be different\n");
        return 0;
    }
    wfp = openWavFile(infilename,"r",&wfin);
    if (wfp == NULL) 
        return 0;
    
    wfout.audioFormat = 3; // float
    wfout.sampleRate = wfp->sampleRate; // same as input
    wfout.bitDepth = 32;
    wfout.numChannels = 1;    
    wfp = openWavFile(outfilename,"w",&wfout);
    if (wfp == NULL) {
        closeWavFile(&wfin);
        return 0;
    }
    float window[winSize];
    float rdBuf[winSize];
    float wrBuf[winSize];

    memset(rdBuf,0,winSize * sizeof(rdBuf[0]));
    memset(wrBuf,0,winSize * sizeof(wrBuf[0]));

    for (int cnt = 0; cnt < (int) wfin.numSamples; cnt += winSize / 2) {
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

        for (int i = 0; i < winSize; i++)
            wrBuf[i] += window[i];

        if (cnt > 0)
            writeWavFile(&wfout,wrBuf,winSize / 2);

        memmove(rdBuf,rdBuf + winSize / 2,(winSize / 2) * sizeof(rdBuf[0]));
        memset(rdBuf + winSize / 2,0,(winSize / 2) * sizeof(rdBuf[0]));
        memmove(wrBuf,wrBuf + winSize / 2,(winSize / 2) * sizeof(wrBuf[0]));
        memset(wrBuf + winSize / 2,0,(winSize / 2) * sizeof(wrBuf[0]));
    }
    closeWavFile(&wfin);        
    closeWavFile(&wfout);        
    return 0;
}
