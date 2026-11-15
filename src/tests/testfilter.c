/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"
#include "hann.h"
#include "filter.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
      fprintf(stderr,"Syntax: testfilter <infilename> <outfilename>\n");
      return -1;
    }
    if (strcmp(argv[1],argv[2]) == 0) {
      fprintf(stderr,"Input and output file names must be different\n");
      return -1;
    }

    char* infilename = argv[1];
    char* outfilename = argv[2];
    WAVFILE wfin, wfout, *wfp;
    
    wfp = openWavFile(infilename,"r",&wfin);
    if (wfp == NULL) {
      fprintf(stderr,"Failed to open wav file '%s' for read\n",infilename);
      return -1;
    }

    FILTER filter;
    int rv = initFilter(&filter,4,"h",wfp->sampleRate,240);
    if (rv == -1) {
        fprintf(stderr,"Failed to initialize filter\n");
        closeWavFile(&wfin);
        return -1;
    }
    printf("filter order %d type %c sample rate %d cutoff %d\n",
           filter.order,filter.type,filter.sampleRate,filter.cutoffFreq);
    printf("aCoeff ");
    for (int i = 0; i < MAX_FILTER_ORDER + 1; i++)
        printf("%13.8f ",filter.aCoeff[i]);
    printf("\n");
    printf("bCoeff ");
    for (int i = 0; i < MAX_FILTER_ORDER + 1; i++)
        printf("%13.8f ",filter.bCoeff[i]);
    printf("\n");
    
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
    size_t winSize = (wfin.sampleRate == 8000) ? 160 : 320;
    float window[winSize];
    float rdBuf[winSize];
    float wrBuf[winSize];
    size_t wcnt = 0;

    memset(rdBuf,0,winSize * sizeof(rdBuf[0]));
    memset(wrBuf,0,winSize * sizeof(wrBuf[0]));

    HANNWIN hannWin;
    int rv2 = hannWindowInit(&hannWin,winSize,1);
    if (rv2 == -1) {
        fprintf(stderr,"Failed to initialize hann window of winSize %zu\n",winSize);
        closeWavFile(&wfin);
        closeWavFile(&wfout);    
        return -1; 
    }

    for (size_t cnt = 0; cnt < wfin.numSamples; cnt += winSize / 2) {
        float fltBuf[winSize / 2];
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
            pcm2flt(pcmBuf,fltBuf,samplesRead);
        }
        else
            samplesRead = readWavFile(&wfin,fltBuf,winSize / 2);

        if (samplesRead < winSize / 2) /* Zero the rest of a partial read */
            memset(fltBuf + samplesRead,0,
                   ((winSize / 2) - samplesRead) * sizeof(fltBuf[0]));

        runFiler(&filter,fltBuf,rdBuf + winSize / 2,winSize / 2);

        if (samplesRead < winSize / 2) {
            memset(rdBuf + (winSize / 2) + samplesRead,0,
                   ((winSize / 2) - samplesRead) * sizeof(rdBuf[0]));
        }

        hannWindow(&hannWin,rdBuf,window);

        for (size_t i = 0; i < winSize; i++)
            wrBuf[i] += window[i];

        wcnt = writeWavFile(&wfout,wrBuf,winSize / 2);
        if (wcnt < winSize / 2) {
            fprintf(stderr,"Failed to write to output file\n");
            break;
        }

        memmove(rdBuf,rdBuf + winSize / 2,(winSize / 2) * sizeof(rdBuf[0]));
        memset(rdBuf + winSize / 2,0,(winSize / 2) * sizeof(rdBuf[0]));
        memmove(wrBuf,wrBuf + winSize / 2,(winSize / 2) * sizeof(wrBuf[0]));
        memset(wrBuf + winSize / 2,0,(winSize / 2) * sizeof(wrBuf[0]));
    }
    /* Write the second half of the last frame */
    wcnt = writeWavFile(&wfout,wrBuf,winSize / 2);
    if (wcnt < winSize / 2)
        fprintf(stderr,"Failed to write to output file\n");
    closeWavFile(&wfin);    
    closeWavFile(&wfout);    
}
