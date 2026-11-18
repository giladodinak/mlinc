/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include "array.h"
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"
#include "hann.h"
#include "lpc.h"
#include "lsp.h"
#include "filter.h"

/* Reads a feature file created by timitfeat, one row per frame, and
 * creates an audio file from it by LPC synthesis of the LPC/LSP features.
 */
#define TIMIT_FEAT_CNT      14
#define MAX_SAMPLES 1000 /* 10 seconds */
float samples[MAX_SAMPLES][TIMIT_FEAT_CNT];
int samplecnt = 0;

/* Rate at which timitfeat computes features */
int audioSampleRate = 8000;
int frameSize = 10; // in milliseconds
int winSize = 160;  // in samples, two frames
int featcnt = 0;    // sample vector dimension

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Syntax: feat2audio <feat infilename> <wav outfilename>\n");
        return 0;
    }
    WAVFILE wfout, *wfp;
    char* featfilename = argv[1];
    char* wavfilename = argv[2];

    if (strcmp(featfilename,wavfilename) == 0) {
        fprintf(stderr,"Input and output file names must be different\n");
        return -1;
    }

    FILE *fp = fopen(featfilename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",featfilename);
        return -1;
    }

    int lineno = 0;
    int vecinx = 0;
    for (;;) {
        char buffer[20000];
        char *line = fgets(buffer,sizeof(buffer),fp);
        if (line == NULL) /* End of file */
            break;
        lineno++;
        int len = strlen(line);
        if (len == 0 || line[len - 1] != '\n') {
            fprintf(stderr,"In file '%s' line %d: line too long "
                                          "or malformed\n",featfilename,lineno);
            fclose(fp);
            return -1;
        }
        /* Remove all whitespace in current line */
        len = 0;
        for (int i = 0; line[i] != '\0'; i++)
            if (!isspace(line[i]))
                line[len++] = line[i];
        line[len] = '\0';
        const char* header = "frame,"; /* Header line starts with this */
        if (strncmp(line,header,strlen(header)) == 0) { /* Header */
            /* Number of values per frame is number of columns - 1 */
            featcnt = 0;
            for (int i = 0; line[i] != '\0'; i++)
                if (line[i] == ',')
                    featcnt++;
            if (featcnt != TIMIT_FEAT_CNT) {
                fprintf(stderr,"In file '%s' line %d: %d values per frame, "
                        "should be %d\n",featfilename,lineno,featcnt,
                        TIMIT_FEAT_CNT);
                fclose(fp);
                return -1;
            }
            continue;
        }
        if (featcnt == 0) {
            fprintf(stderr,"In file '%s' line %d: missing header line\n",
                                                          featfilename,lineno);
            fclose(fp);
            return -1;
        }
        /* Replace commas with spaces */
        for (int i = 0; line[i] != '\0'; i++)
            if (line[i] == ',')
                line[i] = ' ';
        /* Read frame index */
        char* end = line;
        int frame = strtol(line,&end,10);
        if (line == end || frame != vecinx) {
            fprintf(stderr,"In file '%s' line %d: malformed line\n",
                                                          featfilename,lineno);
            fclose(fp);
            return -1;
        }
        line = end;
        /* Read frame's values */
        for (int j = 0; j < featcnt; j++) {
            float feat;
            feat = strtof(line,&end);
            if (line == end) {
                fprintf(stderr,
                        "In file '%s' line %d: malformed feature #%d\n",
                        featfilename,lineno,j);
                fclose(fp);
                return -1;
             }
             samples[vecinx][j] = feat; 
             line = end;
        }
        vecinx++;
        samplecnt = vecinx;
        if (vecinx >= MAX_SAMPLES) {
            fprintf(stderr,
                    "In file '%s' line %d: reached %d samples, "
                    "ignoring the rest\n",featfilename,lineno,MAX_SAMPLES);
            break;
        }
    }
    fclose(fp);
    
    if (samplecnt == 0) {
        fprintf(stderr,
                "no vaild sample data in file '%s' - aborting\n",featfilename);
        return -1;
    }

    printf("winSize %d samples, frameSize %d millisec\n",winSize,frameSize);
 
    wfout.audioFormat = 3; // float
    wfout.sampleRate = audioSampleRate;
    wfout.bitDepth = 32;
    wfout.numChannels = 1;  
    wfp = openWavFile(wavfilename,"w",&wfout);
    if (wfp == NULL)
        return -1;

    float window[winSize];
    float wrBuf[winSize];

    memset(wrBuf,0,winSize * sizeof(wrBuf[0]));

    for (int i = 0; i < samplecnt; i++) {
        // double zcr = samples[i][0]; // not used
        double sigma = exp(samples[i][1] * -30.0) - 1e-07;
        int lpcOrder = featcnt - 2; // zcr, sigma
        double lspCoeff[lpcOrder + 1];
        for (int k = 0; k < lpcOrder; k++)
            lspCoeff[k] = samples[i][k + 2];
        lspCoeff[lpcOrder] = 0.0;
        double lpcCoeff[lpcOrder + 1];
        lsp2lpc(lspCoeff,lpcCoeff,lpcOrder);
        LPCsynthesis(lpcCoeff,lpcOrder,sigma,winSize,window);
        for (int j = 0; j < winSize; j++)
            wrBuf[j] += window[j];
        writeWavFile(&wfout,wrBuf,winSize / 2);
        memmove(wrBuf,wrBuf + winSize / 2,(winSize / 2) * sizeof(wrBuf[0]));
        memset(wrBuf + winSize / 2,0,(winSize / 2) * sizeof(wrBuf[0]));
    }
    closeWavFile(&wfout);
    return 0;
}
