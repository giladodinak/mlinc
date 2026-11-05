/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "wav.h"
#include "sphere.h"
#include "ulaw.h"
#include "pcm.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"Syntax: sph2wav <infilename> <outfilename>\n");
        return -1;
    }
    if (strcmp(argv[1],argv[2]) == 0) {
        fprintf(stderr,"Input and output file names must be different\n");
        return -1;
    }

    char* infilename = argv[1];
    char* outfilename = argv[2];
    SPHFILE sfin, *sfp;
    WAVFILE wfout, *wfp;

    sfp = openSphereFile(infilename,"r",&sfin);
    if (sfp == NULL) 
        return -1;

    if (sfin.audioFormat != 1 && sfin.audioFormat != 3 && sfin.audioFormat != 7) {
        fprintf(stderr,"Unsupported input audio format %d\n", sfin.audioFormat);
        closeSphereFile(&sfin);
        return -1;
    }
    if (sfin.numChannels != 1) {
        fprintf(stderr,"Only mono input supported; input has %d channels\n",sfin.numChannels);
        closeSphereFile(&sfin);
        return -1;
    }
    if (sfin.sampleRate != 8000 && sfin.sampleRate != 16000) {
        fprintf(stderr,"Unsupported input audio sample rate %d\n", sfin.sampleRate);
        closeSphereFile(&sfin);
        return -1;
    }
    
    wfout.audioFormat = 3; /* float */
    wfout.sampleRate = sfp->sampleRate; /* same as input */
    wfout.bitDepth = 32;
    wfout.numChannels = 1;  
    wfp = openWavFile(outfilename,"w",&wfout);
    if (wfp == NULL) {
        closeSphereFile(&sfin);
        return -1;
    }
    int sampleRate = sfin.sampleRate;
    size_t bufSize = (sampleRate == 8000) ? 160 : 320;

    for (size_t cnt = 0; cnt < sfin.numSamples; cnt += bufSize) {
        float fltBuf[bufSize];
        size_t samplesRead = 0;
        if (sfin.audioFormat != 3) {
            int16_t pcmBuf[bufSize];
            if (sfin.audioFormat == 7) {
                uint8_t ulawBuf[bufSize];
                samplesRead = readSphereFile(&sfin,ulawBuf,bufSize);
                ulaw2pcm(ulawBuf,pcmBuf,samplesRead);
            }
            else {
                samplesRead = readSphereFile(&sfin,pcmBuf,bufSize);
                if (sfin.endianess == 'b') {
                    for (size_t i = 0; i < samplesRead; i++)
                        pcmBuf[i] = __builtin_bswap16(pcmBuf[i]);
                }
            }
            pcm2flt(pcmBuf,fltBuf,samplesRead);
        }
        else {
            samplesRead = readSphereFile(&sfin,fltBuf,bufSize);
            if (sfp->endianess == 'b') {
                uint32_t *bits = (uint32_t*) fltBuf;
                for (size_t i = 0; i < samplesRead; i++)
                    bits[i] = __builtin_bswap32(bits[i]);
            }
        } 

        if (samplesRead < bufSize) { /* Output always multiple of bufSize */
            memset(fltBuf + samplesRead,0,
                     (bufSize - samplesRead) * sizeof(fltBuf[0]));
        }
        size_t samplesWritten = writeWavFile(&wfout,fltBuf,bufSize);
        if (samplesWritten < bufSize) {
            fprintf(stderr,"Failed to write to output file - aborting\n");
            break;
        }
    }
    closeSphereFile(&sfin);
    if (closeWavFile(&wfout) == NULL)
        return -1;
}
