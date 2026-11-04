/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include "float.h"
#include "sphere.h"
#include "ulaw.h"
#include "pcm.h"

/* SPHERE header exmaple
NIST_1A 
   1024
sample_coding -s3 pcm
channel_count -i 1
sample_rate -i 16000
sample_n_bytes -i 2
sample_count -i 3356677
sample_byte_format -s2 10
end_head
*/

static void hdr2sf(SPHFILE* sf, const char* header)
{
    char hdr[1024];
    memcpy(hdr,header,1024);
    hdr[1023] = '\0';
    long sampleNbytes = 0, sampleCount = 0;
    for (char *s = hdr; *s != '\0';) {
        char e1[1024] = "", e2[1024] = "", e3[1024] = "";
        char* p = index(s,'\n');
        if (p == NULL)
            break;
        *p = '\0';
        sscanf(s,"%s %s %s",e1,e2,e3);
        if (strncmp(e1,"end_head",1023) == 0)
            break;
        if (strncmp(e1,"sample_coding",1023) == 0) {
            if (strncasecmp(e3,"pcm",3) == 0)
                sf->audioFormat = 1;
            else
            if (strncasecmp(e3,"float",5) == 0)
                sf->audioFormat = 3;
            else
            if (strncasecmp(e3,"uLaw",4) == 0)
                sf->audioFormat = 7;
            else
                sf->audioFormat = 0;                
        }
        else
        if (strncmp(e1,"channel_count",1023) == 0)
            sf->numChannels = (uint16_t) atol(e3);
        else
        if (strncmp(e1,"sample_rate",1023) == 0)
            sf->sampleRate = (uint32_t) atol(e3);
        else
        if (strncmp(e1,"sample_n_bytes",1023) == 0)
            sampleNbytes = atol(e3);
        else
        if (strncmp(e1,"sample_count",1023) == 0)
            sampleCount = atol(e3);
        else
        if (strncmp(e1,"sample_byte_format",1023) == 0)
            sf->endianess = (e3[0] == '1') ? 'b' : 'l';
        s = p + 1; // does not work as 3rd part of for loop
    }
    sf->bitDepth = sampleNbytes * 8;
    sf->dataSize = sampleCount * sampleNbytes;
    if (sf->audioFormat == 0 && sampleNbytes == 2)
        sf->audioFormat = 1; // Assume PCM
}

void printSphereFileInfo(SPHFILE *sf, char *mode)
{
    char *format, *endianess;
    switch (sf->audioFormat) {
        case 1: format = "PCM"; break;
        case 3: format = "float"; break;
        case 7: format = "uLaw"; break;
        default: format = "unknown";
    }
    switch (sf->endianess) {
        case 'l': endianess = "little-endinan"; break;
        case 'b': endianess = "big-endinan"; break;
        default: endianess = "unknown";
    }
    printf("Audio Format: %s\n",format);
    printf("Endianess: %s\n",endianess);
    printf("Sample Rate: %d Hz\n", sf->sampleRate);
    printf("Bit Depth: %d bits\n", sf->bitDepth);
    printf("Number of Channels: %d\n", sf->numChannels);
    if (*mode == 'w') return;
    printf("Number of Samples per Channel: %d\n", sf->numSamplesPerChannel);
    printf("Total Number of Samples: %d\n", sf->numSamples);
    printf("Data Size: %d bytes\n", sf->dataSize);
    fflush(stdout);
}

SPHFILE* openSphereFile(const char* filename, char *mode, SPHFILE *sf)
{
    FILE* fileHandle = NULL;
    char hdr[1024];
    memset(hdr,0,sizeof(hdr));
    memset(sf,0,sizeof(*sf));
    
    if (*mode != 'r') {
        fprintf(stderr,"In openSphereFile('%s'): invalid mode '%s'; only 'r' supported.\n",filename,mode);
        return NULL;
    }
    //printf("Openning '%s' for %s.\n",filename,(*mode == 'r') ? "read" : "write");
    
    if (*mode == 'r') {
        fileHandle = fopen(filename,"rb");
        if (fileHandle == NULL) {
            fprintf(stderr,"In openSphereFile('%s'): failed to open the file for read.\n",filename);
            return NULL;
        }
        if (fread(hdr,sizeof(hdr[0]),sizeof(hdr),fileHandle) != sizeof(hdr)) {
            fprintf(stderr,"In openSphereFile('%s'): failed to read SPHERE header.\n",filename);
            fclose(fileHandle);
            return NULL;
        }
        if (strncmp(hdr,"NIST_1A\n",8) || strncmp(hdr + 8,"   1024\n",8)) {
            fprintf(stderr,"In openSphereFile('%s'): not in NIST_1A format.\n",filename);
            fclose(fileHandle);
            return NULL;
        }
        hdr2sf(sf,hdr);
        sf->numSamplesPerChannel = sf->dataSize / (sf->bitDepth / 8);
        sf->numSamples = sf->numSamplesPerChannel * sf->numChannels;
        //printSphereFileInfo(sf,mode);
        if (sf->audioFormat != 1) {
            fprintf(stderr,"In openSphereFile('%s'): unsupported audio format %d; only PCM (1) supported.\n",filename,sf->audioFormat);
            fclose(fileHandle);
            return NULL;
        }
        sf->mode = *mode;
        sf->fileHandle = fileHandle;
    }
    return sf;
}

SPHFILE* closeSphereFile(SPHFILE* sf) 
{
    int rv = 0;
    FILE *fileHandle = sf->fileHandle;
    sf->fileHandle = NULL;
    if (fileHandle == NULL)
      return sf;
    rv = fclose(fileHandle);
    if (rv != 0) {
        fprintf(stderr,"In closeSphereFile(): failed to close the file.\n");
        return NULL;
    }
    return sf;
}

size_t readSphereFile(SPHFILE* sf, void *buffer, size_t numSamples) 
{
    return fread(buffer,sf->bitDepth/8,numSamples,sf->fileHandle);
}

size_t seekSphereFile(SPHFILE* sf, size_t offsetSamples)
{
    if (sf->mode != 'r')
        return -1;
    long pos = 1024 + offsetSamples * sf->bitDepth / 8;
    return fseek(sf->fileHandle,pos,SEEK_SET);
}

size_t readSphereAudio(SPHFILE* sf, float *fltBuf, size_t numSamples) 
{
    size_t bufSize = 1024;
    size_t cnt = 0;
    while (cnt < numSamples) {
        size_t toRead = numSamples - cnt;
        size_t readSize = (toRead  > bufSize) ? bufSize : toRead;
        size_t samplesRead = 0;
        if (sf->audioFormat != 3) {
            int16_t pcmBuf[bufSize];
            if (sf->audioFormat == 7) {
                uint8_t ulawBuf[bufSize];
                samplesRead = readSphereFile(sf,ulawBuf,readSize);
                ulaw2pcm(ulawBuf,pcmBuf,samplesRead);
            }
            else
                samplesRead = readSphereFile(sf,pcmBuf,readSize);
            if (sf->endianess == 'b') {
                for (size_t i = 0; i < samplesRead; i++)
                    pcmBuf[i] = __builtin_bswap16(pcmBuf[i]);
            } 
            pcm2flt(pcmBuf,fltBuf + cnt,samplesRead);
        }
        else {
            samplesRead = readSphereFile(sf,fltBuf,readSize);
            if (sf->endianess == 'b') {
                for (size_t i = 0; i < samplesRead; i++)
                    fltBuf[i] = __builtin_bswap32(fltBuf[i]);
            }
        }
        cnt += samplesRead;
        if (samplesRead < bufSize)
            break;
    }
    return cnt;
}

