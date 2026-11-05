/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "wav.h"

#pragma pack(push, 1) // Ensure the structure is packed without padding
typedef struct {
    // RIFF header
    char riffHeader[4];     // "RIFF"
    uint32_t fileSize;      // Total file size - 8
    // WAV format
    char wavHeader[4];      // "WAVE"
    char fmtHeader[4];      // "fmt "
    uint32_t fmtSize;       // Size of the fmt chunk (always 16, no extension)
    uint16_t audioFormat;   // Audio format (1:PCM 3:float 7:uLaw)
    uint16_t numChannels;   // Number of channels
    uint32_t sampleRate;    // Sample rate
    uint32_t byteRate;      // Byte rate (SampleRate * NumChannels * BitsPerSample / 8)
    uint16_t blockAlign;    // Block align (NumChannels * BitsPerSample / 8)
    uint16_t bitsPerSample; // Bits per sample (8, 16, 32)
    // Data chunk
    char dataHeader[4];     // "data"
    uint32_t dataSize;      // Size of the data chunk
} WAVHDR;
#pragma pack(pop)

static void hdr2wf(WAVFILE* wf, const char* hdr)
{
    const WAVHDR *wh = (WAVHDR *) hdr;
    wf->audioFormat = wh->audioFormat; // 1:PCM 7:uLaw
    wf->endianess = 'l'; // always little-endian
    wf->numChannels = wh->numChannels;
    wf->sampleRate = wh->sampleRate;
    wf->bitDepth = wh->bitsPerSample;
    wf->dataSize = wh->dataSize;
}

static void wf2hdr(char* hdr, const WAVFILE* wf)
{
    WAVHDR wh = {
        .riffHeader = {'R','I','F','F'},
        .fileSize = wf->dataSize + sizeof(WAVHDR) - 8,
        .wavHeader = {'W','A','V','E'},
        .fmtHeader = {'f','m','t',' '},
        .fmtSize = 16,
        .audioFormat = wf->audioFormat,
        .numChannels = wf->numChannels,
        .sampleRate = wf->sampleRate,
        .byteRate = wf->sampleRate * wf->numChannels * (wf->bitDepth / 8),
        .blockAlign = wf->numChannels * (wf->bitDepth / 8),
        .bitsPerSample = wf->bitDepth,
        .dataHeader = {'d','a','t','a'},
        .dataSize = wf->dataSize
    };
    memcpy(hdr,&wh,sizeof(wh));
}

static void printwf(WAVFILE *wf, char *mode)
{
    char *format, *endianess;
    switch (wf->audioFormat) {
        case 1: format = "PCM"; break;
        case 3: format = "float"; break;
        case 7: format = "uLaw"; break;
        default: format = "unknown";
    }
    switch (wf->endianess) {
        case 'l': endianess = "little-endian"; break;
        case 'b': endianess = "big-endian"; break;
        default: endianess = "unknown";
    }
    printf("Audio Format: %s\n",format);
    printf("Endianess: %s\n",endianess);
    printf("Sample Rate: %d Hz\n", wf->sampleRate);
    printf("Bit Depth: %d bits\n", wf->bitDepth);
    printf("Number of Channels: %d\n", wf->numChannels);
    if (*mode == 'w') return;
    printf("Number of Samples per Channel: %d\n", wf->numSamplesPerChannel);
    printf("Total Number of Samples: %d\n", wf->numSamples);
    printf("Data Size: %d bytes\n", wf->dataSize);
}

WAVFILE* openWavFile(const char* filename, char* mode, WAVFILE* wf) 
{
    FILE* fileHandle = NULL;
    char hdr[sizeof(WAVHDR)];
    
    if (*mode != 'r' && *mode != 'w') {
        fprintf(stderr,"In openWavFile('%s'): invalid mode '%s'; only 'r' and 'w' supported.\n",filename,mode);
        return NULL;
    }
    
    if (*mode == 'r') {
        fileHandle = fopen(filename,"rb");
        if (fileHandle == NULL) {
            fprintf(stderr,"In openWavFile('%s'): failed to open the file for read.\n",filename);
            return NULL;
        }
        if (fread(hdr,sizeof(hdr[0]),sizeof(hdr),fileHandle) != sizeof(hdr)) {
            fprintf(stderr,"In openWavFile('%s'): failed to read WAV header.\n",filename);
            fclose(fileHandle);
            return NULL;
        }
        if (hdr[0]!='R' || hdr[1]!='I' || hdr[2]!='F' || hdr[3]!='F' ||
            hdr[8]!='W' || hdr[9]!='A' || hdr[10]!='V' || hdr[11]!='E') {
            fprintf(stderr,"In openWavFile('%s'): not a WAV file.\n",filename);
            fclose(fileHandle);
            return NULL;
        }
        hdr2wf(wf,hdr);
        if (wf->audioFormat != 1 && wf->audioFormat != 3 && wf->audioFormat != 7) {
            fprintf(stderr,"In openWavFile('%s'): unsupported audio format %d; only PCM (1), float (3) and uLaw (7) supported.\n",filename,wf->audioFormat);
            fclose(fileHandle);
            return NULL;
        }
        wf->numSamples =  wf->dataSize / (wf->bitDepth / 8);
        wf->numSamplesPerChannel = wf->numSamples / wf->numChannels;
        wf->fileHandle = fileHandle;
        wf->mode = *mode;
        printwf(wf,mode);
    }
    if (*mode == 'w') {
        wf->endianess = 'l'; // always little-endian
        printwf(wf,mode);
        fileHandle = fopen(filename,"wb");
        if (fileHandle == NULL) {
            fprintf(stderr,"In openWavFile('%s'): failed to open the file for write.\n",filename);
            return NULL;
        }
        wf->dataSize = wf->numSamplesPerChannel = wf->numSamples = 0; // updated in closeWavFile()
        wf->fileHandle = fileHandle;
        wf->mode = *mode;
        wf2hdr(hdr,wf);
        if (fwrite(hdr,sizeof(hdr[0]),sizeof(hdr),fileHandle) != sizeof(hdr)) {
            fprintf(stderr,"In openWavFile('%s'): failed to write the WAV header.\n",filename);
            fclose(fileHandle);
            return NULL;
        }
    }
    return wf;
}

WAVFILE* closeWavFile(WAVFILE* wf) 
{
    int rv = 0;
    FILE *fileHandle = wf->fileHandle;
    wf->fileHandle = NULL;
    if (fileHandle == NULL)
        return wf;
    if (wf->mode == 'w') {
        rv = fflush(fileHandle);
        if (rv == 0) {
            char hdr[sizeof(WAVHDR)];
            long pos = ftell(fileHandle);
            uint32_t size = (uint32_t) (pos - sizeof(hdr));
            wf->dataSize = size;
            rewind(fileHandle);
            wf2hdr(hdr,wf);
            if (fwrite(hdr,sizeof(hdr[0]),sizeof(hdr),fileHandle) != sizeof(hdr))
                rv = -1;
        }
        if (rv != 0) {
            fprintf(stderr,"In closeWavFile(): failed to update data size; it is incorrect.\n");
            fclose(fileHandle);
            return NULL;
        }
    }
    rv = fclose(fileHandle);
    if (rv != 0) {
        fprintf(stderr,"In closeWavFile(): failed to close the file.\n");
        return NULL;
    }
    return wf;
}

size_t readWavFile(WAVFILE* wf, void *buffer, size_t numSamples) 
{
    return fread(buffer,wf->bitDepth/8,numSamples,wf->fileHandle);
}

size_t seekWavFile(WAVFILE* wf, size_t offsetSamples)
{
    if (wf->mode != 'r')
        return -1;
    long pos = sizeof(WAVHDR) + offsetSamples * wf->bitDepth / 8;
    return fseek(wf->fileHandle,pos,SEEK_SET);
}

size_t writeWavFile(WAVFILE* wf, void *buffer, size_t numSamples) 
{
    return fwrite(buffer,wf->bitDepth/8,numSamples,wf->fileHandle);
}

