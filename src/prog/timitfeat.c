/* Copyright (c) 2023-2024 Gilad Odinak */

/* This program reads files from the original TIMIT dataset and creates
 * feature files that are used for phoneme training and testing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include "random.h"
#include "sphere.h"
#include "filter.h"
#include "zcr.h"
#include "hann.h"
#include "lpc.h"
#include "lsp.h"
#include "featfile.h"

/* Audio is divided into frames. For each frame compute LPC parameters,
 * and the residual error variance. Convert the LPC's to LSP's and the 
 * variance to standard deviation, sigma*. Calculate zero crossing ratio.
 * These values comprise a frame's feature vector. Feature vectors are
 * grouped into segments. Each segment spans one phoneme. If a segment
 * (phoneme) is shorter than one frame, no data is collected for it; 
 * if it is longer than maximum number of frames, only that number of
 * frames around the center of the phoneme is used.
 * *sigma = normalized lpc error residual = -log(err_res + 1e-7) / 30;
 */
#define MAXFILELEN   10 /* Audio length in *seconds*         */
#define FRAMETIME    10 /* Audio length in *milliseconds*    */
#define FRAMEFEATCNT 14 /* Features per audio frame          */
#define MAXSEGMENT   32 /* Max number of frames in a segment */
#define LPCORDER (FRAMEFEATCNT - 2) /* preceeded by zcr and sigma    */

#ifndef __APPLE__
static_assert (LPCORDER % 2 == 0,"LPC order must be even number");
#endif

#define NUMPHN 64
#define SIZEPHN 8
typedef struct phninfo_t {
    uint32_t startPos; // in samples
    uint32_t endPos;
    char phoneme[SIZEPHN];
    int label;
} PHNINFO;

typedef struct phnfile_t {
    FILE *fileHandle;
    char mode;         // 'r'
} PHNFILE;

PHNFILE* openPhonemeFile(const char* filename, char *mode, PHNFILE* pf);
PHNFILE* closePhonemeFile(PHNFILE* pf);
size_t readPhonemeFile(PHNFILE* pf, size_t cnt, PHNINFO *phninfo);

/* These variables hold data per processed file */
#define FRAMEARRSIZE ((MAXFILELEN * 1000) / FRAMETIME)
double frame_features[FRAMEARRSIZE][FRAMEFEATCNT]; /* Features of audio file */
int frmcnt = 0; /* Actual number of rows in frame_features[][] */

#define PHSIZE ((MAXFILELEN * 1000) / FRAMETIME) 
PHNINFO phonemes[PHSIZE]; /* Phonemes of the entire audio file     */
int phncnt = 0;           /* Acutal number of entries in phoneme[] */

/* Maximum number of features in one segment */
#define MAX_FEATURES MAXSEGMENT * (FRAMEFEATCNT + 1)

/* Check if a float value is a number. Works even when using -ffast-math */
static int isnumber(float f)
{
    char s[32]; 
    snprintf(s,sizeof(s),"%f  ",f);
    /* not a number strings are nan, inf, -nan, -inf, +nan, +inf */
    if (s[0] == 'n' || s[0] == 'i' || s[1] == 'n' || s[1] == 'i')
        return 0;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Syntax: timitfeat <filelist file> \\\n"
            "            <timit files directory> <feature files directory>\n");
        return 0;
    }

    char *filelist = argv[1];
    FILE *fp = fopen(filelist,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",filelist);
        return 0;
    }
    char *timitdir = argv[2];
    char *featdir = argv[3];
    const int maxpath = 512;
    char buffer[3 * maxpath];
    if ((int) strlen(timitdir) >= maxpath) {
        fprintf(stderr,"Directory name too long: '%s'\n",timitdir);
        return 0;
    }
    if ((int) strlen(featdir) >= maxpath) {
        fprintf(stderr,"Directory name too long: '%s'\n",featdir);
        return 0;
    }
    init_lrng(42);
    int fileno;
    for (fileno = 0;; fileno++) { /* Process one audio, phoneme file pair */
        /* Construct file path to the files */
        strcpy(buffer,timitdir);
        int pfxlen = strlen(buffer);
        if (buffer[pfxlen - 1] != '/')
            strcat(buffer,"/");
        char* fn = buffer + strlen(buffer);
        fn = fgets(fn,maxpath,fp);
        if (fn == NULL || strlen(fn) == 0)
            break;      /* End of file list */
        fn = buffer;    /* fn now points to file path */
        fn[strlen(fn) - 1] = '\0'; /* Punch out end of line char */
        char *ext = rindex(fn,'.');
        if (ext != NULL)
            *ext = '\0'; /* Remove extension if any */
        printf("Processing file %s.WAV, with %s.PHN\n",fn,fn);
        fflush(stdout);
        
        /* Audio file */
        strcat(fn,".WAV");
        SPHFILE sfin, *sfp;
        sfp = openSphereFile(fn,"r",&sfin);
        if (sfp == NULL) {
            fprintf(stderr,"Failed to open '%s' for read - skipping\n",fn);
            continue;
        }
        int sample_rate = sfin.sampleRate;
        int frame_sample_cnt = ((sample_rate * FRAMETIME) / 1000);
        size_t winSize = 2 * frame_sample_cnt;
        float window[winSize];
        HANNWIN hannWin;
        int rv = hannWindowInit(&hannWin,winSize);
        if (rv == -1) {
            fprintf(stderr,"Failed to initialize hann window for '%s' - skipping\n",fn);
            closeSphereFile(sfp);
            continue;
        }
        float rdBuf[winSize];
        memset(rdBuf,0,winSize * sizeof(rdBuf[0]));
        
        FILTER filter1, filter2, filter3;
        int rv3 = initFilter(&filter3,4,"h",sample_rate,180);
        int rv2 = initFilter(&filter2,4,"l",sample_rate,3600);
        int rv1 = initFilter(&filter1,1,"h",sample_rate,2000);
        if (rv3 == -1 ||rv2 == -1 || rv1 == -1) {
            fprintf(stderr,"Failed to initialize filter(s) for '%s' - skipping\n",fn);
            closeSphereFile(sfp);
            continue;
        }
        
        ext = rindex(fn,'.'); /* fn ends with ".WAV" */
        /* Phoneme file */
        strcpy(ext,".PHN"); /* Replace previous file name suffix */
        PHNFILE pfin, *pfp;
        pfp = openPhonemeFile(fn,"r",&pfin);
        if (pfp == NULL) {
            fprintf(stderr,"Failed to open '%s' for read - skipping\n",fn);
            closeSphereFile(sfp);
            continue;
        }

        /* Load phonemes */
        phncnt = readPhonemeFile(pfp,PHSIZE,phonemes);
        if (phncnt < 1) {
            fprintf(stderr,"Failed to read phonemes from '%s' - skipping\n",fn);
            closeSphereFile(sfp);
            closePhonemeFile(pfp);
            continue;
        }
        closePhonemeFile(pfp);

        if (phncnt < 3) {
            fprintf(stderr,"Not enough phonemes in '%s' - skipping\n",fn);
            closeSphereFile(sfp);
            continue;
        }

        /* Extract features from audio */
        frmcnt = 0;
        size_t fsize = winSize / 2;
        while (frmcnt < FRAMEARRSIZE) {
            float fltBuf[fsize];
            
            size_t fcnt = readSphereAudio(sfp,fltBuf,fsize);
            if (fcnt <= 0) /* EOF (or error) */
                break; 
            if (fcnt < fsize) /* Complete partial frame */
                memset(fltBuf + fcnt,0,(fsize - fcnt) * sizeof(fltBuf[0]));

            /* Add noise floor */
            for (size_t i = 0; i < fsize; i++)
                fltBuf[i] += nrand(0,1) * 0.001;

            /* Shape spectrum */
            runFiler(&filter3,fltBuf,fltBuf,fsize);
            runFiler(&filter2,fltBuf,fltBuf,fsize);
            runFiler(&filter1,fltBuf,fltBuf,fsize);

            double zcr = zeroCrossings(fltBuf,fsize);

            for (int i = 0; i < (int) fsize; i++)
                rdBuf[fsize + i] = fltBuf[i];

            hannWindow(&hannWin,rdBuf,window,winSize);

            double lpcCoeffs[LPCORDER + 1];
            double lspCoeffs[LPCORDER + 1];
            double error = computeLPC(window,winSize,LPCORDER,lpcCoeffs);
            lpc2lsp(lpcCoeffs,lspCoeffs,LPCORDER);
            frame_features[frmcnt][0] = zcr;
            frame_features[frmcnt][1] = sqrt(error);
            for (int i = 2; i < FRAMEFEATCNT; i++)
                frame_features[frmcnt][i] = lspCoeffs[i - 2];

            for (int i = 0; i < FRAMEFEATCNT; i++) {
                if (!isnumber(frame_features[frmcnt][i])) {
                    printf("in %s frame %d feature %d is not a number\n",
                                                                  fn,frmcnt,i);
                    frame_features[frmcnt][i] = 0;
                }
            }          
             
            frmcnt++;
            
            memmove(rdBuf,rdBuf + fsize,fsize * sizeof(rdBuf[0]));
            memset(rdBuf + fsize,0,fsize * sizeof(rdBuf[0]));
        }
        closeSphereFile(sfp);

        /* TIMIT file names start with either TEST or TRAIN.
         * Skip file path prefix that precedes that, if one exists
         */
        char *tfn = strstr(fn,"TRAIN/");
        if (tfn == NULL)
            tfn = strstr(fn,"TEST/");
        if (tfn != NULL)
            fn = tfn;
                
        /* Create an output file path that is the path of the 
         * input file in the output directory, with FEAT extension.
         */
        ext = rindex(fn,'.'); /* fn ends with ".PHN" */
        *ext = '\0'; /* Remove previous file name suffix */
        
        char outfn[3 * maxpath];
        snprintf(outfn,sizeof(outfn) - 1,"%s/%s.FEAT",featdir,fn);
        for (int i = strlen(featdir) + 1; i < (int) strlen(outfn); i++)
            if (outfn[i] == '/')
                outfn[i] = '_';

        FILE *fp1 = fopen(outfn,"wb");
        if (fp1 == NULL) {
            fclose(fp);
            fprintf(stderr,"Failed to open '%s' for write - aborting\n",outfn);
            return 0;
        }
        
        static char commas[MAX_FEATURES + 1] = { [0 ... MAX_FEATURES-1] = ',', '\0' };
        fprintf(fp1,"phoneme,label,start,end,file,fsize,nfrm,%s\n",commas);

        for (int phninx = 0; phninx < phncnt; phninx++) {
            char *phoneme = phonemes[phninx].phoneme;
            int label = phonemes[phninx].label;
            int seg_start_sample = (int) phonemes[phninx].startPos;
            int seg_end_sample = (int) phonemes[phninx].endPos;
            int seg_start_frame = seg_start_sample / frame_sample_cnt;
            int seg_end_frame = seg_end_sample / frame_sample_cnt;
            int fsize = FRAMEFEATCNT;
            int nfrm = seg_end_frame - seg_start_frame;
            double (*ff)[FRAMEFEATCNT] = &frame_features[0]; 
            double one_frame_features[1][FRAMEFEATCNT];

            if (nfrm > MAXSEGMENT) {
                int midway = (seg_start_frame + seg_end_frame) / 2;
                seg_start_frame = midway - MAXSEGMENT / 2;
                seg_end_frame = midway + MAXSEGMENT / 2;
                nfrm = seg_end_frame - seg_start_frame;
            }
            if (nfrm == 0) { /* Segment too short */
                /* synthesize a frame that is 
                 * an average of previous and this frames
                 */
                for (int i = 0; i < FRAMEFEATCNT; i++)
                    one_frame_features[0][i] = ff[seg_start_frame][i];
                if (seg_start_frame > 0) {
                    for (int i = 0; i < FRAMEFEATCNT; i++)
                        one_frame_features[0][i] += ff[seg_start_frame - 1][i];
                    for (int i = 0; i < FRAMEFEATCNT; i++)
                        one_frame_features[0][i] /= 2;
                }
                ff = &one_frame_features[0];
                seg_start_frame = 0;
                seg_end_frame = 1;
                nfrm = 1;
            }

            double stime = ((double)seg_start_sample) / ((double)sample_rate);
            double etime = ((double)seg_end_sample) / ((double)sample_rate);
            fprintf(fp1,"%s,%2d,%5.3lf,%5.3lf,%s,%2d,%4d,",
                        phoneme,label,stime,etime,fn,fsize,nfrm);

            for(int i = seg_start_frame; i < seg_end_frame; i++) {
                fprintf(fp1,"%12.4le,",ff[i][0]); /* zcr */
                fprintf(fp1,"%12.4le,",-log(ff[i][1] + 1e-7) / 30); /* normalized sigma */
                for (int j = 2; j < fsize; j++)
                    fprintf(fp1,"%7.4lf,",ff[i][j]);
                assert(i - seg_start_frame < nfrm);
            }
            int nc = MAX_FEATURES - 
                        (seg_end_frame - seg_start_frame) * fsize;
            fprintf(fp1,"%*.*s\n",nc,nc,commas);
            fflush(fp1);
        }
        fclose(fp1);
    }

    fclose(fp);
    printf("Processed %d files\n",fileno);

    double sum[FRAMEFEATCNT] = {0.0};
    for (int i = 0; i < frmcnt; i++)
        for (int j = 0; j < FRAMEFEATCNT; j++)
            sum[j] += frame_features[i][j];

    double mean[FRAMEFEATCNT];
    for (int j = 0; j < FRAMEFEATCNT; j++)
        mean[j] = sum[j] / frmcnt;

    double var[FRAMEFEATCNT] = {0.0};
    for (int i = 0; i < frmcnt; i++)
        for (int j = 0; j < FRAMEFEATCNT; j++)
            var[j] += pow(frame_features[i][j] - mean[j],2);
    
    double stddev[FRAMEFEATCNT];
    for (int j = 0; j < FRAMEFEATCNT; j++) 
        stddev[j] = sqrt(var[j] / frmcnt);

    printf("\nfeature mean and stddev:\n");
    for (int j = 0; j < FRAMEFEATCNT; j++) 
        printf("%7.4lf %8.4lf \n",mean[j],stddev[j]);    
    printf("\n");
}

PHNFILE* openPhonemeFile(const char* filename, char *mode, PHNFILE* pf)
{
    FILE* fileHandle = NULL;   
    if (*mode != 'r') {
        fprintf(stderr,"In openPhonemeFile('%s'): invalid mode '%s'; only 'r' supported.\n",filename,mode);
        return NULL;
    }
    //printf("Openning '%s' for read.\n",filename);

    fileHandle = fopen(filename,"rb");
    if (fileHandle == NULL) {
        fprintf(stderr,"In openPhonemeFile('%s'): failed to open the file for read.\n",filename);
        return NULL;
    }
    pf->fileHandle = fileHandle;
    pf->mode = *mode;   
    return pf;
}

PHNFILE* closePhonemeFile(PHNFILE* pf)
{
    FILE *fileHandle = pf->fileHandle;
    int rv;
    pf->fileHandle = NULL;
    if (fileHandle == NULL)
      return pf;
    rv = fclose(fileHandle);
    if (rv != 0) {
        fprintf(stderr,"In closePhonemeFile(): failed to close the file.\n");
        return NULL;
    }
    return pf;
}

size_t readPhonemeFile(PHNFILE* pf, size_t cnt, PHNINFO *phninfo)
{
    size_t rcnt = 0;
    for (rcnt = 0; rcnt < cnt; rcnt++) {
        char line[100];
        char *s = fgets(line,sizeof(line),pf->fileHandle);
        if (s == NULL)
            break;
        uint32_t startPos;
        uint32_t endPos;
        char phoneme[5];
        int e = sscanf(line,"%u %u %4s",&startPos,&endPos,phoneme);
        if (e < 3) {
            fprintf(stderr,"In readPhonemeFile(): malformed line '%s'\n",line);
            break;
        }
        phninfo->startPos = startPos;
        phninfo->endPos = endPos;
        strcpy(phninfo->phoneme,phoneme);
        phninfo->label = SIL;
        for (int i = 0; i < TIMIT_PHONEME_CNT; i++) {
            if (strcmp(timit_phoneme_names[i],phoneme) == 0) {
                phninfo->label = i;
                break;
            }
        }
        phninfo++;
    };
    return rcnt;
}
