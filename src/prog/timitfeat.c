/* Copyright (c) 2023-2024 Gilad Odinak */

/* This program reads files from the original TIMIT dataset and creates
 * feature files that are used for phoneme training and testing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <sys/stat.h>
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
 * These values comprise a frame's feature vector.
 * *sigma = normalized lpc error residual = -log(err_res + 1e-7) / 30;
 *
 * For each audio file two output files are written:
 *
 * <name>.FEAT - all frames of the utterance, in time order, one row per
 *               frame: frame index followed by the frame's features.
 *               Frames are never dropped, cropped or synthesized.
 *
 * <name>.LAB  - the utterance's phonemes, in order, one row per phoneme:
 *               phoneme name, label, start and end time in seconds, and
 *               start and end position in (fractional) frame units.
 *               Phonemes shorter than one frame are kept.
 *
 * Frame i is analyzed with a window spanning frames i-1 and i, so its
 * center is at time i * FRAMETIME.
 *
 * Uses LPC based features, which may be more resilient to background noise
 * than FFT based features. Reference:
 *  https://www.isca-archive.org/eurospeech_2001/wet01_eurospeech.pdf
 */
#define MAXFILELEN   10 /* Audio length in *seconds*         */
#define FRAMETIME    10 /* Audio length in *milliseconds*    */
#define TARGETRATE 8000 /* Features computed at this rate Hz */
#define FRAMEFEATCNT 14 /* Features per audio frame          */
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

#define MAXPATH 512

int create_dir_path(const char* path);
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
            "Syntax: timitfeat <filelist file> "
            "<timit files directory> <feature files directory>\n");
        return 0;
    }

    char *filelist = argv[1];
    FILE *fp = fopen(filelist,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",filelist);
        return -1;
    }
    char *timitdir = argv[2];
    char *featdir = argv[3];
    char buffer[3 * MAXPATH];
    if ((int) strlen(timitdir) >= MAXPATH) {
        fprintf(stderr,"Directory name too long: '%s'\n",timitdir);
        return -1;
    }
    if ((int) strlen(featdir) >= MAXPATH) {
        fprintf(stderr,"Directory name too long: '%s'\n",featdir);
        return -1;
    }
    if (create_dir_path(featdir) != 0) {
        fprintf(stderr,"Directory does not exist and cannot be created: '%s'\n",featdir);
        return -1;
    }
    init_lrng(42);
    /* Feature statistics accumulated over all files */
    double stat_sum[FRAMEFEATCNT] = {0.0};
    double stat_sumsq[FRAMEFEATCNT] = {0.0};
    long stat_cnt = 0;
    int fileno;
    for (fileno = 0;; fileno++) { /* Process one audio, phoneme file pair */
        /* Construct file path to the files */
        strcpy(buffer,timitdir);
        int pfxlen = strlen(buffer);
        if (buffer[pfxlen - 1] != '/')
            strcat(buffer,"/");
        char* fn = buffer + strlen(buffer);
        fn = fgets(fn,MAXPATH,fp);
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
        int decim = sample_rate / TARGETRATE; /* Decimation factor */
        if (decim < 1 || decim * TARGETRATE != sample_rate) {
            fprintf(stderr,"Sample rate %d of '%s' is not a multiple of %d"
                           " - skipping\n",sample_rate,fn,TARGETRATE);
            closeSphereFile(sfp);
            continue;
        }
        /* Input samples per frame; also maps PHN sample positions to frames */
        int frame_sample_cnt = ((sample_rate * FRAMETIME) / 1000);
        /* Samples per frame after decimation */
        int out_frame_cnt = frame_sample_cnt / decim;
        size_t winSize = 2 * out_frame_cnt;
        float window[winSize];
        HANNWIN hannWin;
        int rv = hannWindowInit(&hannWin,winSize,1);
        if (rv == -1) {
            fprintf(stderr,"Failed to initialize hann window for '%s' - skipping\n",fn);
            closeSphereFile(sfp);
            continue;
        }
        float rdBuf[winSize];
        memset(rdBuf,0,winSize * sizeof(rdBuf[0]));

        FILTER filter1, filter2, filter3, filter4;
        int rv3 = initFilter(&filter3,4,"h",sample_rate,180);
        int rv2 = initFilter(&filter2,4,"l",sample_rate,3400);
        int rv1 = initFilter(&filter1,1,"h",sample_rate,2000);
        int rv4 = initFilter(&filter4,4,"l",sample_rate,3400);
        if (rv3 == -1 ||rv2 == -1 || rv1 == -1 || rv4 == -1) {
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
        size_t fsize = frame_sample_cnt; /* Input samples per frame  */
        size_t osize = out_frame_cnt;    /* Output samples per frame */
        while (frmcnt < FRAMEARRSIZE) {
            float fltBuf[fsize];
            float decBuf[osize];

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
            runFiler(&filter4,fltBuf,fltBuf,fsize);

            /* Decimate to TARGETRATE */
            for (size_t i = 0; i < osize; i++)
                decBuf[i] = fltBuf[i * decim];

            double zcr = zeroCrossings(decBuf,osize);

            for (int i = 0; i < (int) osize; i++)
                rdBuf[osize + i] = decBuf[i];

            hannWindow(&hannWin,rdBuf,window);

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

            memmove(rdBuf,rdBuf + osize,osize * sizeof(rdBuf[0]));
            memset(rdBuf + osize,0,osize * sizeof(rdBuf[0]));
        }
        closeSphereFile(sfp);

        /* Accumulate statistics; feature 1 as written to the file */
        for (int i = 0; i < frmcnt; i++)
            for (int j = 0; j < FRAMEFEATCNT; j++) {
                double v = (j == 1) ? -log(frame_features[i][1] + 1e-7) / 30
                                    : frame_features[i][j];
                stat_sum[j] += v;
                stat_sumsq[j] += v * v;
            }
        stat_cnt += frmcnt;

        /* TIMIT file names start with either TEST or TRAIN.
         * Skip file path prefix that precedes that, if one exists
         */
        char *tfn = strstr(fn,"TRAIN/");
        if (tfn == NULL)
            tfn = strstr(fn,"TEST/");
        if (tfn != NULL)
            fn = tfn;

        /* Create output file paths that are the path of the input file
         * in the output directory, with FEAT and LAB extensions.
         */
        ext = rindex(fn,'.'); /* fn ends with ".PHN" */
        *ext = '\0'; /* Remove previous file name suffix */

        char outfn[3 * MAXPATH];
        snprintf(outfn,sizeof(outfn) - 6,"%s/%s",featdir,fn);
        for (int i = strlen(featdir) + 1; i < (int) strlen(outfn); i++)
            if (outfn[i] == '/')
                outfn[i] = '_';
        char *outext = outfn + strlen(outfn); /* Extension goes here */

        /* Warn if phonemes extend past the extracted frames */
        int last_frame = phonemes[phncnt - 1].endPos / frame_sample_cnt;
        if (last_frame > frmcnt + 1)
            fprintf(stderr,"Phonemes of '%s' end at frame %d, audio has "
                           "%d frames\n",fn,last_frame,frmcnt);

        /* Feature file: all frames, in time order, one row per frame */
        strcpy(outext,".FEAT");
        FILE *fp1 = fopen(outfn,"wb");
        if (fp1 == NULL) {
            fclose(fp);
            fprintf(stderr,"Failed to open '%s' for write - aborting\n",outfn);
            return -1;
        }
        fprintf(fp1,"frame,zcr,sigma");
        for (int j = 2; j < FRAMEFEATCNT; j++)
            fprintf(fp1,",lsp%d",j - 1);
        fprintf(fp1,"\n");
        for (int i = 0; i < frmcnt; i++) {
            int frminx = i;
            fprintf(fp1,"%4d",frminx);
            float zcr = frame_features[i][0];
            float sigma = -log(frame_features[i][1] + 1e-7) / 30;
            fprintf(fp1,",%12.4le,%12.4le",zcr,sigma);
            for (int j = 2; j < FRAMEFEATCNT; j++) /* lsp */
                fprintf(fp1,",%7.4lf",frame_features[i][j]);
            fprintf(fp1,"\n");
        }
        fclose(fp1);

        /* Label file: all phonemes, in order, one row per phoneme.
         * Times in seconds; positions in fractional frame units.
         */
        strcpy(outext,".LAB");
        FILE *fp2 = fopen(outfn,"wb");
        if (fp2 == NULL) {
            fclose(fp);
            fprintf(stderr,"Failed to open '%s' for write - aborting\n",outfn);
            return -1;
        }
        fprintf(fp2,"phoneme,label,start,end,sfrm,efrm\n");
        for (int phninx = 0; phninx < phncnt; phninx++) {
            double ssmp = (double) phonemes[phninx].startPos;
            double esmp = (double) phonemes[phninx].endPos;
            fprintf(fp2,"%s,%2d,%5.3lf,%5.3lf,%7.2lf,%7.2lf\n",
                        phonemes[phninx].phoneme,phonemes[phninx].label,
                        ssmp / sample_rate,esmp / sample_rate,
                        ssmp / frame_sample_cnt,esmp / frame_sample_cnt);
        }
        fclose(fp2);
    }

    fclose(fp);
    printf("Processed %d files\n",fileno);

    if (stat_cnt > 0) {
        double mean[FRAMEFEATCNT];
        double stddev[FRAMEFEATCNT];
        for (int j = 0; j < FRAMEFEATCNT; j++) {
            mean[j] = stat_sum[j] / stat_cnt;
            double var = stat_sumsq[j] / stat_cnt - mean[j] * mean[j];
            stddev[j] = sqrt(var > 0 ? var : 0);
        }
        printf("\nfeature mean and stddev:\n");
        for (int j = 0; j < FRAMEFEATCNT; j++)
            printf("%7.4lf %8.4lf \n",mean[j],stddev[j]);
        printf("\n");
    }
}

int create_dir_path(const char* path)
{
    char buf[MAXPATH];
    mode_t mode = 0755;
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(buf))
        return -1;

    memcpy(buf,path,len + 1);

    for (char* p = buf + 1; ; p++) {
        if (*p == '/' || *p == '\0') {
            char c = *p;
            *p = '\0';
            if (mkdir(buf,mode) != 0) {
                struct stat st;
                if (errno != EEXIST ||
                                    stat(buf,&st) != 0 || !S_ISDIR(st.st_mode))
                    return -1;
            }
            if (c == '\0')
                break;
            *p = c;
        }
    }
    return 0;
}

PHNFILE* openPhonemeFile(const char* filename, char *mode, PHNFILE* pf)
{
    FILE* fileHandle = NULL;
    if (*mode != 'r') {
        fprintf(stderr,"In openPhonemeFile('%s'): invalid mode '%s'; only 'r' supported.\n",filename,mode);
        return NULL;
    }

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
