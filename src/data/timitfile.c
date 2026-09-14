/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read speech feature files            */
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "sphere.h"
#include "timitfile.h"

int read_timit_files(const char* file_list,
                     int max_samples, int sample_dim,
                     int max_sequences, int* seq_length,
                     fArr2D x_, iVec y_)
{
    /* M = max_samples */
    const int N = sample_dim;
    typedef float (*ArrMN)[N];
    typedef int (*VecM);
    ArrMN x = (ArrMN) x_;
    VecM y = (VecM) y_;
    
    printf("Reading file list from %s\n",file_list);
    char** filename = allocmem(1,max_sequences,char*);
    FILE* fp = fopen(file_list,"rb");
    if (fp == NULL) {
        fprintf(stderr,"%s: failed to open for read\n",file_list);
        return -1;
    }
    int file_cnt;
    for (file_cnt = 0; file_cnt < max_sequences; file_cnt++) {
        char buffer[500];
        char *fn = fgets(buffer,sizeof(buffer),fp);
        if (fn == NULL) /* End of file list */
            break;
        buffer[sizeof(buffer) - 1] = '\0';
        char *eol = index(fn,'\n');
        if (eol != NULL) *eol = '\0';
        char *ext = rindex(fn,'.');
        if (ext != NULL) *ext = '\0';
        filename[file_cnt] = allocmem(1,strlen(fn) + 1,char);
        strcpy(filename[file_cnt],fn);
    }
    fclose(fp);
    
    printf("Reading %d file pairs\n",file_cnt);
    fflush(stdout);
    int sample_cnt, seq_cnt;
    for (sample_cnt = 0, seq_cnt = 0; seq_cnt < file_cnt;) {
        char filepath[500];
        snprintf(filepath,sizeof(filepath),"%s.WAV",filename[seq_cnt]);
        SPHFILE sf;
        SPHFILE* sfp = openSphereFile(filepath,"r",&sf); 
        if (sfp == NULL) {
            fprintf(stderr,"%s: failed to open for read - skipping\n",filepath);
            continue;
        }
        snprintf(filepath,sizeof(filepath),"%s.PHN",filename[seq_cnt]);
        FILE* fp = fopen(filepath,"rb");
        if (fp == NULL) {
            closeSphereFile(sfp);
            fprintf(stderr,"%s: failed to open for read - skipping\n",filepath);
            continue;
        }
        
        int seq_len = 0;
        for (int lineno = 0; ; lineno++) {
            char phoneme[10];
            int phoneme_class;
            int phoneme_start; /* Number of audio data points */
            int phoneme_end;   /* Number of audio data points */
            int cnt = fscanf(fp,"%d %d %9s%*[^\n]\n",
                             &phoneme_start,&phoneme_end,phoneme);
            if (cnt == EOF)
                break;
            if (cnt < 3) {
                fprintf(stderr,"In file '%s': malformed line %d"
                    " - skipping rest of file\n",filepath,lineno + 1);
                break;
            }
            for (phoneme_class = 0; 
                            phoneme_class < TIMIT_CLASS_CNT; phoneme_class++) {
                if (strcmp(phoneme_names[phoneme_class],phoneme) == 0)
                    break;
            }
            if (phoneme_class >= TIMIT_CLASS_CNT) {
                fprintf(stderr,"In file '%s', line %d: unknwon phoneme '%s'"
                    " - skipping rest of file\n",filepath,lineno + 1,phoneme);
                break;
            }
            /* Force phoneme length to multiples of sample_dim (see below) */
            if (phoneme_start < seq_len * N)
                phoneme_start = seq_len * N;
            /* Read audio data and convert to N dimensional vector samples */
            for (int i = phoneme_start; i < phoneme_end; i += N) {
                cnt = (int) readSphereFile(sfp,x[sample_cnt],N);
                if (cnt < N) /* Discard partial vector at end of file */
                    break;
                y[sample_cnt] = phoneme_class;
                sample_cnt++;
                seq_len++;
                if (sample_cnt >= max_samples)
                    break;
            }
            /* Notice that the above loop may read audio data past
             * end of current phoneme; next phoneme start point adjust above.
             */
        }
        fclose(fp);
        closeSphereFile(sfp);
        seq_length[seq_cnt] = seq_len;
        seq_cnt++;
        if (sample_cnt >= max_samples || seq_cnt >= max_sequences)
            break;
    }
    printf("%d files, %d sequences, %d samples\n",file_cnt,seq_cnt,sample_cnt);
    return seq_cnt;
}
