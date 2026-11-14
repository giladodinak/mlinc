/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <string.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "model.h"
#include "modelio.h"
#include "onehot.h"
#include "featfile.h"
#include "beamsrch.h"
#include "alignseq.h"

/* Each sequence contains multiple phonemes, find how many in total */
int count_phoneme(int*  yc, int len)
{
    int cnt = 0;
    for (int i = 0; i < len; i++) {
        if (yc[i] >= EOP) {
            yc[i] -= EOP;
            cnt++;
        }
    }
    return cnt;
}


/* Removes consecutive duplicate labels and blanks from the labels array.
 * Modifies the array labels in place and returns the new length.
 */
int dedup_labels(iVec restrict labels, int len, int blank_inx)
{
    int j, k;
    for (j = 0, k = 0; j < len; j++)
        if (labels[j] != blank_inx)
            if (k == 0 || labels[k - 1] != labels[j])
                labels[k++] = labels[j];
    return k;
}


/* Removes silence from a sequence of phoneme labels.
 * Preserves consecutive identical phonemes.
 * Modifies the array labels in place and returns the new length.
 */
int reference_labels(iVec restrict labels, int len, int blank_inx)
{
    int j, k;
    for (j = 0, k = 0; j < len; j++)
        if (labels[j] != blank_inx)
            labels[k++] = labels[j];
    return k;
}


int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"syntax: timittest <model file> <feature file>\n");
        return -1;
    }        
    const char* modelfile = argv[1];
    const char* featfile = argv[2];

    MODEL* m = load_model(modelfile);
    if (m == NULL) {
        fprintf(stderr,"Failed to load model file '%s'\n",modelfile);
        return -1;
    }

    int D = m->input_dim;   /* Input vectors dimension          */
    int N = m->output_dim;  /* Number of output classes         */
    int MTe = 1000;         /* Max number of samples 10 seconds */
    float xTe[MTe][D];      /* Input feature vectors            */
    int   yTec[MTe];        /* Phoneme indices (true labels)    */
    int   phTe[MTe];        /* Phoneme sequence (true labels)   */
    int   phcnt = 0;        /* Number of labels in phTe         */

    FILE* fp = fopen(featfile,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open file '%s' for read\n",featfile);
        model_free(m);
        return -1;    
    }
    /* Label file has the feature file's name, with LAB extension */
    char labfile[strlen(featfile) + 5];
    strcpy(labfile,featfile);
    char* ext = strrchr(labfile,'.');
    strcpy((ext != NULL) ? ext : labfile + strlen(labfile),".LAB");
    FILE* lfp = fopen(labfile,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"Failed to open file '%s' for read\n",labfile);
        fclose(fp);
        model_free(m);
        return -1;    
    }
    int cnt = read_feature_file(fp,lfp,MTe,xTe,yTec,MTe,phTe,&phcnt);
    fclose(fp);
    fclose(lfp);
    if (cnt == 0) {
        printf("feature file does not contain any data\n");
        model_free(m);
        return 0;
    }
    MTe = cnt; /* Actual number of samples */
    int ytc[phcnt + 1]; /* True labels             */
    memcpy(ytc,phTe,phcnt * sizeof(int));
    int ytc_len = reference_labels(ytc,phcnt,SIL);
    /* Each sequence contains multiple phonemes, find how many in total */
    int PTe = count_phoneme(yTec,MTe);

    printf("%d phonemes, %d samples\n\n",PTe,MTe);

    float yp[MTe][N];   /* Predicted probabilities */
    int ypc[MTe];       /* Predicted labels        */

    /* Predict lables from features */    
    model_predict(m,xTe,yp,MTe);

    int ypc_len;
    if (m->loss_func == 'C') {
        /* CTC beam search returns already collapsed, SIL terminated labels */
        int beamwidth = 10;
        int sequences[beamwidth][MTe + 1];
        float scores[beamwidth];
        ctc_beam_search(yp,MTe,N,beamwidth,sequences,scores);
        for (ypc_len = 0; sequences[0][ypc_len] != SIL; ypc_len++);
        memcpy(ypc,sequences[0],ypc_len * sizeof(int));
    }
    else {
        onehot_decode(yp,ypc,MTe,N);
        ypc_len = dedup_labels(ypc,MTe,SIL);
    }

    /* Align the true and predicted phoneme sequences */
    int rlen = ((ytc_len > ypc_len) ? ytc_len : ypc_len) * 2;
    if (rlen == 0) rlen = 1;
    int ypc2[rlen];
    int ytc2[rlen];
    int dist = alignseq(ypc,ypc_len,ytc,ytc_len,ypc2,ytc2,rlen,SIL);

    /* Print the sequences */
    const char** rpn = reduced_phoneme_names;
    printf("True phonemes:      ");
    for (int i = 0; i < rlen; i++) {
        if (ytc2[i] != SIL || ypc2[i] != SIL)
            printf("%-3s ",ytc2[i] == SIL ? "-" : rpn[ytc2[i]]);
        else
            break;
    }
    printf("\n");
    printf("Predicted phonemes: ");
    for (int i = 0; i < rlen; i++) {
        if (ytc2[i] != SIL || ypc2[i] != SIL)
            printf("%-3s ",ypc2[i] == SIL ? "-" : rpn[ypc2[i]]);
        else
            break;
    }
    printf("\n");

    /* Count matches */
    int pcnt = 0, mcnt = 0;
    for (int i = 0; i < rlen; i++) {
        if (ytc2[i] != SIL || ypc2[i] != SIL) {
            pcnt++;
            if (ytc2[i] == ypc2[i])
                mcnt++;
        }
        else
            break;
    }
    printf("%d out of %d matched (%5.3f)\n",mcnt,pcnt,pcnt ? (float) mcnt / pcnt : 0.0f);
    if (ytc_len > 0)
        printf("Phoneme error rate (PER) %5.3f\n",(float) dist / ytc_len);

    model_free(m);
}
