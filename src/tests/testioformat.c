/* Copyright (c) 2026 Gilad Odinak */
/* Tests backward compatibility and idempotence of layer file formats */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "random.h"

#include "dense.h"
#include "denseio.h"
#include "lstm.h"
#include "lstmio.h"
#include "embedding.h"
#include "embedio.h"
#include "lmemb.h"
#include "lmembio.h"
#include "mha.h"
#include "mhaio.h"
#include "addnorm.h"
#include "addnormio.h"
#include "transformer.h"
#include "xfmrio.h"
#include "negsample.h"
#include "negsampio.h"
#include "smsftmax.h"
#include "smsftmaxio.h"

static int write_layers(const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open '%s' for write\n",filename);
        return 0;
    }

    init_lrng(42);

    DENSE* dense = dense_create(7,"gelu",1);
    dense_init(dense,5,3,1);
    if (!write_dense(dense,0,fp)) goto err;
    dense_free(dense);

    LSTM* lstm = lstm_create(7,1,1);
    lstm_init(lstm,5,3,1);
    if (!write_lstm(lstm,0,fp)) goto err;
    lstm_free(lstm);

    EMBEDDING* embedding = embedding_create(7,4,0);
    embedding_init(embedding,11,3,1);
    if (!write_embedding(embedding,fp)) goto err;
    embedding_free(embedding);

    LMEMB* lmemb = lmemb_create(7,4,0);
    lmemb_init(lmemb,11,3,1,0);
    if (!write_lmemb(lmemb,0,fp)) goto err;
    lmemb_free(lmemb);

    MHA* mha = mha_create(2,4,0);
    mha_init(mha,8,3,1,0.25);
    if (!write_mha(mha,0,fp)) goto err;
    mha_free(mha);

    ADDNORM* addnorm = addnorm_create();
    addnorm_init(addnorm,8,12);
    if (!write_addnorm(addnorm,fp)) goto err;
    addnorm_free(addnorm);

    TRANSFORMER* transformer = transformer_create(2,4,8,16,0);
    transformer_init(transformer,3,1,0.25);
    if (!write_transformer(transformer,0,fp)) goto err;
    transformer_free(transformer);

    NEGSAMPLE* negsample = negsample_create(11,3);
    negsample_init(negsample,7,12,1);
    if (!write_negsample(negsample,0,fp)) goto err;
    negsample_free(negsample);

    SMSFTMAX* smsftmax = smsftmax_create(11,3);
    smsftmax_init(smsftmax,7,12);
    if (!write_smsftmax(smsftmax,fp)) goto err;
    smsftmax_free(smsftmax);

    fclose(fp);
    return 1;

err:
    fclose(fp);
    return 0;
}

static int rewrite_layers(const char* input, const char* output)
{
    FILE* fi = fopen(input,"rb");
    if (fi == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",input);
        return 0;
    }
    FILE* fo = fopen(output,"wb");
    if (fo == NULL) {
        fprintf(stderr,"Failed to open '%s' for write\n",output);
        fclose(fi);
        return 0;
    }

    DENSE* dense = read_dense(fi);
    if (dense == NULL || !write_dense(dense,0,fo)) goto err;
    dense_free(dense);

    LSTM* lstm = read_lstm(fi);
    if (lstm == NULL || !write_lstm(lstm,0,fo)) goto err;
    lstm_free(lstm);

    EMBEDDING* embedding = read_embedding(fi);
    if (embedding == NULL || !write_embedding(embedding,fo)) goto err;
    embedding_free(embedding);

    LMEMB* lmemb = read_lmemb(fi);
    if (lmemb == NULL || !write_lmemb(lmemb,0,fo)) goto err;
    lmemb_free(lmemb);

    MHA* mha = read_mha(fi);
    if (mha == NULL || !write_mha(mha,0,fo)) goto err;
    mha_free(mha);

    ADDNORM* addnorm = read_addnorm(fi);
    if (addnorm == NULL || !write_addnorm(addnorm,fo)) goto err;
    addnorm_free(addnorm);

    TRANSFORMER* transformer = read_transformer(fi);
    if (transformer == NULL || !write_transformer(transformer,0,fo)) goto err;
    transformer_free(transformer);

    NEGSAMPLE* negsample = read_negsample(fi);
    if (negsample == NULL || !write_negsample(negsample,0,fo)) goto err;
    negsample_free(negsample);

    SMSFTMAX* smsftmax = read_smsftmax(fi);
    if (smsftmax == NULL || !write_smsftmax(smsftmax,fo)) goto err;
    smsftmax_free(smsftmax);

    int c;
    do { /* Discard trailing white space, if any */
        c = fgetc(fi);
    } while (c != EOF && (c == ' ' || c == '\t' || c == '\r' || c == '\n'));

    if (c != EOF) {
        fprintf(stderr,"Unexpected data remains at end of '%s'\n",input);
        goto err;
    }

    fclose(fi);
    fclose(fo);
    return 1;

err:
    fclose(fi);
    fclose(fo);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && strcmp(argv[1],"write") == 0) {
        if (!write_layers(argv[2]))
            return 1;
        return 0;
    }

    if (argc == 4 && strcmp(argv[1],"rewrite") == 0) {
        if (!rewrite_layers(argv[2],argv[3]))
            return 1;
        return 0;
    }

    fprintf(stderr,"Usage:\n");
    fprintf(stderr,"  %s write <output-file>\n",argv[0]);
    fprintf(stderr,"  %s rewrite <input-file> <output-file>\n",argv[0]);
    return 1;
}
