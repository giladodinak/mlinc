/* Copyright (c) 2026 Gilad Odinak */

/* lmgen - load a trained decoder-only LM and generate text.
 *
 * Loads a model saved by lmtrain, accepts a prompt plus the tunable
 * lm_generate() parameters, generates a continuation, and prints it.
 *
 * Usage forms:
 *   lmgen [options] <prompt words...>   one-shot: continue the prompt, exit
 *   lmgen [options]                     interactive: read prompts until ^D
 *
 * All options precede the prompt. Everything after the last option is the
 * prompt (joined with single spaces). If no prompt is given, lmgen enters
 * interactive mode: it reads a line, prints the continuation, and waits for
 * the next line, until end-of-input (^D), at which point it prints "Bye".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include "mem.h"
#include "float.h"
#include "etime.h"
#include "blascpu.h"
#include "array.h"
#include "random.h"
#include "vocab.h"
#include "activation.h"
#include "lmemb.h"
#include "transformer.h"
#include "smsftmax.h"
#include "layer.h"
#include "lm.h"
#include "lmio.h"

static const char* usage =
"Usage: lmgen [options] [prompt words...]\n"
"Loads a trained model and generates a continuation of the prompt.\n"
"With no prompt, enters interactive mode (^D to quit).\n"
"All options must precede the prompt.\n"
"Options:\n"
"  -h                     Show this help message, then exit\n"
"  -l <model_file>        Model to load (default lmtrain.model)\n"
"  --model=<file>\n"
"  -n <int>               Max new tokens to generate (default 20)\n"
"  --max-tokens=<int>\n"
"  -t <float>             Sampling temperature (default 0.8)\n"
"  --temperature=<float>\n"
"  -k <int>               Top-k sampling cutoff (default 40)\n"
"  --top-k=<int>\n"
"  -w <int>               Repetition-penalty window size (default 32)\n"
"  --repeat-window=<int>\n"
"  -p <float>             Repetition penalty (default 1.3)\n"
"  --repeat-penalty=<float>\n"
"  -s <int>               RNG seed for reproducible sampling (default: clock)\n"
"  --seed=<int>\n"
"  -c <list>              Cores for openblas use (default 0,2,4,6)\n"
"  --cores=<list>\n"
;

/* Generate a continuation for the given prompt tokens and print it.
 * Argument order matches lm_generate() as called in lmtrain.c:
 *   (m, vocab->hmap, tokens, n_tokens, max_new_tokens, out, out_size,
 *    temperature, top_k, repeat_window, repeat_penalty)
 * The 'repeat_window' argument (the int between top_k and the float penalty)
 * is inferred positionally from lmtrain.c; rename here if the header differs.
 */
static void generate_and_print(LM* m, const VOCAB* vocab, int* toks, int n,
                               int max_new, float temperature, int top_k,
                               int repeat_window, float repeat_penalty)
{
    if (n <= 0) {
        fprintf(stderr,"No usable prompt tokens - nothing to generate\n");
        return;
    }
    int out_size = (n + max_new + 8) * 32;
    char* out = allocmem(1,out_size,char);
    lm_generate(m,vocab->hmap,toks,n,max_new,out,out_size,
                temperature,top_k,repeat_window,repeat_penalty);
    printf("%s\n",out);
    fflush(stdout);
    freemem(out);
}

int main(int argc, char** argv)
{
    char* model_file     = "lmtrain.model";
    int   max_new_tokens = 20;
    float temperature    = 0.8;
    int   top_k          = 40;
    int   repeat_window  = 32;
    float repeat_penalty = 3.7;
    int   seed           = 0;     /* 0 -> derive from clock */
    int   seed_set       = 0;
    char* blas_cores     = "0,2,4,6";

    static struct option long_opts[] = {
        {"model",          required_argument, 0, 'l'},
        {"max-tokens",     required_argument, 0, 'n'},
        {"temperature",    required_argument, 0, 't'},
        {"top-k",          required_argument, 0, 'k'},
        {"repeat-window",  required_argument, 0, 'w'},
        {"repeat-penalty", required_argument, 0, 'p'},
        {"seed",           required_argument, 0, 's'},
        {"cores",          required_argument, 0, 'c'},
        {"help",           no_argument,       0, 'h'},
        {0,0,0,0}
    };

    /* Leading '+' stops option parsing at the first non-option argument, so
     * every argument after the options is treated as the prompt verbatim
     * (even if a prompt word happens to start with '-').
     */
    int opt;
    while ((opt = getopt_long(argc,argv,"+l:n:t:k:w:p:s:c:h",
                              long_opts,NULL)) != -1) {
        switch (opt) {
            case 'l': model_file     = optarg;       break;
            case 'n': max_new_tokens = atoi(optarg); break;
            case 't': temperature    = atof(optarg); break;
            case 'k': top_k          = atoi(optarg); break;
            case 'w': repeat_window  = atoi(optarg); break;
            case 'p': repeat_penalty = atof(optarg); break;
            case 's': seed = atoi(optarg); seed_set = 1; break;
            case 'c': blas_cores     = optarg;       break;
            case 'h': printf("%s",usage); return 0;
            default:  fprintf(stderr,"%s",usage);    return -1;
        }
    }

    if (max_new_tokens < 1) {
        fprintf(stderr,"lmgen: invalid max-tokens %d\n",max_new_tokens);
        return -1;
    }

    char* p = blas_cores;
    int cores[256];
    int core_cnt = 0;
    while (*p != '\0' && core_cnt < (int)(sizeof(cores)/sizeof(cores[0]))) {
        cores[core_cnt++] = strtol(p,&p,10);
        if (*p == ',') p++;
    }
    openblas_use_cpus(cores,core_cnt);

    /* Load the trained model and its vocabulary. */
    VOCAB* vocab = NULL;
    LMPARAM st;
    LM* m = load_lm(model_file,&vocab,&st);
    if (m == NULL) {
        fprintf(stderr,"lmgen: failed to load model from '%s'\n",model_file);
        return -1;
    }

    /* Seed the sampler: fixed seed for reproducibility, else from the clock. */
    if (!seed_set)
        seed = (int) (current_time() * 1000.0);
    init_lrng(seed);

    int max_prompt_tokens = (m->T > 0) ? m->T : 128;
    int* prompt_tokens = allocmem(1,max_prompt_tokens,int);

    printf("Loaded '%s': vocab %d, model_dim %d, context %d\n",
           model_file,m->V,m->E,m->T);
    fflush(stdout);

    /* Assemble any command-line prompt from the arguments after the options. */
    int have_prompt = (optind < argc);
    if (have_prompt) {
        char prompt[4096];
        prompt[0] = '\0';
        size_t len = 0;
        for (int i = optind; i < argc && len < sizeof(prompt) - 1; i++) {
            if (i > optind && len < sizeof(prompt) - 1)
                prompt[len++] = ' ';
            len += snprintf(prompt + len,sizeof(prompt) - len,"%s",argv[i]);
        }
        int n = vocab_tokenize(vocab,prompt,prompt_tokens,max_prompt_tokens);
        generate_and_print(m,vocab,prompt_tokens,n,max_new_tokens,
                           temperature,top_k,repeat_window,repeat_penalty);
    } else {
        /* Interactive mode: read a prompt, print a continuation, repeat. */
        char line[4096];
        for (;;) {
            printf("> ");
            fflush(stdout);
            if (fgets(line,sizeof(line),stdin) == NULL) {
                printf("\nBye\n");
                break;
            }
            int n = vocab_tokenize(vocab,line,prompt_tokens,max_prompt_tokens);
            if (n <= 0)
                continue; /* blank line or all-OOV: prompt again */
            generate_and_print(m,vocab,prompt_tokens,n,max_new_tokens,
                               temperature,top_k,repeat_window,repeat_penalty);
        }
    }

    freemem(prompt_tokens);
    lm_free(m);
    vocab_free(vocab);
    return 0;
}
