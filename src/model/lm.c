/* Copyright (c) 2026 Gilad Odinak       */
/* Decoder-only language model functions */
#include "mem.h"
#include "array.h"
#include "hash.h"
#include "lmemb.h"
#include "transformer.h"
#include "smsftmax.h"
#include "layer.h"
#include "lm.h"

LM* lm_create(int vocab, int model_dim, int heads, int seq_len,
              int batch, int layers, int ffn_dim, int n_neg,
              float dropout, char optimizer)
{
    LM* m = allocmem(1,1,LM);
    m->V = vocab; m->E = model_dim; m->T = seq_len; m->B = batch;
    m->N = layers; m->BT = batch * seq_len; m->n_neg = n_neg;

    /* Embedding: owns its weights (untied), training mode. */
    m->emb = lmemb_create(model_dim,seq_len,/*padinx=*/0);
    lmemb_init(m->emb,vocab,batch,/*training=*/1,/*tied=*/0);

    /* Transformer stack, each wrapped in a LAYER */
    m->tr = allocmem(layers,1,LAYER);
    for (int i = 0; i < layers; i++) {
        TRANSFORMER* t = transformer_create(heads,seq_len,model_dim,
                                            ffn_dim,/*lookahead=*/0);
        transformer_init(t,batch,/*training=*/1,dropout);
        m->tr[i].type = 't';
        m->tr[i].transformer = t;
        layer_alloc_grads(&m->tr[i],optimizer);
    }

    /* Output layer */
    m->head = smsftmax_create(vocab,n_neg);
    smsftmax_init(m->head,model_dim,m->BT);
    m->gHead = allocmem(1,1,fArr2D*);
    m->gHead[0] = allocmem(vocab,model_dim,float);

    m->acts = allocmem(layers + 1,1,fArr2D*);
    for (int i = 0; i <= layers; i++)
        m->acts[i] = allocmem(m->BT,model_dim,float);
    m->dtop  = allocmem(m->BT,model_dim,float);
    m->dcur  = allocmem(m->BT,model_dim,float);
    m->dnext = allocmem(m->BT,model_dim,float);

    m->pad_mask = allocmem(m->BT,1,int);
    m->labels = allocmem(m->BT,1,float);
    m->ids = allocmem(m->BT,1,int);
    return m;
}

void lm_free(LM* m)
{
    lmemb_free(m->emb);
    for (int i = 0; i < m->N; i++) {
        transformer_free(m->tr[i].transformer);
        for (int j = 0; j < m->tr[i].num_grads; j++)
            freemem(m->tr[i].grads[j]);
        freemem(m->tr[i].grads);
    }
    freemem(m->tr);
    smsftmax_free(m->head);
    freemem(m->gHead[0]);
    freemem(m->gHead);
    for (int i = 0; i <= m->N; i++)
        freemem(m->acts[i]);
    freemem(m->acts);
    freemem(m->dtop);
    freemem(m->dcur);
    freemem(m->dnext);
    freemem(m->pad_mask);
    freemem(m->labels);
    freemem(m->ids);
    freemem(m);
}

/* Forward the whole stack. Returns head-input activations [BT][E] */
fArr2D lm_forward(LM* m, int training)
{
    /* Embedding gather: acts[0] = emb(ids). lmemb_forward writes into
     * its own l->h; copy into acts[0] so the stack owns a stable buffer.
     */
    fArr2D h = lmemb_forward(m->emb,m->ids,0);
    fltcpy(m->acts[0],h,m->BT * m->E);

    for (int i = 0; i < m->N; i++)
        transformer_forward(m->tr[i].transformer,
                            m->acts[i],m->pad_mask,
                            training,m->acts[i+1],i);
    return m->acts[m->N];
}

/* Backward the whole stack given dh at the head input (in m->dtop) */
void lm_backward(LM* m)
{
    /* Walk transformers in reverse. dcur holds grad w.r.t. layer i+1's
     * output; dnext receives grad w.r.t. layer i's input.
     */
    fltcpy(m->dcur,m->dtop,m->BT * m->E);
    for (int i = m->N - 1; i >= 0; i--) {
        transformer_backward(m->tr[i].transformer,
                             m->dcur,m->acts[i],m->dnext,i);
        /* Swap dcur <-> dnext for next (lower) layer */
        fArr2D tmp = m->dcur; m->dcur = m->dnext; m->dnext = tmp;
    }
    /* dcur now holds grad w.r.t. embedding output: scatter into gWx */
    lmemb_backward(m->emb,m->dcur,0);
}

/* One optimizer step across all trainable layers */
void lm_update(LM* m, char optimizer, float lr, float wd, int update_cnt)
{
    /* Transformers via the LAYER machinery. */
    for (int i = 0; i < m->N; i++)
        layer_update(&m->tr[i],optimizer,lr,wd,update_cnt);

    /* Head: sparse SGD over touched rows (its own scheme). */
    smsftmax_update(m->head,m->gHead[0],lr,wd);

    /* Embedding: sparse row update over rows touched this batch. Weight
     * decay omitted for embeddings (standard practice).
     */
    {
        typedef float (*ArrVE)[m->E];
        ArrVE Wx  = (ArrVE) m->emb->Wx;
        ArrVE gWx = (ArrVE) m->emb->gWx;
        for (int r = 0; r < m->emb->ntouched; r++) {
            int row = m->emb->touched[r];
            for (int j = 0; j < m->E; j++)
                Wx[row][j] -= lr * gWx[row][j];
        }
    }
}

/* Generates tokens continuing from a prompt.
 *
 * Parameters:
 *   m           - Language model
 *   hmap        - Token id <-> string map, used to render tokens
 *   seed        - Prompt token ids.
 *   seedlen     - Number of prompt tokens
 *   steps       - Number of tokens to generate
 *   buffer      - Output string buffer; null-terminated on return.
 *   buflen      - Capacity of `buffer` in characters. If the text would
 *                 overflow, output is truncated at a token boundary
 *   temperature - Sampling temperature
 *   rep_win_len - Size of the recent-token ring buffer used by the penalty
 *   rep_penalty - Repetition penalty multiplier (<= 1 to disable)
 *
 * References:
 * [1] Hinton, Vinyals & Dean (2015), Distilling the Knowledge in a
 *     Neural Network. Eq. (1): temperature-scaled softmax.
 *     https://arxiv.org/abs/1503.02531
 *
 * [2] Keskar et al. (2019), CTRL: A Conditional Transformer Language
 *     Model for Controllable Generation. Sec. 4.1: temperature sampling,
 *     greedy decoding, top-k sampling, and repetition penalties.
 *     https://arxiv.org/abs/1909.05858
 *
 * [3] Fan, Lewis & Dauphin (2018), Hierarchical Neural Story Generation.
 *     Sec. 5.4: top-k random sampling.
 *     https://arxiv.org/abs/1805.04833
 *
 * [4] Hollows (2026), Gauge Dependence and Structured-Output Corruption
 *     in Sign-Branched Repetition Penalties. Repetition penalty applied
 *     to normalized log-probabilities to remove dependence on the
 *     arbitrary additive offset of raw logits.
 *     https://arxiv.org/abs/2607.09791
 */
int lm_generate(LM* m, HASHMAP* hmap, 
                const int* seed, int seedlen,
                int steps, char *buffer, int buflen,
                float temperature, int top_k, 
                int rep_win_len, float rep_penalty)
{
    int T = m->T, K = m->V, E = m->E;

    if (seed == NULL || seedlen <= 0 || seedlen > T) 
        return -1;
    if (buffer == NULL || buflen <= 0)
        return -1;
    buffer[0] = '\0';
    int bufext = 0;
    int s = 0;

    float* logits = allocmem(1,K,float);

    /* Ring of recently generated tokens for the repetition penalty */
    int* recent = allocmem(rep_win_len,1,int);
    int nrecent = 0;

    /* Initialize context with seed */
    int* ctx = allocmem(T,1,int);
    int ctxlen = seedlen; 
    for (int i = 0; i < ctxlen; i++)
        ctx[i] = seed[i];

    /* Initialize output buffer with seed */
    for (int i = 0; i < seedlen && buflen > 1; i++) {
        bufext = snprintf(buffer,buflen,"%s ",hashmap_inx2str(hmap,ctx[i]));
        if (bufext < 0 || bufext >= buflen)
            break;
        buffer += bufext;
        buflen -= bufext;
    }

    for (s = 0; s < steps; s++) {
        for (int i = 0; i < ctxlen; i++)
            m->ids[i] = ctx[i];
        for (int i = ctxlen; i < m->BT; i++)
            m->ids[i] = 0;
        for (int i = 0; i < ctxlen; i++)
            m->pad_mask[i] = 1;
        for (int i = ctxlen; i < m->BT; i++)
            m->pad_mask[i] = 0;

        typedef float (*ArrBTE)[E];
        ArrBTE H = lm_forward(m,0);

        smsftmax_logits(m->head,(fArr2D) H[ctxlen - 1],(fArr2D) logits,1);

        /* log-softmax: log p_i = z_i - logsumexp(z)
         * (Ref. #4, gauge-fixing)
         */
        float mx = logits[1];
        for (int k = 2; k < K; k++)
            if (logits[k] > mx)
                mx = logits[k];

        float sum = 0;
        for (int k = 1; k < K; k++)
            sum += expf(logits[k] - mx);

        float logsum = mx + logf(sum);
        for (int k = 1; k < K; k++)
            logits[k] -= logsum;

        /* Repetition penalty applied once per unique recent token
         * (Ref. #2 Sec. 4.1, Ref. #4).
         * log p <= 0, so multiplying by rep_penalty > 1 lowers the score.
         */
        if (rep_penalty > 1) {
            for (int i = 0; i < nrecent; i++) {
                int t = recent[i];
                if (t <= 0 || t >= K) continue;

                int seen = 0;
                for (int j = 0; j < i; j++)
                    if (recent[j] == t) {
                        seen = 1;
                        break;
                    }

                if (!seen)
                    logits[t] *= rep_penalty;
            }
        }

        int nxt;

        if (temperature > 0) {
            /* Temperature scaling of log-probability scores.
             * Ref. #1 Eq. (1), Ref. #2 Sec. 4.1.
             */
            for (int k = 1; k < K; k++)
                logits[k] /= temperature;

            /* Top-k sampling: restrict sampling to the top_k highest
             * scoring tokens.
             * Ref. #2 Sec. 4.1, Ref. #3 Sec. 5.4.
             */
            if (top_k > 0 && top_k < K - 1) {
                float neginf = -1e30;
                float tmp[K];
                for (int k = 1; k < K; k++)
                    tmp[k] = logits[k];

                float thresh = neginf;
                for (int r = 0; r < top_k; r++) {
                    float mv = neginf;
                    int mi = -1;

                    for (int k = 1; k < K; k++)
                        if (tmp[k] > mv) {
                            mv = tmp[k];
                            mi = k;
                        }

                    if (mi < 0) break;
                    thresh = mv;
                    tmp[mi] = neginf;
                }

                for (int k = 1; k < K; k++)
                    if (logits[k] < thresh)
                        logits[k] = neginf;
            }

            /* Convert scores to probabilities and sample.
             * Ref. #1 Eq. (1), Ref. #2 Sec. 4.1, Ref. #3 Sec. 5.4.
             */
            typedef float (*Arr1K1)[K - 1];
            softmax((Arr1K1) (logits + 1),1,K - 1);

            float r = urand(0,1);
            float c = 0;
            nxt = 0;
            for (int k = 1; k < K && nxt == 0; k++)
                if (logits[k] > 0) 
                    nxt = k;
           
            for (int k = 1; k < K; k++) {
                c += logits[k];
                if (r <= c) {
                    nxt = k;
                    break;
                }
            }
        }
        else { /* Greedy decoding (Ref. #2 Sec. 4.1) */
            nxt = 1;
            for (int k = 2; k < K; k++)
                if (logits[k] > logits[nxt])
                    nxt = k;
        }

        bufext = snprintf(buffer,buflen,"%s ",hashmap_inx2str(hmap,nxt));
        if (bufext < 0 || bufext >= buflen)
            break;
        buffer += bufext;
        buflen -= bufext;

        /* Record the emitted token in the repetition window */
        if (rep_penalty > 1) {
            if (nrecent < rep_win_len)
                recent[nrecent++] = nxt;
            else {
                for (int i = 1; i < rep_win_len; i++)
                    recent[i - 1] = recent[i];
                recent[rep_win_len - 1] = nxt;
            }
        }

        if (ctxlen < T)
            ctx[ctxlen++] = nxt;
        else {
            for (int i = 1; i < T; i++)
                ctx[i - 1] = ctx[i];
            ctx[T - 1] = nxt;
        }
    }
    if (bufext > 1 && buffer[bufext - 1] == ' ')
        buffer[bufext - 1] = '\0';
    freemem(ctx);
    freemem(logits);
    freemem(recent);
    return s; /* Actual number of words generated (excluding seed) */
}
