/* Copyright (c) 2026 Gilad Odinak       */
/* Decoder-only language model functions */
#include <string.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include "mem.h"
#include "float.h"
#include "random.h"
#include "arraytypes.h"
#include "array.h"
#include "hash.h"
#include "vocab.h"
#include "lmemb.h"
#include "transformer.h"
#include "smsftmax.h"
#include "layer.h"
#include "lm.h"


typedef struct lmstat_s {
    iVec nn_pad_mask; /* Backend-resident copy of pad_mask [BT] */
} LMSTAT;

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

    /* pad mask (common to all layers) */
    LMSTAT* s = allocmem(1,1,LMSTAT);
    s->nn_pad_mask = (iVec) nnallocmem(m->BT,1,int);
    m->devstat = s;

    /* Transformer stack, each wrapped in a LAYER */
    m->tr = allocmem(layers,1,LAYER);
    for (int i = 0; i < layers; i++) {
        TRANSFORMER* t = transformer_create(heads,seq_len,model_dim,ffn_dim,0);
        transformer_init(t,batch,1,dropout);
        transformer_set_pad_mask(t,s->nn_pad_mask);
        m->tr[i].type = 't';
        m->tr[i].transformer = t;
        layer_alloc_opt_state(&m->tr[i],optimizer);
    }

    /* Output layer */
    m->head = smsftmax_create(vocab,n_neg);
    smsftmax_init(m->head,model_dim,m->BT);
    m->gHead = allocmem(1,1,fArr2D*);
    m->gHead[0] = nnallocmem(vocab,model_dim,float);

    m->acts = allocmem(layers + 1,1,fArr2D*);
    for (int i = 0; i <= layers; i++)
        m->acts[i] = nnallocmem(m->BT,model_dim,float);
    m->dtop  = nnallocmem(m->BT,model_dim,float);
    m->dcur  = nnallocmem(m->BT,model_dim,float);
    m->dnext = nnallocmem(m->BT,model_dim,float);

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
        layer_free_opt_state(&m->tr[i]);
    }
    freemem(m->tr);
    smsftmax_free(m->head);
    nnfreemem(m->gHead[0]);
    freemem(m->gHead);
    for (int i = 0; i <= m->N; i++)
        nnfreemem(m->acts[i]);
    freemem(m->acts);
    nnfreemem(m->dtop);
    nnfreemem(m->dcur);
    nnfreemem(m->dnext);
    freemem(m->pad_mask);
    freemem(m->labels);
    freemem(m->ids);
    if (m->devstat != NULL) {
        LMSTAT* s = (LMSTAT*) m->devstat;
        nnfreemem(s->nn_pad_mask);
        freemem(s);
    }
    freemem(m);
}

/* Forward the whole stack. Returns head-input activations [BT][E] */
fArr2D lm_forward(LM* m, int training)
{
    /* Embedding gather: acts[0] = emb(ids). lmemb_forward writes into
     * its own l->h; copy into acts[0] so the stack owns a stable buffer.
     */
    fArr2D h = lmemb_forward(m->emb,m->ids,0);
    nnfltcpy(m->acts[0],h,m->BT * m->E);

    LMSTAT* s = (LMSTAT*) m->devstat;
    nnmemcpy(s->nn_pad_mask,m->pad_mask,m->BT * sizeof(int));
    for (int i = 0; i < m->N; i++)
        transformer_forward(m->tr[i].transformer,
                            m->acts[i],training,m->acts[i + 1],i);
    return m->acts[m->N];
}

/* Backward the whole stack given dh at the head input (in m->dtop) */
void lm_backward(LM* m)
{
    /* Walk transformers in reverse. dcur holds grad w.r.t. layer i+1's
     * output; dnext receives grad w.r.t. layer i's input.
     */
    nnfltcpy(m->dcur,m->dtop,m->BT * m->E);
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

    /* Embedding: sparse row update. Weight decay omitted. */
    lmemb_update(m->emb,lr);
}

/* Enables or disables single-token cached decoding.
 *
 * When enabled (on != 0), sets the row-processing sub-layers (FFN and the
 * two AddNorms of every transformer) to a batch size of 1 so they process
 * a single token per call, as required by transformer_forward_step and
 * lm_forward_step. When disabled, restores them to the model's training
 * row count m->BT.
 *
 * The MHA sub-layer is not switched here: its cached single-token path
 * (mha_forward_step) does not use the batch size. The restore value is
 * derived from m.
 */
static void lm_use_cache(LM* m, int on)
{
    int rows = on ? 1 : m->BT;
    for (int i = 0; i < m->N; i++) {
        TRANSFORMER* t = m->tr[i].transformer;
        t->ffn1->B  = rows;
        t->ffn2->B  = rows;
        t->norm1->B = rows;
        t->norm2->B = rows;
    }
}

/* Single-token cached forward. Embeds one token id and runs it through
 * the stack using each layer's cached step at absolute position 'offset'.
 * Returns the head-input activation row [1][E]. Requires lm_use_cache(m,1)
 * first.
 *
 * Uses acts[i] row 0 as the inter-layer buffer (only one row).
 */
fArr2D lm_forward_step(LM* m, int token_id, int offset)
{
    int E = m->E;

    /* Embed just the single token: copy its row from the embedding table.
     * lmemb_forward would embeds all BT positions; here only one exists.
     */
    float* Wx = (float*) m->emb->Wx;
    nnfltcpy(m->acts[0],Wx + token_id * E,E);

    for (int i = 0; i < m->N; i++)
        transformer_forward_step(m->tr[i].transformer,
                                 m->acts[i],offset,m->acts[i+1],i);
    return m->acts[m->N];
}

/* Generates tokens continuing from a prompt.
 *
 * Autoregressive generation: each new token is sampled conditional on
 * the prompt and all tokens generated so far. The prompt and generated
 * tokens remain available as left context through the MHA KV cache.
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
 *   rep_penalty - Repetition penalty multiplier (<= 1 to disable)
 *   rep_win_len - Size of the recent-token ring buffer used by the penalty
 *
 * References:
 * [1] Hinton, Vinyals & Dean (2015), Distilling the Knowledge in a
 *     Neural Network. Eq. (1): temperature-scaled softmax.
 *     https://arxiv.org/pdf/1503.02531
 *
 * [2] Keskar et al. (2019), CTRL: A Conditional Transformer Language
 *     Model for Controllable Generation. Sec. 4.1: temperature sampling,
 *     greedy decoding, top-k sampling, and repetition penalties.
 *     https://arxiv.org/pdf/1909.05858
 *
 * [3] Fan, Lewis & Dauphin (2018), Hierarchical Neural Story Generation.
 *     Sec. 5.4: top-k random sampling.
 *     https://arxiv.org/pdf/1805.04833
 *
 * [4] Hollows (2026), Gauge Dependence and Structured-Output Corruption
 *     in Sign-Branched Repetition Penalties. Repetition penalty applied
 *     to normalized log-probabilities to remove dependence on the
 *     arbitrary additive offset of raw logits.
 *     https://arxiv.org/pdf/2607.09791
 */
int lm_generate(LM* m, HASHMAP* hmap,
                const int* seed, int seedlen,
                int steps, char *buffer, int buflen,
                float temperature, int top_k,
                float rep_penalty, int rep_win_len)
{
    int T = m->T, K = m->V;

    if (seed == NULL || seedlen <= 0 || seedlen > T)
        return -1;
    if (buffer == NULL || buflen <= 0)
        return -1;
    buffer[0] = '\0';
    int bufext = 0;
    int bufpos = 0;
    int s = 0;

    float* logits = allocmem(1,K,float);

    /* Ring of recently generated tokens for the repetition penalty */
    int* recent = (rep_win_len > 0) ? allocmem(rep_win_len,1,int) : NULL;
    int nrecent = 0;

    /* Output buffer starts with the seed tokens: these are the
     * conditioning context x_1..x_m, echoed back so the printed
     * text reads prompt + continuation.
     */
    for (int i = 0; i < seedlen && buflen > 1; i++) {
        bufext = snprintf(buffer + bufpos,buflen - bufpos,
                          "%s ",hashmap_inx2str(hmap,seed[i]));
        if (bufext < 0 || bufext >= buflen - bufpos)
            break;
        bufpos += bufext;
    }

    /* Switch to single-row processing and use the MHA KV cache */
    lm_use_cache(m,1);

    /* Prefill: this is where the prompt enters the computation. Feeding
     * the tokens one at a time (offsets 0..seedlen-1) writes each one's
     * key/value vectors into the KV cache, so later attention can attend
     * to the whole prompt. The head-input activation H kept from the LAST
     * prompt token is h_m = f(x_1..x_m): the attention-weighted summary
     * that defines the first generated token's distribution P(. | x_1..x_m).
     */
    fArr2D H = NULL; /* Holds the current hidden activation */
    for (int i = 0; i < seedlen; i++)
        H = lm_forward_step(m,seed[i],i);

    int pos = seedlen; /* Absolute position of next token */

    for (s = 0; s < steps; s++) {

        /* Project the current hidden state onto the vocabulary: z = W_o h_t.
         * h_t already encodes the prompt (and tokens generated so far) via
         * attention, so these logits ARE the prompt's effect on the next token
         */
        smsftmax_logits(m->head,(fArr2D) H,(fArr2D) logits,1);

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

        /* Repetition penalty applied once per unique recent token.
         * Scaling a normalized log-prob (which is <= 0) by rep_penalty > 1 
         * makes it more negative, i.e. lowers the probability of tokens
         * already emitted in the window.
         * Ref. #2 Sec. 4.1, Ref. #4.
         */
        if (rep_penalty > 1) {
            for (int i = 0; i < nrecent; i++) {
                int t = recent[i];
                if (t <= 0 || t >= K)
                    continue;

                /* Skip tokens already penalized earlier in the window,
                 * so a repeated token is down-weighted once, not compounded.
                 */
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
            /* Temperature scaling of log-probability scores: z_i / T.
             * T<1 sharpens the distribution (more prompt-driven, closer to
             * greedy), T>1 flattens it toward uniform.
             * Ref. #1 Eq. (1), Ref. #2 Sec. 4.1.
             */
            for (int k = 1; k < K; k++)
                logits[k] /= temperature;

            /* Top-k sampling: keep only the top_k highest-scoring tokens by
             * finding the k-th largest score (thresh) and masking everything
             * below it to -inf, so all probability mass falls on those k.
             * Ref. #2 Sec. 4.1, Ref. #3 Sec. 5.4.
             */
            if (top_k > 0 && top_k < K - 1) {
                float neginf = -1e30;
                float tmp[K];
                for (int k = 1; k < K; k++)
                    tmp[k] = logits[k];

                /* Selection pass: pull off the top_k maxima to
                 * locate the cutoff score without a full sort.
                 */
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

            /* Convert scores to probability distribution over the
             * (surviving) vocabulary: p_i = softmax(z/T)_i.
             * Ref. #1 Eq. (1), Ref. #2 Sec. 4.1, Ref. #3 Sec. 5.4.
             */
            softmax1(logits + 1,K - 1);

            /* Inverse-transform sampling: draw r ~ U(0,1) and walk the CDF;
             * choose the first token whose cumulative probability reaches r.
             * nxt tracks the last token with nonzero probability, so it
             * also serves as a fallback if rounding leaves the CDF below r.
             */
            float r = urand(0,1);
            float cdf = 0;
            nxt = 0;

            for (int k = 1; k < K; k++) {
                if (logits[k] <= 0)
                    continue;
                nxt = k;
                cdf += logits[k];
                if (r <= cdf)
                    break;
            }
        }
        else {
           /* Greedy decoding: argmax_i p_i, i.e. always the most likely
            * next token (Ref. #2 Sec. 4.1)
            */
            nxt = 1;
            for (int k = 2; k < K; k++)
                if (logits[k] > logits[nxt])
                    nxt = k;
        }

        /* Emit the chosen token's surface string into the output buffer. */
        bufext = snprintf(buffer + bufpos,buflen - bufpos,
                          "%s ",hashmap_inx2str(hmap,nxt));
        if (bufext < 0 || bufext >= buflen - bufpos)
            break;
        bufpos += bufext;

        /* Record the emitted token in the repetition window so the
         * repetition penalty can down-weight it on later steps.
         */
        if (rep_win_len > 0 && rep_penalty > 1) {
            if (nrecent < rep_win_len)
                recent[nrecent++] = nxt;
            else {
                for (int i = 1; i < rep_win_len; i++)
                    recent[i - 1] = recent[i];
                recent[rep_win_len - 1] = nxt;
            }
        }

        /* Extend the conditioning set: feed the chosen token back at the next
         * position, advancing the KV cache so H = h_{t+1} conditions on
         * x_1..x_m, y_1..y_s for the following step.
         */
        H = lm_forward_step(m,nxt,pos);
        pos++;
    }
    if (bufpos > 0 && buffer[bufpos - 1] == ' ')
        buffer[bufpos - 1] = '\0';
    lm_use_cache(m,0); /* Restore batch row counts */
    freemem(logits);
    freemem(recent);
    return s; /* Actual number of words generated (excluding seed) */
}
