/* Copyright (c) 2023-2026 Gilad Odinak */
/* Splits text into word tokens         */
#include "tokenizer.h"

/* Extracts the next token from a string.
 *
 * Parameters:
 *   str   - A null-terminated string; not modified
 *   flags - Bitwise OR of TOK_SPECIAL and/or TOK_NUMBER
 *   token - Output buffer of at least tlen chars
 *   tlen  - Size of token buffer
 *
 * Returns:
 *   Position past the token in str; NULL when there are no more tokens.
 *   Writes the token into token buffer (null terminated).
 *   Tokens longer than tlen - 1 are skipped.
 */
const char* text_next_token(const char* str, int flags, char* token, int tlen)
{
    char* p = (char*) str; /* Not modified */
    for (char* w = token_start(p,flags); w != NULL; w = token_start(p,flags)) {
        char* e = token_end(w,flags);
        if (e == w || (*w == '<' && *(e - 1) != '>')) {
            p = w + 1; /* '<' not starting a complete special token */
            continue;
        }
        if (*(e - 1) == '\'' || *(e - 1) == '.' || *(e - 1) == ',')
            e--;
        int len = e - w;
        p = e;
        if (len < tlen) {
            for (int i = 0; i < len; i++)
                token[i] = TOK_TOLOWER(w[i]);
            token[len] = '\0';
            return e;
        }
    }
    return NULL;
}
