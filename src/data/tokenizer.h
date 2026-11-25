/* Copyright (c) 2023-2026 Gilad Odinak */
/* Splits text into word tokens         */
#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <ctype.h>
#include <string.h>

/* Tokenizer flags (bitwise OR; 0 = words only) */
#define TOK_SPECIAL 1   /* <name> tokens  */
#define TOK_NUMBER  2   /* numeric tokens */

/* Tokenization rules:
 *  - A word is a sequence of letters and apostrophes. An apostrophe at the
 *    beginning or end of the sequence is not part of the word.
 *  - All letters are converted to lowercase.
 *  - With TOK_SPECIAL, '<' followed by letters/digits/'_' and '>' is one
 *    token (lowercased); any other '<' is a delimiter.
 *  - With TOK_NUMBER, digits are word characters, and '.' or ',' between
 *    two digits is part of the word: 42, 3.14, 1,000, 2nd, mp3 are single
 *    tokens. Otherwise digits are delimiters.
 *  - Any other character is a delimiter.
 */

#define TOK_ISALPHA(c) isalpha((unsigned char) (c))
#define TOK_ISDIGIT(c) isdigit((unsigned char) (c))
#define TOK_ISALNUM(c) isalnum((unsigned char) (c))
#define TOK_TOLOWER(c) tolower((unsigned char) (c))

/* Returns a pointer to the first character that can start a token, or NULL:
 * a letter; a digit if TOK_NUMBER is set; '<' if TOK_SPECIAL is set.
 */
static inline char* token_start(char* s, int flags)
{
    const int num = (flags & TOK_NUMBER);
    const int spc = (flags & TOK_SPECIAL);
    for (; *s != '\0'; s++) {
        if (TOK_ISALPHA(*s) || (num && TOK_ISDIGIT(*s)) || (spc && *s == '<'))
            return s;
    }
    return NULL;
}

/* Returns a pointer past the end of the token that starts at s:
 *   word    - letters and apostrophes; with TOK_NUMBER also digits, and
 *             single '.' or ',' between digits (2nd, mp3, 3.14, 1,000)
 *   special - '<', one or more letters/digits/'_', '>'
 * Returns s if s does not start a valid special token. A token that may
 * continue past the end of the buffer ends at the terminating '\0'.
 */
static inline char* token_end(char* s, int flags)
{
    char* e = s;
    if (*s == '<') {
        for (e++; TOK_ISALNUM(*e) || *e == '_'; e++);
        if (*e == '>' && e > s + 1)
            return e + 1;
        return (*e == '\0') ? e : s;
    }
    int num = flags & TOK_NUMBER;
    for (;;) {
        while (TOK_ISALPHA(*e) || *e == '\'' || (num && TOK_ISDIGIT(*e))) e++;
        if (num && e > s && (*e == '.' || *e == ',')) {
            if (TOK_ISDIGIT(e[-1]) && (TOK_ISDIGIT(e[1]) || e[1] == '\0')) {
                e++;
                continue;
            }
        }
        return e;
    }
}

/* token_is_number - True if token is a number token as produced with
 * TOK_NUMBER: starts with a digit and contains only digits, '.' and ','
 * (e.g. 42, 3.14, 1,000; not 2nd or mp3).
 */
static inline int token_is_number(const char* token)
{
    if (!TOK_ISDIGIT(token[0]))
        return 0;
    return token[strspn(token,"0123456789.,")] == '\0';
}

/* token_is_special - True if token is a special token (<name>), as produced
 * with TOK_SPECIAL.
 */
static inline int token_is_special(const char* token)
{
    return token[0] == '<';
}

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
const char* text_next_token(const char* str, int flags, char* token, int tlen);

#endif
