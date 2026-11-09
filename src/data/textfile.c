/* Copyright (c) 2023-2026 Gilad Odinak */
/* Reads and tokenizes text files       */
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "mem.h"
#include "hash.h"
#include "textfile.h"

#define ISALPHA(c) isalpha((unsigned char) (c))
#define ISDIGIT(c) isdigit((unsigned char) (c))
#define ISALNUM(c) isalnum((unsigned char) (c))
#define TOLOWER(c) tolower((unsigned char) (c))

/* Returns a pointer to the first character that can start a token, or NULL:
 * a letter; a digit if TF_NUMBER is set; '<' if TF_SPECIAL is set.
 */
static inline char* token_start(char* s, int flags)
{
    const int num = (flags & TF_NUMBER);
    const int spc = (flags & TF_SPECIAL);
    for (; *s != '\0'; s++) {
        if (ISALPHA(*s) || (num && ISDIGIT(*s)) || (spc && *s == '<'))
            return s;
    }
    return NULL;
}

/* Returns a pointer past the end of the token that starts at s:
 *   word    - letters and apostrophes; with TF_NUMBER also digits, and
 *             single '.' or ',' between digits (2nd, mp3, 3.14, 1,000)
 *   special - '<', one or more letters/digits/'_', '>'
 * Returns s if s does not start a valid special token. A token that may
 * continue past the end of the buffer ends at the terminating '\0'.
 */
static inline char* token_end(char* s, int flags)
{
    char* e = s;
    if (*s == '<') {
        for (e++; ISALNUM(*e) || *e == '_'; e++);
        if (*e == '>' && e > s + 1)
            return e + 1;
        return (*e == '\0') ? e : s;
    }
    int num = flags & TF_NUMBER;
    for (;;) {
        while (ISALPHA(*e) || *e == '\'' || (num && ISDIGIT(*e))) e++;
        if (num && e > s && (*e == '.' || *e == ',')) {
            if (ISDIGIT(e[-1]) && (ISDIGIT(e[1]) || e[1] == '\0')) {
                e++;
                continue;
            }
        }
        return e;
    }
}

/* Extracts the next token from a string.
 *
 * Parameters:
 *   str   - A null-terminated string
 *   flags - Bitwise OR of TF_SPECIAL and/or TF_NUMBER
 *   token - Output buffer of at least len chars
 *   tlen  - Size of token buffer.
 *
 * Returns:
 *   Position past the token in str; NULL when there are no more tokens.
 *   Writes the token into token buffer (null terminted).
 *   Tokens longer than tlen - 1 are skipped.
 *
 * Example:
 *   char tok[101];
 *   for (const char* p = text_next_token(text,flags,tok,101);
 *        p != NULL;
 *        p = text_next_token(p,flags,tok,100))
 *       ... use tok ...
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
                token[i] = TOLOWER(w[i]);
            token[len] = '\0';
            return e;
        }
    }
    return NULL;
}

/* Processes an ASCII text file to create a word vocabulary, a word frequency
 * table, and/or an array of word tokens.
 *
 * Parameters:
 *   file_name  - Name of file to be processed.
 *   file_dir   - Directory prefix where the file is located (optional).
 *   hmap       - Hashmap that stores the word vocabulary (optional).
 *   add_new    - If non-zero, new words are added to the hashmap.
 *   max_vocab  - Maximum number of words to include in the vocabulary.
 *   word_cnt   - An output array of size max_vocab, where each entry is 
 *                incremented by the number of times the corresponding word
 *                appears in the text file (optional, requires hmap).
 *   file_words - An output array of size max_words, which stores the hashmap
 *                index for each word encountered in the file. If add_new is
 *                zero, words not in the vocabulary are ignored (skipped).
 *   max_words  - Size of file_words array
 *   flags      - Bitwise OR of optional token types (0 = words only):
 *                TF_SPECIAL - "<name>" tokens, e.g. <bos>, <eos>, <sep>
 *                TF_NUMBER  - numbers, e.g. 42, 3.14, 1,000
 *
 * Returns:
 *  - If hmap is not NULL: Returns the number of words that were not skipped.
 *  - If hmap is NULL: Returns the total number of words.
 *  - Returns -1 on error.
 *
 * Notes:
 *  - Only consider alphabetic character sequences (and apostrophe), delimited
 *    by non-alphabetic characters, as words.
 *  - Convert all alphabetic characters to lowercase before further processing.
 *  - Apostrophe in the middle of alphabetic character string, but not at the
 *    begining or end, is considered to be part of a word.
 *  - With TF_SPECIAL, '<' followed by letters/digits/'_' and '>' is one
 *    token (lowercased); any other '<' is a delimiter.
 *  - With TF_NUMBER, digits are word characters, and '.' or ',' between
 *    two digits is part of the word: 42, 3.14, 1,000, 2nd, mp3 are single
 *    tokens. Otherwise digits are delimiters.
 */
int process_text_file(const char* file_name,
                      const char* file_dir,
                      HASHMAP* hmap, int add_new,
                      int max_vocab, WRDCNT* word_cnt,
                      int *file_words, int max_words, int flags)
{
    int maxpath = 512;
    char filepath[2 * maxpath];

    if (file_dir == NULL)
        file_dir = ".";
    int dir_len = strlen(file_dir);
    if (dir_len > 0 && file_dir[dir_len - 1] == '/')
        dir_len--;
    if (dir_len > maxpath) {
        fprintf(stderr,"In process_text_file: "
                "file_dir too long: %d\n",dir_len);
        return -1;
    }
    int name_len = strlen(file_name);
    if (name_len > maxpath) {
        fprintf(stderr,"In process_text_file: "
                "file_name too long: %d\n",name_len);
        return -1;
    }
    snprintf(filepath,sizeof(filepath),
             "%.*s/%.*s",dir_len,file_dir,name_len,file_name);

    FILE* fp = fopen(filepath,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In process_text_file: "
                "failed to open data file '%s' for read\n",file_name);
        return -1;
    }

    char buffer[20000];
    int file_word_cnt = 0; /* Number of words in the file */
    const int max_word_len = 100;
    char token[100 + 1]; /* Lowercased copy of current token */
    int skip = 0; /* Set when an over-long token continues past the buffer */
    for (int cnt, off = 0;;) {
        cnt = fread(buffer + off,1,sizeof(buffer) - off - 1,fp) + off;
        buffer[cnt] = '\0';
        off = 0;
        char* w = token_start(buffer,flags);
        if (skip && w == buffer && *w != '<') { /* Skip rest of the token */
            char* e = token_end(w,flags);
            skip = (e == buffer + cnt); /* Still continues past buffer */
            w = skip ? NULL : token_start(e,flags);
        }
        else
            skip = 0;
        while (w != NULL) {
            char* e = token_end(w,flags);
            if (e == w) { /* '<' not starting a special token: skip it  */
                w = token_start(w + 1,flags);
                continue;
            }
            int len = e - w;
            if (e == buffer + cnt) { /* Last char part of a word         */
                if (!feof(fp)) { /* Word may continue past end of buffer */
                    if (len <= max_word_len) {
                        memmove(buffer,w,len); /* Discard processed data */
                        off = len; /* Will read more data starting here  */
                    }
                    else
                        skip = 1; /* Too long: skip its continuation     */
                    break;
                }
            }
            if (*w == '<' && *(e - 1) != '>') { /* Unclosed at file end  */
                w = token_start(w + 1,flags);
                continue;
            }
            /* Exclude trailing apostrophe (word) or separator (number)  */
            if (*(e - 1) == '\'' || *(e - 1) == '.' || *(e - 1) == ',')
                e--;
            len = e - w;
            if (len <= max_word_len) {
                /* Copy, as the char at e may begin the next token       */
                for (int i = 0; i < len; i++)
                    token[i] = TOLOWER(w[i]);
                token[len] = '\0';
                if (hmap != NULL) {
                    int inx = hashmap_str2inx(hmap,token,add_new);
                    if (inx >= 0 && inx < max_vocab) {
                        if (word_cnt != NULL) {
                            word_cnt[inx].inx = inx;
                            word_cnt[inx].cnt++;
                        }
                        if (file_words != NULL) {
                            if (file_word_cnt < max_words)
                                file_words[file_word_cnt] = inx;
                            else {
                                fprintf(stderr,
                                    "\nFile contains more than %d words\n",
                                    max_words);
                                return file_word_cnt;
                            }
                        }
                        /* Count only words that are not skipped */
                        file_word_cnt++;
                    }
                }
                else
                    file_word_cnt++; /* Count all words */
            }
            w = token_start(e,flags); /* Continue at end of prev token   */
        }
        if (feof(fp))
            break;
    }
    fclose(fp);
    return file_word_cnt;
}

/* Reads a list file containing file names (one per line), filters to include
 * only those files that end with the ".txt" extension (case-insensitive),
 * verifies that these files exist under the given data directory, and returns
 * an array of the valid file names (without paths).
 *
 * Parameters:
 *   list_file  - Path to a text file containing file names, one per line.
 *   data_dir   - Directory prefix where the data files are located.
 *   num_files  - Pointer to an integer where the function stores the count of
 *                valid files found.
 *
 * Returns an array of pointers to file names, or NULL on failure.
 * The returned array should be freed by calling free_text_file_list().
 */
char** read_text_file_list(const char* list_file,
                           const char* data_dir,
                           int* num_files)
{
    char** file_list = NULL;
    FILE* lfp;
    int maxpath = 512;
    char filepath[maxpath * 3];
    strncpy(filepath,data_dir,maxpath);
    filepath[maxpath - 1] = '\0';
    int pfxlen = strlen(filepath);
    if (filepath[pfxlen - 1] != '/') {
        strcat(filepath,"/");
        pfxlen = strlen(filepath);
    }

    lfp = fopen(list_file,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"In read_file_list: "
                "failed to open list file '%s' for read\n",list_file);
        return NULL;
    }

    int max_files = 0;
    for (;;) {
        char* filename = filepath + pfxlen;
        filename = fgets(filename,maxpath - pfxlen,lfp);
        if (filename == NULL || strlen(filename) == 0)
            break; /* End of file list */
        max_files++;
    }
    fclose(lfp);

    lfp = fopen(list_file,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"In read_file_list: "
                "failed to open list file '%s' for read\n",list_file);
        return NULL;
    }

    file_list = allocmem(max_files + 1,1,char*);

    int file_cnt = 0;
    for (;;) {
        char* filename = filepath + pfxlen;
        filename = fgets(filename,maxpath - pfxlen,lfp);
        if (filename == NULL || strlen(filename) == 0)
            break; /* End of file list */

        filename[strcspn(filename, "\r\n")] = '\0';
        const int fnlen = strlen(filename);

        const char* ext = ".txt";
        int off = fnlen - strlen(ext);
        if (off <= 0 || strcasecmp(filename + off, ext) != 0)
            continue; /* Skip files whose name does not end with ext */

        FILE* fp = fopen(filepath, "rb");
        if (fp == NULL) {
            fprintf(stderr,"In read_file_list: "
                    "failed to open data file '%s' for read\n",filename);
            continue;
        }
        fclose(fp);
        file_list[file_cnt] = allocmem(fnlen + 1,1,char);
        strcpy(file_list[file_cnt],filename);
        file_cnt++;
    }
    *num_files = file_cnt;
    return file_list;
}

/* Frees the memory allocated and returned by read_text_file_list()
 *
 * file_list - An array of pointers to file names
 * num_files - The number of entries in file_list array
 */
void free_text_file_list(char** file_list, int num_files)
{
    for (int i = 0; i < num_files; i++)
        freemem(file_list[i]);
    freemem(file_list);
}
