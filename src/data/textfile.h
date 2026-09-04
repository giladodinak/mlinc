/* Copyright (c) 2023-2026 Gilad Odinak */
/* Reads and tokenizes text files       */
#ifndef TEXTFILE_H
#define TEXTFILE_H
#include "hash.h"

/* Stores frequency of a word in the dataset */
typedef struct wrdcnt_s {
    int inx;   /* Word index in hashmap                            */
    int cnt;   /* Number of times word was encountered in the txet */
} WRDCNT;

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
 *  - Apostrophe in the middle of alphabetic character string, but not agt the 
 *    begining or end, is considered to be part of a word.
 */
int process_text_file(const char* file_name,
                      const char* file_dir,
                      HASHMAP* hmap, int add_new,
                      int max_vocab, WRDCNT* word_cnt,
                      int *file_words, int max_words);

/* Reads a list file containing file names (one per line), filters to include
 * only those files that end with the ".txt" extension (case-insensitive),
 * verifies that these files exist under the given data directory, and returns
 * an array of the valid file names (without paths).
 *
 * Parameters:
 *   list_file  - Path to a text file containing file names, one per line.
 *   data_dir   - Directory prefix where the data files are expected to be located.
 *   num_files  - Pointer to an integer where the function stores the count of
 *                valid files found.
 *
 * Returns an array of pointers to file names, or NULL on failure.
 * The returned array should be freed by calling free_text_file_list().
 *
 * Note that the function only returns file names, not full file paths.
 */
char** read_text_file_list(const char* list_file,
                           const char* data_dir,
                           int* num_files);

/* Frees the memory allocated and returned by read_text_file_list()
 *
 * file_list - An array of pointers to file names
 * num_files - The number of entries in file_list array
 */
void free_text_file_list(char** file_list, int num_files);

#endif
