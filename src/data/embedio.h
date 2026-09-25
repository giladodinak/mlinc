/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store embedding layer */
#ifndef EMBEDIO_H
#define EMBEDIO_H
#include <stdio.h>
#include "embedding.h"

/* read_embedding - Read an embedding layer from a file
 * 
 * Reads an embedding layer from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read embedding layer if successful, NULL otherwise
 */
EMBEDDING* read_embedding(FILE* fp);

/* write_embedding - Write an embedding layer to a file
 * 
 * Writes the embedding layer pointed to by d to the file pointed to by fp. 
 * 
 * Parameters:
 *   e  - Pointer to the embedding layer to be written
 *   fp - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_embedding(const EMBEDDING* e, FILE* fp);

/* load_embedding - Load an embedding layer from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads an embedding layer from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the embedding layer from
 * 
 * Returns:
 *   Pointer to the loaded embedding layer if successful, NULL otherwise
 */
EMBEDDING* load_embedding(const char* filename);

/* store_embedding - Store an embedding layer into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the embedding layer pointed to by d into it.
 * 
 * Parameters:
 *   e        - Pointer to the embedding layer to be stored
 *   filename - Name of the file to store the embedding layer in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_embedding(const EMBEDDING* e, const char* filename);

#endif
