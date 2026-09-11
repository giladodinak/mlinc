/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store multi-layer neural network model */
#ifndef MODELIO_H
#define MODELIO_H
#include <stdio.h>
#include "model.h"

/* read_model - Read a model from a file
 * 
 * Reads a model from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read model if successful, NULL otherwise
 */
MODEL* read_model(FILE* fp);

/* write_model - Write a model to a file
 * 
 * Writes the model pointed to by m to the file pointed to by fp. 
 * 
 * Parameters:
 *   m     - Pointer to the model to be written
 *   final - If not zero, store the model as final: optimizer state is
 *           omitted and trainable layers are recorded as inference-only.
 *   fp    - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_model(const MODEL* m, int final, FILE* fp);

/* load_model - Load a model from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads a model from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the model from
 * 
 * Returns:
 *   Pointer to the loaded model if successful, NULL otherwise
 */
MODEL* load_model(const char* filename);

/* store_model - Store a model into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the model pointed to by m into it.
 * 
 * Parameters:
 *   m        - Pointer to the model to be stored
 *   filename - Name of the file to store the model in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_model(const MODEL* m, const char* filename);

#endif
