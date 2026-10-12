/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store arrays   */
#ifndef ARRAYIO_H
#define ARRAYIO_H
#include <stdio.h>
#include "float.h"
#include "arraytypes.h"

/* read_array - Read values from a file into a 2D array
 * 
 * Reads values from the file pointed to by fp and stores them into
 * the 2D array a. The dimensions of the array are specified by M and N.
 * 
 * Parameters:
 *   a        - Pointer to the 2D array where values will be stored
 *   M        - Number of rows in the 2D array
 *   N        - Number of columns in the 2D array
 *   fp       - Pointer to a FILE object representing the input file
 *   exc_last - Flag indicating whether to read and discard one value when
 *              reaching end of a row (i.e. the bias)
 * Returns:
 *   1 if all values are successfully read into the array, 0 otherwise
 */
int read_array(fArr2D a, int M, int N, FILE* fp, int exc_last);

/* write_array - Write values from a 2D array to a file
 * 
 * Writes values from the 2D array a to the file pointed to by fp. 
 * The dimensions of the array are specified by M and N. The values 
 * are formatted according to the provided format string fmt.
 * 
 * Parameters:
 *   a_       - Pointer to the 2D array containing values to be written
 *   M        - Number of rows in the 2D array
 *   N        - Number of columns in the 2D array
 *   fp       - Pointer to a FILE object representing the output file
 *   fmt      - Format string specifying the format for writing values 
 *              (default: "%g ")
 *   exc_last - Flag indicating whether to skip the last value of each 
 *              row (i.e. the bias) when writing 
 * 
 * Returns:
 *   1 if all values are successfully written to the file, 0 otherwise
 */ 
int write_array(const fArr2D a, int M, int N, 
                FILE* fp, const char* fmt, int exc_last);

/* load_array - Load values from a file into a 2D array
 * 
 * Opens the file specified by the filename parameter for reading 
 * and loads the values into the 2D array a. The dimensions of the 
 * array are specified by M and N.
 * Parameters:
 *   a        - Pointer to the 2D array where values will be stored
 *   M        - Number of rows in the 2D array
 *   N        - Number of columns in the 2D array
 *   filename - Name of the file to read values from
 *   exc_last - Flag indicating whether to read and discard one value when
 *              reaching end of a row (i.e. the bias)
 * 
 * Returns:
 *   1 if all values are successfully read into the array, 0 otherwise
 */
int load_array(fArr2D a, int M, int N, const char* filename, int exc_last);

/* store_array - Store values from a 2D array into a file
 * 
 * Opens the file specified by the filename parameter for writing 
 * and stores the values from the 2D array a_ into it. The dimensions 
 * of the array are specified by M and N. Values are formatted according 
 * to the provided format string fmt.
 * 
 * Parameters:
 *   a        - Pointer to the 2D array containing values to be written
 *   M        - Number of rows in the 2D array
 *   N        - Number of columns in the 2D array
 *   filename - Name of the file to store values in
 *   fmt      - Format string specifying the format for writing values
 *   exc_last - Flag indicating whether to skip the last value of each 
 *              row (i.e. the bias) when writing 
 * 
 * Returns:
 *   1 if all values are successfully written to the file, 0 otherwise
 */
int store_array(const fArr2D a, int M, int N, 
                const char* filename, const char* fmt, int exc_last);

/* print_array - Print values from a 2D array to standard output
 * 
 * Prints the values from the 2D array a to standard output. The dimensions 
 * of the array are specified by M and N. The array is printed with the 
 * provided name and formatted according to the given format string fmt. 
 * If the exc_last flag is set, the last value of each row is excluded 
 * from printing.
 * 
 * Parameters:
 *   a        - 2D array containing values to be printed
 *   M        - Number of rows in the 2D array
 *   N        - Number of columns in the 2D array
 *   name     - Name of the array to be printed
 *   fmt      - Format string specifying the format for printing values
 *   exc_last - Flag indicating whether to skip the last value of each 
 *              row (i.e. the bias) when printing
 * 
 * Returns:
 *   None
 */
void print_array(const fArr2D a, int M, int N, 
                 const char* name, const char* fmt, int exc_last);
#endif
