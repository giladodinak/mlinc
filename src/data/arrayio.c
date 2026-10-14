/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store arrays   */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "arrayio.h"

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
int read_array(fArr2D a_, int M, int N, FILE* fp, int exc_last)
{
    typedef float (*ArrMN)[N];
#ifdef USE_CUDA
    fArr2D host = allocmem(M,N,float);
    ArrMN a = (ArrMN) host;
#else
    ArrMN a = (ArrMN) a_;
#endif
    int tot = 0, cnt = 0;
    float value;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, tot++) {
            cnt = fscanf(fp,FMTF,&value);
            if (cnt == EOF || cnt <= 0) {
                fprintf(stderr,"In read_array: failed to read value at row %d, col %d\n",i,j);
                break;
            }
            a[i][j] = value;
        }
        if (cnt > 0 && exc_last) {
            cnt = fscanf(fp,FMTF,&value);
            if (cnt == EOF || cnt <= 0) {
                fprintf(stderr,"In read_array: failed to read (and discard) value at row %d, past col %d\n",i,N);
                break;
            }
        }
    }
#ifdef USE_CUDA
    if (tot == M * N)
        nnfltcpy(a_,host,M * N);
    freemem(host);
#endif
    return (tot == M * N) ? 1 : 0;
}

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
 *              (default: "%.8g ")
 *   exc_last - Flag indicating whether to skip the last value of each 
 *              row (i.e. the bias) when writing 
 * 
 * Returns:
 *   1 if all values are successfully written to the file, 0 otherwise
 */ 
int write_array(const fArr2D a_, int M, int N, 
                FILE* fp, const char* fmt, int exc_last)
{
    if (fmt == NULL)
        fmt = "%.6g ";
    typedef float (*ArrMN)[N];
#ifdef USE_CUDA
    fArr2D host = allocmem(M,N,float);
    nnfltcpy(host,a_,M * N);
    ArrMN a = (ArrMN) host;
#else
    ArrMN a = (ArrMN) a_;
#endif
    int len = 0;
    exc_last = (exc_last) ? 1 : 0;
    for (int i = 0; i < M && len >= 0; i++) {
        for (int j = 0; j < N - exc_last && len >= 0; j++)
            len = fprintf(fp,fmt,a[i][j]);
        if (len >= 0)
            len = fprintf(fp,"\n");
    }
    if (len < 0) 
        fprintf(stderr,"In write_array: failed to write array data\n");
#ifdef USE_CUDA
    freemem(host);
#endif
    return (len >= 0) ? 1 : 0;
}

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
int load_array(fArr2D a, int M, int N, const char* filename, int exc_last)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_array: failed to open file '%s' for read\n",filename);
        return 0;
    }
    int ok = read_array(a,M,N,fp,exc_last);
    fclose(fp);
    return ok;
}

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
                const char* filename, const char* fmt, int exc_last)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_array: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_array(a,M,N,fp,fmt,exc_last);
    fclose(fp);
    return ok;
}

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
                 const char* name, const char* fmt, int exc_last)
{
    printf("%s %d X %d\n",name,M,N);
    int ok = write_array(a,M,N,stdout,fmt,exc_last);
    fflush(stdout);
    if (!ok)
        fprintf(stderr,"In print_array: failed to print array %s %d X %d\n",name,M,N);
}    

