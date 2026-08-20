# Machine Learning from scratch in C

## Overview

This project aims to provide a simple implementation of some
Machine Learning (ML) basic building blocks, linear neural networks
(Multi Layer Perceptron), multi-layer recurrent neural networks
(specifically LSTM), and Transformers. These are implemented in 
simple C, making all algorithm details evident.

#### ML algorithms and building blocks implemented

-   Neural network layers
    -   Linear (Dense) fully connected neural network
    -   LSTM - Long Short Term Memory neural network (Hochreiter et al)
    -   Embedding (Mikolov et al)
    -   MHA - Multi-Head Attention (Vaswani et al)
    -   Transformer (decoder) (Vaswani et al)
    -   Multi-layer neural network model
-   Activation functions and their derivatives
    -   Sigmoid
    -   ReLU - Rectified Linear Unit
    -   GeLU - Gaussian Error Linear Unit
    -   Softmax
-   Loss functions and their derivatives
    -   Mean Square Error - regression
    -   Cross Entropy - classification
    -   CTC - Connectionist Temporal Classification (Graves)
-   Optimizer
    -   ADAMW - Adaptive Moment Estimation with Weight decay
        (Loshchilov - Hutter)
-   Sequencing
    -   Beam search
    -   Sequence alignment (Needleman - Wunsch)
    -   Edit distance (Levenshtein)
-   Search
    -   Approximate Nearest Neighbors - Annoy (Erik Bernhardsson)
-   Decomposition
    -   SVD - Singular Value Decomposition (Golub - Reinsch)
    -   PCA - Principal Components Analysis
    -   QR Decomposition

### Implementation notes

The code should be quite portable, as it does not explicitly use any
special hardware, and only relies on compiler (GCC) optimizations.\
\
The ML algorithms implemented here involve working with arrays of real 
numbers. This code uses the C float type to store floating-point values 
that approximate real numbers. In C, there are a few ways to store arrays 
of numbers. This code stores two-dimensional arrays in continuous memory, 
in a row-major fashion. For example, an array A of 2 rows and 3 columns 
that stores the numbers 1 to 6 is laid out in memory as 6 consecutive 
floats storing the values 1, 2, 3, 4, 5, 6. A\[1\]\[2\] references the 
second row\'s third element, which stores the value 6.\
\
Often, arrays passed to various functions do not have predetermined
sizes. The function\'s array parameter is of generic type, and the
actual number of rows and columns is passed as separate parameters. For
better readability, the code makes extensive use of typedefs and casts
so that inside a function the array elements are accessed in a natural
way, i.e., as row and column subscripts, for example A\[1\]\[2\]. See
[array.h](src/numeric/array.h), for more details.\
\
The code generally uses heap memory to prevent stack overflow when
dealing with large arrays. The [allocmem()](src/mem.h) macro
ensures that the allocated memory is properly initialized to all zeros;
if not enough memory is available, it prints an error message and
termintes the program.\
\
The neural network architecture does not incorporate a separate bias
term. Instead it uses the \"Bias Trick\": it incorporates the bias term
directly into the weight matrix. This is done by adding an extra input
to each neuron that is always set to 1, called the bias input. The
corresponding weight for this bias input effectively becomes the bias
term. This way, the bias can be treated as just another weight,
simplifying matrix operations.

### Browsing the code

A good starting point is [testmodel.c](src/tests/testmodel.c),
which is a program that tests different Neural Network models against
simple test cases. This program provides a good idea on how models are
constructed, trained and used to make predictions.

### Compiling the code

This code should compile and run on Linux and MacOS.\
Clone the repository (or dwonload the source code) into a sub-directory
for example \~/mlinc . In this directory will find a Makefile. run\
`$ make`\
The resulting programs will be in bin sub-directory. Try running
the memory allocation test program to verify binary compatibility\
`$ bin/testmem`

### Using BLAS (Basic Linear Algebra Subprograms)

When enabled, BLAS substitutes some of the matrix multiplications functions
in [array.h](src/numeric/array.h) with highly optimized multi-core library
routines, To enable BLAS, compile the code with the USEBLAS flag\
`$ make USEBLAS=Yes`

### Plotting

Some test programs use C bindings to Python matplotlib to display
training results. This requires that Python and matplotlib be present,
see Plot.mk file for expected location of python libraries. Plotting can
be turned off when compiling the code, using the NOPLOT flag\
`$ make NOPLOT=Yes`

### Obtaining test datasets

Scripts in the scripts sub-directory fetch publicly available datasets
for use in test programs. For example:\
`$ scripts/getirisdata.sh`

### Running the code

In the project root directory, run these commands:\
`$ bin/testmodel`\
`$ bin/testembed`\
`$ bin/har`\
`$ bin/timit`\
`$ bin/word2vec`

### License

The use of this software is governed by the [MIT license](LICENSE.md)
