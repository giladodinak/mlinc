# Copyright (c) 2023 Gilad Odinak

DEBUG=     # Set to yes to disable optimizations and add gdb support
MEMCHK=    # Set to yes to add memory errors detection
PROFILE=   # Set to yes to enable profiling support (disables inlining)
USEDOUBLE= # Set to yes to use double precision math
NOPLOT=    # Set to yes to disable plotting, and use of python matplotlib
USEBLAS=   # Set to yes to use openblas/cblas for matrix multiplications
USECUDA=   # Set to yes to use CUDA/cuBLAS implementations where available
USECLANG=  # Set to yes to use clang instead of gcc
USEGCCENV= # Set to use gcc headers and libraries with clang
MARCH =    # Set to target architecture, if not same as this machine
CARCH =    # Set to cuda target hardware architecture

CC = gcc
CPPC = g++
LINK = g++
AR = ar

CFLAGS = -Wall -Wextra

LFLAGS =
LIBS = -lm

SRC_DIR = ./src
BUILD_DIR = ./build
BIN_DIR = ./bin

NUM_DIR    = $(SRC_DIR)/numeric
DECOMP_DIR = $(SRC_DIR)/decomp
STATS_DIR  = $(SRC_DIR)/stats
SEARCH_DIR = $(SRC_DIR)/search
DATA_DIR   = $(SRC_DIR)/data
MODEL_DIR  = $(SRC_DIR)/model
AUDIO_DIR  = $(SRC_DIR)/audio
FEAT_DIR   = $(SRC_DIR)/feat
TESTS_DIR  = $(SRC_DIR)/tests
PROG_DIR   = $(SRC_DIR)/prog

# Environ.mk and Plot.mk may add to INC_DIRS and LIB_DIRS
INC_DIRS = -I$(SRC_DIR) -I$(NUM_DIR) -I$(DECOMP_DIR) \
		   -I$(STATS_DIR) -I$(SEARCH_DIR) -I$(DATA_DIR) \
		   -I$(MODEL_DIR) -I$(AUDIO_DIR) -I$(FEAT_DIR)
           
LIB_DIRS =

PROGRAMS = sph2wav feat2audio word2vec wordembd simlex bats \
           har timitfeat timit timittest charlm lmtrain lmgen
TESTS = testmem cblastest cudatest \
        testrandom testarray testhash testannoy \
		testhann testfilter testlpc testlsp \
		testqr testsvd testpca \
		testadamw testctc testnorm \
		testdense testlstm testmodel \
		testembed testlmemb testmha testxfmr testsmsftmax

SRCS = $(shell find $(SRC_DIR) -name '*.c')
HDRS = $(shell find $(SRC_DIR) -name '*.h')
AOBJS= $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
OBJS = $(filter-out $(BUILD_DIR)/prog/%.o,$(filter-out \
                                              $(BUILD_DIR)/tests/%.o,$(AOBJS)))

LIB = $(BUILD_DIR)/libmlinc.a

.DEFAULT_GOAL := all

include Environ.mk # OS and processor depndencies

ifneq ($(USECLANG),) # USECLANG is not blank - use clang instead of gcc
include Clang.mk
endif

ifneq ($(USECUDA),)  # USECUDA is not blank - use CUDA/cuBLAS implementations
include NVcc.mk
endif

ifeq ($(NOPLOT),)
include Plot.mk   # plotting (matlibplot for C++) support, if available
endif

.PRECIOUS: $(OBJS) $(AOBJS) $(LIB) # Do not delete object files after link

all: $(PROGRAMS) $(TESTS)

$(PROGRAMS): %: $(BIN_DIR)/%
$(TESTS): %: $(BIN_DIR)/%

# Build the static library from all shared object files.
$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $(OBJS)

$(BIN_DIR)/%: $(BUILD_DIR)/prog/%.o $(LIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(LFLAGS) $(CFLAGS) -o $@ $(BUILD_DIR)/prog/$*.o \
	      -Wl,--start-group $(LIB) -Wl,--end-group $(LIB_DIRS) $(LIBS)

$(BIN_DIR)/%: $(BUILD_DIR)/tests/%.o $(LIB)
	@mkdir -p $(BIN_DIR)
	$(LINK) $(LFLAGS) $(CFLAGS) -o $@ $(BUILD_DIR)/tests/$*.o \
	      -Wl,--start-group $(LIB) -Wl,--end-group $(LIB_DIRS) $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HDRS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC_DIRS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
