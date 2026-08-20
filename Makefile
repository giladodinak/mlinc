# Copyright (c) 2023 Gilad Odinak

DEBUG =     # Set to yes to disable optimizations and add gdb support
MEMCHK =    # Set to yes to add memory errors detection
PROFILE =   # Set to yes to enable profiling support (disables inlining)
USEDOUBLE = # Set to yes to use double precision math
MARCH =     # Set to target architecture, if not same as this machine
NOPLOT =    # Set to yes to disable plotting, and use of python matplotlib
USEBLAS=    # Set to yes to use openblas/cblas for matrix multiplications
USECLANG=   # Set to yes to use clang instead of gcc
USEGCCENV=  # Set to use gcc headers and libraries with clang

CC = gcc
CPPC = g++
LINK = g++

# Environ.mk and Plot.mk add more flags to CFLAGS LFLAGS LIBS
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
           har timitfeat timit timittest charlm
TESTS = testmem cblastest testarray testrandom testhash testannoy \
		testhann testfilter testlpc testlsp \
		testqr testsvd testpca \
		testadamw testctc testnorm \
		testdense testlstm testmodel \
		testembed testmha testxfmr

SRCS = $(shell find $(SRC_DIR) -name '*.c')
HDRS = $(shell find $(SRC_DIR) -name '*.h')
AOBJS= $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
OBJS = $(filter-out $(BUILD_DIR)/prog/%.o,$(filter-out \
                                              $(BUILD_DIR)/tests/%.o,$(AOBJS)))

include Environ.mk # OS and processor depndencies

ifneq ($(USECLANG),) # USECLANG is not blank - use clang instead of gcc
include Clang.mk
endif

ifeq ($(NOPLOT),)
include Plot.mk   # plotting (matlibplot for C++) support, if available
endif

.PRECIOUS: $(OBJS) $(AOBJS) # Do not delete object files after link

all: $(PROGRAMS) $(TESTS)

$(PROGRAMS): %: $(BIN_DIR)/%
$(TESTS): %: $(BIN_DIR)/%

$(BIN_DIR)/%: $(BUILD_DIR)/prog/%.o $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(LFLAGS) $(CFLAGS) -o $@ $(BUILD_DIR)/prog/$*.o \
	      $(OBJS) $(LIB_DIRS) $(LIBS)

$(BIN_DIR)/%: $(BUILD_DIR)/tests/%.o $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(LINK) $(LFLAGS) $(CFLAGS) -o $@ $(BUILD_DIR)/tests/$*.o \
	      $(OBJS) $(LIB_DIRS) $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HDRS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC_DIRS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
