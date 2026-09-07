# Copyright (c) 2026 Gilad Odinak

# CUDA source tree mirrors SRC_DIR.  For example:
#   src/model/mha.c
#   cuda/model/mha.cu
#
# A CUDA source replaces the corresponding C object.  C files without
# CUDA counterparts continue to be compiled normally by CC.

NVCC = nvcc
CUDA_DIR = ./cuda
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_86

TESTS += cudatest-nvcc testarray-gpu

CUSRCS = $(shell find $(CUDA_DIR) -name '*.cu' 2>/dev/null)
CUHDRS = $(shell find $(CUDA_DIR) \( -name '*.h' \) 2>/dev/null)
CUOBJS = $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUSRCS))

# CUDA headers take precedence over corresponding SRC_DIR headers for .cu files.
CUINC_DIRS = $(addprefix -I,$(shell find $(CUDA_DIR) -type d 2>/dev/null))

# CUDA module objects exclude programs and tests, which are linked separately.
CUMOD_OBJS = $(filter-out $(BUILD_DIR)/prog/%.o, \
             $(filter-out $(BUILD_DIR)/tests/%.o,$(CUOBJS)))

# Remove CPU objects having CUDA counterparts, then add CUDA module objects.
OBJS := $(filter-out $(CUMOD_OBJS),$(OBJS)) $(CUMOD_OBJS)

CFLAGS += -DUSE_CUDA -I$(CUDA_HOME)/include -fno-omit-frame-pointer
LIB_DIRS += -L$(CUDA_HOME)/lib64
LIBS += -lcublas -lcudart

NVFLAGS = -arch=$(CUDA_ARCH) -DUSE_CUDA -Drestrict=__restrict__ 
NVFLAGS += $(CUINC_DIRS) $(INC_DIRS) -I$(CUDA_HOME)/include

ifeq ($(DEBUG),)    # DEBUG is blank - not debug 
NVFLAGS += -O3 -Xcompiler=-O3,-march=$(MARCH),-ffast-math,-fno-omit-frame-pointer --use_fast_math
else
NVFLAGS += -g -G
endif

#ifneq ($(PROFILE),) # PROFILE is not blank - add profilng support
NVFLAGS += -g
#endif

ifneq ($(USEDOUBLE),)
NVFLAGS += -DUSE_DOUBLE
endif

# Static pattern rule takes precedence for objects with CUDA counterparts.
$(CUOBJS): $(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu $(HDRS) $(CUHDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c -o $@ $<
