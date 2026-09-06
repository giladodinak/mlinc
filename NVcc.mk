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

CUSRCS = $(shell find $(CUDA_DIR) -name '*.cu' 2>/dev/null)
CUOBJS = $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUSRCS))

# Remove CPU objects having a CUDA counterpart, then add the CUDA objects.
OBJS := $(filter-out $(CUOBJS),$(OBJS)) $(CUOBJS)

CFLAGS += -DUSE_CUDA -I$(CUDA_HOME)/include
LIB_DIRS += -L$(CUDA_HOME)/lib64
LIBS += -lcublas -lcudart

NVFLAGS = -O3 -arch=$(CUDA_ARCH) -DUSE_CUDA $(INC_DIRS) -I$(CUDA_HOME)/include

ifneq ($(DEBUG),)
NVFLAGS := -g -G -arch=$(CUDA_ARCH) -DUSE_CUDA $(INC_DIRS) -I$(CUDA_HOME)/include
endif

ifneq ($(USEDOUBLE),)
NVFLAGS += -DUSE_DOUBLE
endif

# Static pattern rule takes precedence for objects with CUDA counterparts.
$(CUOBJS): $(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c -o $@ $<
