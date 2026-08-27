# Copyright (c) 2023 Gilad Odinak

ifeq ($(shell uname), Linux)
OSTYPE=linux
else ifeq ($(shell uname), Darwin)
OSTYPE=macos
else
$(error Unsupported OS)
endif

ifeq ($(DEBUG),)    # DEBUG is blank - not debug 
ifeq ($(MARCH),)
MARCH = native
endif
CFLAGS += -O3 -march=$(MARCH) -ffast-math
else
CFLAGS += -ggdb     # gdb support
endif

ifneq ($(USEBLAS),) # USEBLAS is not blank - use openblas/cblas
  CFLAGS += -DUSE_BLAS
  ifeq ($(OSTYPE),linux)
    ifneq ($(wildcard /opt/OpenBLAS/lib/libopenblas.so),) # Use latest OpenBLAS
      CFLAGS  += -I/opt/OpenBLAS/include
      LFLAGS += -L/opt/OpenBLAS/lib -Wl,-rpath,/opt/OpenBLAS/lib
    endif
    LIBS += -lopenblas
  else ifeq ($(OSTYPE),macos)
    LIBS += -lcblas
  endif
endif

ifneq ($(MEMCHK),)  # MEMCHK is not blank - add memory error detector
CFLAGS += -fsanitize=address
LFLAGS += -static-libasan
endif

ifneq ($(PROFILE),) # PROFILE is not blank - add profilng support
CFLAGS += -g -fno-omit-frame-pointer # -fno-inline
endif

ifneq ($(USEDOUBLE),) # USEDOUBLE is not blank - use double instead of float
    CFLAGS += -DUSE_DOUBLE # bit exact math with python
endif

ifeq ($(OSTYPE),macos)
LFLAGS += -w # Suppress OS version message spam on macos
endif

