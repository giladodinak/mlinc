
ifneq ($(USEGCCENV),) # USEGCCENV is not blank - use gcc headers and libraries

GCC_INCLUDE_DIRS := $(shell gcc -xc -xc++ -E -v /dev/null 2>&1 | \
    awk '/#include <...> search starts here:/{flag=1; next} \
         /End of search list./{flag=0} flag {print "-isystem " $$1}')

INC_DIRS += $(GCC_INCLUDE_DIRS)

GCC_PREFIX := $(shell gcc -print-file-name=crtbegin.o | sed 's|/crtbegin.o||')
GCC_LIBDIR := $(shell gcc -print-file-name=libgcc.a | sed 's|/libgcc.a||')

CLANG_B := -B$(GCC_PREFIX)
CLANG_L := -L$(GCC_LIBDIR) -L/usr/lib64

LFLAGS += $(CLANG_B) $(CLANG_L)

endif

CFLAGS += -fno-finite-math-only -fno-unsafe-math-optimizations -Wno-format-security -Wno-vla-cxx-extension -Wno-vla-extension -Wno-literal-conversion

CC = clang
CPPC = clang++
LINK = clang++
