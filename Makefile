# User defined parameters
KERNELS          = omp3
COMPILER         = INTEL
MPI              = yes
DECOMP					 = TILES
MAC_RPATH				 = -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL     = -O3 -g -qopenmp -no-prec-div -std=gnu99 -DINTEL \
									 $(MAC_RPATH) -Wall -qopt-report=5 #-xhost
CFLAGS_INTEL_KNL = -O3 -g -qopenmp -no-prec-div -std=gnu99 -DINTEL \
									 -xMIC-AVX512 -Wall -qopt-report=5
CFLAGS_GCC       = -O3 -g -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_CRAY      = -lrt -hlist=a
OPTIONS          = -DENABLE_PROFILING 

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

ifeq ($(DECOMP), TILES)
OPTIONS += -DTILES
endif
ifeq ($(DECOMP), ROWS)
OPTIONS += -DROWS
endif
ifeq ($(DECOMP), COLS)
OPTIONS += -DCOLS
endif

# Default compiler
MULTI_COMPILER_CC   = cc
MULTI_COMPILER_CPP  = CC
MULTI_LINKER    		= $(MULTI_COMPILER_CC)
MULTI_FLAGS     		= $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS   		= $(MULTI_FLAGS) -lm
MULTI_BUILD_DIR 		= ../obj
MULTI_DIR       		= ..

ifeq ($(KERNELS), cuda)
include Makefile.cuda
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(MULTI_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

wet: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(OBJS) $(MULTI_LDFLAGS) -o wet.$(KERNELS)

# Rule to make controlling code
$(MULTI_BUILD_DIR)/%.o: %.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/%.o: $(MULTI_DIR)/%.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(MULTI_BUILD_DIR)/* wet.exe *.vtk *.bov *.dat *.optrpt *.cub *.ptx

