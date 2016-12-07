# User defined parameters
KERNELS 	  	= cuda
COMPILER    	= GCC
CFLAGS_INTEL	= -O3 -g -qopenmp -no-prec-div -std=gnu99 -DINTEL -xhost -Wall -qopt-report=5
CFLAGS_GCC		= -O3 -g -fopenmp -std=gnu99 -march=native -Wall
CFLAGS_CRAY		= -lrt -hlist=a
OPTIONS		  	= -DENABLE_PROFILING -DDEBUG #-DMPI 

# Default compiler
MULTI_COMPILER  = gcc
MULTI_LINKER    = $(MULTI_COMPILER)
MULTI_FLAGS     = $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS   =
MULTI_BUILD_DIR = ../obj
MULTI_DIR       = ..

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(MULTI_DIR)/$(KERNELS)/*.c)

ifeq ($(KERNELS), cuda)
include Makefile.cuda
endif

SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			 += $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

wet: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(MULTI_FLAGS) $(OBJS) $(MULTI_LDFLAGS) -o wet.exe

# Rule to make controlling code
$(MULTI_BUILD_DIR)/%.o: %.c Makefile 
	$(MULTI_COMPILER) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/%.o: $(MULTI_DIR)/%.c Makefile 
	$(MULTI_COMPILER) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(MULTI_BUILD_DIR)/* wet.exe *.vtk *.bov *.dat *.optrpt *.cub *.ptx

