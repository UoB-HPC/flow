# User defined parameters
KERNELS 	  	= cuda
COMPILER    	= GCC
MPI						= yes
CFLAGS_INTEL	= -O3 -g -qopenmp -no-prec-div -std=gnu99 -DINTEL -xhost -Wall -qopt-report=5
CFLAGS_GCC		= -O3 -g -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_CRAY		= -lrt -hlist=a
OPTIONS		  	= -DENABLE_PROFILING -DDEBUG 

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

# Default compiler
MULTI_COMPILER_CC   = mpicc
MULTI_COMPILER_CPP  = mpic++
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
	$(MULTI_LINKER) $(OBJS) $(MULTI_LDFLAGS) -o wet.exe

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

