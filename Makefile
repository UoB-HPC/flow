# User defined parameters
KERNELS 	  	= omp4
COMPILER    	= CRAY
CFLAGS_INTEL	= -O3 -g -qopenmp -no-prec-div -std=gnu99 -DINTEL -xhost -Wall -qopt-report=5
CFLAGS_CRAY		= -lrt -hlist=a
OPTIONS		  	= -DENABLE_PROFILING -DMPI -DDEBUG

# Default compiler
MULTI_COMPILER  = cc
#MULTI_COMPILER  = mpiicc
#MULTI_COMPILER  = mpicc
MULTI_LINKER    = $(MULTI_COMPILER)
MULTI_FLAGS     = $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS   =
MULTI_BUILD_DIR = ../obj
MULTI_DIR       = ..

SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			 = $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

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
	rm -rf $(MULTI_BUILD_DIR)/* wet.exe *.vtk *.bov *.dat *.optrpt

