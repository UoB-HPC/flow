# User defined parameters
KERNELS 	  	= omp3
COMPILER    	= INTEL
CFLAGS_INTEL	= -O3 -g -qopenmp -no-prec-div -xhost -std=gnu99
OPTIONS		  	= -DENABLE_PROFILING -DMPI -DDEBUG

# Default compiler
MULTI_COMPILER   = mpicc
MULTI_LINKER     = mpicc
MULTI_FLAGS      = $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS    =
MULTI_BUILD_DIR  = ../obj
MULTI_SHARED_DIR = ../shared

SRC  			 = $(wildcard *.c)
SRC 			+= $(wildcard $(MULTI_SHARED_DIR)/*.c)
SRC_CLEAN  = $(subst $(MULTI_SHARED_DIR)/,,$(SRC))
OBJS 			 = $(patsubst %.c, $(MULTI_BUILD_DIR)/$(KERNELS)/%.o, $(SRC_CLEAN))

wet: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(MULTI_FLAGS) $(OBJS) $(MULTI_LDFLAGS) -o wet.exe

# Rule to make controlling code
$(MULTI_BUILD_DIR)/$(KERNELS)/%.o: %.c Makefile 
	$(MULTI_COMPILER) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/$(KERNELS)/%.o: ../shared/%.c Makefile 
	$(MULTI_COMPILER) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(MULTI_BUILD_DIR)/* wet.exe *.vtk *.bov *.dat

