# User defined parameters
KERNELS          	 = omp3
COMPILER         	 = GCC
MPI              	 = no
DECOMP					 	 = TILES
OPTIONS          	 = #-DENABLE_PROFILING 
ARCH_COMPILER_CC   = gcc
ARCH_COMPILER_CPP  = g++

# Compiler-specific flags
MAC_RPATH				 	 = -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL     	 = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
								 	   $(MAC_RPATH) -Wall -qopt-report=5 #-xhost
CFLAGS_INTEL_KNL 	 = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
								 	   -xMIC-AVX512 -Wall -qopt-report=5
CFLAGS_GCC       	 = -O3 -march=native -fopenmp -std=gnu99
CFLAGS_GCC_KNL   	 = -O3 -fopenmp -std=gnu99 \
										 -mavx512f -mavx512cd -mavx512er -mavx512pf #-fopt-info-vec-all
CFLAGS_GCC_POWER   = -O3 -mcpu=power8 -mtune=power8 -fopenmp -std=gnu99
CFLAGS_CRAY      	 = -lrt -hlist=a
CFLAGS_XL		 			 = -O5 -qsmp=omp -qarch=pwr8 -qtune=pwr8 -qaltivec
CFLAGS_XL_OMP4		 = -O5 -qsmp -qoffload
CFLAGS_CLANG_OMP4  = -O3 -Wall -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps \
										 -fopenmp=libomp --cuda-path=$(CUDA_PATH) -DCLANG
CFLAGS_CLANG			 = -std=gnu99 -fopenmp=libiomp5 -march=native -Wall
CFLAGS_PGI_NV			 = -fast -acc -ta=tesla:cc60 -Minfo=acc
CFLAGS_PGI_MC			 = -ta=multicore -fast 
CFLAGS_INTEL_RAJA  = -O3 -qopenmp -std=c++11 -DINTEL -Wall
CFLAGS_NVCC_RAJA   = -O3 -arch=sm_60 -x cu -std=c++11 --expt-extended-lambda -DRAJA_USE_CUDA

ifeq ($(KERNELS), cuda)
  CHECK_CUDA_ROOT = yes
endif
ifeq ($(COMPILER), CLANG_OMP4)
  CHECK_CUDA_ROOT = yes
endif

ifeq ($(CHECK_CUDA_ROOT), yes)
ifeq ("${CUDA_PATH}", "")
$(error "$$CUDA_PATH is not set, please set this to the root of your CUDA install.")
endif
endif

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -g -DDEBUG 
endif

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
ARCH_LINKER    			= $(ARCH_COMPILER_CC)
ARCH_FLAGS     			= $(CFLAGS_$(COMPILER))
ARCH_LDFLAGS   			= $(ARCH_FLAGS) -lm
ARCH_BUILD_DIR 			= ../obj/flow/
ARCH_DIR       			= ..

ifeq ($(KERNELS), raja)
ifeq ("${RAJA_PATH}", "")
$(error "$$RAJA_PATH is not set, please set this to the root of your RAJA install.")
endif
  OPTIONS += -I$(RAJA_PATH)/include/
  ARCH_LDFLAGS = -arch=sm_60 -Xcompiler "-fopenmp" -lRAJA -L$(RAJA_PATH)/lib
endif

ifeq ($(KERNELS), cuda)
include Makefile.cuda
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(ARCH_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(ARCH_DIR)/*.c))
SRC_CLEAN  = $(subst $(ARCH_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(ARCH_BUILD_DIR)/%.o, $(SRC_CLEAN))

flow: make_build_dir $(OBJS) Makefile
	$(ARCH_LINKER) $(OBJS) $(ARCH_LDFLAGS) -o flow.$(KERNELS)

# Rule to make controlling code
$(ARCH_BUILD_DIR)/%.o: %.c Makefile 
	$(ARCH_COMPILER_CC) $(ARCH_FLAGS) $(OPTIONS) -c $< -o $@

$(ARCH_BUILD_DIR)/%.o: $(ARCH_DIR)/%.c Makefile 
	$(ARCH_COMPILER_CC) $(ARCH_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(ARCH_BUILD_DIR)/
	@mkdir -p $(ARCH_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(ARCH_BUILD_DIR)/* flow.exe *.vtk *.bov *.dat *.optrpt *.cub *.ptx *.i *.bc *.o *.s *.lk

