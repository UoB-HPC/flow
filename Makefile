
MPI     = yes
OPTIONS = -g -DENABLE_PROFILING -qopt-report=5
CFLAGS  = -O3 -std=gnu99 -xhost -qopenmp
LDFLAGS = #-lrt

ifeq ($(MPI), yes)
	CC = mpiicc
	OPTIONS += -DMPI
else
	CC = icc#mpiicc
endif

all:
	$(CC) $(CFLAGS) $(OPTIONS) $(LDFLAGS) main.c profiler.c -o hydro.exe

clean:
	rm -rf hydro.exe *.bov *.dat

