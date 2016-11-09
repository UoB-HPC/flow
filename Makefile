CC = icc#mpiicc
OPTIONS = -g -DENABLE_PROFILING -qopt-report=5
CFLAGS  = -O3 -std=gnu99 -xhost -qopenmp
LDFLAGS = #-lrt

all:
	$(CC) $(CFLAGS) $(OPTIONS) $(LDFLAGS) main.c profiler.c -o hydro.exe

clean:
	rm -rf hydro.exe *.bov *.dat

