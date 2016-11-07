CC = icc
CFLAGS  = -O3 -std=gnu99 -xhost -qopenmp
LDFLAGS = #-lrt
OPTIONS = -g -DENABLE_PROFILING -qopt-report=5

all:
	$(CC) $(CFLAGS) $(OPTIONS) $(LDFLAGS) main.c profiler.c -o hydro.exe

clean:
	rm -rf hydro.exe *.bov *.dat

