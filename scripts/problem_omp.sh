
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 100 100 50 > problem_100_to_5000_50/OUT_100_omp3
echo "finished 100"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 200 200 50 > problem_100_to_5000_50/OUT_200_omp3
echo "finished 200"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 400 400 50 > problem_100_to_5000_50/OUT_400_omp3
echo "finished 400"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 800 800 50 > problem_100_to_5000_50/OUT_800_omp3
echo "finished 800"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 1000 1000 50 > problem_100_to_5000_50/OUT_1000_omp3
echo "finished 1000"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 2000 2000 50 > problem_100_to_5000_50/OUT_2000_omp3
echo "finished 2000"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 3000 3000 50 > problem_100_to_5000_50/OUT_3000_omp3
echo "finished 3000"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 4000 4000 50 > problem_100_to_5000_50/OUT_4000_omp3
echo "finished 4000"
OMP_NUM_THREADS=44 KMP_AFFINITY=compact aprun -n 1 -d 44 -N 1 -cc depth ./wet.omp3 5000 5000 50 > problem_100_to_5000_50/OUT_5000_omp3
echo "finished 5000"

