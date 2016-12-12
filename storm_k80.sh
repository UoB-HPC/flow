
OMP_NUM_THREADS=1 srun -n 1 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu1
echo "Finished 1 GPU"
OMP_NUM_THREADS=1 srun -n 2 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu2
echo "Finished 2 GPU"
OMP_NUM_THREADS=1 srun -n 3 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu3
echo "Finished 3 GPU"
OMP_NUM_THREADS=1 srun -n 4 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu4
echo "Finished 4 GPU"
OMP_NUM_THREADS=1 srun -n 5 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu5
echo "Finished 5 GPU"
OMP_NUM_THREADS=1 srun -n 6 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu6
echo "Finished 6 GPU"
OMP_NUM_THREADS=1 srun -n 7 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu7
echo "Finished 7 GPU"
OMP_NUM_THREADS=1 srun -n 8 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu8
echo "Finished 8 GPU"
OMP_NUM_THREADS=1 srun -n 9 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu9
echo "Finished 9 GPU"
OMP_NUM_THREADS=1 srun -n 10 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu10
echo "Finished 10 GPU"
OMP_NUM_THREADS=1 srun -n 11 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu11
echo "Finished 11 GPU"
OMP_NUM_THREADS=1 srun -n 12 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu12
echo "Finished 12 GPU"
OMP_NUM_THREADS=1 srun -n 13 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu13
echo "Finished 13 GPU"
OMP_NUM_THREADS=1 srun -n 14 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu14
echo "Finished 14 GPU"
OMP_NUM_THREADS=1 srun -n 15 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu15
echo "Finished 15 GPU"
OMP_NUM_THREADS=1 srun -n 16 ./wet.cuda 5000 5000 50 > strong_5000_50_storm_k80/OUT_gpu16
echo "Finished 16 GPU"

