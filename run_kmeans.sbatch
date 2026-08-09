#!/bin/bash
#SBATCH --partition rtx3070			### Specify partition name where to run a job.
#SBATCH --time 0-00:30:00			### Job running time limit. Format: D-H:MM:SS
#SBATCH --output kmeans-%J.out			### Output log for running job - %J is the job number variable
#SBATCH --error kmeans-%J.err			### Output log for running job - %J is the job number variable

#SBATCH --job-name 'kmeans_mpi'

# Reserve the full 8 cores on each of the 4 nodes (32 cores total) as
# plain slots -- --cpus-per-task is intentionally NOT set here, since
# each sweep run below picks its own ranks-per-node/threads-per-rank
# split of those 8 cores via --map-by ppr:R:node + OMP_NUM_THREADS.
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4

module load anaconda
source activate myenv

export OMPI_CC=gcc
export OMPI_CXX=g++

# compile both the sequential baseline and the MPI+OpenMP version
gcc -Wall -Wextra -O2 -std=c17 -fopenmp -o kmeans_seq main.c kmeans.c io.c -lm
mpicc -Wall -Wextra -O2 -std=c17 -fopenmp -o kmeans_mpi main_mpi.c kmeans.c io.c -lm

# --- Sequential baseline: what all the parallel configs below must beat ---
./kmeans_seq input_large.txt output_seq.txt 2> time_sequential.log
echo "Done: sequential baseline"

# Scaling sweep: same 4-node x 8-core allocation, varying how it's
# split between MPI ranks and OpenMP threads, so we can measure where
# the speedup actually comes from before committing to one config.
# Each --map-by ppr:R:node places R ranks per node; OMP_NUM_THREADS
# fills the rest of that node's cores with threads per rank
# (R * OMP_NUM_THREADS = 8, one node's core count, in every row below).

# --- 1 rank/node (4 ranks total), 8 threads/rank ---
export OMP_NUM_THREADS=8
mpirun -np 4 --map-by ppr:1:node ./kmeans_mpi input_large.txt output_4x8.txt \
  2> time_4ranks_8threads.log
echo "Done: 4 ranks x 8 threads"

# --- 2 ranks/node (8 ranks total), 4 threads/rank ---
export OMP_NUM_THREADS=4
mpirun -np 8 --map-by ppr:2:node ./kmeans_mpi input_large.txt output_8x4.txt \
  2> time_8ranks_4threads.log
echo "Done: 8 ranks x 4 threads"

# --- 4 ranks/node (16 ranks total), 2 threads/rank ---
export OMP_NUM_THREADS=2
mpirun -np 16 --map-by ppr:4:node ./kmeans_mpi input_large.txt output_16x2.txt \
  2> time_16ranks_2threads.log
echo "Done: 16 ranks x 2 threads"

# --- 8 ranks/node (32 ranks total), pure MPI, 1 thread/rank ---
export OMP_NUM_THREADS=1
mpirun -np 32 --map-by ppr:8:node ./kmeans_mpi input_large.txt output_32x1.txt \
  2> time_32ranks_1thread.log
echo "Done: 32 ranks x 1 thread"

# --- 1 rank total (single node), 8 threads: pure OpenMP, no MPI parallelism ---
export OMP_NUM_THREADS=8
mpirun -np 1 --map-by ppr:1:node ./kmeans_mpi input_large.txt output_1x8.txt \
  2> time_1rank_8threads.log
echo "Done: 1 rank x 8 threads"

echo "All runs complete. Compare time_*.log for computation times."
