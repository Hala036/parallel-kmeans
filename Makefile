# Local (Mac) builds: no OpenMP runtime available under Apple clang, so
# #pragma omp lines are silently ignored -- plain sequential loops.
# Use `make cluster` on the cluster (GCC + MPI + OpenMP all present) for
# the real hybrid MPI+OpenMP build.
CC       = gcc
MPICC    = mpicc
CFLAGS   = -Wall -Wextra -O2 -std=c17

SEQ_TARGET = kmeans_seq
MPI_TARGET = kmeans_mpi

.PHONY: all seq mpi cluster clean

all: seq mpi

seq: $(SEQ_TARGET)
mpi: $(MPI_TARGET)

$(SEQ_TARGET): main.c kmeans.c io.c kmeans.h
	$(CC) $(CFLAGS) -o $@ main.c kmeans.c io.c -lm

$(MPI_TARGET): main_mpi.c kmeans.c io.c kmeans.h
	$(MPICC) $(CFLAGS) -o $@ main_mpi.c kmeans.c io.c -lm

# Cluster build: adds -fopenmp for real MPI+OpenMP hybrid parallelism.
cluster: CFLAGS += -fopenmp
cluster: $(MPI_TARGET)

clean:
	rm -f $(SEQ_TARGET) $(MPI_TARGET) *.o
