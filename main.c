#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }

    Params params;
    Point *points;
    if (read_input(argv[1], &params, &points) != 0) {
        return 1;
    }

    Cluster *centers = malloc((size_t)params.K * sizeof(Cluster));

    /* Spec: "Use first K points at t=0 as initial positions of the
     * centers" -- seeded once, here, not re-derived at every t. Every
     * subsequent t starts K-means from the *previous* t's converged
     * centers (points move via update_positions, but centers have no
     * velocity of their own -- they're recomputed by run_kmeans from
     * wherever the points ended up). This also means cluster_id is
     * never reset to -1 between timesteps: a point's previous
     * assignment is exactly the natural starting guess for the next
     * t, consistent with "keep its center for the next iteration"
     * treating center/assignment state as carried forward, not
     * discarded, across iterations. */
    init_centers(points, params.K, centers);

    /* Timed region excludes file I/O -- this is what should be
     * compared against the MPI version's MPI_Wtime-measured region
     * (see main_mpi.c) to demonstrate parallel speedup. */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int found = 0;
    double found_t = 0.0, found_q = 0.0;

    int n_steps = (int)(params.T / params.dT) + 1;

    for (int step = 0; step < n_steps && !found; step++) {
        double t = step * params.dT;

        update_positions(points, params.N, t);

        run_kmeans(points, params.N, centers, params.K, params.LIMIT);

        double q = quality_measure(points, params.N, centers, params.K);

        if (q < params.QM) {
            found = 1;
            found_t = t;
            found_q = q;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    fprintf(stderr, "Sequential computation time: %.6f s\n", elapsed);

    write_output(argv[2], found, found_t, found_q, centers, params.K);

    if (found) {
        printf("First occurrence t = %g with q = %g\n", found_t, found_q);
    } else {
        printf("There was no t found.\n");
    }

    free(points);
    free(centers);
    return 0;
}
