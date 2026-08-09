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

    // timed region excludes file I/O, to match what main_mpi.c measures
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int found = 0;
    double found_t = 0.0, found_q = 0.0;

    int n_steps = (int)(params.T / params.dT) + 1;

    for (int step = 0; step < n_steps && !found; step++) {
        double t = step * params.dT;

        update_positions(points, params.N, t);

        // re-seed centers from the first K points and re-run K-Means fresh at every t
        init_centers(points, params.K, centers);
        for (int i = 0; i < params.N; i++) {
            points[i].cluster_id = -1;
        }

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
