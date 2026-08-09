#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

/* Splits N items across `size` ranks as evenly as possible (the first
 * N % size ranks get one extra item), filling counts[]/displs[] for
 * use with MPI_Scatterv / MPI_Gatherv. */
static void compute_counts_displs(int N, int size, int *counts, int *displs) {
    int base = N / size;
    int rem = N % size;
    int offset = 0;
    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "usage: %s <input file> <output file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    Params params;
    double *all_x0 = NULL, *all_y0 = NULL, *all_vx = NULL, *all_vy = NULL;
    double *seed_x0 = malloc(0), *seed_y0 = malloc(0), *seed_vx = malloc(0), *seed_vy = malloc(0);

    /* --- Rank 0 reads the whole file; nobody else has touched disk. --- */
    if (rank == 0) {
        Point *points;
        if (read_input(argv[1], &params, &points) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        all_x0 = malloc((size_t)params.N * sizeof(double));
        all_y0 = malloc((size_t)params.N * sizeof(double));
        all_vx = malloc((size_t)params.N * sizeof(double));
        all_vy = malloc((size_t)params.N * sizeof(double));
        free(seed_x0); free(seed_y0); free(seed_vx); free(seed_vy);
        seed_x0 = malloc((size_t)params.K * sizeof(double));
        seed_y0 = malloc((size_t)params.K * sizeof(double));
        seed_vx = malloc((size_t)params.K * sizeof(double));
        seed_vy = malloc((size_t)params.K * sizeof(double));
        for (int i = 0; i < params.N; i++) {
            all_x0[i] = points[i].x0;
            all_y0[i] = points[i].y0;
            all_vx[i] = points[i].vx;
            all_vy[i] = points[i].vy;
        }
        /* Seed points = first K points (spec: "use first K points at
         * t=0 as initial positions of the centers"). Grab their
         * (x0,y0,vx,vy) now, while points[] still exists, so every
         * rank can later recompute centers at any t with zero
         * per-timestep communication (see init_centers_from_seed). */
        for (int k = 0; k < params.K; k++) {
            seed_x0[k] = points[k].x0;
            seed_y0[k] = points[k].y0;
            seed_vx[k] = points[k].vx;
            seed_vy[k] = points[k].vy;
        }
        free(points);
    }

    /* --- Broadcast the scalar params to everyone. --- */
    MPI_Bcast(&params.N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.dT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.LIMIT, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.QM, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int N = params.N, K = params.K;

    /* Non-root ranks don't have real seed values yet -- (re)allocate
     * to the correct size K now that K is known everywhere, so the
     * MPI_Bcast below has valid, correctly-sized buffers on every
     * rank. Rank 0's buffers already hold the real data and are the
     * same size, so this is a cheap no-op there. */
    if (rank != 0) {
        free(seed_x0); free(seed_y0); free(seed_vx); free(seed_vy);
        seed_x0 = malloc((size_t)K * sizeof(double));
        seed_y0 = malloc((size_t)K * sizeof(double));
        seed_vx = malloc((size_t)K * sizeof(double));
        seed_vy = malloc((size_t)K * sizeof(double));
    }

    /* --- Figure out this rank's slice of the N points. --- */
    int *counts = malloc((size_t)size * sizeof(int));
    int *displs = malloc((size_t)size * sizeof(int));
    compute_counts_displs(N, size, counts, displs);
    int local_N = counts[rank];

    double *lx0 = malloc((size_t)local_N * sizeof(double));
    double *ly0 = malloc((size_t)local_N * sizeof(double));
    double *lvx = malloc((size_t)local_N * sizeof(double));
    double *lvy = malloc((size_t)local_N * sizeof(double));

    MPI_Scatterv(all_x0, counts, displs, MPI_DOUBLE, lx0, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_y0, counts, displs, MPI_DOUBLE, ly0, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_vx, counts, displs, MPI_DOUBLE, lvx, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_vy, counts, displs, MPI_DOUBLE, lvy, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Point *local_points = malloc((size_t)local_N * sizeof(Point));
    for (int i = 0; i < local_N; i++) {
        local_points[i].x0 = lx0[i];
        local_points[i].y0 = ly0[i];
        local_points[i].vx = lvx[i];
        local_points[i].vy = lvy[i];
        /* Not yet assigned -- cluster_id is carried forward across
         * timesteps now (see the t=0 seeding comment below), so this
         * initial value matters: it must not equal any real cluster
         * index, so t=0's first assign_clusters call correctly counts
         * every point as "changed". */
        local_points[i].cluster_id = -1;
    }
    free(lx0); free(ly0); free(lvx); free(lvy);
    if (rank == 0) { free(all_x0); free(all_y0); free(all_vx); free(all_vy); }

    /* --- Every rank keeps a tiny replicated copy of the K seed
     * points' (x0,y0,vx,vy), so centers can be computed locally at
     * every t with zero communication (K < 20, cheap to replicate).
     * Rank 0 filled seed_x0/y0/vx/vy above; broadcast to everyone. --- */
    MPI_Bcast(seed_x0, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(seed_y0, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(seed_vx, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(seed_vy, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Cluster *centers = malloc((size_t)K * sizeof(Cluster));
    double *sum_x = malloc((size_t)K * sizeof(double));
    double *sum_y = malloc((size_t)K * sizeof(double));
    int *count = malloc((size_t)K * sizeof(int));
    double *gsum_x = malloc((size_t)K * sizeof(double));
    double *gsum_y = malloc((size_t)K * sizeof(double));
    int *gcount = malloc((size_t)K * sizeof(int));

    /* Full arrays, rebuilt every timestep (after convergence) on every
     * rank via MPI_Allgatherv so the diameter step -- which needs
     * cross-rank point pairs to be exact -- can be split across all
     * ranks by point index instead of running on rank 0 alone. */
    double *full_x = malloc((size_t)N * sizeof(double));
    double *full_y = malloc((size_t)N * sizeof(double));
    int *full_cid = malloc((size_t)N * sizeof(int));
    Point *full_points = malloc((size_t)N * sizeof(Point));

    /* Diameter work is split by point index (this rank's slice of the
     * outer i-loop over all N points), not by cluster -- reusing the
     * same counts/displs as the point scatter. Splitting by cluster
     * instead would badly imbalance ranks whenever cluster sizes are
     * very unequal (one rank could land the one giant cluster, since
     * K < 20 gives few, coarse buckets to divide among ranks); N is
     * usually >> size, so an i-range split stays even regardless of
     * how points fall into clusters. */
    int i_lo = displs[rank];
    int i_hi = i_lo + counts[rank];

    double *partial_diam = malloc((size_t)K * sizeof(double));
    double *diam = malloc((size_t)K * sizeof(double));

    double *local_x = malloc((size_t)local_N * sizeof(double));
    double *local_y = malloc((size_t)local_N * sizeof(double));
    int *local_cid = malloc((size_t)local_N * sizeof(int));

    /* Spec: "Use first K points at t=0 as initial positions of the
     * centers" -- seeded once, here (via init_centers_from_seed at
     * t=0), not re-derived at every t. Every subsequent t starts
     * K-means from the *previous* t's converged centers, carried
     * forward across timesteps -- consistent with "keep its center
     * for the next iteration" treating center state as persistent,
     * not discarded and rebuilt from the seed each time. cluster_id
     * is likewise never reset between timesteps: each rank's points
     * already have real (-1 initially, then real cluster ids after
     * t=0) assignments from read_input, giving assign_clusters a
     * meaningful "did this change" baseline from the first call. */
    init_centers_from_seed(centers, K, seed_x0, seed_y0, seed_vx, seed_vy, 0.0);

    int found = 0;
    double found_t = 0.0, found_q = 0.0;
    int n_steps = (int)(params.T / params.dT) + 1;

    /* Timed region excludes file I/O and the scatter of input data,
     * matching what main.c measures (see its comment) -- starts once
     * every rank is ready to begin the per-timestep computation.
     * MPI_Barrier ensures all ranks start the clock together. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int step = 0; step < n_steps && !found; step++) {
        double t = step * params.dT;

        update_positions(local_points, local_N, t);

        /* --- Inner K-Means loop (steps 2-5), synchronized across ranks --- */
        for (int iter = 0; iter < params.LIMIT; iter++) {
            int changed_local = assign_clusters(local_points, local_N, centers, K);
            int changed_global = 0;
            MPI_Allreduce(&changed_local, &changed_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (changed_global == 0) break;

            compute_partial_sums(local_points, local_N, K, sum_x, sum_y, count);
            MPI_Allreduce(sum_x, gsum_x, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(sum_y, gsum_y, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(count, gcount, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            centers_from_sums(centers, K, gsum_x, gsum_y, gcount);
        }

        /* --- Step 6: every rank needs the full, up-to-date (this
         * timestep, post-convergence) point set to compute exact
         * diameters -- a cluster's two farthest points can live on
         * different ranks. MPI_Allgatherv gives every rank the same
         * complete, current snapshot (not stale from a prior t). --- */
        for (int i = 0; i < local_N; i++) {
            local_x[i] = local_points[i].x;
            local_y[i] = local_points[i].y;
            local_cid[i] = local_points[i].cluster_id;
        }
        MPI_Allgatherv(local_x, local_N, MPI_DOUBLE, full_x, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_y, local_N, MPI_DOUBLE, full_y, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_cid, local_N, MPI_INT, full_cid, counts, displs, MPI_INT, MPI_COMM_WORLD);
        for (int i = 0; i < N; i++) {
            full_points[i].x = full_x[i];
            full_points[i].y = full_y[i];
            full_points[i].cluster_id = full_cid[i];
        }

        /* Each rank computes partial diameters using only its slice
         * [i_lo,i_hi) of point indices as the outer loop (still
         * compared against all N points) -- spreads the O(N^2/K)
         * diameter cost evenly across nodes by point count, regardless
         * of how unevenly points fall into the K clusters.
         * MPI_MAX-combining is exact: untouched contributions are 0.0,
         * which never wins against a real (non-negative) diameter. */
        cluster_diameters_i_range(full_points, N, K, i_lo, i_hi, partial_diam);
        MPI_Allreduce(partial_diam, diam, K, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double q = quality_measure_from_diam(diam, centers, K);
        int this_found = (q < params.QM) ? 1 : 0;

        if (this_found) {
            found = 1;
            found_t = t;
            found_q = q;
        }
    }

    /* All ranks reach here at roughly the same time (the loop's MPI
     * collectives keep them in lockstep each iteration), so measuring
     * on rank 0 alone is representative -- no barrier needed here. */
    double t_end = MPI_Wtime();
    if (rank == 0) {
        fprintf(stderr, "Parallel computation time: %.6f s (%d ranks x %s threads)\n",
                t_end - t_start, size, getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "1");
    }

    if (rank == 0) {
        write_output(argv[2], found, found_t, found_q, centers, K);
        if (found) {
            printf("First occurrence t = %g with q = %g\n", found_t, found_q);
        } else {
            printf("There was no t found.\n");
        }
    }

    free(local_points); free(counts); free(displs);
    free(seed_x0); free(seed_y0); free(seed_vx); free(seed_vy);
    free(centers); free(sum_x); free(sum_y); free(count);
    free(gsum_x); free(gsum_y); free(gcount);
    free(local_x); free(local_y); free(local_cid);
    free(full_x); free(full_y); free(full_cid); free(full_points);
    free(partial_diam); free(diam);

    MPI_Finalize();
    return 0;
}
