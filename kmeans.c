#include <math.h>
#include <stdlib.h>
#include "kmeans.h"

static double dist(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

void update_positions(Point *points, int N, double t) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        points[i].x = points[i].x0 + t * points[i].vx;
        points[i].y = points[i].y0 + t * points[i].vy;
    }
}

void init_centers(const Point *points, int K, Cluster *centers) {
    for (int k = 0; k < K; k++) {
        centers[k].cx = points[k].x;
        centers[k].cy = points[k].y;
    }
}

void init_centers_from_seed(Cluster *centers, int K,
                             const double *seed_x0, const double *seed_y0,
                             const double *seed_vx, const double *seed_vy,
                             double t) {
    for (int k = 0; k < K; k++) {
        centers[k].cx = seed_x0[k] + t * seed_vx[k];
        centers[k].cy = seed_y0[k] + t * seed_vy[k];
    }
}

int assign_clusters(Point *points, int N, const Cluster *centers, int K) {
    int changed = 0;

    // each point only reads centers[] and writes its own cluster_id, so this is safe to parallelize directly
    #pragma omp parallel for reduction(+:changed)
    for (int i = 0; i < N; i++) {
        int best_k = 0;
        double best_dist = dist(points[i].x, points[i].y,
                                 centers[0].cx, centers[0].cy);

        for (int k = 1; k < K; k++) {
            double d = dist(points[i].x, points[i].y,
                             centers[k].cx, centers[k].cy);
            if (d < best_dist) {
                best_dist = d;
                best_k = k;
            }
        }

        if (points[i].cluster_id != best_k) {
            changed++;
            points[i].cluster_id = best_k;
        }
    }

    return changed;
}

void compute_partial_sums(const Point *points, int N, int K,
                           double *sum_x, double *sum_y, int *count) {
    for (int k = 0; k < K; k++) {
        sum_x[k] = 0.0;
        sum_y[k] = 0.0;
        count[k] = 0;
    }

    // array-section reduction: each thread gets private sum_x/sum_y/count, combined at the end
    #pragma omp parallel for reduction(+:sum_x[:K], sum_y[:K], count[:K])
    for (int i = 0; i < N; i++) {
        int k = points[i].cluster_id;
        sum_x[k] += points[i].x;
        sum_y[k] += points[i].y;
        count[k]++;
    }
}

void centers_from_sums(Cluster *centers, int K,
                        const double *sum_x, const double *sum_y, const int *count) {
    for (int k = 0; k < K; k++) {
        if (count[k] > 0) {
            centers[k].cx = sum_x[k] / count[k];
            centers[k].cy = sum_y[k] / count[k];
        }
        // else: empty cluster, leave centers[k] as it was
    }
}

void update_centers(const Point *points, int N, Cluster *centers, int K) {
    double *sum_x = malloc((size_t)K * sizeof(double));
    double *sum_y = malloc((size_t)K * sizeof(double));
    int *count = malloc((size_t)K * sizeof(int));

    // sequential case: same as the parallel path, just with one "rank" owning all N points
    compute_partial_sums(points, N, K, sum_x, sum_y, count);
    centers_from_sums(centers, K, sum_x, sum_y, count);

    free(sum_x);
    free(sum_y);
    free(count);
}

int run_kmeans(Point *points, int N, Cluster *centers, int K, int LIMIT) {
    int iter = 0;

    for (iter = 0; iter < LIMIT; iter++) {
        int changed = assign_clusters(points, N, centers, K);
        if (changed == 0) {
            break;  // converged, centers are already consistent with this assignment
        }
        update_centers(points, N, centers, K);
    }

    return iter;
}

double cluster_diameter(const Point *points, int N, int cluster_id) {
    double max_d = 0.0;

    // workload is triangular (shrinks as i grows), so schedule(dynamic) avoids leaving threads idle
    #pragma omp parallel for schedule(dynamic) reduction(max:max_d)
    for (int i = 0; i < N; i++) {
        if (points[i].cluster_id != cluster_id) continue;
        for (int j = i + 1; j < N; j++) {
            if (points[j].cluster_id != cluster_id) continue;
            double d = dist(points[i].x, points[i].y, points[j].x, points[j].y);
            if (d > max_d) max_d = d;
        }
    }

    return max_d;
}

// splits the diameter scan across ranks by point index rather than by cluster, so it stays balanced when cluster sizes are uneven; combine results with MPI_MAX
void cluster_diameters_i_range(const Point *points, int N, int K,
                                int i_lo, int i_hi, double *diam_out) {
    for (int k = 0; k < K; k++) diam_out[k] = 0.0;

    #pragma omp parallel
    {
        double *local_max = calloc((size_t)K, sizeof(double));

        #pragma omp for schedule(dynamic)
        for (int i = i_lo; i < i_hi; i++) {
            int ci = points[i].cluster_id;
            for (int j = i + 1; j < N; j++) {
                if (points[j].cluster_id != ci) continue;
                double d = dist(points[i].x, points[i].y, points[j].x, points[j].y);
                if (d > local_max[ci]) local_max[ci] = d;
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                if (local_max[k] > diam_out[k]) diam_out[k] = local_max[k];
            }
        }

        free(local_max);
    }
}

double quality_measure_from_diam(const double *diam, const Cluster *centers, int K) {
    double sum = 0.0;
    int pairs = 0;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if (i == j) continue;
            double D_ij = dist(centers[i].cx, centers[i].cy,
                                centers[j].cx, centers[j].cy);
            if (D_ij > 0.0) {  // guards against a degenerate 0 distance between centers
                sum += diam[i] / D_ij;
            }
            pairs++;
        }
    }

    return pairs > 0 ? sum / pairs : 0.0;
}

double quality_measure(const Point *points, int N, const Cluster *centers, int K) {
    double *diam = malloc((size_t)K * sizeof(double));
    for (int k = 0; k < K; k++) {
        diam[k] = cluster_diameter(points, N, k);
    }

    double q = quality_measure_from_diam(diam, centers, K);

    free(diam);
    return q;
}
