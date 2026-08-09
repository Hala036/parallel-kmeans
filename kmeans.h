#ifndef KMEANS_H
#define KMEANS_H

/* One data point: initial position + velocity, plus which cluster
 * it currently belongs to. cluster_id gets updated every K-Means
 * iteration, so it lives on the point itself rather than in a
 * separate parallel array. */
typedef struct {
    double x0, y0;      /* initial position at t = 0 */
    double vx, vy;       /* velocity */
    double x, y;          /* current position, filled in by update_positions() */
    int cluster_id;      /* index of the cluster this point currently belongs to */
} Point;

/* The parameters read from the first line of input.txt */
typedef struct {
    int N;          /* number of points */
    int K;          /* number of clusters */
    double T;       /* end of time interval */
    double dT;      /* time step */
    int LIMIT;      /* max K-Means iterations per timestep */
    double QM;      /* quality threshold to stop at */
} Params;

/* A cluster center. Kept separate from Point since centers are not
 * data points themselves -- they're averages, and K << N. */
typedef struct {
    double cx, cy;
} Cluster;

/* Reads input.txt: first line is "N K T dT LIMIT QM", followed by
 * N lines of "x y vx vy". Allocates and fills *points (caller must
 * free it) and fills *params. Returns 0 on success, -1 on failure. */
int read_input(const char *filename, Params *params, Point **points);

/* Writes the result to filename in the required output format.
 * If found != 0, writes "First occurrence..." + centers.
 * If found == 0, writes "There was no t found." */
int write_output(const char *filename, int found, double t, double q,
                  const Cluster *centers, int K);

/* Sets points[i].x / .y to the position at time t, using the point's
 * initial position and velocity (x0,y0,vx,vy are never modified). */
void update_positions(Point *points, int N, double t);

/* Step 1 (only called once, at the very start): copies the first K
 * points' current (x,y) into centers[0..K-1]. */
void init_centers(const Point *points, int K, Cluster *centers);

/* Same idea, but from the K "seed" points' initial position/velocity
 * directly, evaluated at time t. Used by the parallel driver: every
 * rank keeps a small replicated copy of the seed points' (x0,y0,vx,vy)
 * and computes centers for this t locally -- no communication needed
 * for this step, since K is tiny (<20). */
void init_centers_from_seed(Cluster *centers, int K,
                             const double *seed_x0, const double *seed_y0,
                             const double *seed_vx, const double *seed_vy,
                             double t);

/* Step 2: assigns every point to its nearest center (sets cluster_id).
 * Returns the number of points whose cluster_id changed from before
 * the call -- this is what step 4's convergence check uses. */
int assign_clusters(Point *points, int N, const Cluster *centers, int K);

/* Step 3: recomputes each center as the average of its member points.
 * A cluster with no members keeps its previous center unchanged. */
/* Step 3, split into two pieces so the MPI version can put an
 * MPI_Allreduce between them:
 *   compute_partial_sums: purely local -- sums/counts this rank's
 *     own points per cluster. Sequential code just calls this on
 *     ALL N points (nothing to reduce with only one "rank").
 *   centers_from_sums: turns (already-global) sums/counts into new
 *     centers. Empty cluster (count==0) keeps its old center. */
void compute_partial_sums(const Point *points, int N, int K,
                           double *sum_x, double *sum_y, int *count);
void centers_from_sums(Cluster *centers, int K,
                        const double *sum_x, const double *sum_y, const int *count);

void update_centers(const Point *points, int N, Cluster *centers, int K);

/* Steps 2-5: repeatedly assigns + recomputes until zero points change
 * cluster or LIMIT iterations have run. Returns the number of
 * iterations actually performed. */
int run_kmeans(Point *points, int N, Cluster *centers, int K, int LIMIT);

/* Diameter of one cluster: the largest distance between any two of
 * its member points. Returns 0.0 for a cluster with 0 or 1 members. */
double cluster_diameter(const Point *points, int N, int cluster_id);

/* Partial per-cluster diameters using only outer-loop point indices
 * in [i_lo, i_hi) (still compared against all N points). Lets the MPI
 * driver split the diameter workload across ranks by point index
 * rather than by cluster, so it stays balanced even when cluster
 * sizes are very unequal. diam_out must have room for K entries;
 * combine partial results from multiple ranks/calls with MPI_MAX. */
void cluster_diameters_i_range(const Point *points, int N, int K,
                                int i_lo, int i_hi, double *diam_out);

/* Step 6: the quality measure q, averaged over all ordered pairs
 * (i,j), i != j, of d_i / D_ij. */
double quality_measure(const Point *points, int N, const Cluster *centers, int K);

/* Same computation as quality_measure, but takes already-computed
 * per-cluster diameters -- used by the MPI driver, which combines
 * diameters computed in parallel across ranks (see
 * cluster_diameters_range) before this final (cheap, O(K^2)) step. */
double quality_measure_from_diam(const double *diam, const Cluster *centers, int K);

#endif
