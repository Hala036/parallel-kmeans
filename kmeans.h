#ifndef KMEANS_H
#define KMEANS_H

typedef struct {
    double x0, y0;     // initial position at t = 0
    double vx, vy;
    double x, y;        // position at the current t
    int cluster_id;
} Point;

typedef struct {
    int N;
    int K;
    double T;       // end of time interval
    double dT;      // time step
    int LIMIT;      // max K-Means iterations per timestep
    double QM;      // quality threshold to stop at
} Params;

typedef struct {
    double cx, cy;
} Cluster;

// Reads "N K T dT LIMIT QM" then N lines of "x y vx vy". Allocates *points.
int read_input(const char *filename, Params *params, Point **points);

int write_output(const char *filename, int found, double t, double q,
                  const Cluster *centers, int K);

void update_positions(Point *points, int N, double t);

// Seeds centers from the first K points' current position.
void init_centers(const Point *points, int K, Cluster *centers);

// Same as init_centers, but computed from replicated seed data instead of a scatter.
void init_centers_from_seed(Cluster *centers, int K,
                             const double *seed_x0, const double *seed_y0,
                             const double *seed_vx, const double *seed_vy,
                             double t);

// Assigns each point to its nearest center; returns how many cluster_ids changed.
int assign_clusters(Point *points, int N, const Cluster *centers, int K);

// Local half of update_centers, so MPI can reduce between the two halves.
void compute_partial_sums(const Point *points, int N, int K,
                           double *sum_x, double *sum_y, int *count);
// Empty clusters (count == 0) keep their previous center.
void centers_from_sums(Cluster *centers, int K,
                        const double *sum_x, const double *sum_y, const int *count);

void update_centers(const Point *points, int N, Cluster *centers, int K);

// Repeats assign/update until nothing changes or LIMIT is hit; returns iterations run.
int run_kmeans(Point *points, int N, Cluster *centers, int K, int LIMIT);

// Largest distance between any two points in the cluster.
double cluster_diameter(const Point *points, int N, int cluster_id);

// Same as cluster_diameter, but only scans point indices [i_lo, i_hi) -- for splitting across MPI ranks.
void cluster_diameters_i_range(const Point *points, int N, int K,
                                int i_lo, int i_hi, double *diam_out);

double quality_measure(const Point *points, int N, const Cluster *centers, int K);

// Same as quality_measure, but from diameters already computed elsewhere (e.g. reduced across ranks).
double quality_measure_from_diam(const double *diam, const Cluster *centers, int K);

#endif
