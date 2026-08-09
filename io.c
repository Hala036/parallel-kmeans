#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

int read_input(const char *filename, Params *params, Point **points) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "read_input: could not open %s\n", filename);
        return -1;
    }

    // first line: N K T dT LIMIT QM
    if (fscanf(fp, "%d %d %lf %lf %d %lf",
               &params->N, &params->K, &params->T,
               &params->dT, &params->LIMIT, &params->QM) != 6) {
        fprintf(stderr, "read_input: malformed header line\n");
        fclose(fp);
        return -1;
    }

    if (params->N <= 0 || params->K <= 0 || params->K > params->N) {
        fprintf(stderr, "read_input: invalid N/K (N=%d, K=%d)\n",
                params->N, params->K);
        fclose(fp);
        return -1;
    }

    Point *pts = malloc((size_t)params->N * sizeof(Point));
    if (!pts) {
        fprintf(stderr, "read_input: allocation of %d points failed\n", params->N);
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < params->N; i++) {
        double x, y, vx, vy;
        if (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &vx, &vy) != 4) {
            fprintf(stderr, "read_input: malformed point at line %d\n", i + 2);
            free(pts);
            fclose(fp);
            return -1;
        }
        pts[i].x0 = x;
        pts[i].y0 = y;
        pts[i].vx = vx;
        pts[i].vy = vy;
        pts[i].x = x;
        pts[i].y = y;
        pts[i].cluster_id = -1;
    }

    fclose(fp);
    *points = pts;
    return 0;
}

int write_output(const char *filename, int found, double t, double q,
                  const Cluster *centers, int K) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "write_output: could not open %s\n", filename);
        return -1;
    }

    if (found) {
        fprintf(fp, "First occurrence t = %g with q = %g\n", t, q);
        fprintf(fp, "Centers of the clusters:\n");
        for (int k = 0; k < K; k++) {
            fprintf(fp, "%g %g\n", centers[k].cx, centers[k].cy);
        }
    } else {
        fprintf(fp, "There was no t found.\n");
    }

    fclose(fp);
    return 0;
}
