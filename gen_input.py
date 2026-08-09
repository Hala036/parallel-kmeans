#!/usr/bin/env python3
"""Generates an input file for the K-Means assignment.

Each of the K clusters starts as a tight group at its own fixed
"home" position (so cluster diameter is small and roughly independent
of N or K -- controlled entirely by `jitter` below) and drifts at a
fixed velocity along a *random* direction, not radially outward. Speed
and direction are randomized per cluster (not just per-cluster angle
around a circle), so clusters cross paths, pass close to each other,
and separate again at different, unpredictable times instead of all
monotonically spreading apart together. That makes the quality measure
(diameter / inter-center distance) fluctuate non-monotonically across
[0, T] -- some clusters are transiently close (poor quality) while
others are far apart, and which timestep first satisfies a given QM
isn't obvious from N or K alone. Unlike a design where quality moves
smoothly in one direction, this keeps the search from being trivially
satisfied at t=0 without needing QM hand-tuned per N.

Cluster diameter (~`jitter`, fixed) and home-position spacing
(~`home_spacing`, fixed) are both independent of N and K, which keeps
the quality range roughly similar across dataset sizes -- but not
exactly: more points per cluster means the *sampled* max pairwise
distance within the jitter box creeps closer to its true maximum, so
diameter (and therefore quality) drifts up somewhat as N grows.
Verified qm=0.6 (the default) finds a match within [0, T] for N in
{2000, 20000, 50000}; for much larger N you may need to bump QM
slightly -- check the actual `q` values a run reports (stderr/output
file) and adjust if it says "There was no t found" when you expected
a match, or matches suspiciously close to t=0.
"""
import math
import random
import sys

def generate(n_points=100000, k_clusters=15, t_end=50.0, dt=0.1,
             limit=1000, qm=0.6, seed=42, out_path="input_large.txt"):
    random.seed(seed)

    lines = [f"{n_points} {k_clusters} {t_end} {dt} {limit} {qm}"]

    jitter = 3.0          # per-point spread around its cluster's home position
    home_spacing = 25.0   # distance between adjacent home positions on the grid
    speed_range = (1.0, 3.0)

    # Home positions on a grid, not a circle -- gives a mix of near
    # and far cluster pairs (unlike a circle, where every adjacent
    # pair is equally spaced), which is part of what makes some
    # clusters pass close to each other while others stay apart.
    side = math.ceil(math.sqrt(k_clusters))
    homes = [(home_spacing * (i % side), home_spacing * (i // side))
             for i in range(k_clusters)]

    base = n_points // k_clusters
    rem = n_points % k_clusters
    for k in range(k_clusters):
        count = base + (1 if k < rem else 0)
        hx, hy = homes[k]
        # Random direction and speed per cluster (not radially
        # outward from a shared center) -- this is what causes
        # clusters to cross paths instead of just spreading apart in
        # lockstep.
        angle = random.uniform(0, 2.0 * math.pi)
        speed = random.uniform(*speed_range)
        vx, vy = speed * math.cos(angle), speed * math.sin(angle)
        for _ in range(count):
            x = hx + random.uniform(-jitter, jitter)
            y = hy + random.uniform(-jitter, jitter)
            lines.append(f"{x:.4f} {y:.4f} {vx:.4f} {vy:.4f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_path}: N={n_points} K={k_clusters} T={t_end} "
          f"dT={dt} LIMIT={limit} QM={qm}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    qm = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    out = sys.argv[4] if len(sys.argv) > 4 else "input_large.txt"
    generate(n_points=n, k_clusters=k, qm=qm, out_path=out)
