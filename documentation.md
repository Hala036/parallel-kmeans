# Hybrid MPI+OpenMP K-Means on Moving Points — Documentation

## 1. What the program does

Given N points that move at constant velocity, the program sweeps time
t = 0, dT, 2dT, ..., T. At each t it re-clusters the current point
positions into K clusters using a simplified K-Means algorithm, then
evaluates a Quality Measure q for that clustering. The program stops
and reports the first t at which q < QM, or reports that no such t
exists if the sweep completes without one.

Two binaries are built from the same shared algorithm core
(`kmeans.c`, `io.c`):
- `kmeans_seq` — single-process baseline, used only for correctness
  cross-checking and the timing comparison.
- `kmeans_mpi` — the hybrid MPI+OpenMP parallel version, targeting a
  4-node × 8-core cluster.

## 2. What was parallelized, and how

### 2.1 Two-level parallelism, matching the two-level hardware

The cluster has two distinct levels of parallelism available, and the
program maps one parallel strategy to each:

- **Across nodes (MPI):** nodes do not share memory, so the only way
  to use more than one node is message passing. The N points are
  **partitioned** across MPI ranks — each rank owns and operates on
  only its own slice of the points.
- **Within a node (OpenMP):** the 8 cores on a single node share
  memory directly, which is cheaper than message passing. Each rank's
  hot loops (position update, cluster assignment, partial-sum
  accumulation, diameter computation) are parallelized with
  `#pragma omp parallel for` across that rank's threads.

### 2.2 Data distribution: points partitioned, centers replicated

The N points are split across ranks with `MPI_Scatterv`, as evenly as
possible (`N / size`, with the first `N % size` ranks getting one
extra point — handles the case where N doesn't divide evenly).

The K cluster **centers are not partitioned** — every rank keeps a
full, identical copy of all K centers (K is small, well under 20, so
this is cheap). This is the key design choice that keeps most of the
algorithm communication-free:

- **Assignment step:** each rank compares its own local points against
  its own local (but globally identical) copy of the centers. Zero
  communication needed — this step is embarrassingly parallel.
- **Update step:** each rank computes *partial* per-cluster sums and
  counts for its own points only (`compute_partial_sums`), then a
  small `MPI_Allreduce` (3 calls, each only K numbers) combines these
  across all ranks and delivers the combined total back to every rank.
  Every rank then independently divides sum/count to get the new
  centers — no rank needs to receive another rank's centers directly,
  since all ranks compute the identical result from the identical
  reduced totals.
- **Convergence check:** each rank counts how many of its own points
  changed cluster; a single `MPI_Allreduce` (SUM) gives every rank the
  true global change count, so all ranks agree on when to stop the
  inner K-Means loop.

### 2.3 Center initialization: re-seeded at every t

Per the assignment spec ("Use first K points at t=0 as initial
positions of the centers"), every rank keeps a small replicated copy
of the K seed points' original (x0, y0, vx, vy). At every t, each rank
calls `init_centers_from_seed` to recompute those K points' position
*at this t* (seed position + t · velocity) and uses that as the
starting centers for this timestep's K-Means run — so each t is an
independent K-Means run seeded from the same K original points,
evaluated at their position at that t. This needs no per-timestep
communication, since every rank already holds the seed points'
(x0,y0,vx,vy) and can compute this locally.

### 2.4 Diameter / quality: split across all ranks, not just rank 0

Assignment and center updates only need *aggregate* per-cluster
numbers (sums, counts), which `Allreduce` handles cleanly. Diameter
computation is different: it needs the max distance between
*individual point pairs*, and a cluster's members can be split across
several ranks — there is no cheap running aggregate (like a sum) that
captures "the two farthest-apart points," so `Allreduce` alone cannot
solve it, and a purely local per-rank diameter would silently miss
the true maximum if the two farthest points happen to live on
different ranks.

**Design chosen:** after K-Means converges for a timestep, every rank's
points ((x, y, cluster_id)) are combined into an identical, complete
snapshot on *every* rank via `MPI_Allgatherv` (not just rank 0). Each
rank then computes partial diameters using only its own slice
`[i_lo, i_hi)` of point indices as the outer loop of the diameter scan
(`cluster_diameters_i_range`), still comparing against the full point
set so no cross-rank pair is missed. `MPI_Allreduce` with `MPI_MAX`
combines the K partial diameters into the true global diameters,
identically on every rank, which is enough for every rank to compute
`q` independently — no further gather or broadcast is needed to reach
the found/not-found decision.

An earlier version of this step gathered to rank 0 only and computed
all K diameters there, single-threaded across ranks (parallelized only
via OpenMP on rank 0's 8 cores). That left 3 of 4 nodes idle during
what is usually the most expensive step per timestep. Splitting the
scan by point index instead of by cluster index was also considered
and rejected: with K under 20, a by-cluster split gives few, uneven
buckets, so a rank could land the one large cluster while others sit
mostly idle — splitting by point index instead stays balanced
regardless of how unevenly points fall into clusters, since N is
usually much larger than K or the rank count.

## 3. Complexity evaluation

Per timestep, the dominant cost is the diameter/quality step: O(N²/K)
sequentially, since each of the K clusters requires comparing every
pair of its member points, and clusters average N/K points. This is
far more expensive than the K-Means inner loop itself, O(I·N·K) for
I iterations, since N is large (up to 300,000) while K stays small
(under 20).

Splitting across P ranks and C OpenMP threads per rank divides the
compute cost of every step by roughly P·C. The diameter step also
needs one `MPI_Allgatherv` (O(N) data) and one small `MPI_Allreduce`
per timestep to combine partial results, and the K-Means inner loop
needs a few small `MPI_Allreduce` calls (K numbers each) per
iteration. These communication costs don't shrink as P grows — they
grow slightly (more participants per collective call) — so at some
point adding more ranks stops helping: the shrinking compute time is
outweighed by the growing communication overhead. This matches what
the timing sweep in §4 actually shows.

## 4. Timing results

Measured on the 4-node × 8-core cluster, same input file across all
configurations (32 physical cores available). The input (N=100,000,
K=15, `input_large.txt`, generated by `gen_input.py`) was built so the
clusters aren't already separated at t=0 and the search has to sweep
through many timesteps before finding one that passes QM — the
originally supplied test file converges at t=0 immediately, which
would make sequential and parallel runtimes both trivially short and
not actually demonstrate a meaningful speedup either way.

| Configuration | Time (s) | Speedup vs. sequential |
|---|---|---|
| Sequential | 9.26 | 1.0x (baseline) |
| 4 ranks × 8 threads | 3.84 | 2.41x |
| 8 ranks × 4 threads | 4.43 | 2.09x |
| 16 ranks × 2 threads | 5.87 | 1.58x |
| 32 ranks × 1 thread | 27.66 | 0.33x (3x slower) |
| 1 rank × 8 threads | 32.68 | 0.28x (3.5x slower) |

**Interpretation:**
- Best configuration: **4 ranks × 8 threads** (one rank per node) —
  minimizes the number of `Allreduce`/`Allgatherv` participants (only
  4) while still using all 32 cores via OpenMP within each node.
- Speedup drops off as rank count increases beyond 4 (8×4, 16×2),
  consistent with `Allreduce` cost growing with participant count —
  more ranks means more, smaller messages and more synchronization
  points per K-Means iteration.
- **1 rank × 8 threads** only uses 8 of the 32 available cores (a
  single node; the other 3 sit idle), which alone explains a
  significant part of its slowdown relative to the 32-core
  configurations.
- **32 ranks × 1 thread** uses all 32 cores but is markedly worse
  than even the single-node 1×8 case at moving data volume — this
  points to core-oversubscription rather than communication volume
  alone (see note below).

**Open item, worth resolving before treating these numbers as final:**
the sbatch script does not explicitly set `--cpus-per-task`, meaning
SLURM's default core-binding may not be spreading OpenMP threads
across distinct physical cores as intended. Re-running the 1×8 and
32×1 configurations with explicit `--cpus-per-task` and
`OMP_PROC_BIND=true` / `OMP_PLACES=cores` set would confirm whether
the pathological slowdown in those two rows is a genuine
communication/oversubscription effect or a binding artifact.

## 5. Load balancing

The goal is to keep every rank and every thread doing roughly the
same amount of work, so no single one becomes a bottleneck the others
sit idle waiting for.

**Splitting points across MPI ranks:** for position updates and
cluster assignment, every point costs the same fixed amount of work,
so splitting the N points evenly by count across ranks (`N/P`, with
any remainder spread one-per-rank) is already balanced.

Diameter computation is trickier, since it needs the max distance
between every pair of points *within the same cluster*, and cluster
sizes vary. Splitting this work by cluster (e.g. "rank 0 handles
clusters 0–3") would be unbalanced: with only ~15 clusters, one rank
could easily get stuck with the one unusually large cluster while the
others sit mostly idle. Instead, the diameter scan is split by point
index — the same even N/P split used everywhere else — so no rank can
get unlucky with an oversized cluster.

**Splitting work across OpenMP threads within a rank:** the diameter
scan compares each point i against every later point j, so smaller
values of i have more comparisons left to do than larger ones — the
workload shrinks as the loop progresses. The default (`static`)
OpenMP scheduling would hand each thread a fixed, equal-sized block of
i values, which is *not* equal work: the thread with the smallest i's
does the most comparisons. `schedule(dynamic)` is used on this loop
instead, so a thread grabs a new small chunk of work as soon as it
finishes its current one, rather than being stuck with a fixed range.
The assignment loop doesn't need this, since its per-point cost is
uniform — it uses the default static schedule.

## 6. Verification performed

- Sequential and parallel outputs cross-checked for exact agreement
  at 1, 2, 3 (uneven N/rank split), and 4 ranks, and at multiple
  OpenMP thread counts.
- Both output paths tested: a result found mid-sweep, and the
  "There was no t found" case.
- A dedicated test case was constructed where cluster members are
  deliberately interleaved across ranks, and another with intentionally
  unequal cluster sizes, to confirm the distributed diameter step
  (§2.4) never misses a cross-rank point pair or under-reports a
  diameter due to imbalance.