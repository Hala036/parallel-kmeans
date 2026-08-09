# Design notes: Parallel K-Means

## Architecture

Hybrid MPI + OpenMP, matching the assumed target (4 nodes x 8 cores):

- **MPI** distributes the N points across ranks (one rank per node).
  Used for everything that needs cross-node communication: scattering
  input, synchronizing convergence and cluster sums each K-Means
  iteration, and combining cluster diameters.
- **OpenMP** parallelizes the per-rank, per-node work across that
  node's 8 cores: point position updates, cluster assignment, center
  sum accumulation, and diameter computation.

`-np 4` (one rank per node) with `OMP_NUM_THREADS=8` is the intended
launch configuration -- see `run_kmeans.sbatch`. Running with `-np 32`
(one rank per core) would oversubscribe every node, since each of the
8 ranks per node would also spawn OpenMP threads competing for the
same 8 cores.

## Data flow per timestep

For each `t = 0, dT, 2dT, ..., T`:

1. `update_positions` (OpenMP): each rank moves its local points to
   their position at `t`.
2. `init_centers_from_seed`: every rank independently recomputes the
   K seed centers' positions at `t` from a small replicated copy of
   the seed points' `(x0,y0,vx,vy)` (captured once from rank 0's first
   K input points, broadcast at startup). No per-timestep
   communication needed for this step since K < 20.
3. Inner K-Means loop (MPI, per iteration):
   - `assign_clusters` (OpenMP): each rank assigns its local points to
     the nearest center.
   - `MPI_Allreduce(SUM)` on the changed-count decides convergence.
   - `compute_partial_sums` (OpenMP) + `MPI_Allreduce(SUM)` on
     per-cluster sums/counts + `centers_from_sums`: recomputes global
     centers from all ranks' local contributions.
4. Diameter / quality (Step 6): see Load Balancing below.
5. If `q < QM`, every rank independently reaches the same answer (no
   broadcast needed) and the outer loop stops.

## Load Balancing

**Points (Steps 2-3):** split across ranks as evenly as possible via
`compute_counts_displs` (first `N % size` ranks get one extra point).
Per-point cost is uniform (one distance check per cluster), so an even
count split is also an even work split.

**Diameters (Step 6)** are the dominant cost -- O(N^2/K) overall --
and needed the most care:

- *Rejected: compute all K diameters on rank 0 alone.* Simple, but
  leaves 3 of 4 nodes idle during what's likely the most expensive
  step every timestep. Fails the "parallel must be faster than
  sequential" requirement once diameter cost dominates.
- *Rejected: split by cluster index (`k_lo..k_hi` per rank).* Spreads
  work across ranks, but K < 20 gives few, coarse buckets -- if
  cluster sizes are very unequal (plausible with real data), one rank
  can land the one giant cluster (O(m^2) cost) while others get
  nearly empty ones. Imbalanced in the same way naive static
  scheduling is.
- *Rejected: partition each cluster's own points across ranks.*
  Unsafe, not just imbalanced -- a cluster's diameter is the max
  distance over all pairs of its members, and its two farthest points
  can easily land on different ranks. Computing diameter from only
  the points physically on one rank would silently under-report it.
- **Chosen: split by global point index.** Every rank first gets the
  full, current-timestep point snapshot via `MPI_Allgatherv` (so no
  cross-rank pair is ever missed -- correctness is independent of how
  the outer loop is split). Each rank then scans only its slice
  `[i_lo, i_hi)` of point indices as the outer loop (reusing the same
  `counts`/`displs` as the point scatter), comparing against all N
  points, and contributes partial per-cluster maxima. `MPI_Allreduce
  (MAX)` combines them exactly. Because the split is over N points
  (usually >> K and >> `size`), it stays balanced regardless of how
  unevenly points fall into clusters -- unlike the by-cluster split.

  Residual imbalance: the per-`i` workload is triangular (smaller `i`
  means more `j > i` to check), so a *contiguous* range of `i` still
  gives slightly more work to lower-index ranks. `cluster_diameter`'s
  OpenMP loop uses `schedule(dynamic)` to smooth this out *within* a
  rank across its 8 threads; there is no equivalent cross-rank
  work-stealing. Not addressed further since it is a second-order
  effect relative to the by-cluster imbalance it replaces, and K < 20,
  N <= 300000 keeps the absolute skew small.

**OpenMP (within a rank):** `assign_clusters` and
`compute_partial_sums` use plain `schedule(static)` (the OpenMP
default) since their per-point cost is uniform. `cluster_diameter` /
`cluster_diameters_i_range` use `schedule(dynamic)` because their
workload per outer index is triangular (see above).

## Complexity

Per timestep, per K-Means iteration: O(N*K/size) local assignment work
per rank, O(K) reduction. Over `iter` iterations (bounded by LIMIT):
O(N*K*iter/size) per rank. Diameter step: O(N^2/(K*size)) per rank
(K roughly-equal-sized clusters assumed for the estimate; see Load
Balancing for why the actual split doesn't depend on this
assumption holding). Total over `T/dT` timesteps dominates with the
diameter term for large N.

## Correctness

Sequential (`main.c`) and MPI (`main_mpi.c`) share `kmeans.c`/`io.c`
and were verified to produce identical output across rank counts
1/2/3/4/5/7, including test cases with cluster members deliberately
interleaved across ranks and with intentionally unequal cluster sizes
(25/3/2 points), confirming the distributed diameter computation does
not miss cross-rank pairs.
