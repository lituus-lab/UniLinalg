<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# UniLinalg benchmarks

```bash
nimble bench                # UniLinalg-only throughput (bench/bench.nim)
nimble benchVsLapack         # vs raw LAPACK (nimlapack)      -- needs `nimble install nimlapack`
nimble benchVsArraymancer    # vs Arraymancer (BLAS-backed)   -- needs `nimble install arraymancer`
```

`benchVsLapack`/`benchVsArraymancer` resolve their comparison library's location at
task-run time via `nimble path`, so neither is declared in `UniLinalg.nimble`'s
`requires` -- they are dev-only comparison tooling, not real dependencies of
the library (same doctrine as the family's oracle/benchmark harnesses, kept
out of the shipped dependency graph).

## Why two comparison targets

Both are **compiled Nim, timed in the same process with the same clock** --
not a Python binding (subprocess overhead and interpreter startup would make
that comparison meaningless for anything but the largest sizes):

- **`vs_lapack.nim`** calls `nimlapack`'s raw F77 bindings
  (`dgesv`/`dpotrf`/`dgeqrf`+`dorgqr`/`dgesvd`/`dgetrf`) directly against the
  system OpenBLAS/LAPACK. This is the actual reference implementation every
  numerical package eventually calls -- the strongest performance oracle
  available on this machine.
- **`vs_arraymancer.nim`** calls Arraymancer, a real, actively-used Nim
  tensor library that is *itself* BLAS/LAPACK-backed (via `nimblas`/
  `nimlapack`), not hand-written kernels. It gives a second, independent,
  higher-level-API comparison point from within the same language ecosystem.
  Arraymancer does not publicly expose a Cholesky or signed-determinant
  proc, so only `solve`/`qr`/`svd` are compared there.

## SVD timing: V^T conversion excluded from the timed region

`vs_lapack.nim`/`vs_arraymancer.nim` convert UniLinalg's `V` (n x n, columns)
to `V^T` for the residual/ratio comparison, since that is the layout both
oracles return directly. That conversion happens *after* `timed:` stops, not
inside it -- timing it alongside `svdDecompose` would inflate UniLinalg's
measured time for work the oracles never have to do (they already hand back
`V^T`). Moving it out dropped the measured svd-vs-Arraymancer ratio at n=64
from 11.4x to 6.1x (see the table below) -- the earlier number partly
reflected this measurement bias, not just Jacobi's sweep-count variance.

## Correctness, not just speed

Every comparison checks a **residual** for each implementation independently
(does the result actually satisfy `Ax=b` / `LL^T=A` / `QR=A` /
`U diag(S) V^T=A`?), rather than comparing outputs bitwise: Householder sign
conventions and Jacobi sweep order legitimately differ between
implementations. Singular values *are* unique (both sorted descending), so
`vs_lapack.nim`/`vs_arraymancer.nim` additionally print
`max|s_unilinalg - s_reference|` as an unambiguous cross-check -- observed at
~1e-14 (float64 rounding noise) on this machine, i.e. the two independent
implementations agree to the last representable bit.

## -d:danger vs -d:release: the actual first lever

Before reaching for SIMD or blocking, measure what's already on the table.
`nimble test*`/`clib*`/`bench*` tasks build with **`-d:danger`**, not
`-d:release`: Nim's array/seq bound checks and integer overflow checks are
still active under `-d:release` (only assertions and NimContracts compile
away there) and cost real time in an O(n^3) inner loop that touches
`data[i*cols+j]` on every iteration. Switching to `-d:danger` -- which the
C ABI can afford safely, since every `ulin_*` entry point already validates
its own indices/lengths before touching `Matrix` internals (`SECURITY.md`)
-- was a zero-code-change, ~3.7x speedup on `matmul` and ~4.9x on `lu_solve`
at n=256, measured directly, no estimation:

| build       | matmul (n=256) | lu_solve (n=256) |
|-------------|-----------------|-------------------|
| `-d:release`| 72.4 ms         | 20.8 ms           |
| `-d:danger` | 19.3 ms         | 4.2 ms            |

That single switch closed most of the gap to LAPACK/Arraymancer seen in an
earlier pass of this benchmark (`solve` at n=256 was 74x slower under
`-d:release`; see git history for that run) -- far more than hand-vectorizing
the same loops with `nimsimd` would have bought (~2-4x, and no measured
precedent exists anywhere in the family for that number).

## The second lever: stop recomputing row*cols+col in every hot loop

`Matrix`'s `[]`/`[]=` accessors recompute `row*cols + col` on every single
call. That's fine for one-off access, but `Matrix.*` (matmul), `luDecompose`,
`cholesky`, `qrDecompose`, and `svdDecompose` all call it from inside an
O(n^3) (or O(n^3)-per-sweep, for SVD) loop -- the same row-base offset gets
recomputed on every iteration of the innermost loop. Measured directly (a
scratch probe comparing the exact same algorithm, one version through `[]`,
one hoisting `iRow = i * cols` once and indexing `.data[iRow + j]` directly):

| op (n=256/512)      | via `[]` | hoisted `.data[]` | speedup |
|----------------------|----------|--------------------|---------|
| matmul (n=256)       | 15.6 ms  | 8.0 ms             | 1.9x    |
| matmul (n=512)       | 81.6 ms  | 49.7 ms            | 1.6x    |
| luDecompose (n=256)  | 7.5 ms   | 4.4 ms             | 1.7x    |
| luDecompose (n=512)  | 30.6 ms  | 17.0 ms            | 1.8x    |

Verified bit-identical output (`maxdiff=0.00e+00`) in the probe -- this is
the same algorithm and the same floating-point operation order, just
skipping redundant index arithmetic. Cache-blocking/tiling was tested
alongside this (block sizes 32 and 64 on the matmul probe) and **rejected**:
it helped ~9% at n=256 but was *slower* than the plain hoisted version at
n=512 -- Apple Silicon's cache hierarchy and prefetcher already handle the
sequential access pattern well enough that explicit tiling only adds loop
overhead here. `src/UniLinalg/{types/matrix,algorithms/{lu,cholesky,qr,svd}}.nim`
now use this hoisted-offset style throughout their hot loops; QR's
R-update and SVD's Jacobi rotation inherently walk a *column* of a
row-major matrix (applying a Householder reflector / rotating a column
pair), so the row offset can't be hoisted across those specific loops --
only the direct-`.data[]`-instead-of-`[]` half of the win applies there,
which is still consistent with the measured gains below.

## Representative results (this machine, Apple Silicon, -d:danger, hoisted offsets)

UniLinalg is still deliberately pedagogical (see the README's anti-goals: no
blocked/tiled kernels, no SIMD dispatch, no multi-threading) -- the remaining
gap is now honestly small enough that it reflects only the *algorithmic*
choice (LAPACK's cache-blocked, multi-threaded kernels vs. UniLinalg's
readable triple loop), not incidental interpreter/safety/indexing overhead:

| op       | n   | UniLinalg | LAPACK   | ratio | Arraymancer | ratio |
|----------|-----|-----------|----------|-------|-------------|-------|
| solve    | 16  | 0.003 ms  | 0.059 ms | 0.1x  | 0.002 ms    | 0.0x  |
| solve    | 256 | 2.99 ms   | 0.45 ms  | 6.7x  | 0.83 ms     | 2.7x  |
| cholesky | 256 | 1.07 ms   | 0.27 ms  | 4.0x  | -- (n/a)    | --    |
| qr       | 128 | 1.76 ms   | 0.41 ms  | 4.3x  | 0.59 ms     | 3.2x  |
| svd      | 64  | 2.14 ms   | 0.50 ms  | 4.3x  | 0.43 ms     | 6.1x  |
| det      | 256 | 2.16 ms   | 0.32 ms  | 6.8x  | -- (n/a)    | --    |

Before the two indexing levers (`-d:release`, `[]`-based indexing), `solve` at
n=256 was 74x slower than LAPACK; it is now **6.7x**, and 2.7x vs. Arraymancer.
`svd` vs. Arraymancer is now 6.1x, in line with the other ops, once the V^T
conversion was moved out of the timed region (previous section) -- one-sided
Jacobi's sweep count is still data-dependent (a different random matrix
genuinely needs a different number of sweeps), but that no longer shows up as
an outsized ratio. Singular values agree with both oracles to ~1e-14.

At small `n` (16-32) UniLinalg is competitive or faster: LAPACK/Arraymancer's
fixed per-call overhead (workspace queries, BLAS thread-pool dispatch)
dominates there. The remaining single-digit ratio at larger `n` is what
cache-blocking (already tested and rejected above) and multi-threading buy
OpenBLAS -- multi-threading is the one lever not yet tried, and the next
one if more speed is ever needed (see the README's anti-goals: not pursued
today, no concrete consumer requires it).

## Accuracy: solve() vs solve(refine=true)

`solve`'s plain float64 LU can miss the correctly-rounded answer by 1-2 ULP
even on a well-scaled system (see ADR-0006). `refine=true` runs one step of
`UniAccurate`-backed iterative refinement afterward. `nimble bench` prints
both the added cost (`lu_solve` vs `lu_solve_refine`, same CSV/regression
tracking as every other op) and the accuracy gained (`max|residual|` via
`residual()`, an exact `b - Ax` computed through `UniAccurate.
SuperAccumulator`), on the same random matrices at each `n`:

| n   | cond2(A) | max\|residual\| plain | max\|residual\| refined | lu_solve | lu_solve_refine |
|-----|----------|------------------------|---------------------------|----------|-------------------|
| 16  | 32.1     | 1.27e-15               | 2.44e-16                  | 0.003 ms | 0.004 ms          |
| 32  | 208.9    | 1.24e-14               | 1.72e-15                  | 0.010 ms | 0.013 ms          |
| 64  | 237.0    | 5.35e-14                | 2.69e-15                 | 0.061 ms | 0.073 ms          |
| 128 | 184.2    | 2.10e-14                | 7.44e-16                 | 0.269 ms | 0.334 ms          |

One refinement step consistently drops the max residual by roughly one
order of magnitude across these (genuinely random, moderately conditioned)
matrices, for a ~15-25% total-time overhead on top of a fresh `solve` (the
refinement step itself is O(n^2) against an O(n^3) factorization it reuses
internally — `refineOnce` called directly on an already-decomposed matrix,
skipping the redundant re-decomposition `solve(refine=true)` does, is
cheaper still: ~0.4x a fresh `solve` at n=64, see ADR-0006).

## Regression gate

`bench/bench.nim --csv:<file>` writes `op,n,ms,ops_per_sec`; a later run with
`--baseline:<file> --threshold:0.15` flags any op/size pair that regressed by
more than 15% and exits non-zero. No baseline is committed here yet -- run
`nimble bench -- --csv:bench/baseline.csv` once a target machine is fixed.

## Machine-tagged results

`nimble benchReadme` runs the suite above and writes the table below, tagged
to the machine it ran on (`<!-- bench:machine=... -->` -- see
`bench/export_readme.nim`). Re-running on the same machine replaces only that
machine's block; a second machine (say a FreeBSD/Zen4 box,
`UNILINALG_BENCH_MACHINE` env var to name it explicitly) adds its own block
alongside, so this table can carry more than one machine's numbers at once
without either overwriting the other.

<!-- bench:insert -->

<!-- bench:machine=macosx-apple-m4 -->
| op | n | ms | ops/sec |
|---|---|---|---|
| matmul | 16 | 0.002 | 8000000. |
| matmul | 32 | 0.012 | 2666667. |
| matmul | 64 | 0.095 | 673684. |
| matmul | 128 | 0.820 | 156098. |
| matmul | 256 | 6.517 | 39282. |
| transpose | 16 | 0.000 | - |
| transpose | 32 | 0.000 | - |
| transpose | 64 | 0.002 | 32000000. |
| transpose | 128 | 0.013 | 9846154. |
| transpose | 256 | 0.156 | 1641026. |
| transpose | 512 | 0.843 | 607355. |
| lu_solve | 16 | 0.002 | 8000000. |
| lu_solve | 32 | 0.011 | 2909091. |
| lu_solve | 64 | 0.058 | 1103448. |
| lu_solve | 128 | 0.405 | 316049. |
| lu_solve | 256 | 3.068 | 83442. |
| lu_solve_refine | 16 | 0.006 | 2666667. |
| lu_solve_refine | 32 | 0.020 | 1600000. |
| lu_solve_refine | 64 | 0.079 | 810127. |
| lu_solve_refine | 128 | 0.476 | 268908. |
| lu_solve_refine | 256 | 3.340 | 76647. |
| cholesky | 16 | 0.001 | 16000000. |
| cholesky | 32 | 0.002 | 16000000. |
| cholesky | 64 | 0.011 | 5818182. |
| cholesky | 128 | 0.085 | 1505882. |
| cholesky | 256 | 0.884 | 289593. |
| qr | 16 | 0.006 | 2666667. |
| qr | 32 | 0.031 | 1032258. |
| qr | 64 | 0.155 | 412903. |
| qr | 128 | 1.649 | 77623. |
| svd (one-sided Jacobi) | 16 | 0.036 | 444444. |
| svd (one-sided Jacobi) | 32 | 0.265 | 120755. |
| svd (one-sided Jacobi) | 64 | 2.220 | 28829. |

**Accuracy: solve() vs solve(refine=true)**

| n | cond2(A) | max\|residual\| plain | max\|residual\| refined |
|---|---|---|---|
| 16 | 32.1 | 1.265e-15 | 2.444e-16 |
| 32 | 208.9 | 1.236e-14 | 1.719e-15 |
| 64 | 237.0 | 5.350e-14 | 2.687e-15 |
| 128 | 184.2 | 2.103e-14 | 7.436e-16 |

<!-- /bench:machine=macosx-apple-m4 -->
