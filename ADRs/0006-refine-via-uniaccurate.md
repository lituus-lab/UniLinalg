<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0006: Opt-in accurate refinement via UniAccurate

- Status: Accepted
- Date: 2026-08-03
- Scope: `algorithms/refine.nim` (the shared `residual`), `lu.nim`
  (`refineOnce`, `solve`'s and `inverse`'s `refine` parameter),
  `cholesky.nim` (`choleskyRefineOnce`), `qr.nim` (`qrSolve`,
  `qrRefineOnce`, `leastSquares`'s `refine` parameter), and the equivalent
  C ABI / Python surface for `solve` only (see "Generalizing beyond LU")

## Context

`solve()`'s plain float64/float32 LU can miss the correctly-rounded answer
by 1-2 ULP even on a well-scaled system: on `x+2y+z=8, 2x+y+3z=13, x+y+z=6`
(exact answer `(1, 2, 3)`, `cond2(A) ~ 29.8`), `solve` returns
`(1.0000000000000007, 2.0, 2.9999999999999996)`. Verified against raw LAPACK
`dgesv` and Arraymancer (both LAPACK-backed): both show noise of the same
order on the identical system, so this is expected float64 rounding for
this conditioning, not a UniLinalg-specific defect.

The first correction attempt tried was `UniAccurate.dot2` (twice-precision
dot product): `b[i] - dot2(row, x)`. It does nothing here. `dot2` rounds
the compensated dot product to one `T` *before* the subtraction; since
`b[i]`'s own ULP (~1.7e-15 near 8.0) is larger than the true error (~2.2e-16
near 1.0), the dot product rounds right back to `b[i]` and the residual
comes out exactly zero. Confirmed by computing the true residual exactly
via `Rational[BigInt]`: `-2.220446049250313e-16`, non-zero.

## Decision

`residual` folds `-b[i]` into the *same* exact accumulation as the
`a[i,j]*x[j]` products, via `UniAccurate.SuperAccumulator`/`addProduct`
(already reachable through the existing `UniMath` dependency — `UniMath/
eft.nim` re-exports the full `UniAccurate` umbrella, so no new `requires`
line), then rounds once. This reproduces the exact residual bit-for-bit and
lets `refineOnce` correct the LU solution using the already-computed
factors. On the example above it recovers `(1.0, 2.0, 3.0)` exactly.

`solve` gains a `refine: bool = false` parameter (default off, so existing
callers are unaffected). `refineOnce`/`residual` are exported standalone for
callers reusing a factorization (`luDecompose` once, `luSolve` several
right-hand sides) who want refinement without re-decomposing.

Not adopted: exact linear algebra via `BigFloat`/`Rational[BigInt]`
(prototyped as a generic solve over `RealField`, confirmed exact) — both
cost 30-240x the float64 baseline for a 3x3 system, and exact rational
elimination has well-known multiplicative denominator growth with size that
was not measured here. `solve`'s `T: SomeFloat` constraint keeps these
usable only by a caller who writes their own generic solve; adopting one as
the default would fight the "pedagogical, dependency-free" scope (README).

## Generalizing beyond LU

`residual` needs only the original `a`, `b`, and a candidate `x` — nothing
LU-specific — so it moved out of `lu.nim` into its own `algorithms/
refine.nim`, generalized to `a`'s actual shape (`b.len == a.rows`,
`x.len == a.cols`) rather than assuming square, and is now shared by three
decompositions:

- **Cholesky**: `choleskyRefineOnce(a, l, b, x)` is the direct analogue of
  `refineOnce` — same square `Ax=b` structure, same technique, verified:
  on `A=[[4,2],[2,3]], b=[1,1]`, `choleskySolve` returns
  `(0.125, 0.24999999999999997)`; one refinement step recovers
  `(0.125, 0.25)` exactly.
- **`inverse`**: built from `n` `luSolve` calls against identity columns;
  gained the same `refine: bool = false` parameter, applying `refineOnce`
  to each column. Verified on a random 40x40 matrix: `max|A*A^-1 - I|`
  dropped from `5.6e-15` to `1.6e-15`.
- **QR / `leastSquares`** (overdetermined, m > n): this is Björck's
  classical iterative refinement for least squares, not a copy-paste of
  the square case — `leastSquares` was split into `qrDecompose` +
  `qrSolve(d, b)` so the correction can reuse the *same* Q, R (A's
  decomposition doesn't depend on the right-hand side). `qrRefineOnce(a,
  d, b, x)` solves `min ‖A dx - r‖` via `qrSolve(d, residual(a, b, x))`.
  Verified on an exactly-consistent random 30x6 system (any residual here
  is pure float64 rounding, not model noise): max residual dropped from
  `3.8e-15` to `7.1e-16`.
- **SVD**: no analogue — this repo exposes no `svdSolve`; `svdDecompose`/
  `rank` don't solve `Ax=b`, so there is nothing here to refine.
- **`det`**: a different accuracy concern (compensated product
  accumulation to avoid round-off in a running product), not iterative
  refinement of a solve — not attempted here.

Every new formula above (the Cholesky forward/back-substitution reuse, the
QR-factor-reuse least-squares correction) is derived from the published
algorithm and written fresh against this repo's existing types, the same
way `lu.nim`/`cholesky.nim`/`qr.nim` were originally written from Golub &
Van Loan — no source was copied from another codebase.

## Consequences

- One refinement step reuses the already-computed factorization rather than
  re-decomposing, so it is cheap relative to a fresh `solve` and gets
  relatively cheaper as the system grows. For the square case (LU,
  Cholesky) this is O(n^2) against an O(n^3) factorization. `residual`
  itself is O(rows * cols): for the general QR least-squares case
  (`m > n`) that is O(mn), not O(n^2) -- still cheap next to
  `qrDecompose`'s own O(m^2 n) (dominated by accumulating the full m x m
  `Q`), just not the same bound as the square case. Measured at n=64
  (square, random matrix, `cond2 ~ 246.5`): baseline `solve()` ~92.3us,
  `refineOnce` ~35.2us (0.38x baseline) for a ~20x reduction in max
  residual (9.7e-15 -> 4.8e-16). See `bench/README.md` for the reproducible
  numbers.
- `refine`/`refineOnce`/`qrRefineOnce`/`choleskyRefineOnce` only ever run
  one step. A caller needing full convergence on a badly conditioned
  system calls the `*RefineOnce` proc again on the result; none of the
  one-shot wrappers (`solve`, `inverse`, `leastSquares`) loop internally
  (no convergence criterion to pick a default for, no risk of a silent
  infinite loop on a matrix refinement cannot help).
- The C ABI (`ulin_matrix_lu_solve`) and Python (`Matrix.solve`) expose the
  same `refine` flag, defaulted off, for parity (ADR-0003). Cholesky/QR/
  `inverse` refinement is Nim-only for now — no C ABI/Python surface yet.
