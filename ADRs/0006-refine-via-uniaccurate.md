<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0006: Opt-in accurate refinement via UniAccurate

- Status: Accepted
- Date: 2026-08-03
- Scope: `algorithms/lu.nim` (`residual`, `refineOnce`, `solve`'s `refine`
  parameter), and the equivalent C ABI / Python surface

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

## Consequences

- One refinement step is O(n^2): reuses the O(n^3) factorization, so it is
  cheap relative to a fresh `solve` and gets relatively cheaper as `n`
  grows. Measured at n=64 (random matrix, `cond2 ~ 246.5`): baseline
  `solve()` ~92.3us, `refineOnce` ~35.2us (0.38x baseline) for a ~20x
  reduction in max residual (9.7e-15 -> 4.8e-16). See `bench/README.md` for
  the reproducible numbers.
- `refine` only ever runs one step. A caller needing full convergence on a
  badly conditioned system calls `refineOnce` again on the result; `solve`
  does not loop internally (no convergence criterion to pick a default
  for, no risk of a silent infinite loop on a matrix refinement cannot
  help).
- The C ABI (`ulin_matrix_lu_solve`) and Python (`Matrix.solve`) expose the
  same `refine` flag, defaulted off, for parity (ADR-0003).
