<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# UniLinalg

A linear algebra library: dense/sparse matrices with LU, Cholesky, QR, SVD,
symmetric eigendecomposition, and `Vector[D,T]`, a fixed-dimension
geometric/physical vector. Depends on
UniMath (`UniLinalg --> UniMath`) for `Vector`'s exact-precision arithmetic;
designed to be consumed by downstream geometry/physics engines.

## What's inside

- **Matrices** — `Matrix[T]` (dense, runtime-sized) and `CsrMatrix[T]`
  (sparse, compressed-row).
- **Decompositions** (`algorithms/`) — partial-pivoting LU (`solve`,
  `inverse`, determinant), un-blocked Cholesky, Householder QR
  (`leastSquares`), compact Householder least squares for tall designs
  (`leastSquaresCompact`), one-sided Jacobi SVD, and symmetric Jacobi
  eigenpairs.
- **Accurate refinement** (`refine.nim`, opt-in `refine = true` on `solve`,
  `inverse`, `leastSquares`, plus per-decomposition `*RefineOnce`) — one step
  of UniAccurate-backed iterative refinement to recover the last few ULP a
  plain float64 factorization can miss.
- **`Vector[D,T]`** — fixed `D` in `{2, 3, 4}`, geometry/physics-oriented
  (dot, cross, length, normalize), constrained to UniMath's `RealField`.

## The Uni* family

UniLinalg is layer 3 of `lituus-lab`'s `Uni*` family: a set of Nim libraries,
each with a C ABI and a Python binding, unified by a shared dependency DAG and
documentation/testing conventions. See
[lituus-lab/.github](https://github.com/lituus-lab/.github) for the family's
purpose and philosophy. UniLinalg depends on UniMath (layer 2) for `Vector`'s
exact-precision roots and on UniAccurate (layer 1, transitively) for the
refinement residual.

## Anti-goals

- **Not a BLAS/LAPACK replacement.** No blocked/tiled kernels, no SIMD
  dispatch. The decompositions are pedagogical-clarity-first, not
  performance-competitive with a vendor library. A future `-d:simd`
  opt-in (the family's established flag name, see UniAccurate) is only
  worth adding if a real consumer needs it.
- **`Vector[D,T]` is not a general-purpose N-dimensional array.** Fixed
  `D` in `{2, 3, 4}` only, geometry/physics-oriented (dot, cross, length,
  normalize). For arbitrary-length vectors, use `Matrix`'s `seq[T]`-returning
  matrix-vector product, or `CsrMatrix`.
- **`Matrix` and `Vector` do not interoperate structurally.** No
  `Matrix[T] * Vector[D,T]` — see the book for why this is a deliberate
  boundary (runtime shape vs. compile-time dimension), not a missing feature.
- **Not a source of new numeric algorithms.** The decompositions are the
  textbook versions (partial-pivoting LU, un-blocked Cholesky, Householder
  QR, one-sided Jacobi SVD, symmetric Jacobi eigenpairs); no research-grade
  variants.

## Provenance & development

Domain content (`Matrix`/`CsrMatrix`/LU/Cholesky/QR/SVD) relocated from the
original `UniversalMath` monorepo's `UniLinalg` package (1.0.0). `Vector[D,T]`
is ported and narrowed from an existing geometry library's vector type. See
ADR-0005 for the two substantive decisions beyond a straight port:
`Vector[D,T]` shipping here constrained to UniMath's `RealField` (not the
narrower `SomeFloat`), and the real `UniLinalg --> UniMath` dependency edge
(`Vector.length()`'s unqualified `sqrt` resolves to UniMath's `BigFloat`/
`Rational`/`Fixed` roots for the exact scalars).

Development used LLM/agent assistance extensively, on the terms described in
`.github/README.md`. One visible consequence: this repo's git history is
short and linear, with commits landing close together in time — that
reflects an LLM/agent rewrite pass over the pre-existing `UniversalMath`
design above, not the decompositions being written at that speed from a
blank page.

## Layout

```text
src/UniLinalg.nim                    umbrella module
src/UniLinalg/types/matrix.nim       Matrix[T] (dense, runtime-sized)
src/UniLinalg/types/sparse.nim       CsrMatrix[T]
src/UniLinalg/types/vector.nim       Vector[D,T] (compile-time-sized, RealField)
src/UniLinalg/types/tolerance.nim    EPSILON_DEFAULT, almostZero, almostEqual
src/UniLinalg/algorithms/{lu,cholesky,qr,svd,eigen_symmetric}.nim   decompositions
src/UniLinalg/algorithms/refine.nim  residual(), shared by each decomposition's refine step
src/UniLinalg/c_api.nim              C ABI (ulin_ prefix)
include/UniLinalg.h                  hand-written C header
tests/ tests/c/                      Nim + C ABI tests
examples/                            Nim + C demos
py/                                  Cython binding + pytest
book/                                nimib book
ADRs/                                0001 sibling deps, 0002 license,
                                      0003 engine&shell, 0004 conventions,
                                      0005 Vector + UniMath, 0006 refine
                                      via UniAccurate, 0007 symmetric eigen,
                                      0008 compact QR least squares
vgraph.cfg tools/vgraph.nim          anti-cycle + sibling-dependency check (ADR-0001)
.github/workflows/ci.yml             3-OS Nim matrix + C ABI + Python
```

## Build

```bash
nimble install -y
nimble test           # Nim, debug (contracts active)
nimble testRelease    # Nim, -d:danger (bound/overflow checks off, contracts compiled away)
nimble testAll        # debug + release + C ABI
nimble ctest          # C ABI: static lib + tests/c
nimble cexample       # C demo
nimble example        # Nim demo
nimble pyTest         # Cython + pytest
nimble coverage       # gcov + lcov -> coverage/
nimble book           # nimib book -> book/index.html
nimble docs           # book + API reference -> pages/
nimble checkVGraph    # no upward import relative to vgraph.cfg's layer order
nimble bench          # throughput + refine-accuracy benchmarks (see bench/README.md)
nimble benchReadme    # bench, then splice a headline table into bench/README.md for this machine
```

`nimble install` fetches the public `lituus-lab/UniMath` dependency directly;
no GitHub credentials are required.

## Benchmarks

See [bench/README.md](https://github.com/lituus-lab/UniLinalg/blob/main/bench/README.md)
for the full write-up (throughput vs. LAPACK/Arraymancer, the `-d:danger`/
indexing levers, and the refine-accuracy table) and machine-tagged results.

## CI

`test`, `cabi` and `python` on ubuntu/macOS/Windows. `consume-cabi` and
`consume-wheel` rebuild against the published artifacts on a machine without
Nim, so what ships is what was tested. `coverage` and `docs` run on ubuntu.

`dco` blocks PRs missing a `Signed-off-by` trailer; `commitizen` blocks PRs
whose commits or title are not [Conventional Commits](https://www.conventionalcommits.org/)
(`CONTRIBUTING.md`).

The same gates run locally with pre-commit: `pip install pre-commit && pre-commit install`
(`CONTRIBUTING.md`).

The `docs` job always builds the book and API reference. Following the other
Uni* repositories, the separate Pages job publishes them only from `main` when
the repository is public.

## AI-assisted contributions

Assistance from AI/LLM tools is welcome on the same terms as any other
contribution.

- **Accountability.** The human contributor is the author and remains fully
  responsible for the change. The DCO sign-off (`Signed-off-by`) is the mechanism:
  by signing you certify the content is yours or properly licensed — this covers
  AI-assisted work, provided you can stand behind it.
- **No third-party contamination.** Ensure AI output introduces no code from a
  third party without a compatible license and attribution. If an LLM reproduced
  protected material, do not submit it.
- **Correctness is yours.** The gates (tests, `nimble lint`, conventional commits,
  pre-commit) catch a lot, but you own the result — review and verify what you
  commit.
- **Atomic commits.** Each commit is one logical change. A PR may stack
  several atomic commits (one per element, say) — one monolithic big-bang
  commit is not.
- **Disclosure.** State in the PR whether AI assistance was used (see the PR
  template). It is not a hard requirement — the DCO remains the gate.

## License

Apache-2.0 (`LICENSE`). DCO sign-off on every commit (`CONTRIBUTING.md`).
