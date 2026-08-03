<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0004: UniLinalg conventions

- Status: Accepted
- Date: 2026-07-26
- Scope: UniLinalg itself

## Layout

```text
UniLinalg.nimble                package + tasks
config.nims                     arch-conditional build flags
src/UniLinalg.nim                umbrella
src/UniLinalg/types/matrix.nim   Matrix[T] (dense, runtime-sized)
src/UniLinalg/types/sparse.nim   CsrMatrix[T]
src/UniLinalg/types/vector.nim   Vector[D,T] (compile-time-sized, RealField)
src/UniLinalg/types/tolerance.nim  EPSILON_DEFAULT, almostZero, almostEqual
src/UniLinalg/algorithms/{lu,cholesky,qr,svd}.nim  decompositions
src/UniLinalg/algorithms/refine.nim  residual(), shared by each decomposition's refine step
src/UniLinalg/c_api.nim          C ABI
include/UniLinalg.h              hand-written C header
tests/ tests/c/                  Nim + C ABI tests
examples/                        Nim + C demos
py/                              Cython binding + pytest
book/                            nimib docs
ADRs/                             0001-0006
vgraph.cfg tools/vgraph.nim      anti-cycle + sibling-dependency check (ADR-0001)
.github/workflows/ci.yml         3-OS Nim + C ABI + Python
LICENSE NOTICE CONTRIBUTING.md SECURITY.md .gitignore README.md AGENTS.md CLAUDE.md
```

## Naming

- Nim package/module: `UniLinalg` (PascalCase).
- C library: `libUniLinalg`. C header: `UniLinalg.h`.
- C symbol prefix: `ulin_`.

## Conventions

- NimContracts `{.contractual.}` + `require:`/`ensure:`/`body:`, compiled away
  under `-d:release`. The C ABI never raises — it maps errors to `ULIN_ERR_*` codes.
- A postcondition is cheaper than the body; it never re-derives the result.
- English comments, terse, describe what is done. No "deprecated".
- `types/` never imports `algorithms/` (invariant inherited from ADR-0001,
  applied here to matrix/sparse/vector/tolerance vs. lu/cholesky/qr/svd).
- Real dependency edge to `UniMath` (not decorative): `Vector.length()`/
  `normalize()` call `UniMath.sqrtNewtonGeneric`. See ADR-0005.

## CI gates

- `nimble testCi` + `testCiRelease` on ubuntu/macOS/Windows.
- `nimble ctest` on ubuntu/macOS/Windows.
- Extension build + `pytest` on ubuntu/macOS/Windows.
- `nimble checkVGraph` — no upward import relative to `vgraph.cfg`'s own
  layer order (`types/` before `algorithms/`), and no undeclared sibling
  dependency (ADR-0001).

## Provenance

Domain content (Matrix/CsrMatrix/LU/Cholesky/QR/SVD) relocated from the
original `UniversalMath` monorepo's `UniLinalg` package (1.0.0). `Vector[D,T]`
is ported and narrowed from an existing geometry library's vector type. See
ADR-0005 for the two substantive decisions beyond a straight port.
