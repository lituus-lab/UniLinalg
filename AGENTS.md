<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# AGENTS.md — UniLinalg

## Build & gates

```bash
nimble install -y
nimble testAll    # Nim debug + -d:danger + C ABI
nimble pyTest     # Cython + pytest (needs libUniLinalg.so)
nimble example
nimble coverage   # gcov + lcov -> coverage/ (needs lcov; linux/macOS)
nimble docs       # nimib book + API reference -> pages/ (needs nimib)
```

`nimble docs` needs a complete Nim distribution: `--project` builds `dochack`,
which Homebrew's `nim` omits (no `tools/`). choosenim and the CI action ship it.

CI: 3-OS Nim matrix + C ABI (linux/macOS) + Python.

## Conventions

- English comments, terse, describe what is done. No "deprecated".
- NimContracts `{.contractual.}` + `require:`/`ensure:`/`body:`, compiled away
  under `-d:release`/`-d:danger`. C ABI never raises — it maps errors to
  `ULIN_ERR_*` codes.
- A postcondition is cheaper than the body: never re-derives the result by
  calling the function itself.
- C ABI: hand-written `include/UniLinalg.h` kept in sync with
  `src/UniLinalg/c_api.nim`; `tests/c` links the header against the lib.
  Built `--app:staticlib`/`--app:lib --noMain --mm:arc -d:danger` -- the C
  ABI is the shipped, perf-sensitive surface, and every `ulin_*` entry point
  already validates handles/indices/lengths itself (`SECURITY.md`) before
  touching `Matrix` internals, so Nim's own bound/overflow checks are
  redundant there. Measured: `-d:danger` vs `-d:release` closes most of the
  gap to LAPACK/Arraymancer (see `bench/README.md`) at zero code cost.
- C symbols `ulin_*` (prefix `ulin_`); lib `libUniLinalg`; header `UniLinalg.h`.
- `types/` (matrix, sparse, vector, tolerance) never imports `algorithms/`
  (lu, cholesky, qr, svd); enforced by `nimble checkVGraph`.
- `Vector[D,T]` is compile-time-sized (`D: static[int]`), constrained to
  UniMath's `RealField` (float32/float64 plus the exact scalars Fixed/
  Rational/BigFloat). `Matrix[T]`/`CsrMatrix[T]` are runtime-sized and
  unconstrained at the type level (`SomeFloat` is enforced by the
  decomposition procs, not the type). The two do not interoperate
  structurally (`Matrix * Vector` is not defined) — see the book
  for the rationale.
- `book/index.nim` is nimib: its code blocks are compiled and run at docs build,
  so prose that outlives its API breaks the build. `py/notebooks/quickstart.ipynb`
  plays the same role for Python and renders natively on GitHub.
- End covered sources with a blank line. Nim maps a trailing statement one line
  past EOF. `nimble coverage` suppresses exactly two lcov categories, both
  compiler artefacts with no source-level fix: `mismatch`, where lcov 2.x and
  gcov disagree on the end line of Nim's generated destructors, and that EOF + 1
  attribution -- `range` on lcov 2.5, `unmapped` on the 2.0 the runners install,
  which is why the task asks the version first. Every other error still fails.

## Scope

A linear algebra library: dense/sparse matrices, the classic decompositions,
and a fixed-dimension geometric vector. Depends on UniMath. Apache-2.0, DCO.
