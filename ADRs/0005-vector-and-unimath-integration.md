<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0005: Vector[D,T] scope and the UniMath dependency

- Status: Accepted
- Date: 2026-07-27
- Scope: the two decisions this repo makes beyond a verbatim relocation of
  `UniversalMath/UniLinalg` 1.0.0

## Context

`UniLinalg` is meant to sit above `UniMath` and below downstream geometry
and physics engines, which need a fixed-dimension vector type. Neither was
true of the code before this repo existed: the mature `UniLinalg` content
being relocated here had no `Vector` type at all (only `Matrix`/`CsrMatrix`),
and had zero dependency on `UniMath` — only on NimContracts.

## Decision 1: Vector[D,T] ships here, constrained to UniMath's RealField

A more general vector type already exists in an existing geometry library,
constrained to a `GeometricNumber` concept so it can eventually be
instantiated over `Rational`/`Fixed`/`BigFloat` as well as `float32`/
`float64`. Reconciling that library's many consumer call sites onto a
`UniLinalg`-hosted `Vector` in the same pass as standing up this repo was
explicitly ruled out as a separate, larger, riskier undertaking.

Instead, `UniLinalg` ships its own `Vector[D: static[int], T: RealField]`,
ported from that implementation but constrained to UniMath's own `RealField`
concept (`OrderedField` plus `sqrt`/`abs`) rather than the other library's
`GeometricNumber`: `float32`/`float64` satisfy it, and so do UniMath's exact
scalars `Rational`/`Fixed`/`BigFloat`, so the port gains exact-arithmetic
vectors for free instead of losing genericity. Reconciling with the other
library's own `Vector`/`GeometricNumber` type is deferred to that eventual
reconciliation, not attempted here.

## Decision 2: a real UniMath dependency, not a decorative one

`Vector.length()`/`normalize()` call unqualified `sqrt` instead of
`std/math.sqrt` directly. This is the one place in this repo where the
declared `UniLinalg --> UniMath` edge is backed by an actual compiled call:

```nim
# before
func length*[D: static[int], T: SomeFloat](v: Vector[D, T]): T {.inline.} =
  sqrt(lengthSquared(v))

# after
import std/math
import UniMath
func length*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  sqrt(lengthSquared(v))
```

The signature is the same call; only the constraint and the imports in scope
change. With both `std/math` and `UniMath` imported, `sqrt`'s overload
resolution routes `float32`/`float64` to the hardware root and the exact
scalars to their UniMath counterpart (`BigFloat.sqrt`, `Rational.sqrt`,
`Fixed`'s router `sqrt` — each defined in its own UniMath module, `export`ed
through the umbrella). No `when` branch is needed: the call site is
identical for every `RealField` instance.

The Matrix decompositions (LU/Cholesky/QR/SVD) deliberately keep
`std/math.sqrt`. They are mature, already-tested code relocated verbatim
from `UniversalMath/UniLinalg` 1.0.0; rewriting their internals to route
through `UniMath` was not required to satisfy "wire a real dependency", and
doing so anyway would add regression risk to code that was ported, not
authored, in this repo. If a future consumer needs UniMath-routed precision
inside the decompositions themselves, that is a separate, scoped change.

## Consequences

- `UniLinalg.nimble` declares `requires "https://github.com/lituus-lab/UniMath#main"`
  as a real, exercised dependency — `nimble checkVGraph` enforces it stays
  declared (ADR-0001), and the C-ABI/Python test suites exercise the
  UniMath-routed `sqrt` path end-to-end (`vec2(3.0, 4.0).length == 5.0`).
- The reconciliation with the other geometry library's own vector type
  (migrating its consumer call sites onto this `Vector`, removing its own)
  remains open, not attempted here.
- Discovered along the way: a bug in `lbartoletti/NimContracts@fix/generic-proc-support`
  (nested generic-proc contractual bodies mis-rewritten by
  `explicitResultBody`) and a related bug in `UniMath`'s own
  `arithmetic/multiplication_big.nim` (a recursive `{.contractual.}` func
  ending in a void tail call) — both fixed upstream, not in this repo.
