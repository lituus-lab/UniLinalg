# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — Tolerance helpers (trimmed from a geometry library's own
# tolerance module)
# =============================================================================
#
# Only the comparison primitives Vector[D,T] needs: EPSILON_DEFAULT, the
# defaulted float32/float64 almostZero/almostEqual, and one exact-scalar
# almostEqual generic over RealField (Fixed/Rational/BigFloat) that the
# dimension- and type-generic Vector requires. Domain-specific epsilons
# (geographic/planimetric/altimetric/angular) and sign/sameSign/oppositeSign
# are GIS/geometry concerns that stay in that library — porting them here
# would be scope creep for a linear-algebra library.

import UniMath

const EPSILON_DEFAULT* = 1e-10
  ## Default tolerance for general float64 comparisons.

const EPSILON_DEFAULT32* = 1e-5'f32
  ## Default tolerance for float32 comparisons. float32 carries ~7 decimal
  ## digits of precision, so EPSILON_DEFAULT (1e-10) cast down would sit at
  ## or below float32's own rounding noise floor -- not a meaningful
  ## tolerance. 1e-5 leaves headroom above that noise for typical magnitudes.

func almostZero*(value: float64, eps: float64 = EPSILON_DEFAULT): bool {.inline.} =
  abs(value) < eps

func almostZero*(value: float32, eps: float32 = EPSILON_DEFAULT32): bool {.inline.} =
  abs(value) < eps

func almostEqual*(a, b: float64, eps: float64 = EPSILON_DEFAULT): bool {.inline.} =
  abs(a - b) < eps

func almostEqual*(a, b: float32, eps: float32 = EPSILON_DEFAULT32): bool {.inline.} =
  abs(a - b) < eps

func almostEqual*[T: RealField](a, b, eps: T): bool {.inline.} =
  ## Exact-scalar tolerance `|a - b| < eps` for any `RealField` (Fixed,
  ## Rational, BigFloat). The concrete float32/float64 overloads above win for
  ## floats; this covers the exact scalars a `Vector[D, T]` now carries. No
  ## default eps — the exact types have no single meaningful literal epsilon.
  abs(a - b) < eps





