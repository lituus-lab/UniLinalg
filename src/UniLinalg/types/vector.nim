# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — Vector[D,T]: fixed-dimension geometric/physical vector
# =============================================================================
#
# Ported from an existing geometry library's vector type. Compile-time sized
# (D: static[int]) — distinct from Matrix[T]'s runtime-sized rows/cols, so the
# two do not interoperate structurally (no Matrix * Vector): see the book for
# the rationale.
#
# The scalar T is a UniMath `RealField` (ordered field + sqrt + abs): float32/
# float64 and the exact scalars Fixed/Rational/BigFloat all qualify.

import UniMath
import contracts
import ./tolerance
export tolerance, math

type
  Vector*[D: static[int], T: RealField] = object
    ## Displacement or direction in D-dimensional space (not a position).
    data*: array[D, T]

  Vector2d* = Vector[2, float64]
  Vector3d* = Vector[3, float64]
  Vector4d* = Vector[4, float64]
  Vector2f* = Vector[2, float32]
  Vector3f* = Vector[3, float32]
  Vector4f* = Vector[4, float32]

# ------------------------------------------------------------------------------
# Constructors
# ------------------------------------------------------------------------------

func initVector*[D: static[int], T: RealField](values: array[D, T]): Vector[D,
    T] {.inline.} =
  result.data = values

func vec2*[T: RealField](x, y: T): Vector[2, T] {.inline.} =
  result.data = [x, y]

func vec3*[T: RealField](x, y, z: T): Vector[3, T] {.inline.} =
  result.data = [x, y, z]

func vec4*[T: RealField](x, y, z, w: T): Vector[4, T] {.inline.} =
  result.data = [x, y, z, w]

func zeroVector*[D: static[int], T: RealField](): Vector[D, T] {.inline.} =
  for i in 0 ..< D:
    result.data[i] = zero(T)

func unitVector*[D: static[int], T: RealField](axis: int): Vector[D,
    T] {.contractual.} =
  ## Unit vector along `axis` (0=X, 1=Y, 2=Z, 3=W, ...).
  ##
  ## Precondition: `axis` is in range (debug-only `require:`, matching the
  ## rest of the library's shape/domain doctrine -- non-blocking in release).
  require: axis in 0 ..< D
  body:
    # Every component starts at zero(T) (matching zeroVector), not the raw
    # zero-initialized memory of an object-based RealField like Rational.
    for i in 0 ..< D:
      result.data[i] = zero(T)
    result.data[axis] = one(T)

# ------------------------------------------------------------------------------
# Accessors
# ------------------------------------------------------------------------------

func `[]`*[D: static[int], T: RealField](v: Vector[D, T],
    i: int): T {.inline.} =
  v.data[i]

func `[]=`*[D: static[int], T: RealField](v: var Vector[D, T], i: int,
    val: T) {.inline.} =
  v.data[i] = val

func x*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  when D >= 1: v.data[0]
  else: {.error: "x requires D >= 1".}

func y*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  when D >= 2: v.data[1]
  else: {.error: "y requires D >= 2".}

func z*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  when D >= 3: v.data[2]
  else: {.error: "z requires D >= 3".}

func w*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  when D >= 4: v.data[3]
  else: {.error: "w requires D >= 4".}

func dim*[D: static[int], T: RealField](v: Vector[D, T]): int {.inline.} =
  D

# ------------------------------------------------------------------------------
# Arithmetic
# ------------------------------------------------------------------------------

func `+`*[D: static[int], T: RealField](v1, v2: Vector[D, T]): Vector[D,
    T] {.inline.} =
  for i in 0 ..< D:
    result.data[i] = v1.data[i] + v2.data[i]

func `-`*[D: static[int], T: RealField](v1, v2: Vector[D, T]): Vector[D,
    T] {.inline.} =
  for i in 0 ..< D:
    result.data[i] = v1.data[i] - v2.data[i]

func `-`*[D: static[int], T: RealField](v: Vector[D, T]): Vector[D,
    T] {.inline.} =
  for i in 0 ..< D:
    result.data[i] = -v.data[i]

func `*`*[D: static[int], T: RealField](v: Vector[D, T], s: T): Vector[D,
    T] {.inline.} =
  for i in 0 ..< D:
    result.data[i] = v.data[i] * s

func `*`*[D: static[int], T: RealField](s: T, v: Vector[D, T]): Vector[D,
    T] {.inline.} =
  v * s

func `/`*[D: static[int], T: RealField](v: Vector[D, T], s: T): Vector[D,
    T] {.inline.} =
  for i in 0 ..< D:
    result.data[i] = v.data[i] / s

func `+=`*[D: static[int], T: RealField](v1: var Vector[D, T], v2: Vector[D,
    T]) {.inline.} =
  for i in 0 ..< D:
    v1.data[i] = v1.data[i] + v2.data[i]

func `-=`*[D: static[int], T: RealField](v1: var Vector[D, T], v2: Vector[D,
    T]) {.inline.} =
  for i in 0 ..< D:
    v1.data[i] = v1.data[i] - v2.data[i]

func `*=`*[D: static[int], T: RealField](v: var Vector[D, T], s: T) {.inline.} =
  for i in 0 ..< D:
    v.data[i] = v.data[i] * s

func `/=`*[D: static[int], T: RealField](v: var Vector[D, T], s: T) {.inline.} =
  for i in 0 ..< D:
    v.data[i] = v.data[i] / s

# ------------------------------------------------------------------------------
# Products
# ------------------------------------------------------------------------------

func dot*[D: static[int], T: RealField](v1, v2: Vector[D, T]): T {.inline.} =
  ## Sum of component-wise products; `dot(v, v) == lengthSquared(v)`.
  result = zero(T)
  for i in 0 ..< D:
    result = result + v1.data[i] * v2.data[i]

func cross*[T: RealField](v1, v2: Vector[3, T]): Vector[3, T] {.inline.} =
  ## 3D cross product, right-hand rule.
  result.data[0] = v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1]
  result.data[1] = v1.data[2] * v2.data[0] - v1.data[0] * v2.data[2]
  result.data[2] = v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0]

func cross2d*[T: RealField](v1, v2: Vector[2, T]): T {.inline.} =
  ## 2D perp-dot product: signed area of the parallelogram, positive if
  ## v2 is counterclockwise from v1.
  v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0]

func perp*[T: RealField](v: Vector[2, T]): Vector[2, T] {.inline.} =
  ## 90 degrees counterclockwise rotation: (-y, x).
  vec2(-v.y, v.x)

func perpCW*[T: RealField](v: Vector[2, T]): Vector[2, T] {.inline.} =
  ## 90 degrees clockwise rotation: (y, -x).
  vec2(v.y, -v.x)

# ------------------------------------------------------------------------------
# Length and normalization
# ------------------------------------------------------------------------------

func lengthSquared*[D: static[int], T: RealField](v: Vector[D,
    T]): T {.inline.} =
  dot(v, v)

func length*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  ## Euclidean length. `sqrt` is unqualified, so overload resolution routes
  ## float to the hardware std/math root and the exact scalars (Fixed/Rational/
  ## BigFloat) to their UniMath roots — a real UniLinalg --> UniMath dependency
  ## edge, not a decorative one (see ADR-0005 and the book).
  sqrt(lengthSquared(v))

func normalize*[D: static[int], T: RealField](v: Vector[D, T]): Vector[D, T] =
  ## Unit vector in the same direction, or the zero vector if `v` is exactly
  ## zero (does not raise). A NaN length (e.g. from a NaN component) is not
  ## exactly zero, so it proceeds through v / len and stays NaN, rather than
  ## being silently masked as the zero vector -- `len > zero(T)` treats NaN
  ## as false (any IEEE754 comparison with NaN except != is false).
  let len = length(v)
  if len == zero(T):
    result = zeroVector[D, T]()
  else:
    result = v / len

func isZero*[D: static[int], T: RealField](v: Vector[D, T], eps: T): bool =
  lengthSquared(v) < eps * eps

# ------------------------------------------------------------------------------
# Equality and display
# ------------------------------------------------------------------------------

func `==`*[D: static[int], T: RealField](v1, v2: Vector[D, T]): bool =
  for i in 0 ..< D:
    if v1.data[i] != v2.data[i]:
      return false
  true

func `!=`*[D: static[int], T: RealField](v1, v2: Vector[D,
    T]): bool {.inline.} =
  not (v1 == v2)

func almostEqual*[D: static[int], T: RealField](v1, v2: Vector[D, T],
    eps: T): bool =
  for i in 0 ..< D:
    if not almostEqual(v1.data[i], v2.data[i], eps):
      return false
  true

func `$`*[D: static[int], T: RealField](v: Vector[D, T]): string =
  result = "Vector["
  for i in 0 ..< D:
    if i > 0: result.add(", ")
    result.add($v.data[i])
  result.add("]")





