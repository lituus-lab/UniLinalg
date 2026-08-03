# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — Vector[D,T] test suite
# =============================================================================

import std/unittest
import ../src/UniLinalg
import UniMath

suite "Vector - constructors":
  test "vec2/vec3/vec4 and component accessors":
    let a = vec2(3.0, 4.0)
    check a.x == 3.0 and a.y == 4.0
    let b = vec3(1.0, 2.0, 3.0)
    check b.x == 1.0 and b.y == 2.0 and b.z == 3.0
    let c = vec4(1.0, 2.0, 3.0, 4.0)
    check c.w == 4.0
    check b.dim == 3

  test "initVector, zeroVector, unitVector":
    let v = initVector([1.0, 2.0, 3.0])
    check v[0] == 1.0 and v[2] == 3.0
    let z = zeroVector[2, float64]()
    check z.x == 0.0 and z.y == 0.0
    let ux = unitVector[3, float64](0)
    let uy = unitVector[3, float64](1)
    check ux.x == 1.0 and ux.y == 0.0
    check uy.x == 0.0 and uy.y == 1.0

  test "float32 aliases":
    let v: Vector2f = vec2(1.0'f32, 2.0'f32)
    check v.x == 1.0'f32
    let v3: Vector3d = vec3(1.0, 2.0, 3.0)
    check v3.z == 3.0

suite "Vector - arithmetic":
  test "add, subtract, negate, scalar mul/div":
    let a = vec2(1.0, 2.0)
    let b = vec2(3.0, 4.0)
    check a + b == vec2(4.0, 6.0)
    check b - a == vec2(2.0, 2.0)
    check -a == vec2(-1.0, -2.0)
    check a * 2.0 == vec2(2.0, 4.0)
    check 2.0 * a == vec2(2.0, 4.0)
    check b / 2.0 == vec2(1.5, 2.0)

  test "in-place operators":
    var v = vec2(1.0, 1.0)
    v += vec2(1.0, 2.0)
    check v == vec2(2.0, 3.0)
    v -= vec2(1.0, 1.0)
    check v == vec2(1.0, 2.0)
    v *= 3.0
    check v == vec2(3.0, 6.0)
    v /= 3.0
    check v == vec2(1.0, 2.0)

suite "Vector - products":
  test "dot product":
    let a = vec2(1.0, 2.0)
    let b = vec2(3.0, 4.0)
    check dot(a, b) == 11.0
    check dot(a, a) == a.lengthSquared

  test "3D cross product":
    let x = vec3(1.0, 0.0, 0.0)
    let y = vec3(0.0, 1.0, 0.0)
    check cross(x, y) == vec3(0.0, 0.0, 1.0)

  test "2D cross2d, perp, perpCW":
    let a = vec2(1.0, 0.0)
    let b = vec2(0.0, 1.0)
    check cross2d(a, b) == 1.0
    check cross2d(b, a) == -1.0
    check perp(a) == vec2(0.0, 1.0)
    check perpCW(a) == vec2(0.0, -1.0)

suite "Vector - length and normalization":
  test "length of a 3-4-5 triangle is exactly 5.0":
    # float length resolves `sqrt` to the hardware std/math root, so this
    # exact input is IEEE-exact.
    let v = vec2(3.0, 4.0)
    check v.length == 5.0
    check v.lengthSquared == 25.0

  test "normalize returns a unit vector in the same direction":
    let v = vec2(3.0, 4.0)
    let n = v.normalize
    check almostEqual(n.length, 1.0, 1e-9)
    check almostEqual(n.x, 0.6, 1e-9) and almostEqual(n.y, 0.8, 1e-9)

  test "normalize of the zero vector returns the zero vector":
    let z = zeroVector[2, float64]()
    check z.normalize == z

  test "normalize propagates NaN instead of masking it as the zero vector":
    let v = vec2(NaN, 1.0)
    let n = v.normalize
    check n.x != n.x # NaN != NaN, unlike an actual zero vector

  test "float32 length is exact for a 3-4-5 triangle":
    let v = vec2(3.0'f32, 4.0'f32)
    check v.length == 5.0'f32

suite "Vector - equality and tolerance":
  test "isZero":
    check vec2(1e-12, 1e-12).isZero(1e-10)
    check not vec2(0.01, 0.0).isZero(1e-10)

  test "exact equality and inequality":
    check vec2(1.0, 2.0) == vec2(1.0, 2.0)
    check vec2(1.0, 2.0) != vec2(1.0, 3.0)

  test "almostEqual on vectors":
    let a = vec2(1.0, 2.0)
    let b = vec2(1.0000001, 2.0000001)
    check almostEqual(a, b, 1e-6)
    check not almostEqual(a, vec2(1.1, 2.0), 1e-6)

  test "string representation":
    check $vec2(3.0, 4.0) == "Vector[3.0, 4.0]"

suite "Vector - exact scalar types (RealField generalization)":
  test "Rational vectors: exact arithmetic, dot, cross and lengthSquared":
    # No sqrt here — every result stays exact, proving Vector is no longer
    # float-locked.
    let a = vec2(initRational(1, 1), initRational(2, 1))
    let b = vec2(initRational(3, 1), initRational(4, 1))
    check a + b == vec2(initRational(4, 1), initRational(6, 1))
    check dot(a, b) == initRational(11, 1)
    check cross2d(a, b) == initRational(-2, 1)
    check a.lengthSquared == initRational(5, 1)
    # Vector tolerance path on an exact scalar: the vector almostEqual routes
    # each component through the RealField overload.
    check almostEqual(a, a, initRational(1, 100))
    check not almostEqual(a, vec2(initRational(2, 1), initRational(2, 1)),
                          initRational(1, 100))

  test "Fixed vectors: exact lengthSquared, length within tolerance":
    let a = vec2(toFixed[int64, 32](3), toFixed[int64, 32](4))
    check a.lengthSquared == toFixed[int64, 32](25)
    # Fixed sqrt is the Newton core; assert directly on the Fixed scalar via
    # the RealField almostEqual overload (no float64 round-trip).
    check almostEqual(a.length, toFixed[int64, 32](5), toFixed[int64, 32](0.001))
