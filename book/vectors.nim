# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "Vectors in geometry and physics"

nbText: """
## Vectors: force, work, and torque

`Vector[D, T]` is a fixed-size counterpart to `Matrix`: `D` (2, 3, or 4) is
decided when the code is written, not at runtime, which is exactly right
for physics and geometry -- a force or a position always has the same
number of components. `zeroVector` and `unitVector` give the two simplest
ones (no displacement at all, and length-1 along one axis); indexing and
the named accessors read the same vector two ways:
"""

nbCode:
  echo "zeroVector = ", zeroVector[3, float64]()
  echo "unitVector along X = ", unitVector[3, float64](0)
  var pos = vec3(1.0, 2.0, 3.0)
  echo "pos[1] = ", pos[1], "  pos.y = ", pos.y, "  dimensions = ", pos.dim
  pos[1] = 20.0
  echo "after pos[1] = 20 -> ", pos

nbText: """
Vectors add and subtract component-wise (combine two forces acting on the
same object into their net effect), scale, and support the in-place
`+=`/`-=`/`*=`/`/=` forms for updating one in a loop without rebuilding it:
"""

nbCode:
  var netForce = vec2(3.0, 0.0)
  let secondForce = vec2(0.0, 4.0)
  netForce += secondForce
  echo "two forces combined = ", netForce
  netForce -= secondForce
  echo "removing the second again = ", netForce

nbText: """
A mechanic tightens a bolt with a 0.3 m wrench, pushing 20 N perpendicular
to it. Torque is the cross product of the wrench's position vector and the
applied force -- only defined in 3D, which is why `cross` requires `Vec3`:
"""

nbCode:
  let wrench = vec3(0.3, 0.0, 0.0)
  let push = vec3(0.0, 20.0, 0.0)
  echo "torque = ", cross(wrench, push), " N*m"

nbText: """
6 N·m -- the "twisting strength" of that push. Work, meanwhile, is a dot
product: force times the *component of displacement in the force's own
direction*. A force of `(3, 4)` N moving something by `(4, 3)` m does:
"""

nbCode:
  echo "work = ", dot(vec2(3.0, 4.0), vec2(4.0, 3.0)), " J"

nbText: """
24 J. But a force exactly perpendicular to the displacement -- like gravity
on someone walking on flat ground, or string tension swinging a ball in a
circle -- does zero work, because the dot product of perpendicular vectors
is always zero:
"""

nbCode:
  echo "work (perpendicular) = ", dot(vec2(3.0, 4.0), vec2(4.0, -3.0)), " J"

nbText: """
A robot walks 3 m east then 4 m north. `length` gives the straight-line
distance back to the start; `normalize` gives the unit direction it walked
in. `lengthSquared` gives the same distance *before* the square root --
cheaper to compute, and enough on its own whenever only *comparing*
distances matters (no need to know exactly how far, just which is
farther). `perp`/`perpCW` rotate a 2D vector 90° left/right -- useful for
"which way is sideways from here" -- and `cross2d` gives the signed area
of the parallelogram the two vectors span (positive when the second is
counterclockwise from the first):
"""

nbCode:
  let walk = vec2(3.0, 4.0)
  echo "distance = ", walk.length, " m  (squared: ", walk.lengthSquared, ")"
  echo "direction walked = ", walk.normalize
  echo "90 degrees left of that direction = ", perp(walk)
  echo "90 degrees right of that direction = ", perpCW(walk)
  echo "cross2d(walk, (4, 3)) = ", cross2d(walk, vec2(4.0, 3.0))

nbText: """
Finally, `isZero` checks whether a vector is (within a tolerance) the zero
vector -- did the object actually stop moving -- and `==`/`!=`/`almostEqual`
compare two vectors exactly or within a tolerance, the same distinction
`Matrix.almostEqual` made for the quadratic-form check earlier:
"""

nbCode:
  echo "did it stop? ", zeroVector[2, float64]().isZero(1e-9)
  echo "still moving? ", walk.isZero(1e-9)
  echo "vec2(1,2) == vec2(1,2)? ", vec2(1.0, 2.0) == vec2(1.0, 2.0)
  echo "vec2(1,2) != vec2(1,3)? ", vec2(1.0, 2.0) != vec2(1.0, 3.0)
  echo "close enough? ", almostEqual(vec2(1.0, 2.0), vec2(1.0000000001, 2.0), 1e-6)

nbText: """
### References

- Wikipedia: [Euclidean vector](https://en.wikipedia.org/wiki/Euclidean_vector)
- Wikipedia: [Dot product](https://en.wikipedia.org/wiki/Dot_product)
- Wikipedia: [Cross product](https://en.wikipedia.org/wiki/Cross_product)
- Wikipedia: [Unit vector](https://en.wikipedia.org/wiki/Unit_vector)
- Wikipedia: [Torque](https://en.wikipedia.org/wiki/Torque)
- Wikipedia: [Work (physics)](https://en.wikipedia.org/wiki/Work_(physics))

## A projectile

A ball is thrown with initial velocity 10 m/s horizontal, 15 m/s vertical.
Taking `g = 10 m/s²` (the usual classroom rounding), its position at time
`t` is `p(t) = v0*t - (0, ½g)*t²` -- pure vector arithmetic, no calculus
needed for a few sample instants:
"""

nbCode:
  let v0 = vec2(10.0, 15.0)
  let halfG = vec2(0.0, 5.0)
  for t in [1.0, 2.0, 3.0]:
    echo "t=", t, "s -> position = ", v0 * t - halfG * (t * t)

nbText: """
The ball is at the same height at `t=1` and `t=2` -- once climbing, once
falling, since the parabola is symmetric around its peak -- and it's back
on the ground exactly at `t=3`.

### References

- Wikipedia: [Projectile motion](https://en.wikipedia.org/wiki/Projectile_motion)
"""

nbSave

nbSave
