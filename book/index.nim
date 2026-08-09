# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib

nbInit
nb.title = "UniLinalg"

nbText: """
# UniLinalg

A linear algebra library for dense and sparse matrices, the classic
decompositions (LU, Cholesky, QR, SVD), and fixed-dimension vectors for
geometry and physics.

This page is a nimib book: every code block below is compiled and run when
the book is built, and the output shown is exactly what the code produced.
The exercises are real math and physics problems, the kind you'd meet in a
lycée or a first-year university course -- solved here with `UniLinalg`
instead of pencil and paper.

## A system of three unknowns

A fruit seller sells apples, pears, and oranges. One customer buys 1 apple,
2 pears, and 1 orange for €8. A second buys 2 apples, 1 pear, and 3 oranges
for €13. A third buys 1 apple, 1 pear, and 1 orange for €6. What does each
fruit cost?

Writing `a`, `p`, `o` for the three prices, the three purchases are three
linear equations in three unknowns:

```
a + 2p + o  = 8
2a + p + 3o = 13
a + p + o   = 6
```
"""

nbCode:
  import UniLinalg

  let prices = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                       2.0, 1.0, 3.0,
                                       1.0, 1.0, 1.0])
  echo "apple, pear, orange = ", solve(prices, [8.0, 13.0, 6.0])

nbText: """
Apples cost €1, pears €2, oranges €3 -- almost. The second and third
numbers above aren't quite exact; hold onto that, there's a section below
explaining exactly why, and how to fix it.

## Is this expression always positive?

Take the algebraic expression `4x² + 4xy + 3y²`. Is it always positive for
every `(x, y)` other than `(0, 0)`? Trying values by hand gets tedious
fast; there's a systematic way.

Write it as `[x y] A [x; y]` for the symmetric matrix `A = [[4, 2], [2,
3]]` (check: that product expands to exactly `4x² + 2xy + 2xy + 3y² = 4x²
+ 4xy + 3y²`). Cholesky factors `A = L Lᵗ` only when `A` is symmetric
*positive-definite* -- and when it succeeds, the expression turns out to be
a sum of squares in disguise (`[x y] A [x; y] = ‖Lᵗ[x; y]‖²`), which can
never be negative:
"""

nbCode:
  let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
  let lower = cholesky(a)
  echo "L = ", lower
  echo "L * L^T == A? ", almostEqual(lower * transpose(lower), a, 1e-9)

nbText: """
Cholesky didn't raise, so yes: `4x² + 4xy + 3y²` is always positive away
from the origin. (A matrix that *isn't* positive-definite -- a saddle-shaped
expression like `x² - y²`, say -- makes `cholesky` raise instead; that
failure *is* the test.)

## Finding a physical law from measurements

A lab measures the position (in metres) of an object moving at constant
speed, at four instants (in seconds):

| t (s)        | 0 | 1 | 2 | 3 |
|--------------|---|---|---|---|
| position (m) | 1 | 3 | 5 | 7 |

Position under constant velocity follows `y = v·t + y0`. With four
measurements and only two unknowns (`v` and `y0`), the system is
overdetermined -- exactly what `leastSquares` is for:
"""

nbCode:
  let times = matrix[float64](4, 2, [0.0, 1.0,
                                      1.0, 1.0,
                                      2.0, 1.0,
                                      3.0, 1.0])
  let positions = [1.0, 3.0, 5.0, 7.0]
  echo "[v, y0] = ", leastSquares(times, positions, refine = true)

nbText: """
Velocity 2 m/s, starting position 1 m. Real (noisy) measurements wouldn't
land on a line exactly -- `leastSquares` finds the line minimizing the
total squared error instead, the same idea behind linear regression in
statistics.

## How many independent measurements do you really have?

Two sensors report `1` and `2`; two more report `2` and `4` -- exactly
double the first pair. The second pair carries no new information: it's
the same measurement, scaled. `rank` catches this directly:
"""

nbCode:
  echo "rank = ", rank(matrix[float64](2, 2, [1.0, 2.0, 2.0, 4.0]))

nbText: """
Rank 1, not 2: only one genuinely independent measurement is in there.
`svdDecompose`'s singular values say the same thing more precisely -- they
measure how much a transformation stretches space along each direction, in
order:
"""

nbCode:
  let diag = matrix[float64](2, 2, [2.0, 0.0, 0.0, 5.0])
  echo "singular values = ", svdDecompose(diag).s

nbText: """
## Why doesn't my computer find exactly 3?

Back to the fruit prices: the plain solve above gave `2.9999999999999996`
for the orange, not `3`. That isn't a bug in the library or in Gaussian
elimination -- `float64` only has about 15-17 significant decimal digits,
and rounding during elimination can land one or two of the last few on the
wrong side of the true answer. `refine=true` runs one extra correction
step (computing the true error in exact arithmetic, then correcting for
it) and recovers the exact answer:
"""

nbCode:
  echo "refined = ", solve(prices, [8.0, 13.0, 6.0], useRefinement = true)

nbText: """
## Vectors: force, work, and torque

A mechanic tightens a bolt with a 0.3 m wrench, pushing 20 N perpendicular
to it. Torque is the cross product of the wrench's position vector and the
applied force:
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
in:
"""

nbCode:
  let walk = vec2(3.0, 4.0)
  echo "distance = ", walk.length, " m"
  echo "direction = ", walk.normalize

nbText: """
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
"""

nbSave
