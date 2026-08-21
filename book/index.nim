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
instead of pencil and paper. Every function the library exports gets used
at least once along the way.

## Matrices: building, reading, and basic arithmetic

A `Matrix[T]` is a rectangular grid of numbers, stored row by row.
`initMatrix` gives you one full of zeros; `identity` gives the special
matrix with `1`s on the diagonal and `0`s everywhere else (the matrix
equivalent of multiplying by `1`). Reading and writing a single entry uses
`[row, col]`, zero-indexed:
"""

nbCode:
  import UniLinalg

  var m = identity[float64](3)
  m[0, 2] = 5.0
  echo "m = ", m
  echo "m[0, 2] = ", m[0, 2]
  echo "m is square? ", m.isSquare

nbText: """
Matrices add, subtract, and scale entry by entry, and two matrices multiply
the way you learned in class: row `i` of the left times column `j` of the
right, summed, becomes entry `(i, j)` of the result. `transpose` flips rows
and columns. A small, hand-checkable warm-up with `A = [[1, 2], [3, 4]]`
and `B = [[5, 6], [7, 8]]`:
"""

nbCode:
  let ma = matrix[float64](2, 2, [1.0, 2.0, 3.0, 4.0])
  let mb = matrix[float64](2, 2, [5.0, 6.0, 7.0, 8.0])
  echo "A + B = ", ma + mb
  echo "B - A = ", mb - ma
  echo "2 * A = ", 2.0 * ma
  echo "A * B = ", ma * mb # row.column, e.g. top-left: 1*5 + 2*7 = 19
  echo "A * [1, 1] = ", ma * [1.0, 1.0] # a matrix times a plain vector
  echo "transpose(A) = ", transpose(ma)

nbText: """
### References

- Wikipedia: [Matrix (mathematics)](https://en.wikipedia.org/wiki/Matrix_(mathematics))
- Wikipedia: [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)
- Wikipedia: [Transpose](https://en.wikipedia.org/wiki/Transpose)

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

Gaussian elimination -- turning this into an equivalent, easier system by
combining rows -- is exactly what `solve` (via `LU decomposition`, named
for the Lower- and Upper-triangular factors elimination produces) does
underneath:
"""

nbCode:
  let prices = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                       2.0, 1.0, 3.0,
                                       1.0, 1.0, 1.0])
  echo "apple, pear, orange = ", solve(prices, [8.0, 13.0, 6.0])

nbText: """
Apples cost €1, pears €2, oranges €3 -- almost. The first and third
numbers above aren't quite exact; that's explained (and fixed) further
down.

`solve` is `luDecompose` (do the elimination once) followed by `luSolve`
(use the result to answer a specific right-hand side) in one step. Splitting
them apart pays off when several customers' totals need solving against
the *same* prices: elimination happens only once, then `luSolve` reuses it
for each new total, here for someone who bought exactly double the first
customer's basket:
"""

nbCode:
  let decomp = luDecompose(prices)
  echo "same customer = ", luSolve(decomp, [8.0, 13.0, 6.0])
  echo "double basket = ", luSolve(decomp, [16.0, 26.0, 12.0])

nbText: """
Two more classic questions about a system like this one: does it have a
unique solution at all (the determinant answers that -- zero means no), and
what if you need to solve it for *many* different totals, so often that
it's worth having the whole solution recipe on hand as one matrix (the
inverse)?
"""

nbCode:
  echo "det(prices) = ", det(prices)
  echo "inverse(prices) = ", inverse(prices, useRefinement = true)

nbText: """
A non-zero determinant confirms the unique solution `solve` found. The
inverse is the matrix that, multiplied by any total, gives the prices
directly -- at the cost of more arithmetic than solving for one specific
total, which is why `solve` is preferred unless the inverse itself is
needed.

### References

- Wikipedia: [System of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations)
- Wikipedia: [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
- Wikipedia: [Pivot element](https://en.wikipedia.org/wiki/Pivot_element) --
  partial pivoting, the numerical-stability strategy `lu.nim` uses.
- Wikipedia: [Determinant](https://en.wikipedia.org/wiki/Determinant)
- Wikipedia: [Invertible matrix](https://en.wikipedia.org/wiki/Invertible_matrix)

## Is this expression always positive?

Take the algebraic expression `4x² + 4xy + 3y²`. Is it always positive for
every `(x, y)` other than `(0, 0)`? Trying values by hand gets tedious
fast; there's a systematic way.

Write it as `[x y] A [x; y]` for the symmetric matrix `A = [[4, 2], [2,
3]]` (check: that product expands to exactly `4x² + 2xy + 2xy + 3y² = 4x²
+ 4xy + 3y²`). Cholesky factors `A = L Lᵗ` -- half the arithmetic of LU,
because it exploits the symmetry -- and it only *succeeds* when `A` is
symmetric *positive-definite*. When it does succeed, the expression turns
out to be a sum of squares in disguise (`[x y] A [x; y] = ‖Lᵗ[x; y]‖²`),
which can never be negative:
"""

nbCode:
  let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
  let lower = cholesky(a)
  echo "L = ", lower
  echo "L * L^T == A? ", almostEqual(lower * transpose(lower), a, 1e-9)
  echo "always positive? isPositiveDefinite = ", isPositiveDefinite(a)

nbText: """
Cholesky didn't raise, and `isPositiveDefinite` agrees: yes, `4x² + 4xy +
3y²` is always positive away from the origin. (A saddle-shaped expression
like `x² - y²` isn't positive-definite; `cholesky` raises on it instead,
and `isPositiveDefinite` returns `false` -- that failure *is* the test.)

Once `L` is known, solving `Ax = b` for a specific `b` reuses it exactly
like `luSolve` reused the LU factors above -- forward-substitute through
`L`, then back-substitute through `Lᵗ`:
"""

nbCode:
  let cx = choleskySolve(lower, [1.0, 1.0])
  echo "choleskySolve = ", cx
  echo "refined = ", choleskyRefineOnce(a, lower, [1.0, 1.0], cx)

nbText: """
### References

- Wikipedia: [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)
- Wikipedia: [Definite quadratic form](https://en.wikipedia.org/wiki/Definite_quadratic_form)

## Finding a physical law from measurements

A lab measures the position (in metres) of an object moving at constant
speed, at four instants (in seconds):

| t (s)        | 0 | 1 | 2 | 3 |
|--------------|---|---|---|---|
| position (m) | 1 | 3 | 5 | 7 |

Position under constant velocity follows `y = v·t + y0`. With four
measurements and only two unknowns (`v` and `y0`), there are more
equations than unknowns -- an *overdetermined* system with no exact
solution in general, only a best fit. `qrDecompose` (Householder QR) turns
the measurement matrix into an orthogonal `Q` and a triangular `R`; the
best-fit line drops out of `R` and `Qᵗ`:
"""

nbCode:
  let times = matrix[float64](4, 2, [0.0, 1.0,
                                      1.0, 1.0,
                                      2.0, 1.0,
                                      3.0, 1.0])
  let positions = [1.0, 3.0, 5.0, 7.0]
  let qrd = qrDecompose(times)
  echo "Q is ", qrd.q.rows, "x", qrd.q.cols, ", R is ", qrd.r.rows, "x", qrd.r.cols
  let coeffs = qrSolve(qrd, positions)
  echo "[v, y0] (plain) = ", coeffs
  echo "[v, y0] (refined) = ", qrRefineOnce(times, qrd, positions, coeffs)

nbText: """
Velocity 2 m/s, starting position 1 m. `leastSquares` is `qrDecompose` +
`qrSolve` in one call, for when the factors don't need reusing:
"""

nbCode:
  echo "leastSquares = ", leastSquares(times, positions, refine = true)

nbText: """
These four points happen to lie exactly on a line; real (noisy)
measurements wouldn't. `leastSquares` finds the line minimizing the total
squared error instead -- the same idea behind linear regression in
statistics.

### References

- Wikipedia: [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
- Wikipedia: [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)
- Wikipedia: [Least squares](https://en.wikipedia.org/wiki/Least_squares)
- Björck, Å. "Iterative refinement of linear least squares solutions I,"
  *BIT Numerical Mathematics* 7, 257-278 (1967) -- the least-squares
  refinement technique used above.

## How many independent measurements do you really have?

Two sensors report `1` and `2`; two more report `2` and `4` -- exactly
double the first pair. The second pair carries no new information: it's
the same measurement, scaled. `rank` catches this directly:
"""

nbCode:
  echo "rank = ", rank(matrix[float64](2, 2, [1.0, 2.0, 2.0, 4.0]))

nbText: """
Rank 1, not 2: only one genuinely independent measurement is in there.
Singular Value Decomposition (SVD) says the same thing more precisely.
Every matrix factors as `A = U * diag(S) * Vᵗ`: `V` rotates, `S` stretches
along each axis by a non-negative amount (the singular values, largest
first), `U` rotates again. A pure axis-scaling matrix like `diag(2, 5)`
makes this especially easy to see -- the singular values are just the
scale factors themselves, and `U`/`V` only need to reorder the axes so the
larger one comes first:
"""

nbCode:
  let diag = matrix[float64](2, 2, [2.0, 0.0, 0.0, 5.0])
  let sv = svdDecompose(diag)
  echo "U = ", sv.u
  echo "singular values = ", sv.s
  echo "V = ", sv.v

nbText: """
A rank-deficient matrix -- like the two proportional sensor readings above
-- has at least one singular value equal to `0`: a direction the
transformation squashes flat, with no independent information along it.

### References

- Wikipedia: [Rank (linear algebra)](https://en.wikipedia.org/wiki/Rank_(linear_algebra))
- Wikipedia: [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- Wikipedia: [Jacobi eigenvalue algorithm](https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm) --
  the one-sided variant `svd.nim` implements.

## Principal directions of a symmetric matrix

Covariance and correlation matrices are symmetric. Their eigenvectors give
orthogonal principal directions, while the eigenvalues quantify the amount
along each direction. `symmetricEigenDecompose` returns those values from
largest to smallest and stores each matching vector in a matrix column.
For `[[2, 1], [1, 2]]`, the sum direction varies three times as much as a unit
axis and the difference direction varies once:
"""

nbCode:
  let covariance = matrix[float64](2, 2, [2.0, 1.0,
                                           1.0, 2.0])
  let eigen = symmetricEigenDecompose(covariance)
  echo "eigenvalues = ", eigen.values
  echo "eigenvectors by column = ", eigen.vectors

nbText: """
The decomposition reconstructs `A = V * diag(values) * Vᵗ`. This primitive
belongs here rather than in a statistics package: PCA can build a covariance
matrix and then delegate the spectral step without maintaining another
eigensolver.

## Why doesn't my computer find exactly 3?

Back to the fruit prices: the plain `solve` near the top of this page gave
`2.9999999999999996` for the orange, not `3`. That isn't a bug in the
library or in Gaussian elimination -- `float64` only has about 15-17
significant decimal digits, and rounding during elimination can land one
or two of the last few on the wrong side of the true answer.

The *exact* error is `b - A*x`, called the residual: recomputed carefully
(in higher precision internally, so the subtraction itself doesn't lose
the very digits being measured), it's a real, tiny, non-zero number, not
nothing:
"""

nbCode:
  let approx = luSolve(decomp, [8.0, 13.0, 6.0])
  echo "residual = ", residual(prices, [8.0, 13.0, 6.0], approx)

nbText: """
`refineOnce` (or `choleskyRefineOnce`/`qrRefineOnce` for the other two
decompositions) uses that residual to correct the approximate answer by
one step, without redoing the elimination from scratch -- and every
`solve`/`leastSquares`/`inverse` above also accepts a `refine`/
`useRefinement` flag that does exactly this automatically:
"""

nbCode:
  echo "corrected = ", refineOnce(prices, decomp, [8.0, 13.0, 6.0], approx)
  echo "same, via the flag = ", solve(prices, [8.0, 13.0, 6.0],
      useRefinement = true)

nbText: """
### References

- Wikipedia: [Iterative refinement](https://en.wikipedia.org/wiki/Iterative_refinement) --
  the technique `refine`/`useRefinement` implements (originally Wilkinson's).
- Wikipedia: [Round-off error](https://en.wikipedia.org/wiki/Round-off_error)

## Sparse matrices: storing only what's there

A small shop tracks how much of each product every customer bought. Most
customers buy only one or two of the shop's products, so most entries in
that table are zero:
"""

nbCode:
  let sales = matrix[float64](3, 3, [5.0, 0.0, 0.0,
                                      0.0, 8.0, 3.0,
                                      0.0, 6.0, 0.0])
  let csr = toCsr(sales)
  echo "non-zero entries stored: ", csr.nnz, " out of ", sales.rows * sales.cols
  echo "csr[1, 2] = ", csr[1, 2]
  echo "back to dense, unchanged? ", toDense(csr) == sales

nbText: """
Compressed Sparse Row (CSR) storage keeps only those `nnz` non-zero values
(plus their positions), instead of the full `rows * cols` grid -- the
larger and emptier the table, the bigger the saving. Multiplying by a
per-product price vector (to get each customer's total spend) only ever
touches the stored non-zeros, never the zeros:
"""

nbCode:
  echo "totals per customer = ", csr * [1.0, 2.0, 3.0]
  echo "transposed (per-product totals view) = ", toDense(transpose(csr))

nbText: """
### References

- Wikipedia: [Sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix)

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
