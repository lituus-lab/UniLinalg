# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "Fitting, and how much data is really there"

nbText: """
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

For tall regression designs that do not need Q itself,
`leastSquaresCompact` applies each Householder reflector directly to the
right-hand side. It retains O(rows*columns) storage instead of materialising a
square `rows*rows` Q matrix; `leastSquares` remains available when explicit
factor reuse or iterative refinement is required.

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
"""

nbSave
