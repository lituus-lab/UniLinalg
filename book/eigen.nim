# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "Eigenvalues, and what the machine finds"

nbText: """
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

nbText: """
The matrix is the one from *Matrices and linear systems* -- three baskets of
apples, pears and oranges. Each chapter is its own program, so it is rebuilt
here rather than carried over:
"""

nbCode:
  let prices = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                      2.0, 1.0, 3.0,
                                      1.0, 1.0, 1.0])
  let decomp = luDecompose(prices)

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
"""

nbSave
