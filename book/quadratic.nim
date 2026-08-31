# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "Quadratic forms"

nbText: """
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
"""

nbSave
