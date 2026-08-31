# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "Matrices and linear systems"

nbText: """
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
"""

nbSave
