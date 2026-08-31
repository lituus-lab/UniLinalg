# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "Sparse matrices"

nbText: """
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
"""

nbSave
