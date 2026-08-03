# UniLinalg — exact residual for iterative refinement
# =============================================================================
#
# Shared by lu.nim/cholesky.nim/qr.nim: correcting an approximate solve
# needs the true b - Ax, computed accurately enough that it survives even
# when it falls below b's own rounding resolution. See ADR-0006.

import contracts
import UniMath
import ../types/matrix

func residual*[T: SomeFloat](a: Matrix[T], b, x: openArray[T]): seq[
    T] {.contractual.} =
  ## Exact `b - Ax` (a is rows x cols, b has one entry per row, x one per
  ## column -- so this covers both square solves and the m > n
  ## least-squares case) via UniAccurate's `SuperAccumulator`: `-b[i]` and
  ## every `a[i,j]*x[j]` product accumulate into the same exact accumulator
  ## before a single rounding. Rounding a compensated dot product to `T`
  ## first and only then subtracting `b[i]` in plain arithmetic instead
  ## throws away exactly the bit refinement needs, whenever the true
  ## residual is smaller than `b[i]`'s own ULP -- verified: that
  ## alternative makes refinement a no-op on the book's own LU example.
  require: b.len == a.rows and x.len == a.cols
  ensure:
    result.len == a.rows
  body:
    result = newSeq[T](a.rows)
    for i in 0 ..< a.rows:
      var acc: SuperAccumulator[T]
      initSuperAccumulator(acc)
      acc.add(-b[i])
      for j in 0 ..< a.cols:
        acc.addProduct(a[i, j], x[j])
      result[i] = -round(acc)
