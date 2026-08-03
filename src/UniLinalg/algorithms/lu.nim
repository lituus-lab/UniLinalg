# UniLinalg — LU decomposition with partial pivoting
# =============================================================================
#
# Gaussian elimination, the algorithm everyone learns first, packaged
# honestly: PA = LU where P is a row permutation.
#
# Why pivoting? Without it, a small (or zero) pivot makes the elimination
# divide by something tiny and the round-off explodes. Partial pivoting
# swaps in the largest remaining entry of the column — cheap and stable
# enough for almost all practical matrices.
#
# Storage trick: L and U share one matrix. U occupies the upper triangle
# (diagonal included); L the strict lower triangle (its unit diagonal is
# implicit).
#
# Contracts (NimContracts): shape preconditions are `require` (debug-only,
# non-blocking in release per the contract doctrine — previously release-active
# `doAssert`s, which violated the "non bloquants en release" doctrine). The
# singular-matrix guard is a body `raise ValueError` (domain-guard doctrine:
# a value-dependent failure that must survive release). Structural
# postconditions (shape of `lu`, `perm`, sign parity) use only non-contracted
# accessors (`rows`/`cols` fields, `isSquare`). The PA=LU reconstruction
# property is exercised against an oracle in `tests/`, not asserted here
# (it would re-multiply the factors). Compiled away under release/danger.
# References: Golub & Van Loan §3.2 (LU with partial pivoting).

import contracts
import ../types/matrix

type
  LuDecomposition*[T] = object
    ## Packed LU factors of PA, the permutation, and the permutation sign.
    lu*: Matrix[T]
    perm*: seq[int] ## perm[i] = original row now sitting at row i
    sign*: int      ## +1 / -1: parity of the permutation (for det)

func luDecompose*[T: SomeFloat](a: Matrix[T]): LuDecomposition[
    T] {.contractual.} =
  ## Factors a square matrix as PA = LU (partial pivoting).
  ## Raises ValueError if the matrix is singular (a pivot is exactly zero
  ## after choosing the column maximum).
  ##
  ## Precondition: `a` is square. Postcondition (on normal return — the
  ## singular `raise` skips the ensure via the contract-raised flag): the
  ## packed factors share `a`'s shape, the permutation has one entry per row,
  ## and the sign is a permutation parity in `{-1, 1}`.
  require: a.isSquare
  ensure:
    result.lu.rows == a.rows
    result.lu.cols == a.cols
    result.perm.len == a.rows
    result.sign == 1 or result.sign == -1
  body:
    let n = a.rows
    result.lu = a
    result.perm = newSeq[int](n)
    for i in 0 ..< n:
      result.perm[i] = i
    result.sign = 1

    for k in 0 ..< n:
      # pivot search: largest |entry| in column k, rows k..n-1
      var pivotRow = k
      var pivotVal = abs(result.lu.data[k * n + k])
      for i in k + 1 ..< n:
        let v = abs(result.lu.data[i * n + k])
        if v > pivotVal:
          pivotVal = v
          pivotRow = i
      if pivotVal == T(0):
        raise newException(ValueError, "luDecompose: singular matrix")
      if pivotRow != k:
        # swap rows k and pivotRow (full rows: L part travels with them)
        for j in 0 ..< n:
          swap(result.lu.data[k * n + j], result.lu.data[pivotRow * n + j])
        swap(result.perm[k], result.perm[pivotRow])
        result.sign = -result.sign

      # eliminate below the pivot. Row-base offsets (kRow/iRow) hoisted out
      # of the inner loop instead of going through `[]`: measured ~1.3-1.8x
      # on this repo's own benchmark (bench/README.md), same algorithm,
      # bit-identical output.
      let kRow = k * n
      let pivot = result.lu.data[kRow + k]
      for i in k + 1 ..< n:
        let iRow = i * n
        let factor = result.lu.data[iRow + k] / pivot
        result.lu.data[iRow + k] = factor # store L entry in the hole
        for j in k + 1 ..< n:
          result.lu.data[iRow + j] = result.lu.data[iRow + j] -
              factor * result.lu.data[kRow + j]
    result

func luSolve*[T: SomeFloat](d: LuDecomposition[T], b: openArray[T]): seq[
    T] {.contractual.} =
  ## Solves Ax = b given the LU factors: permute b, forward-substitute
  ## through L (unit diagonal), back-substitute through U.
  ##
  ## Precondition: `b` length matches the factor size. Postcondition: the
  ## solution vector has one entry per row.
  require: b.len == d.lu.rows
  ensure:
    result.len == d.lu.rows
  body:
    let n = d.lu.rows
    result = newSeq[T](n)
    # apply permutation: y = Pb
    for i in 0 ..< n:
      result[i] = b[d.perm[i]]
    # forward substitution: Lz = y
    for i in 1 ..< n:
      var acc = result[i]
      for j in 0 ..< i:
        acc = acc - d.lu[i, j] * result[j]
      result[i] = acc
    # back substitution: Ux = z
    for i in countdown(n - 1, 0):
      var acc = result[i]
      for j in i + 1 ..< n:
        acc = acc - d.lu[i, j] * result[j]
      result[i] = acc / d.lu[i, i]
    result

func solve*[T: SomeFloat](a: Matrix[T], b: openArray[T]): seq[
    T] {.contractual.} =
  ## One-shot Ax = b. Decomposes then solves; reuse luDecompose when
  ## solving several right-hand sides against the same matrix.
  ##
  ## Precondition: `a` is square and `b` matches its size. Postcondition:
  ## the solution has one entry per row of `a`.
  require: a.isSquare and b.len == a.rows
  ensure:
    result.len == a.rows
  body:
    luSolve(luDecompose(a), b)

func det*[T: SomeFloat](a: Matrix[T]): T {.contractual.} =
  ## Determinant through LU: det(A) = sign(P) * product of U's diagonal.
  ## Returns 0 for singular matrices instead of raising (the singular
  ## `ValueError` from `luDecompose` is caught here).
  ##
  ## Precondition: `a` is square (debug-only `require:`, matching the contract
  ## doctrine — shape preconditions are non-blocking in release). No value
  ## `ensure:` — the determinant is a float product whose correctness is
  ## exercised against an oracle in `tests/`, not asserted inline (an `ensure:`
  ## recomputing it would re-decompose / re-multiply the factors).
  require: a.isSquare
  body:
    let d =
      try:
        luDecompose(a)
      except ValueError:
        return T(0)
    result = T(d.sign)
    for i in 0 ..< a.rows:
      result = result * d.lu[i, i]

func inverse*[T: SomeFloat](a: Matrix[T]): Matrix[T] {.contractual.} =
  ## Inverse through n solves against the identity columns.
  ## Prefer solve() when you only need A^-1 * b — it is cheaper and more
  ## accurate than forming the inverse explicitly.
  ##
  ## Precondition: `a` is square (debug-only `require:`). Postcondition: the
  ## inverse has the same shape as `a` (structural, uses only the non-contracted
  ## `rows`/`cols` accessors). A singular `a` propagates `luDecompose`'s
  ## `ValueError` (body path, survives release).
  require: a.isSquare
  ensure:
    result.rows == a.rows
    result.cols == a.cols
  body:
    let n = a.rows
    let d = luDecompose(a)
    result = initMatrix[T](n, n)
    var e = newSeq[T](n)
    for j in 0 ..< n:
      for i in 0 ..< n: e[i] = T(0)
      e[j] = T(1)
      let col = luSolve(d, e)
      for i in 0 ..< n:
        result[i, j] = col[i]





