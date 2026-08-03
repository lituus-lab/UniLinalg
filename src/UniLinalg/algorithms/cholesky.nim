# UniLinalg — Cholesky decomposition (symmetric positive-definite)
# =============================================================================
#
# For a symmetric positive-definite (SPD) matrix, A = L * L^T with L lower
# triangular. Half the work of LU, no pivoting needed, and a built-in
# SPD certificate: if a diagonal term goes non positive during the
# factorization, the matrix was not SPD — the algorithm doubles as the test.
#
# SPD matrices are everywhere least squares lives: normal equations
# (A^T A), covariance matrices, stiffness matrices.
#
# Contracts (NimContracts): `require` for the square shape (debug-only,
# doctrine: non-blocking in release — previously a release-active `doAssert`);
# the not-SPD guard is a body `raise ValueError` (domain-guard doctrine).
# The lower-triangular postcondition uses the non-contracted `isLowerTriangular`
# helper (recursion doctrine: ensures call only non-contracted helpers).
# Compiled away under release/danger. References: Golub & Van Loan §4.2.

import contracts
import std/math
import ../types/matrix

func isLowerTriangular*[T](m: Matrix[T]): bool {.inline.} =
  ## True iff `m[i, j] == 0` for all `j > i` (strict upper triangle is zero).
  ## Non-contracted so it may be used in other procs' `ensure` blocks.
  for i in 0 ..< m.rows:
    for j in i + 1 ..< m.cols:
      if m[i, j] != T(0): return false
  true

func cholesky*[T: SomeFloat](a: Matrix[T]): Matrix[T] {.contractual.} =
  ## Returns the lower-triangular L with A = L * L^T.
  ## Raises ValueError if A is not symmetric positive-definite.
  ## Symmetry is trusted from the lower triangle (the upper one is ignored).
  ##
  ## Precondition: `a` is square. Postcondition (on normal return — the
  ## not-SPD `raise` skips the ensure): `result` is square, matches `a`'s
  ## shape, and is lower-triangular. (The A = L·L^T reconstruction is
  ## exercised against an oracle in `tests/`, not asserted here.)
  require: a.isSquare
  ensure:
    result.isSquare
    result.rows == a.rows
    result.isLowerTriangular
  body:
    let n = a.rows
    result = initMatrix[T](n, n)
    for j in 0 ..< n:
      # diagonal entry: l_jj = sqrt(a_jj - sum_k l_jk^2). jRow/iRow are the
      # row-base offsets into result.data, hoisted out of the k loops instead
      # of going through `[]`: same pattern as lu.nim, bit-identical output.
      let jRow = j * n
      var diag = a[j, j]
      for k in 0 ..< j:
        let ljk = result.data[jRow + k]
        diag = diag - ljk * ljk
      if diag <= T(0):
        raise newException(ValueError,
          "cholesky: matrix is not positive definite (pivot " & $j & ")")
      let ljj = sqrt(diag)
      result.data[jRow + j] = ljj
      # below the diagonal: l_ij = (a_ij - sum_k l_ik l_jk) / l_jj
      for i in j + 1 ..< n:
        let iRow = i * n
        var acc = a[i, j]
        for k in 0 ..< j:
          acc = acc - result.data[iRow + k] * result.data[jRow + k]
        result.data[iRow + j] = acc / ljj
    result

func choleskySolve*[T: SomeFloat](l: Matrix[T], b: openArray[T]): seq[
    T] {.contractual.} =
  ## Solves Ax = b given L from cholesky(A):
  ## forward-substitute Lz = b, then back-substitute L^T x = z.
  ##
  ## Precondition: `l` is square and `b`'s length matches its size.
  ## Postcondition: the solution has one entry per row of `l`.
  require: l.isSquare and b.len == l.rows
  ensure:
    result.len == l.rows
  body:
    let n = l.rows
    result = newSeq[T](n)
    # forward: Lz = b
    for i in 0 ..< n:
      var acc = b[i]
      for j in 0 ..< i:
        acc = acc - l[i, j] * result[j]
      result[i] = acc / l[i, i]
    # backward: L^T x = z  (walk L by columns to avoid materializing L^T)
    for i in countdown(n - 1, 0):
      var acc = result[i]
      for j in i + 1 ..< n:
        acc = acc - l[j, i] * result[j]
      result[i] = acc / l[i, i]
    result

func isPositiveDefinite*[T: SomeFloat](a: Matrix[T]): bool =
  ## SPD test by attempted factorization — the cheapest reliable check.
  ## Not contracted: it is a predicate composed of `cholesky`, and contracting
  ## it would only assert `result == (a.isSquare and ...)` which adds nothing.
  if not a.isSquare:
    return false
  try:
    discard cholesky(a)
    true
  except ValueError:
    false





