# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
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
import UniMath
import ../types/matrix
import ./refine
export refine

func isLowerTriangular*[T](m: Matrix[T]): bool {.inline.} =
  ## True iff `m[i, j] == 0` for all `j > i` (strict upper triangle is zero).
  ## Non-contracted so it may be used in other procs' `ensure` blocks.
  for i in 0 ..< m.rows:
    for j in i + 1 ..< m.cols:
      if m[i, j] != T(0): return false
  true

func cholesky*[T: SomeFloat](a: Matrix[T]): Matrix[T] {.contractual.} =
  ## Returns the lower-triangular L with A = L * L^T.
  ## Raises ValueError if A is not symmetric, or not positive-definite.
  ##
  ## Precondition: `a` is square. Postcondition (on normal return — the
  ## not-symmetric/not-SPD `raise`s skip the ensure): `result` is square,
  ## matches `a`'s shape, and is lower-triangular. (The A = L·L^T
  ## reconstruction is exercised against an oracle in `tests/`, not
  ## asserted here.)
  require: a.isSquare
  ensure:
    result.isSquare
    result.rows == a.rows
    result.isLowerTriangular
  body:
    let n = a.rows
    # Symmetry is a domain precondition, not a shape one -- checked here
    # (not via `require:`) so it still holds under -d:danger, matching the
    # not-SPD guard below: a silently-accepted asymmetric input would
    # factor only its lower triangle and return a plausible-looking but
    # wrong L for a different matrix than the one passed in.
    for i in 0 ..< n:
      for j in 0 ..< i:
        if a[i, j] != a[j, i]:
          raise newException(ValueError,
            "cholesky: matrix is not symmetric (a[" & $i & "," & $j &
            "] != a[" & $j & "," & $i & "])")
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
    # forward: Lz = b. Checks every diagonal once, in order -- the backward
    # loop below reuses the same n diagonals, already known nonzero by then,
    # so it does not need its own check.
    for i in 0 ..< n:
      var acc = b[i]
      for j in 0 ..< i:
        acc = acc - l[i, j] * result[j]
      if l[i, i] == T(0):
        raise newException(ValueError, "choleskySolve: zero diagonal entry " &
          "at " & $i & " (not a valid Cholesky factor)")
      result[i] = acc / l[i, i]
    # backward: L^T x = z  (walk L by columns to avoid materializing L^T)
    for i in countdown(n - 1, 0):
      var acc = result[i]
      for j in i + 1 ..< n:
        acc = acc - l[j, i] * result[j]
      result[i] = acc / l[i, i]
    result

func choleskyRefineOnce*[T: SomeFloat](a, l: Matrix[T],
    b, x: openArray[T]): seq[T] {.contractual.} =
  ## One step of iterative refinement on a choleskySolve() result, reusing
  ## the exact `residual()` (needs the original `a`, not just the factor
  ## `l`) and the already-computed Cholesky factor -- same technique and
  ## same reasoning as LU's `refineOnce`. O(n^2), reuses the O(n^3)
  ## factorization. See ADR-0006.
  require: a.isSquare and l.isSquare and b.len == a.rows and x.len == a.rows
  ensure:
    result.len == a.rows
  body:
    let r = residual(a, b, x)
    let dx = choleskySolve(l, r)
    result = newSeq[T](a.rows)
    for i in 0 ..< a.rows:
      result[i] = x[i] + dx[i]

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





