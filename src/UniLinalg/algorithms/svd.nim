# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — Singular Value Decomposition (one-sided Jacobi)
# =============================================================================
#
# A = U * diag(S) * V^T, the decomposition that explains every matrix:
# rotate (V^T), stretch along axes (S), rotate again (U).
#
# One-sided Jacobi is the most pedagogical SVD algorithm: repeatedly pick
# two columns of a working copy of A and rotate them until they become
# orthogonal. When every pair is orthogonal, the column norms ARE the
# singular values, the normalized columns are U, and the accumulated
# rotations are V. Slow next to LAPACK's bidiagonalization, but each step
# is understandable — and it is notably accurate on small singular values.

import std/math
import ../types/matrix
import contracts

type
  SvdDecomposition*[T] = object
    u*: Matrix[T] ## m x n, orthonormal columns
    s*: seq[T]    ## n singular values, descending, non-negative
    v*: Matrix[T] ## n x n orthogonal

func svdDecompose*[T: SomeFloat](a: Matrix[T],
                                 maxSweeps: int = 60,
                                 tol: T = T(1e-12)): SvdDecomposition[T]
                                 {.contractual.} =
  ## One-sided Jacobi SVD for m x n with m >= n.
  ## Sweeps stop when every column pair is orthogonal to relative `tol`.
  ##
  ## Shape precondition `a.rows >= a.cols` is a debug-only `require:` (NOT a
  ## release-active `doAssert`) — matching the lu/cholesky/qr doctrine. The
  ## ensure asserts the structural shapes (U m×n, S length n, V n×n);
  ## orthonormality and descending/non-negative singular values are numerical
  ## properties covered by the SVD tests, not cheap invariants.
  require:
    a.rows >= a.cols
    tol > T(0)
    maxSweeps > 0
  ensure:
    result.u.rows == a.rows and result.u.cols == a.cols and
    result.s.len == a.cols and
    result.v.rows == a.cols and result.v.cols == a.cols
  body:
    let m = a.rows
    let n = a.cols
    var w = a # working copy whose columns we orthogonalize
    result.v = identity[T](n)
    # Direct .data[] indexing (vs `[]`) matches lu.nim/cholesky.nim's measured
    # win. The Jacobi rotation inherently operates on a *pair of columns* of
    # a row-major matrix, so the row offset (iRow) can't be reused across
    # the p/q accesses within one i -- it's still hoisted per-i to avoid
    # recomputing it (and w[i,p]/w[i,q] each get read once, not twice as the
    # `[]`-based original did for app/apq).
    let wCols = n
    let vCols = n

    for sweep in 0 ..< maxSweeps:
      var offDiagonal = false
      for p in 0 ..< n - 1:
        for q in p + 1 ..< n:
          # 2x2 Gram entries of columns p and q
          var app = T(0)
          var aqq = T(0)
          var apq = T(0)
          for i in 0 ..< m:
            let iRow = i * wCols
            let wip = w.data[iRow + p]
            let wiq = w.data[iRow + q]
            app = app + wip * wip
            aqq = aqq + wiq * wiq
            apq = apq + wip * wiq
          if abs(apq) <= tol * sqrt(app) * sqrt(aqq):
            continue # this pair is already orthogonal
          offDiagonal = true
          # Jacobi rotation annihilating the (p, q) Gram entry
          let tau = (aqq - app) / (T(2) * apq)
          let t =
            if tau >= T(0):
              T(1) / (tau + sqrt(T(1) + tau * tau))
            else:
              T(-1) / (-tau + sqrt(T(1) + tau * tau))
          let c = T(1) / sqrt(T(1) + t * t)
          let s = c * t
          # rotate columns p and q of W and of V
          for i in 0 ..< m:
            let iRow = i * wCols
            let wp = w.data[iRow + p]
            let wq = w.data[iRow + q]
            w.data[iRow + p] = c * wp - s * wq
            w.data[iRow + q] = s * wp + c * wq
          for i in 0 ..< n:
            let iRow = i * vCols
            let vp = result.v.data[iRow + p]
            let vq = result.v.data[iRow + q]
            result.v.data[iRow + p] = c * vp - s * vq
            result.v.data[iRow + q] = s * vp + c * vq
      if not offDiagonal:
        break # converged before maxSweeps

    # singular values = column norms; U = normalized columns
    result.s = newSeq[T](n)
    result.u = initMatrix[T](m, n)
    let uCols = n
    for j in 0 ..< n:
      var norm = T(0)
      for i in 0 ..< m:
        let wij = w.data[i * wCols + j]
        norm = norm + wij * wij
      norm = sqrt(norm)
      result.s[j] = norm
      if norm > T(0):
        for i in 0 ..< m:
          result.u.data[i * uCols + j] = w.data[i * wCols + j] / norm
      # a zero column (rank deficiency) leaves a zero column in U

    # sort singular values descending, permuting U and V columns alongside
    for i in 0 ..< n - 1:
      var maxIdx = i
      for j in i + 1 ..< n:
        if result.s[j] > result.s[maxIdx]:
          maxIdx = j
      if maxIdx != i:
        swap(result.s[i], result.s[maxIdx])
        for r in 0 ..< m:
          swap(result.u.data[r * n + i], result.u.data[r * n + maxIdx])
        for r in 0 ..< n:
          swap(result.v.data[r * n + i], result.v.data[r * n + maxIdx])

func rank*[T: SomeFloat](a: Matrix[T], tol: T = T(1e-10)): int {.contractual.} =
  ## Numerical rank: singular values above tol * largest.
  ##
  ## Thin delegation to `svdDecompose` (which guards the shape precondition);
  ## the ensure bounds the result to a valid rank range cheaply.
  ensure:
    result >= 0 and result <= a.cols
  body:
    let d = svdDecompose(a)
    if d.s.len == 0 or d.s[0] == T(0):
      return 0
    for s in d.s:
      if s > tol * d.s[0]:
        inc result





