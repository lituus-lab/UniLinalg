# UniLinalg — QR decomposition (Householder reflections)
# =============================================================================
#
# A = Q * R with Q orthogonal (Q^T Q = I) and R upper triangular.
#
# The Householder idea: a reflection across a well-chosen hyperplane sends a
# whole column onto the x-axis in one move. n-1 reflections triangularize
# the matrix; their product is Q. Reflections are orthogonal, so — unlike
# Gaussian elimination — nothing ever amplifies: QR is the numerically
# robust way to solve LEAST SQUARES problems.
#
# Sign choice: v = x + sign(x_0)*|x|*e_0 — adding (never subtracting) the
# norm avoids the classic cancellation when x is already nearly axial.

import std/math
import ../types/matrix
import contracts

type
  QrDecomposition*[T] = object
    q*: Matrix[T] ## orthogonal, rows x rows
    r*: Matrix[T] ## upper triangular (trapezoidal if rectangular), rows x cols

func qrDecompose*[T: SomeFloat](a: Matrix[T]): QrDecomposition[
    T] {.contractual.} =
  ## Householder QR for any m x n matrix with m >= n.
  ##
  ## Shape precondition `a.rows >= a.cols` is a debug-only `require:` (NOT a
  ## release-active `doAssert`) — matching the lu/cholesky doctrine: shape
  ## preconditions never block a release build. Orthogonality (Q^T Q = I) and
  ## upper-triangularity of R are numerical properties verified by the SVD/QR
  ## tests, not cheap invariants; the ensure asserts the structural shapes only.
  require:
    a.rows >= a.cols
  ensure:
    result.q.rows == a.rows and result.q.cols == a.rows and
    result.r.rows == a.rows and result.r.cols == a.cols
  body:
    let m = a.rows
    let n = a.cols
    result.r = a
    result.q = identity[T](m)
    # R is m x n, Q is m x m: row-base offsets use each matrix's own stride.
    # Direct .data[] indexing (vs `[]`) matches lu.nim/cholesky.nim's measured
    # win; the R-update/Q-Gram loops below walk a *column* of a row-major
    # matrix (inherent to applying a Householder reflector), so the row
    # offset itself can't be hoisted across those i/j loops -- only the
    # Q-accumulation loop's outer `i` is invariant across its inner `j` loop.
    let rCols = n
    let qCols = m

    for k in 0 ..< min(n, m - 1):
      # build the Householder vector v from column k, rows k..m-1
      var normX = T(0)
      for i in k ..< m:
        let rik = result.r.data[i * rCols + k]
        normX = normX + rik * rik
      normX = sqrt(normX)
      if normX == T(0):
        continue # column already zero below the diagonal
      var v = newSeq[T](m - k)
      for i in k ..< m:
        v[i - k] = result.r.data[i * rCols + k]
      # add the norm on the side of v[0]'s sign (cancellation-safe)
      if v[0] >= T(0):
        v[0] = v[0] + normX
      else:
        v[0] = v[0] - normX
      var vNorm2 = T(0)
      for x in v:
        vNorm2 = vNorm2 + x * x
      if vNorm2 == T(0):
        continue

      # apply H = I - 2 v v^T / (v^T v) to R (columns k..n-1)
      for j in k ..< n:
        var dot = T(0)
        for i in k ..< m:
          dot = dot + v[i - k] * result.r.data[i * rCols + j]
        let scale = T(2) * dot / vNorm2
        for i in k ..< m:
          result.r.data[i * rCols + j] = result.r.data[i * rCols + j] -
              scale * v[i - k]
      # accumulate Q = Q * H (apply H to Q's columns from the right:
      # Q H = Q - (2/vTv) (Q v) v^T)
      for i in 0 ..< m:
        let iRowQ = i * qCols
        var dot = T(0)
        for j in k ..< m:
          dot = dot + result.q.data[iRowQ + j] * v[j - k]
        let scale = T(2) * dot / vNorm2
        for j in k ..< m:
          result.q.data[iRowQ + j] = result.q.data[iRowQ + j] - scale * v[j - k]

      # clean the exact zeros below the diagonal of column k
      for i in k + 1 ..< m:
        result.r.data[i * rCols + k] = T(0)

func leastSquares*[T: SomeFloat](a: Matrix[T], b: openArray[T]): seq[
    T] {.contractual.} =
  ## Minimizes the residual norm ‖Ax - b‖ for an overdetermined system (m >= n) through QR:
  ## R x = Q^T b, back-substituted on the top n rows.
  ## More stable than the normal equations (A^T A x = A^T b), whose
  ## conditioning is squared.
  ##
  ## Shape preconditions are debug-only `require:`; the rank-deficient case is a
  ## body `raise ValueError` (release-safe domain guard — a `require:` would
  ## compile away and let the back-substitution divide by zero silently).
  require:
    a.rows >= a.cols
    b.len == a.rows
  ensure:
    result.len == a.cols
  body:
    let n = a.cols
    let d = qrDecompose(a)
    # qtb = Q^T b
    var qtb = newSeq[T](a.rows)
    for i in 0 ..< a.rows:
      var acc = T(0)
      for k in 0 ..< a.rows:
        acc = acc + d.q[k, i] * b[k]
      qtb[i] = acc
    # Rank-deficiency threshold relative to R's largest diagonal magnitude:
    # an exact-zero test misses near-singular systems that would otherwise
    # divide by a near-zero pivot and silently blow up the result.
    var maxDiag = T(0)
    for i in 0 ..< n:
      maxDiag = max(maxDiag, abs(d.r[i, i]))
    let diagTol = T(1e-12) * maxDiag
    # back substitution on the n x n top of R
    result = newSeq[T](n)
    for i in countdown(n - 1, 0):
      var acc = qtb[i]
      for j in i + 1 ..< n:
        acc = acc - d.r[i, j] * result[j]
      if abs(d.r[i, i]) <= diagTol:
        raise newException(ValueError, "leastSquares: rank-deficient matrix")
      result[i] = acc / d.r[i, i]





