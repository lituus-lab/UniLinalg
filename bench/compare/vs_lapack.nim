## UniLinalg vs raw LAPACK (nimlapack) -- compiled-code A/B comparison
## ====================================================================
##
## Not a Python binding: both sides are compiled Nim in the same binary,
## timed with the same clock, so the ratio is machine-independent. LAPACK
## (dgesv/dpotrf/dgeqrf+dorgqr/dgesvd) is the reference performant oracle --
## the actual battle-tested Fortran routines every numerical package
## eventually calls, linked here via nimlapack against the system
## OpenBLAS/LAPACK (see README's Reference section).
##
## Correctness is checked independently per implementation (residual: does
## each solution actually satisfy Ax=b / LL^T=A / QR=A / U*diag(S)*V^T=A to
## within tolerance), not by comparing outputs bitwise -- Householder sign
## conventions and Jacobi sweep order legitimately differ between
## implementations, but singular values and residuals do not.
##
## Usage: nim c -r -d:release --path:<nimlapack> bench/compare/vs_lapack.nim [--csv:<file>]
import std/[random, monotimes, times, strformat, math, os, strutils]
import nimlapack
import ../../src/UniLinalg

# ------------------------------------------------------------------------------
# Row-major <-> column-major (LAPACK is Fortran, column-major)
# ------------------------------------------------------------------------------

proc toColMajor(m: Matrix[float64]): seq[float64] =
  result = newSeq[float64](m.rows * m.cols)
  for i in 0 ..< m.rows:
    for j in 0 ..< m.cols:
      result[j * m.rows + i] = m[i, j]

proc fromColMajor(data: seq[float64], rows, cols: int): Matrix[float64] =
  result = initMatrix[float64](rows, cols)
  for i in 0 ..< rows:
    for j in 0 ..< cols:
      result[i, j] = data[j * rows + i]

# ------------------------------------------------------------------------------
# Thin wrappers around the raw F77 LAPACK calls nimlapack exposes.
# ------------------------------------------------------------------------------

proc lapackSolve(a: Matrix[float64], b: seq[float64]): seq[float64] =
  var n = cint(a.rows)
  var nrhs: cint = 1
  var acol = toColMajor(a)
  var bcol = b
  var ipiv = newSeq[cint](a.rows)
  var info: cint
  dgesv(addr n, addr nrhs, addr acol[0], addr n, addr ipiv[0], addr bcol[0],
      addr n, addr info)
  doAssert info == 0, "dgesv failed, info=" & $info
  bcol

proc lapackCholesky(a: Matrix[float64]): Matrix[float64] =
  ## Lower-triangular L, A = L L^T. dpotrf only touches the requested
  ## triangle; the other side of the raw buffer is untouched garbage from
  ## the input, so the upper triangle is zeroed explicitly before returning.
  var n = cint(a.rows)
  var acol = toColMajor(a)
  var info: cint
  dpotrf("L".cstring, addr n, addr acol[0], addr n, addr info)
  doAssert info == 0, "dpotrf failed (not SPD?), info=" & $info
  result = fromColMajor(acol, a.rows, a.rows)
  for i in 0 ..< a.rows:
    for j in i + 1 ..< a.rows:
      result[i, j] = 0.0

proc lapackQR(a: Matrix[float64]): tuple[q, r: Matrix[float64]] =
  ## Thin/economy QR: Q is m x n with orthonormal columns, R is n x n
  ## upper triangular. (UniLinalg's own QR is "full": Q is m x m. The two
  ## conventions are cross-checked independently below, not against
  ## each other -- see the module doc.)
  let m = a.rows
  let n = a.cols
  var mc = cint(m)
  var nc = cint(n)
  var acol = toColMajor(a)
  var tau = newSeq[float64](n)
  var lwork: cint = -1
  var work = newSeq[float64](1)
  var info: cint
  # Workspace query, then the real call -- standard LAPACK idiom.
  dgeqrf(addr mc, addr nc, addr acol[0], addr mc, addr tau[0], addr work[0],
      addr lwork, addr info)
  lwork = cint(work[0])
  work = newSeq[float64](max(1, int(lwork)))
  dgeqrf(addr mc, addr nc, addr acol[0], addr mc, addr tau[0], addr work[0],
      addr lwork, addr info)
  doAssert info == 0, "dgeqrf failed, info=" & $info

  # Extract R (upper triangle of the first n rows) before dorgqr overwrites
  # the buffer with Q.
  let rFull = fromColMajor(acol, m, n)
  var r = initMatrix[float64](n, n)
  for i in 0 ..< n:
    for j in i ..< n:
      r[i, j] = rFull[i, j]

  var kc = cint(n)
  lwork = -1
  dorgqr(addr mc, addr nc, addr kc, addr acol[0], addr mc, addr tau[0],
      addr work[0], addr lwork, addr info)
  lwork = cint(work[0])
  work = newSeq[float64](max(1, int(lwork)))
  dorgqr(addr mc, addr nc, addr kc, addr acol[0], addr mc, addr tau[0],
      addr work[0], addr lwork, addr info)
  doAssert info == 0, "dorgqr failed, info=" & $info
  let q = fromColMajor(acol, m, n)
  (q, r)

proc lapackSvd(a: Matrix[float64]): tuple[u: Matrix[float64], s: seq[float64],
    vt: Matrix[float64]] =
  ## Economy SVD: U is m x k, S has k values, Vt is k x n, k = min(m,n).
  let m = a.rows
  let n = a.cols
  let k = min(m, n)
  var mc = cint(m)
  var nc = cint(n)
  var acol = toColMajor(a)
  var s = newSeq[float64](k)
  var u = newSeq[float64](m * k)
  var vt = newSeq[float64](k * n)
  var ldu = cint(m)
  var ldvt = cint(k)
  var lwork: cint = -1
  var work = newSeq[float64](1)
  var info: cint
  dgesvd("S".cstring, "S".cstring, addr mc, addr nc, addr acol[0], addr mc,
      addr s[0], addr u[0], addr ldu, addr vt[0], addr ldvt, addr work[0],
      addr lwork, addr info)
  lwork = cint(work[0])
  work = newSeq[float64](max(1, int(lwork)))
  dgesvd("S".cstring, "S".cstring, addr mc, addr nc, addr acol[0], addr mc,
      addr s[0], addr u[0], addr ldu, addr vt[0], addr ldvt, addr work[0],
      addr lwork, addr info)
  doAssert info == 0, "dgesvd failed, info=" & $info
  (fromColMajor(u, m, k), s, fromColMajor(vt, k, n))

proc lapackDet(a: Matrix[float64]): float64 =
  var n = cint(a.rows)
  var acol = toColMajor(a)
  var ipiv = newSeq[cint](a.rows)
  var info: cint
  dgetrf(addr n, addr n, addr acol[0], addr n, addr ipiv[0], addr info)
  doAssert info >= 0, "dgetrf failed, info=" & $info
  if info > 0: return 0.0 # exactly singular
  var sign = 1.0
  for i in 0 ..< a.rows:
    if int(ipiv[i]) != i + 1: sign = -sign
  var d = sign
  for i in 0 ..< a.rows:
    d = d * acol[i * a.rows + i]
  d

# ------------------------------------------------------------------------------
# Residual checks (independent per implementation -- see module doc)
# ------------------------------------------------------------------------------

proc maxAbsDiff(a, b: Matrix[float64]): float64 =
  doAssert a.rows == b.rows and a.cols == b.cols
  for i in 0 ..< a.data.len:
    result = max(result, abs(a.data[i] - b.data[i]))

proc solveResidual(a: Matrix[float64], b, x: seq[float64]): float64 =
  let r = a * x
  for i in 0 ..< b.len:
    result = max(result, abs(r[i] - b[i]))

proc choleskyResidual(a, l: Matrix[float64]): float64 =
  maxAbsDiff(l * transpose(l), a)

proc qrResidual(a, q, r: Matrix[float64]): tuple[recon, orthog: float64] =
  let n = q.cols
  var idn = initMatrix[float64](n, n)
  for i in 0 ..< n: idn[i, i] = 1.0
  (maxAbsDiff(q * r, a), maxAbsDiff(transpose(q) * q, idn))

proc svdResidual(a, u: Matrix[float64], s: seq[float64], vt: Matrix[
    float64]): float64 =
  var sMat = initMatrix[float64](s.len, s.len)
  for i in 0 ..< s.len: sMat[i, i] = s[i]
  maxAbsDiff(u * sMat * vt, a)

# ------------------------------------------------------------------------------
# Bench + compare
# ------------------------------------------------------------------------------

proc randMatrix(n: int, seed: int64): Matrix[float64] =
  var r = initRand(seed)
  result = initMatrix[float64](n, n)
  for i in 0 ..< n:
    for j in 0 ..< n:
      result[i, j] = r.rand(2.0) - 1.0

proc randSpdMatrix(n: int, seed: int64): Matrix[float64] =
  let a = randMatrix(n, seed)
  result = transpose(a) * a
  for i in 0 ..< n:
    result[i, i] = result[i, i] + float64(n)

proc randVec(n: int, seed: int64): seq[float64] =
  var r = initRand(seed)
  result = newSeq[float64](n)
  for i in 0 ..< n: result[i] = r.rand(2.0) - 1.0

template timed(body: untyped): float =
  let t0 = getMonoTime()
  body
  inMicroseconds(getMonoTime() - t0).float / 1000.0 # ms

type CsvRow = object
  op: string
  tool: string
  n: int
  ms: float

var csvRows: seq[CsvRow]

proc report(op: string, n: int, uniMs, lapackMs: float, uniOk, lapackOk: bool) =
  csvRows.add CsvRow(op: op, tool: "unilinalg", n: n, ms: uniMs)
  csvRows.add CsvRow(op: op, tool: "lapack", n: n, ms: lapackMs)
  let ratio = uniMs / lapackMs
  let uniTag = if uniOk: "ok" else: "FAIL"
  let lapackTag = if lapackOk: "ok" else: "FAIL"
  echo &"  {op:<10} n={n:>4}  unilinalg={uniMs:>9.4f}ms [{uniTag}]  " &
      &"lapack={lapackMs:>9.4f}ms [{lapackTag}]  ratio={ratio:>7.1f}x"

const Tol = 1e-8

proc compareAll() =
  echo "== UniLinalg vs raw LAPACK (nimlapack, OpenBLAS) =="
  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 21)
    let b = randVec(n, 22)
    var x, xL: seq[float64]
    let uniMs = timed: x = solve(a, b)
    let lapackMs = timed: xL = lapackSolve(a, b)
    report("solve", n, uniMs, lapackMs, solveResidual(a, b, x) < Tol,
        solveResidual(a, b, xL) < Tol)

  for n in [16, 32, 64, 128, 256]:
    let a = randSpdMatrix(n, 23)
    var l, lL: Matrix[float64]
    let uniMs = timed: l = cholesky(a)
    let lapackMs = timed: lL = lapackCholesky(a)
    report("cholesky", n, uniMs, lapackMs, choleskyResidual(a, l) < Tol,
        choleskyResidual(a, lL) < Tol)

  for n in [16, 32, 64, 128]:
    let a = randMatrix(n, 24)
    var qU, rU, qL, rL: Matrix[float64]
    let uniMs = timed:
      let d = qrDecompose(a)
      qU = d.q
      rU = d.r
    let lapackMs = timed:
      (qL, rL) = lapackQR(a)
    let (uniRecon, uniOrth) = qrResidual(a, qU, rU)
    let (lapackRecon, lapackOrth) = qrResidual(a, qL, rL)
    report("qr", n, uniMs, lapackMs, uniRecon < Tol and uniOrth < Tol,
        lapackRecon < Tol and lapackOrth < Tol)

  for n in [16, 32, 64]:
    let a = randMatrix(n, 25)
    var sU, sL: seq[float64]
    var uU, vU, uL, vL: Matrix[float64]
    var dU: SvdDecomposition[float64]
    let uniMs = timed:
      dU = svdDecompose(a)
    uU = dU.u
    sU = dU.s
    # V^T conversion happens outside the timed block: LAPACK's dgesvd returns
    # V^T directly, so timing UniLinalg's V->V^T conversion alongside it would
    # unfairly inflate UniLinalg's measured time for the same mathematical result.
    vU = transpose(dU.v)
    let lapackMs = timed:
      (uL, sL, vL) = lapackSvd(a)
    # Singular values are unique (both sorted descending) -- the strongest,
    # unambiguous cross-check between the two implementations.
    var svalDiff = 0.0
    for i in 0 ..< n: svalDiff = max(svalDiff, abs(sU[i] - sL[i]))
    report("svd", n, uniMs, lapackMs, svdResidual(a, uU, sU, vU) < Tol,
        svdResidual(a, uL, sL, vL) < Tol)
    echo &"    max|s_unilinalg - s_lapack| = {svalDiff:.3e}"

  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 26)
    var dU, dL: float64
    let uniMs = timed: dU = det(a)
    let lapackMs = timed: dL = lapackDet(a)
    let relDiff = abs(dU - dL) / max(1.0, abs(dL))
    report("det", n, uniMs, lapackMs, relDiff < 1e-6, true)

proc writeCsv(path: string) =
  var s = "op,tool,n,ms\n"
  for r in csvRows:
    s.add &"{r.op},{r.tool},{r.n},{r.ms:.4f}\n"
  writeFile(path, s)
  echo "wrote ", path

when isMainModule:
  var csvOut = ""
  for i in 1 .. paramCount():
    let a = paramStr(i)
    if a.startsWith("--csv:"): csvOut = a[6 .. ^1]
  compareAll()
  if csvOut.len > 0: writeCsv(csvOut)
