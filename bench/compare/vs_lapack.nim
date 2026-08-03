# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniLinalg vs raw LAPACK (nimlapack) -- compiled-code A/B comparison
## ====================================================================
##
## Compares solve/cholesky/qr/svd/det against raw LAPACK
## (dgesv/dpotrf/dgeqrf+dorgqr/dgesvd/dgetrf via nimlapack), both compiled
## Nim in the same binary, same clock. See bench/README.md for why this
## comparison exists and how correctness is checked independently per side.
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

## Each *Core proc below takes an already-column-major buffer and does only
## the real LAPACK call(s) -- no toColMajor/fromColMajor format conversion.
## Timing wraps the *Core call alone (see compareAll): converting UniLinalg's
## row-major Matrix to/from LAPACK's column-major layout is marshalling
## overhead with nothing to do with LAPACK's own algorithmic cost, and
## counting it inside LAPACK's measured time would understate LAPACK's real
## speed relative to UniLinalg's native row-major path.

proc lapackSolveCore(acol, bcol: var seq[float64], n: int) =
  var nn = cint(n)
  var nrhs: cint = 1
  var ipiv = newSeq[cint](n)
  var info: cint
  dgesv(addr nn, addr nrhs, addr acol[0], addr nn, addr ipiv[0], addr bcol[0],
      addr nn, addr info)
  doAssert info == 0, "dgesv failed, info=" & $info

proc lapackCholeskyCore(acol: var seq[float64], n: int) =
  var nn = cint(n)
  var info: cint
  dpotrf("L".cstring, addr nn, addr acol[0], addr nn, addr info)
  doAssert info == 0, "dpotrf failed (not SPD?), info=" & $info

proc lapackCholeskyShape(acol: seq[float64], n: int): Matrix[float64] =
  ## dpotrf only touches the requested triangle; the other side of the raw
  ## buffer is untouched garbage from the input, so the upper triangle is
  ## zeroed explicitly here (outside timing, alongside the format conversion).
  result = fromColMajor(acol, n, n)
  for i in 0 ..< n:
    for j in i + 1 ..< n:
      result[i, j] = 0.0

proc lapackQRCore(acol, tau, acolForR: var seq[float64], m, n: int) =
  ## Thin/economy QR: Q is m x n with orthonormal columns, R is n x n
  ## upper triangular. (UniLinalg's own QR is "full": Q is m x m. The two
  ## conventions are cross-checked independently below, not against
  ## each other -- see the module doc.)
  var mc = cint(m)
  var nc = cint(n)
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

  # Snapshot the raw buffer before dorgqr overwrites it with Q -- R is
  # extracted from this snapshot outside timing. A plain seq copy, not a
  # layout conversion, and unavoidably part of the in-place LAPACK algorithm
  # (dorgqr has no option to preserve its input), so it stays inside timing.
  acolForR = acol

  var kc = cint(n)
  lwork = -1
  dorgqr(addr mc, addr nc, addr kc, addr acol[0], addr mc, addr tau[0],
      addr work[0], addr lwork, addr info)
  lwork = cint(work[0])
  work = newSeq[float64](max(1, int(lwork)))
  dorgqr(addr mc, addr nc, addr kc, addr acol[0], addr mc, addr tau[0],
      addr work[0], addr lwork, addr info)
  doAssert info == 0, "dorgqr failed, info=" & $info

proc lapackQRShape(acolForR: seq[float64], m, n: int): Matrix[float64] =
  ## Upper triangle of the first n rows of the pre-dorgqr buffer.
  let rFull = fromColMajor(acolForR, m, n)
  result = initMatrix[float64](n, n)
  for i in 0 ..< n:
    for j in i ..< n:
      result[i, j] = rFull[i, j]

proc lapackSvdCore(acol, s, u, vt: var seq[float64], m, n: int) =
  ## Economy SVD: U is m x k, S has k values, Vt is k x n, k = min(m,n).
  let k = min(m, n)
  var mc = cint(m)
  var nc = cint(n)
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

proc lapackDetCore(acol: var seq[float64], n: int): float64 =
  ## The sign/diagonal-product loop after dgetrf computes the actual
  ## determinant value from the LU factors -- unlike fromColMajor elsewhere,
  ## this is not format conversion, so it stays inside timing. (The diagonal
  ## entries also happen to sit at the same flat index in row- and
  ## column-major layout for a square matrix, but that's incidental; the loop
  ## is genuine algorithm work either way.)
  var nn = cint(n)
  var ipiv = newSeq[cint](n)
  var info: cint
  dgetrf(addr nn, addr nn, addr acol[0], addr nn, addr ipiv[0], addr info)
  doAssert info >= 0, "dgetrf failed, info=" & $info
  if info > 0: return 0.0 # exactly singular
  var sign = 1.0
  for i in 0 ..< n:
    if int(ipiv[i]) != i + 1: sign = -sign
  result = sign
  for i in 0 ..< n:
    result = result * acol[i * n + i]

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
var anyFailure = false

proc report(op: string, n: int, uniMs, lapackMs: float, uniOk, lapackOk: bool) =
  csvRows.add CsvRow(op: op, tool: "unilinalg", n: n, ms: uniMs)
  csvRows.add CsvRow(op: op, tool: "lapack", n: n, ms: lapackMs)
  if not uniOk or not lapackOk: anyFailure = true
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
    var x: seq[float64]
    let uniMs = timed: x = solve(a, b)
    var acol = toColMajor(a)
    var bcol = b
    let lapackMs = timed: lapackSolveCore(acol, bcol, n)
    let xL = bcol
    report("solve", n, uniMs, lapackMs, solveResidual(a, b, x) < Tol,
        solveResidual(a, b, xL) < Tol)

  for n in [16, 32, 64, 128, 256]:
    let a = randSpdMatrix(n, 23)
    var l: Matrix[float64]
    let uniMs = timed: l = cholesky(a)
    var acol = toColMajor(a)
    let lapackMs = timed: lapackCholeskyCore(acol, n)
    let lL = lapackCholeskyShape(acol, n)
    report("cholesky", n, uniMs, lapackMs, choleskyResidual(a, l) < Tol,
        choleskyResidual(a, lL) < Tol)

  for n in [16, 32, 64, 128]:
    let a = randMatrix(n, 24)
    var qU, rU: Matrix[float64]
    let uniMs = timed:
      let d = qrDecompose(a)
      qU = d.q
      rU = d.r
    var acol = toColMajor(a)
    var tau = newSeq[float64](n)
    var acolForR = newSeq[float64](a.rows * n)
    let lapackMs = timed:
      lapackQRCore(acol, tau, acolForR, a.rows, n)
    let rL = lapackQRShape(acolForR, a.rows, n)
    let qL = fromColMajor(acol, a.rows, n)
    let (uniRecon, uniOrth) = qrResidual(a, qU, rU)
    let (lapackRecon, lapackOrth) = qrResidual(a, qL, rL)
    report("qr", n, uniMs, lapackMs, uniRecon < Tol and uniOrth < Tol,
        lapackRecon < Tol and lapackOrth < Tol)

  for n in [16, 32, 64]:
    let a = randMatrix(n, 25)
    var sU: seq[float64]
    var uU, vU: Matrix[float64]
    var dU: SvdDecomposition[float64]
    let uniMs = timed:
      dU = svdDecompose(a)
    uU = dU.u
    sU = dU.s
    # V^T conversion happens outside the timed block: LAPACK's dgesvd returns
    # V^T directly, so timing UniLinalg's V->V^T conversion alongside it would
    # unfairly inflate UniLinalg's measured time for the same mathematical result.
    vU = transpose(dU.v)
    let k = n # randMatrix is square here, so min(m, n) collapses to n.
    var acol = toColMajor(a)
    var s = newSeq[float64](k)
    var u = newSeq[float64](n * k)
    var vt = newSeq[float64](k * n)
    let lapackMs = timed:
      lapackSvdCore(acol, s, u, vt, n, n)
    let sL = s
    let uL = fromColMajor(u, n, k)
    let vL = fromColMajor(vt, k, n)
    # Singular values are unique (both sorted descending) -- the strongest,
    # unambiguous cross-check between the two implementations.
    var svalDiff = 0.0
    for i in 0 ..< n: svalDiff = max(svalDiff, abs(sU[i] - sL[i]))
    report("svd", n, uniMs, lapackMs, svdResidual(a, uU, sU, vU) < Tol,
        svdResidual(a, uL, sL, vL) < Tol)
    echo &"    max|s_unilinalg - s_lapack| = {svalDiff:.3e}"

  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 26)
    var dU: float64
    let uniMs = timed: dU = det(a)
    var acol = toColMajor(a)
    var dL: float64
    let lapackMs = timed: dL = lapackDetCore(acol, n)
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
  if anyFailure: quit(1)
