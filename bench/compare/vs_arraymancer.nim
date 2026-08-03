# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniLinalg vs Arraymancer -- compiled-code A/B comparison
## ============================================================
##
## Compares solve/qr/svd against Arraymancer (BLAS/LAPACK-backed), both
## compiled Nim in one binary, same clock. See bench/README.md for why this
## comparison exists alongside vs_lapack.nim, and why only these three ops.
##
## Usage: nim c -r -d:release --path:<arraymancer> bench/compare/vs_arraymancer.nim [--csv:<file>]
import std/[random, monotimes, times, strformat, math, os, strutils]
import arraymancer
import ../../src/UniLinalg

# ------------------------------------------------------------------------------
# Matrix[float64] <-> Tensor[float64]
# ------------------------------------------------------------------------------

proc toTensor2D(m: Matrix[float64]): Tensor[float64] =
  var rowsSeq: seq[seq[float64]]
  for i in 0 ..< m.rows:
    var row: seq[float64]
    for j in 0 ..< m.cols: row.add m[i, j]
    rowsSeq.add row
  rowsSeq.toTensor()

proc fromTensor2D(t: Tensor[float64]): Matrix[float64] =
  let r = t.shape[0]
  let c = t.shape[1]
  result = initMatrix[float64](r, c)
  for i in 0 ..< r:
    for j in 0 ..< c:
      result[i, j] = t[i, j]

proc toSeq1D(t: Tensor[float64]): seq[float64] =
  for i in 0 ..< t.shape[0]: result.add t[i]

proc toTensor1D(v: seq[float64]): Tensor[float64] =
  v.toTensor()

# ------------------------------------------------------------------------------
# Residual checks (independent per implementation, same doctrine as vs_lapack.nim)
# ------------------------------------------------------------------------------

proc maxAbsDiff(a, b: Matrix[float64]): float64 =
  doAssert a.rows == b.rows and a.cols == b.cols
  for i in 0 ..< a.data.len:
    result = max(result, abs(a.data[i] - b.data[i]))

proc solveResidual(a: Matrix[float64], b, x: seq[float64]): float64 =
  let r = a * x
  for i in 0 ..< b.len:
    result = max(result, abs(r[i] - b[i]))

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

proc report(op: string, n: int, uniMs, amMs: float, uniOk, amOk: bool) =
  csvRows.add CsvRow(op: op, tool: "unilinalg", n: n, ms: uniMs)
  csvRows.add CsvRow(op: op, tool: "arraymancer", n: n, ms: amMs)
  if not uniOk or not amOk: anyFailure = true
  let ratio = uniMs / amMs
  let uniTag = if uniOk: "ok" else: "FAIL"
  let amTag = if amOk: "ok" else: "FAIL"
  echo &"  {op:<10} n={n:>4}  unilinalg={uniMs:>9.4f}ms [{uniTag}]  " &
      &"arraymancer={amMs:>9.4f}ms [{amTag}]  ratio={ratio:>7.1f}x"

const Tol = 1e-8

proc compareAll() =
  echo "== UniLinalg vs Arraymancer (BLAS/LAPACK-backed Nim tensor library) =="
  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 31)
    let b = randVec(n, 32)
    var x: seq[float64]
    let uniMs = timed: x = solve(a, b)
    # Tensor conversion happens outside the timed block, same reasoning as
    # vs_lapack.nim: it is marshalling overhead, not part of what Arraymancer's
    # own solve() computes, and would understate its real speed if counted.
    let at = toTensor2D(a)
    let bt = toTensor1D(b)
    var xt: Tensor[float64]
    let amMs = timed: xt = solve(at, bt)
    let xA = toSeq1D(xt)
    report("solve", n, uniMs, amMs, solveResidual(a, b, x) < Tol,
        solveResidual(a, b, xA) < Tol)

  for n in [16, 32, 64, 128]:
    let a = randMatrix(n, 33)
    var qU, rU: Matrix[float64]
    let uniMs = timed:
      let d = qrDecompose(a)
      qU = d.q
      rU = d.r
    let at = toTensor2D(a)
    var qt, rt: Tensor[float64]
    let amMs = timed:
      (qt, rt) = qr(at)
    let qA = fromTensor2D(qt)
    let rA = fromTensor2D(rt)
    let (uniRecon, uniOrth) = qrResidual(a, qU, rU)
    let (amRecon, amOrth) = qrResidual(a, qA, rA)
    report("qr", n, uniMs, amMs, uniRecon < Tol and uniOrth < Tol,
        amRecon < Tol and amOrth < Tol)

  for n in [16, 32, 64]:
    let a = randMatrix(n, 34)
    var sU: seq[float64]
    var uU, vU: Matrix[float64]
    var dU: SvdDecomposition[float64]
    let uniMs = timed:
      dU = svdDecompose(a)
    uU = dU.u
    sU = dU.s
    # V^T conversion happens outside the timed block: Arraymancer's svd()
    # returns V^T directly, so timing UniLinalg's V->V^T conversion alongside
    # it would unfairly inflate UniLinalg's measured time for the same result.
    vU = transpose(dU.v)
    let at = toTensor2D(a)
    var ut, st, vht: Tensor[float64]
    let amMs = timed:
      (ut, st, vht) = svd(at, float64)
    let uA = fromTensor2D(ut)
    let sA = toSeq1D(st)
    let vA = fromTensor2D(vht)
    var svalDiff = 0.0
    for i in 0 ..< n: svalDiff = max(svalDiff, abs(sU[i] - sA[i]))
    report("svd", n, uniMs, amMs, svdResidual(a, uU, sU, vU) < Tol,
        svdResidual(a, uA, sA, vA) < Tol)
    echo &"    max|s_unilinalg - s_arraymancer| = {svalDiff:.3e}"

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
