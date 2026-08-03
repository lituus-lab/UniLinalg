# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniLinalg benchmark harness
## ============================
##
## Deterministic micro-benchmarks for Matrix ops and the four decompositions.
## Prints a table and, with `--csv:<file>`, writes `op,n,ms,ops_per_sec`. With
## `--baseline:<file>`, compares against a previous CSV and flags regressions
## beyond `--threshold` (default 0.15 = 15%), exiting non-zero.
##
## Philosophy: benchmarks justify a claim of speed, they do not replace
## correctness testing (tests/test_linalg.nim) or cross-validation against
## a reference (bench/compare/vs_*.nim).
import std/[random, monotimes, times, strformat, tables, os, strutils]
import ../src/UniLinalg

type Row = object
  op: string
  n: int
  ms: float

proc randMatrix(n: int, seed: int64): Matrix[float64] =
  var r = initRand(seed)
  result = initMatrix[float64](n, n)
  for i in 0 ..< n:
    for j in 0 ..< n:
      result[i, j] = r.rand(2.0) - 1.0

proc randSpdMatrix(n: int, seed: int64): Matrix[float64] =
  ## A^T A + n*I is always symmetric positive-definite.
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

var rows: seq[Row]
proc record(op: string, n: int, ms: float) =
  rows.add Row(op: op, n: n, ms: ms)
  echo &"  {op:<22} n={n:>6}  {ms:>10.3f} ms  {(n.float / (ms / 1000.0)):>14.0f} ops/s"

proc benchAll() =
  echo "== UniLinalg benchmarks (deterministic, -d:danger recommended) =="

  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 11)
    let b = randMatrix(n, 12)
    let ms = timed: discard a * b
    record("matmul", n, ms)

  for n in [16, 32, 64, 128, 256, 512]:
    let a = randMatrix(n, 13)
    let ms = timed: discard transpose(a)
    record("transpose", n, ms)

  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 14)
    let b = randVec(n, 15)
    let ms = timed: discard solve(a, b)
    record("lu_solve", n, ms)

  for n in [16, 32, 64, 128, 256]:
    let a = randMatrix(n, 14)
    let b = randVec(n, 15)
    let ms = timed: discard solve(a, b, refine = true)
    record("lu_solve_refine", n, ms)

  for n in [16, 32, 64, 128, 256]:
    let a = randSpdMatrix(n, 16)
    let ms = timed: discard cholesky(a)
    record("cholesky", n, ms)

  for n in [16, 32, 64, 128]:
    let a = randMatrix(n, 17)
    let ms = timed: discard qrDecompose(a)
    record("qr", n, ms)

  for n in [16, 32, 64]:
    let a = randMatrix(n, 18)
    let ms = timed: discard svdDecompose(a)
    record("svd (one-sided Jacobi)", n, ms)

# --------------------------------------------------------------------------
# Precision parity: solve() vs solve(refine=true) -- see ADR-0006
# --------------------------------------------------------------------------

proc maxResidual(a: Matrix[float64], b, x: seq[float64]): float64 =
  result = 0.0
  for v in residual(a, b, x):
    if abs(v) > result: result = abs(v)

proc benchRefineParity() =
  echo ""
  echo "Precision parity: solve() vs solve(refine=true)"
  echo repeat('-', 70)
  for n in [16, 32, 64, 128]:
    let a = randMatrix(n, 14)
    let b = randVec(n, 15)
    let sv = svdDecompose(a)
    let cond2 = sv.s[0] / sv.s[sv.s.len - 1]
    let plain = solve(a, b)
    let refined = solve(a, b, refine = true)
    echo &"  n={n:>4}  cond2={cond2:>10.2f}  " &
        &"max|residual| plain={maxResidual(a, b, plain):.3e}  " &
        &"refined={maxResidual(a, b, refined):.3e}"

# --------------------------------------------------------------------------
# CSV + regression gate
# --------------------------------------------------------------------------

proc writeCsv(path: string) =
  var s = "op,n,ms,ops_per_sec\n"
  for r in rows:
    s.add &"{r.op},{r.n},{r.ms:.3f},{(r.n.float / (r.ms / 1000.0)):.0f}\n"
  writeFile(path, s)
  echo "wrote ", path

proc compareBaseline(path: string, threshold: float): int =
  if not fileExists(path):
    echo "no baseline at ", path, " (skipping regression check)"
    return 0
  var base = initTable[string, float]()
  for line in lines(path):
    let f = line.split(',')
    if f.len >= 3 and f[0] != "op":
      base[f[0] & "/" & f[1]] = parseFloat(f[2])
  var regressions = 0
  for r in rows:
    let key = r.op & "/" & $r.n
    if key in base and base[key] > 0:
      let delta = (r.ms - base[key]) / base[key]
      if delta > threshold:
        echo &"  REGRESSION {key}: {base[key]:.3f} -> {r.ms:.3f} ms (+{delta * 100:.0f}%)"
        inc regressions
  echo &"regression check vs {path}: {regressions} over +{threshold * 100:.0f}%"
  regressions

when isMainModule:
  var csvOut, baseline = ""
  var threshold = 0.15
  for i in 1 .. paramCount():
    let a = paramStr(i)
    if a.startsWith("--csv:"): csvOut = a[6 .. ^1]
    elif a.startsWith("--baseline:"): baseline = a[11 .. ^1]
    elif a.startsWith("--threshold:"): threshold = parseFloat(a[12 .. ^1])
  benchAll()
  benchRefineParity()
  if csvOut.len > 0: writeCsv(csvOut)
  if baseline.len > 0:
    if compareBaseline(baseline, threshold) > 0: quit(1)
