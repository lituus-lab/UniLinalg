# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniLinalg benchmark harness
## ============================
##
## Deterministic micro-benchmarks for Matrix ops and the four decompositions.
## `--csv:<file>` writes `op,n,ms,dim_per_sec` (n/seconds, not a FLOP count).
## `--baseline:<file> --threshold:<f>` flags regressions past that fraction
## (default 0.15) and exits non-zero. See bench/README.md for methodology.
import std/[random, monotimes, times, strformat, tables, os, strutils, algorithm]
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

const
  WarmupIters = 2
  SampleCount = 5 # odd: the median is one element, no averaging needed.

template timed(body: untyped): float =
  ## Warms up (discards the timing) then takes the median of several
  ## samples, in ms. A single wall-clock sample is noisy enough on its own
  ## (OS scheduling jitter, cold cache) to trip --threshold spuriously; the
  ## median is far less sensitive to a single outlier than the mean would be.
  for _ in 1 .. WarmupIters:
    body
  var samples: array[SampleCount, float]
  for i in 0 ..< SampleCount:
    let t0 = getMonoTime()
    body
    samples[i] = inMicroseconds(getMonoTime() - t0).float / 1000.0 # ms
  sort(samples)
  samples[SampleCount div 2]

proc fmtDimPerSec(n: int, ms: float): string =
  # Sub-microsecond ops (small n, cheap op) can measure as exactly 0 ms on this
  # timer's resolution; n/0 is an infinite, not a real, throughput.
  if ms <= 0: "-" else: (&"{(n.float / (ms / 1000.0)):.0f}")

var rows: seq[Row]
proc record(op: string, n: int, ms: float) =
  rows.add Row(op: op, n: n, ms: ms)
  echo &"  {op:<22} n={n:>6}  {ms:>10.3f} ms  {fmtDimPerSec(n, ms):>14} dim/s"

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
    let ms = timed: discard solve(a, b, useRefinement = true)
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
# Precision parity: solve() vs solve(useRefinement=true) -- see ADR-0006
# --------------------------------------------------------------------------

proc maxResidual(a: Matrix[float64], b, x: seq[float64]): float64 =
  result = 0.0
  for v in residual(a, b, x):
    if abs(v) > result: result = abs(v)

type ParityRow = object
  n: int
  cond2, plainRes, refinedRes: float64
var parityRows: seq[ParityRow]

proc benchRefineParity() =
  echo ""
  echo "Precision parity: solve() vs solve(useRefinement=true)"
  echo repeat('-', 70)
  for n in [16, 32, 64, 128]:
    let a = randMatrix(n, 14)
    let b = randVec(n, 15)
    let sv = svdDecompose(a)
    let cond2 = sv.s[0] / sv.s[sv.s.len - 1]
    let plain = solve(a, b)
    let refined = solve(a, b, useRefinement = true)
    let plainRes = maxResidual(a, b, plain)
    let refinedRes = maxResidual(a, b, refined)
    parityRows.add ParityRow(n: n, cond2: cond2, plainRes: plainRes, refinedRes: refinedRes)
    echo &"  n={n:>4}  cond2={cond2:>10.2f}  " &
        &"max|residual| plain={plainRes:.3e}  " &
        &"refined={refinedRes:.3e}"

# --------------------------------------------------------------------------
# CSV + regression gate
# --------------------------------------------------------------------------

proc writeCsv(path: string) =
  var s = "op,n,ms,dim_per_sec\n"
  for r in rows:
    s.add &"{r.op},{r.n},{r.ms:.3f},{fmtDimPerSec(r.n, r.ms)}\n"
  writeFile(path, s)
  echo "wrote ", path

proc writeMd(path: string) =
  var s = "| op | n | ms | dim/sec |\n|---|---|---|---|\n"
  for r in rows:
    s.add &"| {r.op} | {r.n} | {r.ms:.3f} | {fmtDimPerSec(r.n, r.ms)} |\n"
  s.add "\n**Accuracy: solve() vs solve(useRefinement=true)**\n\n"
  s.add "| n | cond2(A) | max\\|residual\\| plain | max\\|residual\\| refined |\n"
  s.add "|---|---|---|---|\n"
  for p in parityRows:
    s.add &"| {p.n} | {p.cond2:.1f} | {p.plainRes:.3e} | {p.refinedRes:.3e} |\n"
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
  var csvOut, baseline, mdOut = ""
  var threshold = 0.15
  for i in 1 .. paramCount():
    let a = paramStr(i)
    if a.startsWith("--csv:"): csvOut = a[6 .. ^1]
    elif a.startsWith("--baseline:"): baseline = a[11 .. ^1]
    elif a.startsWith("--threshold:"): threshold = parseFloat(a[12 .. ^1])
    elif a.startsWith("--md:"): mdOut = a[5 .. ^1]
  benchAll()
  benchRefineParity()
  if csvOut.len > 0: writeCsv(csvOut)
  if mdOut.len > 0: writeMd(mdOut)
  if baseline.len > 0:
    if compareBaseline(baseline, threshold) > 0: quit(1)
