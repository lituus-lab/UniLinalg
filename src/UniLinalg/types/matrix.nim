# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — Dense matrix type and basic operations
# =============================================================================
#
# A pedagogical, dependency-free dense matrix: row-major `seq[T]` storage,
# explicit dimensions, no magic. The decompositions (LU, Cholesky, QR, SVD)
# live in `algorithms/`.
#
# Genericity: storage and ring operations (+, -, *) work for any numeric T.
# Decompositions need division and sqrt and are constrained to floats there.
#
# Contracts (NimContracts): shape preconditions on the constructors, ring
# operations, and accessors are debug-only `require:` — non-blocking in
# release. No `ensure:` here: every result shape in this file is a direct
# function of its own inputs (already checked by `require:`), so a
# structural postcondition would just restate the precondition. Compiled
# away under -d:release/-d:danger.

import contracts

type
  Matrix*[T] = object
    ## Dense row-major matrix. Element (i, j) lives at `data[i*cols + j]`.
    rows*, cols*: int
    data*: seq[T]

# ------------------------------------------------------------------------------
# Constructors
# ------------------------------------------------------------------------------

func initMatrix*[T](rows, cols: int): Matrix[T] {.contractual.} =
  ## Zero matrix of the given shape.
  require: rows > 0 and cols > 0
  body:
    Matrix[T](rows: rows, cols: cols, data: newSeq[T](rows * cols))

func matrix*[T](rows, cols: int, elements: openArray[T]): Matrix[
    T] {.contractual.} =
  ## Matrix from a flat row-major element list.
  require: rows > 0 and cols > 0 and elements.len == rows * cols
  body:
    result = initMatrix[T](rows, cols)
    for i in 0 ..< elements.len:
      result.data[i] = elements[i]

func identity*[T](n: int): Matrix[T] {.contractual.} =
  ## Identity matrix of size n.
  require: n > 0
  body:
    result = initMatrix[T](n, n)
    for i in 0 ..< n:
      result.data[i * n + i] = T(1)

# ------------------------------------------------------------------------------
# Element access
# ------------------------------------------------------------------------------

func `[]`*[T](m: Matrix[T], i, j: int): T {.inline, contractual.} =
  ## Element (i, j), zero-based.
  require: i >= 0 and i < m.rows and j >= 0 and j < m.cols
  body:
    m.data[i * m.cols + j]

func `[]=`*[T](m: var Matrix[T], i, j: int, v: T) {.inline, contractual.} =
  require: i >= 0 and i < m.rows and j >= 0 and j < m.cols
  body:
    m.data[i * m.cols + j] = v

func isSquare*[T](m: Matrix[T]): bool {.inline.} =
  m.rows == m.cols

# ------------------------------------------------------------------------------
# Ring operations
# ------------------------------------------------------------------------------

func `+`*[T](a, b: Matrix[T]): Matrix[T] {.contractual.} =
  ## Matrix sum.
  require: a.rows == b.rows and a.cols == b.cols
  body:
    result = initMatrix[T](a.rows, a.cols)
    for i in 0 ..< a.data.len:
      result.data[i] = a.data[i] + b.data[i]

func `-`*[T](a, b: Matrix[T]): Matrix[T] {.contractual.} =
  ## Matrix difference.
  require: a.rows == b.rows and a.cols == b.cols
  body:
    result = initMatrix[T](a.rows, a.cols)
    for i in 0 ..< a.data.len:
      result.data[i] = a.data[i] - b.data[i]

func `*`*[T](a, b: Matrix[T]): Matrix[T] {.contractual.} =
  ## Matrix product — the schoolbook triple loop, i-k-j order so the inner
  ## loop walks both operands contiguously (cache-friendly row-major).
  ##
  ## Row-base offsets (aRow/bRow/rRow) are hoisted out of the k/j loops
  ## instead of going through `[]` (which recomputes `row*cols+col` on every
  ## call) -- same algorithm, bit-identical output. See bench/README.md for
  ## measured numbers.
  require: a.cols == b.rows
  body:
    let aCols = a.cols
    let bCols = b.cols
    result = initMatrix[T](a.rows, bCols)
    for i in 0 ..< a.rows:
      let aRow = i * aCols
      let rRow = i * bCols
      for k in 0 ..< aCols:
        let aik = a.data[aRow + k]
        let bRow = k * bCols
        for j in 0 ..< bCols:
          result.data[rRow + j] = result.data[rRow + j] + aik * b.data[bRow + j]

func `*`*[T](m: Matrix[T], v: openArray[T]): seq[T] {.contractual.} =
  ## Matrix-vector product.
  require: m.cols == v.len
  body:
    result = newSeq[T](m.rows)
    let mCols = m.cols
    for i in 0 ..< m.rows:
      let iRow = i * mCols
      var acc = T(0)
      for j in 0 ..< mCols:
        acc = acc + m.data[iRow + j] * v[j]
      result[i] = acc

func `*`*[T](s: T, m: Matrix[T]): Matrix[T] =
  ## Scalar product.
  result = initMatrix[T](m.rows, m.cols)
  for i in 0 ..< m.data.len:
    result.data[i] = s * m.data[i]

func transpose*[T](m: Matrix[T]): Matrix[T] =
  result = initMatrix[T](m.cols, m.rows)
  for i in 0 ..< m.rows:
    for j in 0 ..< m.cols:
      result[j, i] = m[i, j]

# ------------------------------------------------------------------------------
# Comparison and display
# ------------------------------------------------------------------------------

func `==`*[T](a, b: Matrix[T]): bool =
  a.rows == b.rows and a.cols == b.cols and a.data == b.data

func almostEqual*[T: SomeFloat](a, b: Matrix[T], eps: T = T(1e-9)): bool =
  ## Element-wise comparison with tolerance (floats accumulate round-off).
  ## False for a negative, NaN, or infinite `eps`, or a NaN element on
  ## either side (never "everything matches").
  if eps < T(0) or eps != eps or eps == T(Inf):
    return false
  if a.rows != b.rows or a.cols != b.cols:
    return false
  for i in 0 ..< a.data.len:
    let d = a.data[i] - b.data[i]
    if d != d or abs(d) > eps: # d != d catches NaN (NaN - x is NaN)
      return false
  true

func `$`*[T](m: Matrix[T]): string =
  result = ""
  for i in 0 ..< m.rows:
    result.add(if i == 0: "[" else: " ")
    for j in 0 ..< m.cols:
      if j > 0: result.add("  ")
      result.add($m[i, j])
    result.add(if i == m.rows - 1: "]" else: "\n")





