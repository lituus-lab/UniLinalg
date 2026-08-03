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
# Contracts (NimContracts): shape preconditions on the constructors and ring
# operations are debug-only `require:` — non-blocking in release, matching the
# contract doctrine adopted by the decomposition modules (lu/cholesky/qr/svd).
# The pre-1.0 version used release-active `doAssert`s here, which violated the
# "non bloquants en release" doctrine and were inconsistent with the rest of
# UniLinalg. Structural postconditions use only the non-contracted `rows`/
# `cols` fields and `data.len`. Compiled away under -d:release/-d:danger.

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
  require: elements.len == rows * cols
  body:
    result = initMatrix[T](rows, cols)
    for i in 0 ..< elements.len:
      result.data[i] = elements[i]

func identity*[T](n: int): Matrix[T] =
  ## Identity matrix of size n.
  result = initMatrix[T](n, n)
  for i in 0 ..< n:
    result.data[i * n + i] = T(1)

# ------------------------------------------------------------------------------
# Element access
# ------------------------------------------------------------------------------

func `[]`*[T](m: Matrix[T], i, j: int): T {.inline.} =
  ## Element (i, j), zero-based.
  m.data[i * m.cols + j]

func `[]=`*[T](m: var Matrix[T], i, j: int, v: T) {.inline.} =
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
  ## call): measured ~1.6-2x on this repo's own bench/compare/vs_lapack.nim,
  ## same algorithm, bit-identical output -- see bench/README.md.
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
    for i in 0 ..< m.rows:
      var acc = T(0)
      for j in 0 ..< m.cols:
        acc = acc + m[i, j] * v[j]
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
  ## False for a negative, NaN, or infinite `eps` (never "everything matches").
  if eps < T(0) or eps != eps or eps == T(Inf):
    return false
  if a.rows != b.rows or a.cols != b.cols:
    return false
  for i in 0 ..< a.data.len:
    if abs(a.data[i] - b.data[i]) > eps:
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





