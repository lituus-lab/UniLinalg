# UniLinalg — Sparse matrices (CSR: Compressed Sparse Row)
# =============================================================================
#
# When a 10000 x 10000 matrix has three non-zeros per row, storing the
# 99.97% of zeros is absurd. CSR stores only the non-zeros, row by row:
#
#   vals    the non-zero values, row-major
#   colIdx  the column of each value
#   rowPtr  where each row starts in vals (n+1 entries; rowPtr[^1] = nnz)
#
# Example:  [5 0 0]          vals   = [5, 8, 3, 6]
#           [0 8 3]   =>     colIdx = [0, 1, 2, 1]
#           [0 6 0]          rowPtr = [0, 1, 3, 4]
#
# The matrix-vector product touches each non-zero exactly once: O(nnz).

import ./matrix
import contracts

type
  CsrMatrix*[T] = object
    rows*, cols*: int
    vals*: seq[T]
    colIdx*: seq[int]
    rowPtr*: seq[int]

func nnz*[T](m: CsrMatrix[T]): int {.inline.} =
  ## Number of stored (non-zero) entries.
  m.vals.len

func toCsr*[T](dense: Matrix[T]): CsrMatrix[T] =
  ## Compress a dense matrix (exact zeros are dropped).
  result.rows = dense.rows
  result.cols = dense.cols
  result.rowPtr = newSeq[int](dense.rows + 1)
  for i in 0 ..< dense.rows:
    result.rowPtr[i] = result.vals.len
    for j in 0 ..< dense.cols:
      let v = dense[i, j]
      if v != T(0):
        result.vals.add(v)
        result.colIdx.add(j)
  result.rowPtr[dense.rows] = result.vals.len

func toDense*[T](m: CsrMatrix[T]): Matrix[T] =
  ## Expand back to a dense matrix.
  result = initMatrix[T](m.rows, m.cols)
  for i in 0 ..< m.rows:
    for k in m.rowPtr[i] ..< m.rowPtr[i + 1]:
      result[i, m.colIdx[k]] = m.vals[k]

func `[]`*[T](m: CsrMatrix[T], i, j: int): T =
  ## Element access by row scan (rows are short in practice).
  for k in m.rowPtr[i] ..< m.rowPtr[i + 1]:
    if m.colIdx[k] == j:
      return m.vals[k]
  T(0)

func `*`*[T](m: CsrMatrix[T], v: openArray[T]): seq[T] {.contractual.} =
  ## Sparse matrix-vector product — THE operation CSR is built for: O(nnz).
  ##
  ## Precondition: `v`'s length matches `m`'s column count (debug-only
  ## `require:`, matching the dense-Matrix doctrine — non-blocking in release).
  require: v.len == m.cols
  body:
    result = newSeq[T](m.rows)
    for i in 0 ..< m.rows:
      var acc = T(0)
      for k in m.rowPtr[i] ..< m.rowPtr[i + 1]:
        acc = acc + m.vals[k] * v[m.colIdx[k]]
      result[i] = acc

func transpose*[T](m: CsrMatrix[T]): CsrMatrix[T] =
  ## CSR transpose through a counting pass (the classic two-pass trick):
  ## count entries per column, prefix-sum into the new rowPtr, then place.
  result.rows = m.cols
  result.cols = m.rows
  result.vals = newSeq[T](m.nnz)
  result.colIdx = newSeq[int](m.nnz)
  result.rowPtr = newSeq[int](m.cols + 1)
  # count entries that will land in each transposed row
  for k in 0 ..< m.nnz:
    inc result.rowPtr[m.colIdx[k] + 1]
  for i in 1 .. m.cols:
    result.rowPtr[i] = result.rowPtr[i] + result.rowPtr[i - 1]
  # place values (cursor per transposed row)
  var cursor = result.rowPtr # copy
  for i in 0 ..< m.rows:
    for k in m.rowPtr[i] ..< m.rowPtr[i + 1]:
      let j = m.colIdx[k]
      result.vals[cursor[j]] = m.vals[k]
      result.colIdx[cursor[j]] = i
      inc cursor[j]





