# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — symmetric eigendecomposition by Jacobi rotations

import contracts
import UniMath
import ../types/matrix

type
  SymmetricEigenDecomposition*[T] = object
    values*: seq[T]
    vectors*: Matrix[T]

func finiteValue[T: SomeFloat](value: T): bool {.inline.} =
  value == value and abs(value) != T(Inf)

func symmetricEigenDecompose*[T: SomeFloat](a: Matrix[T],
    maxSweeps = 64, tolerance = T(1e-12)): SymmetricEigenDecomposition[T]
    {.contractual.} =
  ## Eigenvalues and column eigenvectors of a finite real symmetric matrix.
  ## Values are sorted from largest to smallest. `tolerance` is relative to
  ## the largest diagonal magnitude and also controls the symmetry check.
  require:
    a.rows > 0 and a.cols > 0
    maxSweeps > 0
    tolerance > T(0) and finiteValue(tolerance)
  ensure:
    result.values.len == a.rows
    result.vectors.rows == a.rows and result.vectors.cols == a.cols
  body:
    if a.rows <= 0 or a.cols <= 0 or a.rows > high(int) div a.cols or
        a.data.len != a.rows * a.cols:
      raise newException(ValueError,
        "symmetricEigenDecompose: invalid packed matrix storage")
    if not a.isSquare:
      raise newException(ValueError,
        "symmetricEigenDecompose: matrix must be square")
    if maxSweeps <= 0 or not finiteValue(tolerance) or tolerance <= T(0) or
        tolerance > T(1):
      raise newException(ValueError,
        "symmetricEigenDecompose: invalid convergence parameters")

    let n = a.rows
    var work = a
    for i in 0 ..< n:
      let iRow = i * n
      for j in 0 ..< n:
        if not finiteValue(work.data[iRow + j]):
          raise newException(ValueError,
            "symmetricEigenDecompose: matrix must be finite")
      for j in 0 ..< i:
        let ji = j * n + i
        let ij = iRow + j
        let scale = max(abs(work.data[ij]), abs(work.data[ji]))
        if scale > T(0) and
            abs(work.data[ij] - work.data[ji]) > tolerance * scale:
          raise newException(ValueError,
            "symmetricEigenDecompose: matrix must be symmetric")
        let mean = (work.data[ij] / T(2)) + (work.data[ji] / T(2))
        work.data[ij] = mean
        work.data[ji] = mean

    result.vectors = identity[T](n)
    var converged = n == 1
    for sweep in 0 ..< maxSweeps:
      var diagonalScale = T(0)
      var largestOffDiagonal = T(0)
      for i in 0 ..< n:
        let iRow = i * n
        diagonalScale = max(diagonalScale, abs(work.data[iRow + i]))
        for j in i + 1 ..< n:
          largestOffDiagonal = max(largestOffDiagonal, abs(work.data[iRow + j]))
      let matrixScale = max(diagonalScale, largestOffDiagonal)
      if matrixScale == T(0) or
          largestOffDiagonal <= tolerance * matrixScale:
        converged = true
        break

      for p in 0 ..< n - 1:
        let pRow = p * n
        for q in p + 1 ..< n:
          let qRow = q * n
          let apq = work.data[pRow + q]
          if abs(apq) <= tolerance * matrixScale:
            continue
          let tau = ((work.data[qRow + q] / T(2)) -
            (work.data[pRow + p] / T(2))) / apq
          let t =
            if tau >= T(0):
              T(1) / (tau + hypot(T(1), tau))
            else:
              T(-1) / (-tau + hypot(T(1), tau))
          let c = T(1) / hypot(T(1), t)
          let s = t * c

          for k in 0 ..< n:
            if k != p and k != q:
              let kRow = k * n
              let akp = work.data[kRow + p]
              let akq = work.data[kRow + q]
              let nextP = c * akp - s * akq
              let nextQ = s * akp + c * akq
              work.data[kRow + p] = nextP
              work.data[pRow + k] = nextP
              work.data[kRow + q] = nextQ
              work.data[qRow + k] = nextQ
          let app = work.data[pRow + p]
          let aqq = work.data[qRow + q]
          work.data[pRow + p] = app - t * apq
          work.data[qRow + q] = aqq + t * apq
          work.data[pRow + q] = T(0)
          work.data[qRow + p] = T(0)

          for k in 0 ..< n:
            let kRow = k * n
            let vkp = result.vectors.data[kRow + p]
            let vkq = result.vectors.data[kRow + q]
            result.vectors.data[kRow + p] = c * vkp - s * vkq
            result.vectors.data[kRow + q] = s * vkp + c * vkq

      if sweep == maxSweeps - 1:
        var residual = T(0)
        var scale = T(0)
        for i in 0 ..< n:
          let iRow = i * n
          scale = max(scale, abs(work.data[iRow + i]))
          for j in i + 1 ..< n:
            residual = max(residual, abs(work.data[iRow + j]))
        let matrixScale = max(scale, residual)
        converged = matrixScale == T(0) or
          residual <= tolerance * matrixScale

    if not converged:
      raise newException(ValueError,
        "symmetricEigenDecompose: Jacobi iteration did not converge")

    result.values = newSeq[T](n)
    for i in 0 ..< n:
      result.values[i] = work.data[i * n + i]
    for i in 0 ..< n - 1:
      var largest = i
      for j in i + 1 ..< n:
        if result.values[j] > result.values[largest]:
          largest = j
      if largest != i:
        swap(result.values[i], result.values[largest])
        for row in 0 ..< n:
          let rowBase = row * n
          let current = result.vectors.data[rowBase + i]
          result.vectors.data[rowBase + i] = result.vectors.data[rowBase + largest]
          result.vectors.data[rowBase + largest] = current

