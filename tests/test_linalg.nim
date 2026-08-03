# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg test suite — known matrices, hand-checkable results
# =============================================================================

import std/[unittest, math, random, fenv]
import ../src/UniLinalg

func near(a, b: float64, eps = 1e-9): bool = abs(a - b) <= eps

suite "Matrix - construction and ring operations":
  test "constructors, access, identity":
    var m = initMatrix[float64](2, 3)
    m[1, 2] = 7.0
    check m[1, 2] == 7.0 and m[0, 0] == 0.0
    let i3 = identity[float64](3)
    check i3[0, 0] == 1.0 and i3[0, 1] == 0.0

  test "addition, subtraction, scalar product":
    let a = matrix[float64](2, 2, [1.0, 2.0, 3.0, 4.0])
    let b = matrix[float64](2, 2, [5.0, 6.0, 7.0, 8.0])
    check (a + b)[1, 1] == 12.0
    check (b - a)[0, 0] == 4.0
    check (2.0 * a)[1, 0] == 6.0

  test "matrix product against a hand computation":
    # [1 2] [5 6]   [19 22]
    # [3 4] [7 8] = [43 50]
    let a = matrix[float64](2, 2, [1.0, 2.0, 3.0, 4.0])
    let b = matrix[float64](2, 2, [5.0, 6.0, 7.0, 8.0])
    let c = a * b
    check c == matrix[float64](2, 2, [19.0, 22.0, 43.0, 50.0])

  test "matrix-vector product and transpose":
    let a = matrix[float64](2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    check a * [1.0, 1.0, 1.0] == @[6.0, 15.0]
    let t = transpose(a)
    check t.rows == 3 and t[2, 1] == 6.0

  test "almostEqual is false on a NaN element, not vacuously true":
    let a = matrix[float64](1, 2, [1.0, NaN])
    let b = matrix[float64](1, 2, [1.0, 2.0])
    check not almostEqual(a, a) # NaN vs itself: never "close enough"
    check not almostEqual(a, b)

  test "almostEqual rejects a negative, NaN, or infinite eps":
    let a = matrix[float64](1, 2, [1.0, 2.0])
    check not almostEqual(a, a, -1e-9)
    check not almostEqual(a, a, NaN)
    check not almostEqual(a, a, Inf)

suite "LU - solve, determinant, inverse":
  test "solve a 3x3 system with a known solution":
    # x + 2y + z = 8 ; 2x + y + 3z = 13 ; x + y + z = 6  -> (1, 2, 3)
    let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                   2.0, 1.0, 3.0,
                                   1.0, 1.0, 1.0])
    let x = solve(a, [8.0, 13.0, 6.0])
    check near(x[0], 1.0) and near(x[1], 2.0) and near(x[2], 3.0)

  test "refine recovers the exactly-rounded answer the plain solve misses":
    # Same system as above: the plain float64 solve lands 1-2 ULP off
    # (1.0000000000000007, 2.0, 2.9999999999999996) even though (1, 2, 3) is
    # exactly representable -- verified against Rational[BigInt]/BigFloat(256)
    # ground truth. useRefinement=true recovers it exactly here.
    let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                   2.0, 1.0, 3.0,
                                   1.0, 1.0, 1.0])
    let plain = solve(a, [8.0, 13.0, 6.0])
    let exact = [1.0, 2.0, 3.0]
    var maxErr = 0.0
    for i in 0 ..< plain.len:
      maxErr = max(maxErr, abs(plain[i] - exact[i]))
    # positive (the ULP-level noise this test guards against) but bounded to
    # a few ULP, not just "not exactly equal" -- a wildly wrong solve would
    # also satisfy a bare != check.
    check maxErr > 0.0
    check maxErr <= 8.0 * epsilon(float64)
    let x = solve(a, [8.0, 13.0, 6.0], useRefinement = true)
    check x == @[1.0, 2.0, 3.0]

  test "refineOnce on top of an existing LU factorization":
    let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                   2.0, 1.0, 3.0,
                                   1.0, 1.0, 1.0])
    let b = [8.0, 13.0, 6.0]
    let d = luDecompose(a)
    let x0 = luSolve(d, b)
    let x1 = refineOnce(a, d, b, x0)
    check x1 == @[1.0, 2.0, 3.0]

  test "residual is exact even when it falls below b's own ULP":
    let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                   2.0, 1.0, 3.0,
                                   1.0, 1.0, 1.0])
    let b = [8.0, 13.0, 6.0]
    let x0 = solve(a, b)
    let r = residual(a, b, x0)
    check r.len == b.len
    var anyNonZero = false
    for i in 0 ..< r.len:
      # bounded relative to b[i]'s own ULP -- "falls below b's own rounding
      # resolution" is the whole point of this test, not a specific bit
      # pattern that could shift by a compiler/FMA-contraction difference.
      check abs(r[i]) <= epsilon(float64) * abs(b[i])
      if r[i] != 0.0: anyNonZero = true
    check anyNonZero

  test "pivoting handles a zero leading entry":
    # without row swaps this matrix divides by zero immediately
    let a = matrix[float64](2, 2, [0.0, 1.0, 1.0, 0.0])
    let x = solve(a, [3.0, 5.0])
    check near(x[0], 5.0) and near(x[1], 3.0)

  test "determinant: known values and permutation sign":
    check near(det(matrix[float64](2, 2, [1.0, 2.0, 3.0, 4.0])), -2.0)
    check near(det(identity[float64](4)), 1.0)
    # row-swapped identity has det -1
    check near(det(matrix[float64](2, 2, [0.0, 1.0, 1.0, 0.0])), -1.0)

  test "singular matrix: det 0 and solve raises":
    let s = matrix[float64](2, 2, [1.0, 2.0, 2.0, 4.0])
    check det(s) == 0.0
    expect ValueError:
      discard solve(s, [1.0, 1.0])

  test "inverse: A * A^-1 == I":
    # det = -1 (hand-checked): the matrix is comfortably invertible
    let a = matrix[float64](3, 3, [2.0, 1.0, 1.0,
                                   1.0, 3.0, 2.0,
                                   1.0, 0.0, 0.0])
    check almostEqual(a * inverse(a), identity[float64](3), 1e-9)

  test "inverse useRefinement=true reduces max|A*A^-1 - I|":
    var r = initRand(11)
    let n = 40
    var a = initMatrix[float64](n, n)
    for i in 0 ..< n:
      for j in 0 ..< n:
        a[i, j] = r.rand(2.0) - 1.0
    let id = identity[float64](n)
    proc maxAbsDiff(m: Matrix[float64]): float64 =
      result = 0.0
      for v in (m - id).data:
        if abs(v) > result: result = abs(v)
    let errPlain = maxAbsDiff(a * inverse(a))
    let errRefined = maxAbsDiff(a * inverse(a, useRefinement = true))
    check errRefined < errPlain

suite "Cholesky - SPD factorization":
  test "known 2x2 factor":
    # [[4, 2], [2, 3]] = L L^T with L = [[2, 0], [1, sqrt(2)]]
    let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
    let l = cholesky(a)
    check near(l[0, 0], 2.0) and near(l[1, 0], 1.0)
    check near(l[1, 1], sqrt(2.0))
    check almostEqual(l * transpose(l), a, 1e-12)

  test "solve through the factor":
    let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
    let x = choleskySolve(cholesky(a), [10.0, 8.0])
    # check by substitution: A x == b
    let back = a * x
    check near(back[0], 10.0) and near(back[1], 8.0)

  test "choleskyRefineOnce recovers the exactly-rounded answer":
    # Same technique as LU's refine (ADR-0006): b=[1,1] against this A
    # lands 1 ULP off in x[1] (0.24999999999999997, not 0.25).
    let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
    let l = cholesky(a)
    let b = [1.0, 1.0]
    let x0 = choleskySolve(l, b)
    let chExact = [0.125, 0.25]
    var chMaxErr = 0.0
    for i in 0 ..< x0.len:
      chMaxErr = max(chMaxErr, abs(x0[i] - chExact[i]))
    check chMaxErr > 0.0
    check chMaxErr <= 1.0 * epsilon(float64)
    let x1 = choleskyRefineOnce(a, l, b, x0)
    check x1 == @[0.125, 0.25]

  test "non-SPD matrix is rejected":
    let notSpd = matrix[float64](2, 2, [1.0, 2.0, 2.0, 1.0]) # eigenvalue -1
    expect ValueError:
      discard cholesky(notSpd)
    check not isPositiveDefinite(notSpd)
    check isPositiveDefinite(matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0]))

  test "asymmetric matrix is rejected, not silently factored from its lower triangle":
    let asym = matrix[float64](2, 2, [4.0, 999.0, 2.0, 3.0]) # lower tri alone is SPD
    expect ValueError:
      discard cholesky(asym)

  test "choleskySolve rejects a zero diagonal instead of dividing by it":
    let notAFactor = matrix[float64](2, 2, [0.0, 0.0, 1.0, 2.0])
    expect ValueError:
      discard choleskySolve(notAFactor, [1.0, 1.0])

suite "QR - Householder":
  test "Q is orthogonal and Q*R rebuilds A":
    let a = matrix[float64](3, 3, [12.0, -51.0, 4.0,
                                   6.0, 167.0, -68.0,
                                   -4.0, 24.0, -41.0])
    let d = qrDecompose(a)
    check almostEqual(transpose(d.q) * d.q, identity[float64](3), 1e-9)
    check almostEqual(d.q * d.r, a, 1e-9)
    # R upper triangular
    check near(d.r[1, 0], 0.0) and near(d.r[2, 0], 0.0) and near(d.r[2, 1], 0.0)

  test "least squares: exact fit recovered":
    # points (0,1), (1,3), (2,5), (3,7) lie exactly on y = 2x + 1
    let a = matrix[float64](4, 2, [0.0, 1.0,
                                   1.0, 1.0,
                                   2.0, 1.0,
                                   3.0, 1.0])
    let coeffs = leastSquares(a, [1.0, 3.0, 5.0, 7.0])
    check near(coeffs[0], 2.0) and near(coeffs[1], 1.0)

  test "least squares: genuine overdetermined minimization":
    # one-column fit y ~ c*x with a pulled point
    let a = matrix[float64](3, 1, [1.0, 2.0, 3.0])
    let coeffs = leastSquares(a, [1.0, 2.0, 6.0])
    # analytic answer: sum(x*y)/sum(x^2) = (1 + 4 + 18)/14 = 23/14
    check near(coeffs[0], 23.0 / 14.0)

  test "leastSquares refine=true reduces max|residual| (Bjorck refinement)":
    # Exactly consistent overdetermined system (b in A's column space), so
    # any residual is pure float64 rounding, not model noise.
    var r = initRand(7)
    let m = 30
    let n = 6
    var a = initMatrix[float64](m, n)
    for i in 0 ..< m:
      for j in 0 ..< n:
        a[i, j] = r.rand(2.0) - 1.0
    var xtrue = newSeq[float64](n)
    for j in 0 ..< n: xtrue[j] = r.rand(10.0) - 5.0
    let b = a * xtrue
    proc maxAbs(v: seq[float64]): float64 =
      result = 0.0
      for x in v:
        if abs(x) > result: result = abs(x)
    let plain = leastSquares(a, b)
    let refined = leastSquares(a, b, refine = true)
    check maxAbs(residual(a, b, refined)) < maxAbs(residual(a, b, plain))

suite "SVD - one-sided Jacobi":
  test "diagonal matrix: singular values sorted":
    let a = matrix[float64](2, 2, [2.0, 0.0, 0.0, 5.0])
    let d = svdDecompose(a)
    check near(d.s[0], 5.0) and near(d.s[1], 2.0)

  test "reconstruction U * diag(S) * V^T == A":
    let a = matrix[float64](3, 2, [1.0, 2.0,
                                   3.0, 4.0,
                                   5.0, 6.0])
    let d = svdDecompose(a)
    var ds = initMatrix[float64](2, 2)
    ds[0, 0] = d.s[0]; ds[1, 1] = d.s[1]
    check almostEqual(d.u * ds * transpose(d.v), a, 1e-9)
    # orthonormal columns of U, orthogonal V
    check almostEqual(transpose(d.u) * d.u, identity[float64](2), 1e-9)
    check almostEqual(transpose(d.v) * d.v, identity[float64](2), 1e-9)

  test "singular values match the sqrt of A^T A eigenvalues (2x2 known)":
    # A = [[1, 1], [0, 1]]: s^2 = (3 +/- sqrt(5)) / 2
    let a = matrix[float64](2, 2, [1.0, 1.0, 0.0, 1.0])
    let d = svdDecompose(a)
    check near(d.s[0], sqrt((3.0 + sqrt(5.0)) / 2.0))
    check near(d.s[1], sqrt((3.0 - sqrt(5.0)) / 2.0))

  test "rank detects deficiency":
    check rank(matrix[float64](2, 2, [1.0, 2.0, 2.0, 4.0])) == 1
    check rank(identity[float64](3)) == 3

suite "Sparse CSR":
  test "round-trip dense <-> CSR and nnz":
    let dense = matrix[float64](3, 3, [5.0, 0.0, 0.0,
                                       0.0, 8.0, 3.0,
                                       0.0, 6.0, 0.0])
    let csr = toCsr(dense)
    check csr.nnz == 4
    check csr.rowPtr == @[0, 1, 3, 4]
    check toDense(csr) == dense
    check csr[1, 2] == 3.0 and csr[0, 1] == 0.0

  test "sparse matvec matches the dense product":
    let dense = matrix[float64](3, 3, [5.0, 0.0, 0.0,
                                       0.0, 8.0, 3.0,
                                       0.0, 6.0, 0.0])
    let v = [1.0, 2.0, 3.0]
    check toCsr(dense) * v == dense * v

  test "CSR transpose":
    let dense = matrix[float64](2, 3, [1.0, 0.0, 2.0,
                                       0.0, 3.0, 0.0])
    let t = transpose(toCsr(dense))
    check toDense(t) == transpose(dense)

  test "initCsrMatrix accepts a well-formed CSR and rejects a malformed one":
    let m = initCsrMatrix[float64](2, 2, @[5.0, 6.0], @[0, 1], @[0, 1, 2])
    check toDense(m) == matrix[float64](2, 2, [5.0, 0.0, 0.0, 6.0])
    # rowPtr not non-decreasing, but shape/length preconditions still hold
    # (3 rows, 3 vals, rowPtr[0] == 0, rowPtr[^1] == vals.len == 3) -- only
    # the body-level ordering check should fire, not require:.
    expect ValueError:
      discard initCsrMatrix[float64](3, 2, @[5.0, 6.0, 7.0], @[0, 1, 0],
                                      @[0, 2, 1, 3])
    expect ValueError: # colIdx out of [0, cols)
      discard initCsrMatrix[float64](2, 2, @[5.0], @[2], @[0, 1, 1])
