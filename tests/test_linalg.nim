# UniLinalg test suite — known matrices, hand-checkable results
# =============================================================================

import std/[unittest, math]
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
    # ground truth. refine=true recovers it exactly here.
    let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                   2.0, 1.0, 3.0,
                                   1.0, 1.0, 1.0])
    let plain = solve(a, [8.0, 13.0, 6.0])
    check plain != @[1.0, 2.0, 3.0] # the ULP-level noise this test guards against
    let x = solve(a, [8.0, 13.0, 6.0], refine = true)
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
    check r == @[-2.220446049250313e-16, 0.0, -2.220446049250313e-16]

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

  test "non-SPD matrix is rejected":
    let notSpd = matrix[float64](2, 2, [1.0, 2.0, 2.0, 1.0]) # eigenvalue -1
    expect ValueError:
      discard cholesky(notSpd)
    check not isPositiveDefinite(notSpd)
    check isPositiveDefinite(matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0]))

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
