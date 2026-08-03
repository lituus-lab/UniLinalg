# UniLinalg: Contract infrastructure tests (Phase 7)
# ===================================================
#
# Two purposes (mirroring the canonical UniRational suite):
# 1. Prove the NimContracts machinery is wired into this package's build:
#    a deliberately-broken postcondition raises `PostConditionDefect` in debug,
#    and a correct one passes. This is the "loop d'auto-contrôle" the plan
#    requires — the contract layer is alive and would catch a regression.
# 2. Smoke-check the structural postconditions the contracts assert on real
#    inputs: LU shape/sign, solve length, and Cholesky lower-triangularity.
#
# Gated to debug only (`when not defined(release) and not defined(danger)`):
# in release/danger the contracts are compiled away (zero overhead, no
# raises), so a violation test has nothing to observe there.

import std/unittest
import contracts
import ../src/UniLinalg

when not defined(release) and not defined(danger):

  suite "contract machinery is active in debug":
    proc deliberatelyBroken(): int {.contractual.} =
      ensure:
        result == 1
      body:
        0

    proc honest(): int {.contractual.} =
      ensure:
        result == 1
      body:
        1

    test "broken postcondition raises PostConditionDefect":
      var caught = false
      try:
        discard deliberatelyBroken()
      except PostConditionDefect:
        caught = true
      check caught

    test "honoured postcondition returns normally":
      check honest() == 1

  suite "LU / Cholesky structural invariants hold":
    test "luDecompose: shape, permutation length, sign parity":
      let a = matrix[float64](2, 2, [4.0, 3.0, 6.0, 3.0])
      let d = luDecompose(a)
      check d.lu.rows == 2 and d.lu.cols == 2
      check d.perm.len == 2
      check d.sign == 1 or d.sign == -1

    test "solve: result length matches matrix size":
      let a = matrix[float64](2, 2, [2.0, 0.0, 0.0, 3.0])
      let x = solve(a, [1.0, 1.0])
      check x.len == 2

    test "cholesky: square, same shape, lower-triangular":
      # Symmetric positive-definite: A = [[4,2],[2,3]].
      let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
      let l = cholesky(a)
      check l.isSquare
      check l.rows == 2
      check l.isLowerTriangular
      # Strict upper triangle must be zero.
      check l[0, 1] == 0.0

    test "choleskySolve: result length matches":
      let a = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
      let l = cholesky(a)
      let x = choleskySolve(l, [1.0, 1.0])
      check x.len == 2

  suite "QR / SVD structural invariants and shape preconditions":
    test "qrDecompose: shape precondition rejects rows < cols":
      # 2x3 (rows < cols) violates the debug-only `require:`.
      let a = matrix[float64](2, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      expect PreConditionDefect:
        discard qrDecompose(a)

    test "qrDecompose: Q is m×m, R is m×n on a valid 3x2 input":
      let a = matrix[float64](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
      let d = qrDecompose(a)
      check d.q.rows == 3 and d.q.cols == 3
      check d.r.rows == 3 and d.r.cols == 2

    test "leastSquares: shape precondition rejects rhs length mismatch":
      let a = matrix[float64](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
      expect PreConditionDefect:
        discard leastSquares(a, [1.0, 1.0]) # len 2 != rows 3

    test "leastSquares: result length == cols":
      let a = matrix[float64](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
      let x = leastSquares(a, [1.0, 1.0, 1.0])
      check x.len == 2

    test "leastSquares: rank-deficient raises ValueError (body guard)":
      # Zero column → R has a zero diagonal → rank-deficient. The guard is a
      # body `raise` (not a `require:`), so it survives release/danger too;
      # verified here in debug as part of the contract suite.
      let a = matrix[float64](3, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      expect ValueError:
        discard leastSquares(a, [1.0, 1.0, 1.0])

    test "svdDecompose: shape precondition rejects rows < cols":
      let a = matrix[float64](2, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      expect PreConditionDefect:
        discard svdDecompose(a)

    test "svdDecompose: U m×n, S len n, V n×n on a valid 3x2 input":
      let a = matrix[float64](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
      let d = svdDecompose(a)
      check d.u.rows == 3 and d.u.cols == 2
      check d.s.len == 2
      check d.v.rows == 2 and d.v.cols == 2

    test "rank: result bounded in [0, cols]":
      let a = matrix[float64](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
      let r = rank(a)
      check r >= 0 and r <= 2

else:
  suite "contracts compiled away in release/danger":
    test "no contract machinery present":
      check true
