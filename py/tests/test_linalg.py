# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import math

import pytest
import unilinalg


def test_version():
    assert unilinalg.version() == "0.1.0"
    assert unilinalg.__version__ == "0.1.0"


def test_matrix_construction_and_indexing():
    m = unilinalg.Matrix(2, 3)
    m[1, 2] = 7.0
    assert m[1, 2] == 7.0
    assert m[0, 0] == 0.0
    assert m.rows == 2 and m.cols == 3


def test_matrix_from_rows():
    m = unilinalg.Matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
    assert m.to_rows() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_arithmetic():
    a = unilinalg.Matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
    b = unilinalg.Matrix.from_rows([[5.0, 6.0], [7.0, 8.0]])
    assert (a + b).to_rows() == [[6.0, 8.0], [10.0, 12.0]]
    assert (b - a).to_rows() == [[4.0, 4.0], [4.0, 4.0]]
    assert (a * 2.0).to_rows() == [[2.0, 4.0], [6.0, 8.0]]
    assert (2.0 * a).to_rows() == [[2.0, 4.0], [6.0, 8.0]]


def test_matrix_matmul():
    a = unilinalg.Matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
    b = unilinalg.Matrix.from_rows([[5.0, 6.0], [7.0, 8.0]])
    assert (a @ b).to_rows() == [[19.0, 22.0], [43.0, 50.0]]


def test_matrix_transpose():
    a = unilinalg.Matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = a.transpose()
    assert t.rows == 3 and t.cols == 2
    assert t[2, 1] == 6.0


def test_solve_known_3x3_system():
    # x + 2y + z = 8; 2x + y + 3z = 13; x + y + z = 6 -> (1, 2, 3).
    a = unilinalg.Matrix.from_rows([[1.0, 2.0, 1.0],
                                    [2.0, 1.0, 3.0],
                                    [1.0, 1.0, 1.0]])
    x = a.solve([8.0, 13.0, 6.0])
    assert math.isclose(x[0], 1.0, abs_tol=1e-9)
    assert math.isclose(x[1], 2.0, abs_tol=1e-9)
    assert math.isclose(x[2], 3.0, abs_tol=1e-9)


def test_solve_singular_raises():
    s = unilinalg.Matrix.from_rows([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(ValueError):
        s.solve([1.0, 1.0])


def test_determinant():
    a = unilinalg.Matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
    assert math.isclose(a.determinant(), -2.0, abs_tol=1e-9)


def test_determinant_non_square_raises():
    a = unilinalg.Matrix(2, 3)
    with pytest.raises(ValueError):
        a.determinant()


def test_cholesky_known_factor():
    a = unilinalg.Matrix.from_rows([[4.0, 2.0], [2.0, 3.0]])
    l = a.cholesky()
    assert math.isclose(l[0, 0], 2.0, abs_tol=1e-9)
    assert math.isclose(l[1, 0], 1.0, abs_tol=1e-9)
    assert math.isclose(l[1, 1], math.sqrt(2.0), abs_tol=1e-9)


def test_cholesky_non_spd_raises():
    not_spd = unilinalg.Matrix.from_rows([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        not_spd.cholesky()


def test_qr_shape():
    a = unilinalg.Matrix.from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    q, r = a.qr()
    assert q.rows == 3 and q.cols == 3
    assert r.rows == 3 and r.cols == 2
    # Reconstruction: Q @ R == A.
    qr = q @ r
    for i in range(a.rows):
        for j in range(a.cols):
            assert math.isclose(qr[i, j], a.to_rows()[i][j], abs_tol=1e-9)
    # Orthogonality: Q^T @ Q == I.
    qtq = q.transpose() @ q
    for i in range(q.cols):
        for j in range(q.cols):
            expected = 1.0 if i == j else 0.0
            assert math.isclose(qtq[i, j], expected, abs_tol=1e-9)


def test_qr_rank_deficient_shape_raises():
    a = unilinalg.Matrix.from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        a.qr()


def test_svd_diagonal_sorted():
    a = unilinalg.Matrix.from_rows([[2.0, 0.0], [0.0, 5.0]])
    u, s, v = a.svd()
    assert math.isclose(s[0], 5.0, abs_tol=1e-9)
    assert math.isclose(s[1], 2.0, abs_tol=1e-9)


def test_matrix_shape_mismatch_raises():
    a = unilinalg.Matrix(2, 2)
    b = unilinalg.Matrix(3, 3)
    with pytest.raises(ValueError):
        a + b
    with pytest.raises(TypeError):
        a + "not a matrix"


def test_vec3_cross():
    x = unilinalg.Vec3(1.0, 0.0, 0.0)
    y = unilinalg.Vec3(0.0, 1.0, 0.0)
    assert x.cross(y) == unilinalg.Vec3(0.0, 0.0, 1.0)


def test_vec2_length_and_normalize():
    v = unilinalg.Vec2(3.0, 4.0)
    assert v.length == 5.0
    n = v.normalize()
    assert math.isclose(n.length, 1.0, abs_tol=1e-9)


def test_vec_arithmetic():
    a = unilinalg.Vec2(1.0, 2.0)
    b = unilinalg.Vec2(3.0, 4.0)
    assert a + b == unilinalg.Vec2(4.0, 6.0)
    assert b - a == unilinalg.Vec2(2.0, 2.0)
    assert a * 2.0 == unilinalg.Vec2(2.0, 4.0)
    assert a.dot(b) == 11.0


def test_vec_type_error_on_bad_component():
    with pytest.raises(TypeError):
        unilinalg.Vec2("x", 1.0)


def test_vec_type_error_on_bad_other():
    v = unilinalg.Vec2(1.0, 2.0)
    with pytest.raises(TypeError):
        v + "not a vector"
