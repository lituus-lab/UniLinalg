# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Zero-copy buffer fast path for Matrix construction/readback and solve()'s
b: every function must return the same value whether given a plain list/
list-of-lists or a contiguous float64 buffer, and anything else must fall
back to the validated path rather than misreading memory.
"""
import array
import random

import pytest

import unilinalg as ul
from unilinalg import _core


def _rows(n=8, seed=3):
    r = random.Random(seed)
    return [[r.uniform(-10, 10) for _ in range(n)] for _ in range(n)]


def test_create_from_buffer_matches_from_rows():
    rows = _rows()
    n = len(rows)
    flat_list = [v for row in rows for v in row]
    flat_array = array.array("d", flat_list)

    m_rows = ul.Matrix.from_rows(rows)
    m_list = _core._MatrixHandle.create_from_buffer(n, n, flat_list)
    m_buf = _core._MatrixHandle.create_from_buffer(n, n, flat_array)

    assert m_rows.to_rows() == [[m_list.get(i, j) for j in range(n)] for i in range(n)]
    assert m_rows.to_rows() == [[m_buf.get(i, j) for j in range(n)] for i in range(n)]


def test_create_from_buffer_shape_mismatch_returns_none():
    assert _core._MatrixHandle.create_from_buffer(3, 3, [1.0] * 8) is None
    assert _core._MatrixHandle.create_from_buffer(3, 3, array.array("d", [1.0] * 8)) is None


def test_create_from_buffer_rejects_non_numeric():
    with pytest.raises(TypeError):
        _core._MatrixHandle.create_from_buffer(2, 2, [1.0, "oops", 2.0, 3.0])


def test_get_buffer_matches_to_rows():
    rows = _rows()
    m = ul.Matrix.from_rows(rows)
    n = len(rows)
    flat = m._h.get_buffer()
    assert [flat[i * n:(i + 1) * n] for i in range(n)] == rows


def test_non_contiguous_slice_falls_back_not_misread():
    rows = _rows(n=16)
    flat = array.array("d", [v for row in rows for v in row])
    strided = memoryview(flat)[::2]
    assert not strided.contiguous
    # A non-contiguous memoryview can't take the zero-copy fast path (that
    # needs a real contiguous buffer), so this falls back to the generic-
    # iterable path -- which validates by iterating, a memoryview supports
    # that -- and builds a real 1xn matrix (n = len(strided)), matching
    # list(strided) rather than misreading the underlying flat buffer.
    n = len(strided)
    m = _core._MatrixHandle.create_from_buffer(1, n, strided)
    assert m is not None
    assert [m.get(0, j) for j in range(n)] == list(strided)


def test_solve_accepts_array_b():
    a = ul.Matrix.from_rows([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
    b_list = [8.0, 13.0, 6.0]
    b_array = array.array("d", b_list)
    x_list = a.solve(b_list)
    x_array = a.solve(b_array)
    assert x_list == x_array
    assert [round(v, 9) for v in x_list] == [1.0, 2.0, 3.0]


def test_sparse_matvec_accepts_array_v():
    dense = ul.Matrix.from_rows([[5.0, 0.0, 0.0],
                                  [0.0, 8.0, 3.0],
                                  [0.0, 6.0, 0.0]])
    sparse = dense.to_sparse()
    v_list = [1.0, 2.0, 3.0]
    v_array = array.array("d", v_list)
    result_list = sparse.matvec(v_list)
    result_array = sparse.matvec(v_array)
    assert result_list == result_array
    assert result_list == [5.0, 25.0, 12.0]


def test_solve_wrong_length_array_raises_value_error():
    a = ul.Matrix.from_rows([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        a.solve(array.array("d", [1.0, 2.0, 3.0]))


def test_solve_rejects_unsized_iterable():
    a = ul.Matrix.from_rows([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        a.solve(v for v in [1.0, 2.0])


def test_from_rows_list_path_unaffected():
    # The original list-of-lists API must still reject non-numeric elements
    # exactly as before -- the buffer fast path is purely additive.
    with pytest.raises(TypeError):
        ul.Matrix.from_rows([[1.0, "oops"], [2.0, 3.0]])
