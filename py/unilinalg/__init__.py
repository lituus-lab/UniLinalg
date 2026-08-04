# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""unilinalg — Python binding over the UniLinalg C library."""
from . import _core

__version__ = _core.version().decode("ascii")


def version():
    """C library version string."""
    return _core.version().decode("ascii")


class Matrix:
    """Dense row-major matrix (float64). Domain-checked wrapper over
    unilinalg._core's raw _MatrixHandle."""

    def __init__(self, rows, cols):
        if not isinstance(rows, int) or not isinstance(cols, int):
            raise TypeError("rows and cols must be int")
        if rows <= 0 or cols <= 0:
            raise ValueError(f"rows and cols must be > 0, got ({rows}, {cols})")
        self._h = _core._MatrixHandle.create(rows, cols)
        if self._h is None:
            raise ValueError(f"failed to create matrix of shape ({rows}, {cols})")

    @classmethod
    def _from_handle(cls, handle):
        if handle is None:
            return None
        obj = cls.__new__(cls)
        obj._h = handle
        return obj

    @classmethod
    def from_rows(cls, rows):
        """Build from a list of row lists, e.g. [[1.0, 2.0], [3.0, 4.0]].
        One bulk C call, not rows*cols individual `set` calls -- see
        `_MatrixHandle.create_from_buffer`."""
        if not rows or not all(isinstance(r, (list, tuple)) for r in rows):
            raise TypeError("rows must be a non-empty list of row lists")
        ncols = len(rows[0])
        if any(len(r) != ncols for r in rows):
            raise ValueError("all rows must have the same length")
        flat = [v for row in rows for v in row]
        handle = _core._MatrixHandle.create_from_buffer(len(rows), ncols, flat)
        if handle is None:
            raise ValueError(
                f"failed to create matrix of shape ({len(rows)}, {ncols})")
        return cls._from_handle(handle)

    @property
    def rows(self):
        return self._h.rows

    @property
    def cols(self):
        return self._h.cols

    def _check_index(self, i, j):
        if not (0 <= i < self.rows) or not (0 <= j < self.cols):
            raise IndexError(f"({i}, {j}) out of range for shape "
                              f"({self.rows}, {self.cols})")

    def __getitem__(self, key):
        i, j = key
        self._check_index(i, j)
        return self._h.get(i, j)

    def __setitem__(self, key, value):
        i, j = key
        self._check_index(i, j)
        self._h.set(i, j, float(value))

    def _check_same_shape(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"expected Matrix, got {type(other).__name__}")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"shape mismatch: ({self.rows}, {self.cols}) vs "
                f"({other.rows}, {other.cols})")

    def __add__(self, other):
        self._check_same_shape(other)
        return Matrix._from_handle(self._h.add(other._h))

    def __sub__(self, other):
        self._check_same_shape(other)
        return Matrix._from_handle(self._h.sub(other._h))

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"expected Matrix, got {type(other).__name__}")
        if self.cols != other.rows:
            raise ValueError(
                f"shape mismatch: ({self.rows}, {self.cols}) @ "
                f"({other.rows}, {other.cols})")
        return Matrix._from_handle(self._h.matmul(other._h))

    def __mul__(self, scalar):
        return Matrix._from_handle(self._h.scale(float(scalar)))

    __rmul__ = __mul__

    def transpose(self):
        return Matrix._from_handle(self._h.transpose())

    def determinant(self):
        """Raises ValueError if the matrix is not square."""
        value, ok = self._h.determinant()
        if not ok:
            raise ValueError("determinant requires a square matrix")
        return value

    def solve(self, b, refine=False):
        """Solves Ax = b. Raises ValueError on a non-square matrix, a shape
        mismatch, or a singular matrix. refine=True runs one step of
        UniAccurate-backed iterative refinement after the solve, correcting
        the 1-2 ULP a plain float64 solve can miss (ADR-0006)."""
        try:
            blen = len(b)
        except TypeError:
            raise TypeError(f"b must be a sized iterable, got {type(b).__name__}")
        if blen != self.rows:
            raise ValueError(f"b has length {blen}, expected {self.rows}")
        result = self._h.lu_solve(b, bool(refine))
        if result is None:
            raise ValueError("solve failed: non-square, shape mismatch, "
                              "or singular matrix")
        return result

    def cholesky(self):
        """Raises ValueError if the matrix is not symmetric positive-definite."""
        h = self._h.cholesky()
        if h is None:
            raise ValueError("cholesky requires a symmetric positive-definite "
                              "matrix")
        return Matrix._from_handle(h)

    def qr(self):
        """Householder QR. Raises ValueError if rows < cols."""
        result = self._h.qr()
        if result is None:
            raise ValueError("qr requires rows >= cols")
        q, r = result
        return (Matrix._from_handle(q), Matrix._from_handle(r))

    def svd(self):
        """One-sided Jacobi SVD. Raises ValueError if rows < cols."""
        result = self._h.svd()
        if result is None:
            raise ValueError("svd requires rows >= cols")
        u, s, v = result
        return (Matrix._from_handle(u), s, Matrix._from_handle(v))

    def to_rows(self):
        """One bulk C call, not rows*cols individual `get` calls -- see
        `_MatrixHandle.get_buffer`."""
        flat = self._h.get_buffer()
        cols = self.cols
        return [flat[i * cols:(i + 1) * cols] for i in range(self.rows)]

    def __repr__(self):
        return f"Matrix({self.to_rows()!r})"


def _check_vec_components(*values):
    for v in values:
        if not isinstance(v, (int, float)):
            raise TypeError(f"vector components must be int or float, "
                             f"got {type(v).__name__}")


class Vec2:
    """2D vector (float64). Domain-checked wrapper over unilinalg._core's
    raw vec2_* free functions."""

    def __init__(self, x, y):
        _check_vec_components(x, y)
        self.x = float(x)
        self.y = float(y)

    def _check_other(self, other):
        if not isinstance(other, Vec2):
            raise TypeError(f"expected Vec2, got {type(other).__name__}")

    def __add__(self, other):
        self._check_other(other)
        return Vec2(*_core.vec2_add(self.x, self.y, other.x, other.y))

    def __sub__(self, other):
        self._check_other(other)
        return Vec2(*_core.vec2_sub(self.x, self.y, other.x, other.y))

    def __mul__(self, s):
        return Vec2(*_core.vec2_scale(self.x, self.y, float(s)))

    __rmul__ = __mul__

    def dot(self, other):
        self._check_other(other)
        return _core.vec2_dot(self.x, self.y, other.x, other.y)

    @property
    def length(self):
        return _core.vec2_length(self.x, self.y)

    def normalize(self):
        return Vec2(*_core.vec2_normalize(self.x, self.y))

    def cross2d(self, other):
        self._check_other(other)
        return _core.vec2_cross2d(self.x, self.y, other.x, other.y)

    def __eq__(self, other):
        return isinstance(other, Vec2) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Vec2({self.x!r}, {self.y!r})"


class Vec3:
    """3D vector (float64)."""

    def __init__(self, x, y, z):
        _check_vec_components(x, y, z)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def _check_other(self, other):
        if not isinstance(other, Vec3):
            raise TypeError(f"expected Vec3, got {type(other).__name__}")

    def __add__(self, other):
        self._check_other(other)
        return Vec3(*_core.vec3_add(self.x, self.y, self.z, other.x, other.y, other.z))

    def __sub__(self, other):
        self._check_other(other)
        return Vec3(*_core.vec3_sub(self.x, self.y, self.z, other.x, other.y, other.z))

    def __mul__(self, s):
        return Vec3(*_core.vec3_scale(self.x, self.y, self.z, float(s)))

    __rmul__ = __mul__

    def dot(self, other):
        self._check_other(other)
        return _core.vec3_dot(self.x, self.y, self.z, other.x, other.y, other.z)

    @property
    def length(self):
        return _core.vec3_length(self.x, self.y, self.z)

    def normalize(self):
        return Vec3(*_core.vec3_normalize(self.x, self.y, self.z))

    def cross(self, other):
        self._check_other(other)
        return Vec3(*_core.vec3_cross(self.x, self.y, self.z, other.x, other.y, other.z))

    def __eq__(self, other):
        return (isinstance(other, Vec3) and self.x == other.x
                and self.y == other.y and self.z == other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"Vec3({self.x!r}, {self.y!r}, {self.z!r})"


__all__ = ["Matrix", "Vec2", "Vec3", "version", "__version__"]
