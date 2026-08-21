# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""unilinalg — Python binding over the UniLinalg C library."""
import math

from . import _core

__version__ = _core.version().decode("ascii")


def version():
    """C library version string."""
    return __version__


def _checked(handle, what):
    """Raise ValueError instead of silently returning a native None -- every
    caller below already rules out the shape/type reasons the C ABI reports
    before making this call, so it should never actually fire, but a silent
    None would otherwise surface later as an unrelated AttributeError
    instead of a clear error at the real failure site."""
    if handle is None:
        raise ValueError(f"{what} failed")
    return handle


class Matrix:
    """Dense row-major matrix (float64). Domain-checked wrapper over
    unilinalg._core's raw _MatrixHandle."""

    def __init__(self, rows, cols):
        if (isinstance(rows, bool) or isinstance(cols, bool)
                or not isinstance(rows, int) or not isinstance(cols, int)):
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
        return Matrix._from_handle(_checked(self._h.add(other._h), "add"))

    def __sub__(self, other):
        self._check_same_shape(other)
        return Matrix._from_handle(_checked(self._h.sub(other._h), "sub"))

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"expected Matrix, got {type(other).__name__}")
        if self.cols != other.rows:
            raise ValueError(
                f"shape mismatch: ({self.rows}, {self.cols}) @ "
                f"({other.rows}, {other.cols})")
        return Matrix._from_handle(_checked(self._h.matmul(other._h), "matmul"))

    def __mul__(self, scalar):
        return Matrix._from_handle(_checked(self._h.scale(float(scalar)), "scale"))

    __rmul__ = __mul__

    def transpose(self):
        return Matrix._from_handle(_checked(self._h.transpose(), "transpose"))

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
        the 1-2 ULP a plain float64 solve can miss."""
        try:
            blen = len(b)
        except TypeError:
            raise TypeError(f"b must be a sized iterable, got {type(b).__name__}") from None
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

    def symmetric_eigen(self, max_sweeps=64, tolerance=1e-12):
        """Eigenvalues and column eigenvectors of a finite symmetric matrix.

        Eigenvalues are returned in descending order. ``tolerance`` is
        relative to the largest diagonal magnitude and must lie in (0, 1].
        Raises ``ValueError`` for a non-square, asymmetric, non-finite, or
        non-convergent input.
        """
        if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int):
            raise TypeError("max_sweeps must be int")
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a real number")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0:
            raise ValueError("tolerance must be finite and in (0, 1]")
        if self.rows != self.cols:
            raise ValueError("symmetric_eigen requires a square matrix")
        result = self._h.symmetric_eigen(max_sweeps, tolerance)
        if result is None:
            raise ValueError("symmetric_eigen requires a finite symmetric "
                             "matrix and a convergent iteration")
        values, vectors = result
        return (values, Matrix._from_handle(vectors))

    def to_rows(self):
        """One bulk C call, not rows*cols individual `get` calls -- see
        `_MatrixHandle.get_buffer`."""
        flat = self._h.get_buffer()
        cols = self.cols
        return [flat[i * cols:(i + 1) * cols] for i in range(self.rows)]

    def almost_equal(self, other, eps):
        """Element-wise comparison with tolerance. False on a shape mismatch
        or a non-Matrix other (never raises)."""
        if not isinstance(other, Matrix):
            return False
        return self._h.almost_equal(other._h, float(eps))

    def to_sparse(self):
        """Compress to CSR (exact zeros dropped)."""
        return Sparse._from_handle(_core._SparseHandle.from_dense(self._h))

    def __repr__(self):
        return f"Matrix({self.to_rows()!r})"


class Sparse:
    """CSR (Compressed Sparse Row) matrix (float64). Domain-checked wrapper
    over unilinalg._core's raw _SparseHandle. Build one via Matrix.to_sparse();
    there is no direct constructor (CSR is a compression of a dense matrix,
    not built from scratch)."""

    @classmethod
    def _from_handle(cls, handle):
        if handle is None:
            return None
        obj = cls.__new__(cls)
        obj._h = handle
        return obj

    @property
    def nnz(self):
        """Number of stored (non-zero) entries."""
        return self._h.nnz

    def to_dense(self):
        return Matrix._from_handle(_checked(self._h.to_dense(), "to_dense"))

    def matvec(self, v):
        """Sparse matrix-vector product. Raises ValueError on a shape
        mismatch."""
        try:
            len(v)
        except TypeError:
            raise TypeError(f"v must be a sized iterable, got {type(v).__name__}") from None
        result = self._h.matvec(v)
        if result is None:
            raise ValueError("matvec: shape mismatch")
        return result

    def __repr__(self):
        return f"Sparse(nnz={self.nnz})"


def _check_vec_components(*values):
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
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

    def perp(self):
        """90 degrees counterclockwise rotation: (-y, x)."""
        return Vec2(*_core.vec2_perp(self.x, self.y))

    def perp_cw(self):
        """90 degrees clockwise rotation: (y, -x)."""
        return Vec2(*_core.vec2_perp_cw(self.x, self.y))

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


class Vec4:
    """4D vector (float64)."""

    def __init__(self, x, y, z, w):
        _check_vec_components(x, y, z, w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def _check_other(self, other):
        if not isinstance(other, Vec4):
            raise TypeError(f"expected Vec4, got {type(other).__name__}")

    def __add__(self, other):
        self._check_other(other)
        return Vec4(*_core.vec4_add(self.x, self.y, self.z, self.w,
                                     other.x, other.y, other.z, other.w))

    def __sub__(self, other):
        self._check_other(other)
        return Vec4(*_core.vec4_sub(self.x, self.y, self.z, self.w,
                                     other.x, other.y, other.z, other.w))

    def __mul__(self, s):
        return Vec4(*_core.vec4_scale(self.x, self.y, self.z, self.w, float(s)))

    __rmul__ = __mul__

    def dot(self, other):
        self._check_other(other)
        return _core.vec4_dot(self.x, self.y, self.z, self.w,
                               other.x, other.y, other.z, other.w)

    @property
    def length(self):
        return _core.vec4_length(self.x, self.y, self.z, self.w)

    def normalize(self):
        return Vec4(*_core.vec4_normalize(self.x, self.y, self.z, self.w))

    def __eq__(self, other):
        return (isinstance(other, Vec4) and self.x == other.x
                and self.y == other.y and self.z == other.z and self.w == other.w)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.w))

    def __repr__(self):
        return f"Vec4({self.x!r}, {self.y!r}, {self.z!r}, {self.w!r})"


__all__ = ["Matrix", "Sparse", "Vec2", "Vec3", "Vec4", "__version__", "version"]
