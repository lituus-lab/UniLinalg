# cython: language_level=3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Raw Cython bindings over the UniLinalg C ABI (no domain checks).
Use the `unilinalg` package (this module's __init__.py) instead."""
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from cpython cimport array

cdef extern from "UniLinalg.h":
    ctypedef void *ulin_matrix
    ctypedef void *ulin_sparse

    ctypedef struct ulin_vec2:
        double x
        double y

    ctypedef struct ulin_vec3:
        double x
        double y
        double z

    int ULIN_OK
    int ULIN_ERR_NULL_HANDLE
    int ULIN_ERR_SHAPE_MISMATCH
    int ULIN_ERR_BUFFER_TOO_SMALL

    int ulin_matrix_almost_equal(ulin_matrix a, ulin_matrix b, double eps)

    const char *ulin_version()
    bint ulin_init()

    ulin_matrix ulin_matrix_create(int rows, int cols)
    void ulin_matrix_destroy(ulin_matrix h)
    int ulin_matrix_rows(ulin_matrix h)
    int ulin_matrix_cols(ulin_matrix h)
    double ulin_matrix_get(ulin_matrix h, int i, int j)
    void ulin_matrix_set(ulin_matrix h, int i, int j, double v)
    ulin_matrix ulin_matrix_add(ulin_matrix a, ulin_matrix b)
    ulin_matrix ulin_matrix_sub(ulin_matrix a, ulin_matrix b)
    ulin_matrix ulin_matrix_mul(ulin_matrix a, ulin_matrix b)
    ulin_matrix ulin_matrix_scale(ulin_matrix h, double s)
    ulin_matrix ulin_matrix_transpose(ulin_matrix h)
    double ulin_matrix_determinant(ulin_matrix h, int *out_ok)
    int ulin_matrix_lu_solve(ulin_matrix h, const double *b, size_t blen,
                              double *out_buf, size_t out_cap, bint refine)
    ulin_matrix ulin_matrix_cholesky(ulin_matrix h)
    int ulin_matrix_qr(ulin_matrix h, ulin_matrix *out_q, ulin_matrix *out_r)
    int ulin_matrix_svd(ulin_matrix h, ulin_matrix *out_u,
                         double *out_s, size_t out_s_cap, ulin_matrix *out_v)
    ulin_matrix ulin_matrix_create_from_buffer(int rows, int cols,
                                                const double *buf, size_t n)
    int ulin_matrix_get_buffer(ulin_matrix h, double *out_buf, size_t out_cap)

    ulin_vec2 ulin_vec2_add(ulin_vec2 a, ulin_vec2 b)
    ulin_vec2 ulin_vec2_sub(ulin_vec2 a, ulin_vec2 b)
    ulin_vec2 ulin_vec2_scale(ulin_vec2 a, double s)
    double ulin_vec2_dot(ulin_vec2 a, ulin_vec2 b)
    double ulin_vec2_length(ulin_vec2 a)
    ulin_vec2 ulin_vec2_normalize(ulin_vec2 a)
    double ulin_vec2_cross2d(ulin_vec2 a, ulin_vec2 b)

    ulin_vec3 ulin_vec3_add(ulin_vec3 a, ulin_vec3 b)
    ulin_vec3 ulin_vec3_sub(ulin_vec3 a, ulin_vec3 b)
    ulin_vec3 ulin_vec3_scale(ulin_vec3 a, double s)
    double ulin_vec3_dot(ulin_vec3 a, ulin_vec3 b)
    double ulin_vec3_length(ulin_vec3 a)
    ulin_vec3 ulin_vec3_normalize(ulin_vec3 a)
    ulin_vec3 ulin_vec3_cross(ulin_vec3 a, ulin_vec3 b)

ulin_init()


def version():
    return ulin_version()


cdef class _MatrixHandle:
    """Raw handle over ulin_matrix. Owns the C-side pointer; freed on dealloc."""
    cdef ulin_matrix _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h != NULL:
            ulin_matrix_destroy(self._h)
            self._h = NULL

    @staticmethod
    def create(int rows, int cols):
        cdef _MatrixHandle obj = _MatrixHandle()
        obj._h = ulin_matrix_create(rows, cols)
        return obj if obj._h != NULL else None

    @staticmethod
    def create_from_buffer(int rows, int cols, values):
        """Matrix from a flat row-major sequence -- one bulk C call instead
        of rows*cols individual `set` calls. `values` already a contiguous
        float64 buffer (array.array('d', ...), a NumPy float64 array, a
        memoryview) is read with no copy; any other iterable (a `list`,
        most commonly) is validated and copied in a single Cython-compiled
        pass -- same dispatch pattern as UniAccurate's `_sum_generic`."""
        cdef _MatrixHandle obj = _MatrixHandle()
        cdef double[::1] view
        cdef Py_ssize_t n, i
        cdef double *buf
        cdef object v
        try:
            view = values
        except (TypeError, ValueError, BufferError):
            n = len(values)
            buf = <double *>malloc(n * sizeof(double)) if n > 0 else NULL
            if n > 0 and buf == NULL:
                raise MemoryError()
            try:
                for i in range(n):
                    v = values[i]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise TypeError(f"elements must be numbers, got {type(v).__name__}")
                    buf[i] = v
                obj._h = ulin_matrix_create_from_buffer(rows, cols, buf, <size_t>n)
            finally:
                free(buf)
        else:
            obj._h = ulin_matrix_create_from_buffer(
                rows, cols, &view[0] if view.shape[0] > 0 else NULL,
                <size_t>view.shape[0])
        return obj if obj._h != NULL else None

    def get_buffer(self):
        """Every element, flat row-major -- one bulk C call instead of
        rows*cols individual `get` calls."""
        cdef Py_ssize_t n = self.rows * self.cols
        cdef array.array out = array.array('d', bytes(n * sizeof(double)))
        cdef double[::1] view = out
        cdef int written = ulin_matrix_get_buffer(self._h, &view[0], <size_t>n)
        if written < 0:
            return None
        return out.tolist()

    @property
    def rows(self):
        return ulin_matrix_rows(self._h)

    @property
    def cols(self):
        return ulin_matrix_cols(self._h)

    def get(self, int i, int j):
        return ulin_matrix_get(self._h, i, j)

    def set(self, int i, int j, double v):
        ulin_matrix_set(self._h, i, j, v)

    def add(self, _MatrixHandle other):
        return _wrap(ulin_matrix_add(self._h, other._h))

    def sub(self, _MatrixHandle other):
        return _wrap(ulin_matrix_sub(self._h, other._h))

    def matmul(self, _MatrixHandle other):
        return _wrap(ulin_matrix_mul(self._h, other._h))

    def scale(self, double s):
        return _wrap(ulin_matrix_scale(self._h, s))

    def transpose(self):
        return _wrap(ulin_matrix_transpose(self._h))

    def determinant(self):
        cdef int ok = 0
        cdef double d = ulin_matrix_determinant(self._h, &ok)
        return (d, bool(ok))

    def lu_solve(self, b, bint refine=False):
        cdef double[::1] bview
        cdef Py_ssize_t n, i
        cdef double *buf
        cdef object v
        cdef array.array out
        cdef double[::1] outv
        cdef int written
        try:
            bview = b
        except (TypeError, ValueError, BufferError):
            n = len(b)
            buf = <double *>malloc(n * sizeof(double)) if n > 0 else NULL
            if n > 0 and buf == NULL:
                raise MemoryError()
            try:
                for i in range(n):
                    v = b[i]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise TypeError(f"elements must be numbers, got {type(v).__name__}")
                    buf[i] = v
                out = array.array('d', bytes(n * sizeof(double)))
                outv = out
                written = ulin_matrix_lu_solve(self._h, buf, <size_t>n,
                                                &outv[0], <size_t>n, refine)
            finally:
                free(buf)
        else:
            n = bview.shape[0]
            out = array.array('d', bytes(n * sizeof(double)))
            outv = out
            written = ulin_matrix_lu_solve(
                self._h, &bview[0] if n > 0 else NULL, <size_t>n,
                &outv[0] if n > 0 else NULL, <size_t>n, refine)
        if written < 0:
            return None
        return out.tolist()[:written]

    def cholesky(self):
        return _wrap(ulin_matrix_cholesky(self._h))

    def almost_equal(self, _MatrixHandle other, double eps):
        return bool(ulin_matrix_almost_equal(self._h, other._h, eps))

    def qr(self):
        cdef ulin_matrix q = NULL
        cdef ulin_matrix r = NULL
        cdef int status = ulin_matrix_qr(self._h, &q, &r)
        if status != ULIN_OK:
            return None
        return (_wrap(q), _wrap(r))

    def svd(self):
        cdef int cols = ulin_matrix_cols(self._h)
        cdef array.array out = array.array('d', bytes(cols * sizeof(double)))
        cdef double[::1] sv = out
        cdef ulin_matrix u = NULL
        cdef ulin_matrix v = NULL
        cdef int status = ulin_matrix_svd(self._h, &u, &sv[0], cols, &v)
        if status != ULIN_OK:
            return None
        return (_wrap(u), out.tolist(), _wrap(v))


cdef _MatrixHandle _wrap(ulin_matrix h):
    if h == NULL:
        return None
    cdef _MatrixHandle obj = _MatrixHandle()
    obj._h = h
    return obj


def vec2_add(x1, y1, x2, y2):
    cdef ulin_vec2 r = ulin_vec2_add(ulin_vec2(x=x1, y=y1), ulin_vec2(x=x2, y=y2))
    return (r.x, r.y)

def vec2_sub(x1, y1, x2, y2):
    cdef ulin_vec2 r = ulin_vec2_sub(ulin_vec2(x=x1, y=y1), ulin_vec2(x=x2, y=y2))
    return (r.x, r.y)

def vec2_scale(x, y, s):
    cdef ulin_vec2 r = ulin_vec2_scale(ulin_vec2(x=x, y=y), s)
    return (r.x, r.y)

def vec2_dot(x1, y1, x2, y2):
    return ulin_vec2_dot(ulin_vec2(x=x1, y=y1), ulin_vec2(x=x2, y=y2))

def vec2_length(x, y):
    return ulin_vec2_length(ulin_vec2(x=x, y=y))

def vec2_normalize(x, y):
    cdef ulin_vec2 r = ulin_vec2_normalize(ulin_vec2(x=x, y=y))
    return (r.x, r.y)

def vec2_cross2d(x1, y1, x2, y2):
    return ulin_vec2_cross2d(ulin_vec2(x=x1, y=y1), ulin_vec2(x=x2, y=y2))

def vec3_add(x1, y1, z1, x2, y2, z2):
    cdef ulin_vec3 r = ulin_vec3_add(ulin_vec3(x=x1, y=y1, z=z1), ulin_vec3(x=x2, y=y2, z=z2))
    return (r.x, r.y, r.z)

def vec3_sub(x1, y1, z1, x2, y2, z2):
    cdef ulin_vec3 r = ulin_vec3_sub(ulin_vec3(x=x1, y=y1, z=z1), ulin_vec3(x=x2, y=y2, z=z2))
    return (r.x, r.y, r.z)

def vec3_scale(x, y, z, s):
    cdef ulin_vec3 r = ulin_vec3_scale(ulin_vec3(x=x, y=y, z=z), s)
    return (r.x, r.y, r.z)

def vec3_dot(x1, y1, z1, x2, y2, z2):
    return ulin_vec3_dot(ulin_vec3(x=x1, y=y1, z=z1), ulin_vec3(x=x2, y=y2, z=z2))

def vec3_length(x, y, z):
    return ulin_vec3_length(ulin_vec3(x=x, y=y, z=z))

def vec3_normalize(x, y, z):
    cdef ulin_vec3 r = ulin_vec3_normalize(ulin_vec3(x=x, y=y, z=z))
    return (r.x, r.y, r.z)

def vec3_cross(x1, y1, z1, x2, y2, z2):
    cdef ulin_vec3 r = ulin_vec3_cross(ulin_vec3(x=x1, y=y1, z=z1), ulin_vec3(x=x2, y=y2, z=z2))
    return (r.x, r.y, r.z)
