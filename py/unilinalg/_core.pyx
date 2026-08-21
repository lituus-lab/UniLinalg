# cython: language_level=3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Raw Cython bindings over the UniLinalg C ABI (no domain checks).
Use the `unilinalg` package (this module's __init__.py) instead."""
from libc.stddef cimport size_t
from cpython cimport array

cdef extern from "UniLinalg.h":
    # Opaque incomplete-struct pointers, matching UniLinalg.h's own
    # ulin_matrix_s*/ulin_sparse_s* -- distinct Cython types, not both a bare
    # void*, so passing a ulin_sparse where a ulin_matrix is expected is a
    # compile error here too.
    ctypedef struct ulin_matrix_s
    ctypedef ulin_matrix_s *ulin_matrix
    ctypedef struct ulin_sparse_s
    ctypedef ulin_sparse_s *ulin_sparse

    ctypedef struct ulin_vec2:
        double x
        double y

    ctypedef struct ulin_vec3:
        double x
        double y
        double z

    ctypedef struct ulin_vec4:
        double x
        double y
        double z
        double w

    int ULIN_OK
    int ULIN_ERR_NULL_HANDLE
    int ULIN_ERR_SHAPE_MISMATCH
    int ULIN_ERR_BUFFER_TOO_SMALL
    int ULIN_ERR_MEMORY

    int ulin_matrix_almost_equal(ulin_matrix a, ulin_matrix b, double eps)

    ulin_sparse ulin_sparse_from_dense(ulin_matrix h)
    ulin_matrix ulin_sparse_to_dense(ulin_sparse h)
    void ulin_sparse_destroy(ulin_sparse h)
    int ulin_sparse_nnz(ulin_sparse h)
    int ulin_sparse_matvec(ulin_sparse h, const double *v, size_t vlen,
                            double *out_buf, size_t out_cap)

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
    int ulin_matrix_symmetric_eigen(ulin_matrix h, double *out_values,
                                     size_t out_values_cap,
                                     ulin_matrix *out_vectors,
                                     int max_sweeps, double tolerance)
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
    ulin_vec2 ulin_vec2_perp(ulin_vec2 a)
    ulin_vec2 ulin_vec2_perp_cw(ulin_vec2 a)

    ulin_vec3 ulin_vec3_add(ulin_vec3 a, ulin_vec3 b)
    ulin_vec3 ulin_vec3_sub(ulin_vec3 a, ulin_vec3 b)
    ulin_vec3 ulin_vec3_scale(ulin_vec3 a, double s)
    double ulin_vec3_dot(ulin_vec3 a, ulin_vec3 b)
    double ulin_vec3_length(ulin_vec3 a)
    ulin_vec3 ulin_vec3_normalize(ulin_vec3 a)
    ulin_vec3 ulin_vec3_cross(ulin_vec3 a, ulin_vec3 b)

    ulin_vec4 ulin_vec4_add(ulin_vec4 a, ulin_vec4 b)
    ulin_vec4 ulin_vec4_sub(ulin_vec4 a, ulin_vec4 b)
    ulin_vec4 ulin_vec4_scale(ulin_vec4 a, double s)
    double ulin_vec4_dot(ulin_vec4 a, ulin_vec4 b)
    double ulin_vec4_length(ulin_vec4 a)
    ulin_vec4 ulin_vec4_normalize(ulin_vec4 a)

ulin_init()


def version():
    return ulin_version()


cdef double[::1] _as_double_view(values) except *:
    """Validated contiguous float64 view over `values`. Zero-copy for an
    already-contiguous float64 buffer (array.array('d', ...), a NumPy
    float64 array, a memoryview); a generic iterable (a `list`, most
    commonly) is validated and copied into a freshly allocated buffer in a
    single Cython-compiled pass -- same dispatch pattern as UniAccurate's
    `_sum_generic`. Shared by create_from_buffer/lu_solve/matvec, which all
    need the same buffer-or-iterable dispatch over a `double*` argument."""
    cdef double[::1] view
    cdef Py_ssize_t n, i
    cdef array.array out
    cdef object v
    try:
        view = values
        return view
    except (TypeError, ValueError, BufferError):
        n = len(values)
        out = array.array('d', bytes(n * sizeof(double)))
        view = out
        for i in range(n):
            v = values[i]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            view[i] = v
        return view


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
        of rows*cols individual `set` calls. See `_as_double_view` for the
        zero-copy/validate-and-copy dispatch."""
        cdef _MatrixHandle obj = _MatrixHandle()
        cdef double[::1] view = _as_double_view(values)
        cdef Py_ssize_t n = view.shape[0]
        obj._h = ulin_matrix_create_from_buffer(
            rows, cols, &view[0] if n > 0 else NULL, <size_t>n)
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
        cdef double[::1] bview = _as_double_view(b)
        cdef Py_ssize_t n = bview.shape[0]
        cdef array.array out = array.array('d', bytes(n * sizeof(double)))
        cdef double[::1] outv = out
        cdef int written = ulin_matrix_lu_solve(
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

    def symmetric_eigen(self, int max_sweeps, double tolerance):
        cdef int rows = ulin_matrix_rows(self._h)
        cdef array.array out = array.array('d', bytes(rows * sizeof(double)))
        cdef double[::1] values = out
        cdef ulin_matrix vectors = NULL
        cdef int status = ulin_matrix_symmetric_eigen(
            self._h, &values[0], rows, &vectors, max_sweeps, tolerance)
        if status == ULIN_ERR_MEMORY:
            raise MemoryError("symmetric eigendecomposition allocation failed")
        if status != ULIN_OK:
            return None
        return (out.tolist(), _wrap(vectors))


cdef _MatrixHandle _wrap(ulin_matrix h):
    if h == NULL:
        return None
    cdef _MatrixHandle obj = _MatrixHandle()
    obj._h = h
    return obj


cdef class _SparseHandle:
    """Raw handle over ulin_sparse (CSR). Owns the C-side pointer; freed on
    dealloc."""
    cdef ulin_sparse _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h != NULL:
            ulin_sparse_destroy(self._h)
            self._h = NULL

    @staticmethod
    def from_dense(_MatrixHandle dense):
        cdef _SparseHandle obj = _SparseHandle()
        obj._h = ulin_sparse_from_dense(dense._h)
        return obj if obj._h != NULL else None

    def to_dense(self):
        return _wrap(ulin_sparse_to_dense(self._h))

    @property
    def nnz(self):
        return ulin_sparse_nnz(self._h)

    def matvec(self, v):
        cdef double[::1] vview = _as_double_view(v)
        cdef Py_ssize_t n = vview.shape[0]
        cdef array.array out = array.array('d', bytes(n * sizeof(double)))
        cdef double[::1] outv = out
        cdef int written = ulin_sparse_matvec(
            self._h, &vview[0] if n > 0 else NULL, <size_t>n,
            &outv[0] if n > 0 else NULL, <size_t>n)
        if written < 0:
            return None
        return out.tolist()[:written]


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

def vec2_perp(x, y):
    cdef ulin_vec2 r = ulin_vec2_perp(ulin_vec2(x=x, y=y))
    return (r.x, r.y)

def vec2_perp_cw(x, y):
    cdef ulin_vec2 r = ulin_vec2_perp_cw(ulin_vec2(x=x, y=y))
    return (r.x, r.y)

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

def vec4_add(x1, y1, z1, w1, x2, y2, z2, w2):
    cdef ulin_vec4 r = ulin_vec4_add(ulin_vec4(x=x1, y=y1, z=z1, w=w1),
                                      ulin_vec4(x=x2, y=y2, z=z2, w=w2))
    return (r.x, r.y, r.z, r.w)

def vec4_sub(x1, y1, z1, w1, x2, y2, z2, w2):
    cdef ulin_vec4 r = ulin_vec4_sub(ulin_vec4(x=x1, y=y1, z=z1, w=w1),
                                      ulin_vec4(x=x2, y=y2, z=z2, w=w2))
    return (r.x, r.y, r.z, r.w)

def vec4_scale(x, y, z, w, s):
    cdef ulin_vec4 r = ulin_vec4_scale(ulin_vec4(x=x, y=y, z=z, w=w), s)
    return (r.x, r.y, r.z, r.w)

def vec4_dot(x1, y1, z1, w1, x2, y2, z2, w2):
    return ulin_vec4_dot(ulin_vec4(x=x1, y=y1, z=z1, w=w1),
                          ulin_vec4(x=x2, y=y2, z=z2, w=w2))

def vec4_length(x, y, z, w):
    return ulin_vec4_length(ulin_vec4(x=x, y=y, z=z, w=w))

def vec4_normalize(x, y, z, w):
    cdef ulin_vec4 r = ulin_vec4_normalize(ulin_vec4(x=x, y=y, z=z, w=w))
    return (r.x, r.y, r.z, r.w)
