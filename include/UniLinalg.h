// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 lituus-lab
#ifndef UNILINALG_H
#define UNILINALG_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UNILINALG_VERSION_MAJOR 0
#define UNILINALG_VERSION_MINOR 1
#define UNILINALG_VERSION_PATCH 0
#define UNILINALG_VERSION "0.1.0"

#define UNILINALG_VERSION_AT_LEAST(ma, mi, pa) \
  ((UNILINALG_VERSION_MAJOR > (ma)) || \
   (UNILINALG_VERSION_MAJOR == (ma) && UNILINALG_VERSION_MINOR > (mi)) || \
   (UNILINALG_VERSION_MAJOR == (ma) && UNILINALG_VERSION_MINOR == (mi) && \
    UNILINALG_VERSION_PATCH >= (pa)))

/* Error codes. ulin_matrix_qr/ulin_matrix_svd only ever return OK,
 * NULL_HANDLE, SHAPE_MISMATCH, or (svd) BUFFER_TOO_SMALL. ulin_matrix_lu_solve
 * returns the negated code (e.g. -ULIN_ERR_SINGULAR) instead of a bare -1;
 * ulin_matrix_cholesky still signals NOT_SPD as a plain NULL, since its
 * pointer return has no room for a reason code without an extra
 * out-parameter. */
#define ULIN_OK 0
#define ULIN_ERR_NULL_HANDLE 1
#define ULIN_ERR_SHAPE_MISMATCH 2
#define ULIN_ERR_SINGULAR 3
#define ULIN_ERR_NOT_SPD 4
#define ULIN_ERR_BUFFER_TOO_SMALL 5

/* Opaque handles. Never dereference; only pass between ulin_* calls.
 * Distinct incomplete-struct types, not both bare void*, so passing a
 * ulin_sparse where a ulin_matrix is expected (or vice versa) is a compiler
 * error instead of silently compiling. */
typedef struct ulin_matrix_s *ulin_matrix;
typedef struct ulin_sparse_s *ulin_sparse;

/* Value-type fixed-dimension vectors (float64 only). D is a compile-time
 * constant per alias, so these pass/return by value -- no handle, no heap
 * allocation. */
typedef struct { double x, y; } ulin_vec2;
typedef struct { double x, y, z; } ulin_vec3;
typedef struct { double x, y, z, w; } ulin_vec4;

/* -----------------------------------------------------------------------
 * Version & lifecycle.
 * ----------------------------------------------------------------------- */

/* Static version string; do not free. */
const char *ulin_version(void);

/* Bring up the Nim/ARC runtime. Call once before any other entry point.
 * Idempotent. Returns true. */
bool ulin_init(void);

/* No-op (matches ulin_init); handles are freed per-call by *_destroy. */
void ulin_cleanup(void);

/* Human-readable message for a ULIN_ERR_* code; "Unknown error" otherwise.
 * Static string; do not free. */
const char *ulin_get_error_string(int error_code);

/* -----------------------------------------------------------------------
 * Matrix — handle-based (rows x cols, row-major, float64).
 * ----------------------------------------------------------------------- */

/* Zero matrix of the given shape. NULL if rows/cols <= 0 or rows*cols
 * overflows what a returned element count can represent. */
ulin_matrix ulin_matrix_create(int rows, int cols);
void ulin_matrix_destroy(ulin_matrix h);

/* 0 on a nil handle. */
int ulin_matrix_rows(ulin_matrix h);
int ulin_matrix_cols(ulin_matrix h);

/* 0.0 on a nil handle or out-of-range index (never raises). */
double ulin_matrix_get(ulin_matrix h, int i, int j);
/* No-op on a nil handle or out-of-range index. */
void ulin_matrix_set(ulin_matrix h, int i, int j, double v);

/* Matrix from a flat row-major buffer (element (i,j) at buf[i*cols+j]) --
 * one bulk copy instead of rows*cols individual ulin_matrix_set calls. NULL
 * if rows/cols <= 0, buf is NULL, n != rows*cols, or rows*cols overflows
 * what a returned element count can represent. */
ulin_matrix ulin_matrix_create_from_buffer(int rows, int cols,
                                            const double *buf, size_t n);
/* Bulk row-major read of every element into out_buf -- one copy instead of
 * rows*cols individual ulin_matrix_get calls. Returns the count written
 * (rows*cols), or the negated ULIN_ERR_* reason (-ULIN_ERR_NULL_HANDLE /
 * -ULIN_ERR_BUFFER_TOO_SMALL) on failure -- always negative, so a plain
 * `< 0` check still works for a caller that only wants pass/fail.
 * rows*cols too large to return as a count is also -ULIN_ERR_BUFFER_TOO_SMALL. */
int ulin_matrix_get_buffer(ulin_matrix h, double *out_buf, size_t out_cap);

/* NULL on a nil handle or a shape mismatch. */
ulin_matrix ulin_matrix_add(ulin_matrix a, ulin_matrix b);
ulin_matrix ulin_matrix_sub(ulin_matrix a, ulin_matrix b);
/* NULL on a nil handle or a.cols != b.rows. */
ulin_matrix ulin_matrix_mul(ulin_matrix a, ulin_matrix b);
ulin_matrix ulin_matrix_scale(ulin_matrix h, double s);
ulin_matrix ulin_matrix_transpose(ulin_matrix h);

/* 0 (false) if either handle is nil. */
int ulin_matrix_almost_equal(ulin_matrix a, ulin_matrix b, double eps);

/* 0.0 with *out_ok = false on a nil handle or a non-square matrix.
 * out_ok may be NULL. */
double ulin_matrix_determinant(ulin_matrix h, int *out_ok);

/* Solves Ax = b, writing x into out_buf (out_cap must be >= rows). `refine`
 * true runs one step of UniAccurate-backed iterative refinement after the
 * solve (ADR-0006). Returns the number of elements written, or the negated
 * ULIN_ERR_* reason (-ULIN_ERR_NULL_HANDLE / -ULIN_ERR_SHAPE_MISMATCH /
 * -ULIN_ERR_BUFFER_TOO_SMALL / -ULIN_ERR_SINGULAR) on failure -- always
 * negative, so a plain `< 0` check still works for a caller that only
 * wants pass/fail. */
int ulin_matrix_lu_solve(ulin_matrix h, const double *b, size_t blen,
                          double *out_buf, size_t out_cap, bool refine);

/* Lower-triangular L with A = L L^T. NULL on a nil handle, a non-square
 * matrix, or a matrix that is not symmetric positive-definite -- undifferentiated,
 * unlike ulin_matrix_lu_solve: a pointer return has no room for a reason code
 * without an extra out-parameter, which this ABI does not add here. */
ulin_matrix ulin_matrix_cholesky(ulin_matrix h);

/* Householder QR: A = Q R. Writes the Q and R handles through out_q/out_r.
 * Returns ULIN_OK, ULIN_ERR_NULL_HANDLE, or ULIN_ERR_SHAPE_MISMATCH
 * (rows < cols) -- neither out-param is written on error. */
int ulin_matrix_qr(ulin_matrix h, ulin_matrix *out_q, ulin_matrix *out_r);

/* One-sided Jacobi SVD: A = U diag(S) V^T. Writes U/V handles and the
 * singular values (descending, cols of them) into the caller's buffers.
 * Returns ULIN_OK, ULIN_ERR_NULL_HANDLE, ULIN_ERR_SHAPE_MISMATCH
 * (rows < cols), or ULIN_ERR_BUFFER_TOO_SMALL. No out-param is written on
 * error. */
int ulin_matrix_svd(ulin_matrix h, ulin_matrix *out_u,
                     double *out_s, size_t out_s_cap, ulin_matrix *out_v);

/* -----------------------------------------------------------------------
 * Sparse — Compressed Sparse Row (CSR), float64.
 * ----------------------------------------------------------------------- */

ulin_sparse ulin_sparse_from_dense(ulin_matrix h);
ulin_matrix ulin_sparse_to_dense(ulin_sparse h);
void ulin_sparse_destroy(ulin_sparse h);
int ulin_sparse_nnz(ulin_sparse h);

/* Sparse matrix-vector product. Returns the number of elements written,
 * or -1 on a nil handle/buffer, shape mismatch, or too-small buffer. */
int ulin_sparse_matvec(ulin_sparse h, const double *v, size_t vlen,
                        double *out_buf, size_t out_cap);

/* -----------------------------------------------------------------------
 * Vector2 (float64).
 * ----------------------------------------------------------------------- */

ulin_vec2 ulin_vec2_add(ulin_vec2 a, ulin_vec2 b);
ulin_vec2 ulin_vec2_sub(ulin_vec2 a, ulin_vec2 b);
ulin_vec2 ulin_vec2_scale(ulin_vec2 a, double s);
double ulin_vec2_dot(ulin_vec2 a, ulin_vec2 b);
double ulin_vec2_length(ulin_vec2 a);
/* Zero vector if a has zero length (never raises). */
ulin_vec2 ulin_vec2_normalize(ulin_vec2 a);
int ulin_vec2_almost_equal(ulin_vec2 a, ulin_vec2 b, double eps);
/* Signed area of the parallelogram; positive if b is CCW from a. */
double ulin_vec2_cross2d(ulin_vec2 a, ulin_vec2 b);
/* 90 degrees CCW rotation: (-y, x). */
ulin_vec2 ulin_vec2_perp(ulin_vec2 a);
/* 90 degrees CW rotation: (y, -x). */
ulin_vec2 ulin_vec2_perp_cw(ulin_vec2 a);

/* -----------------------------------------------------------------------
 * Vector3 (float64).
 * ----------------------------------------------------------------------- */

ulin_vec3 ulin_vec3_add(ulin_vec3 a, ulin_vec3 b);
ulin_vec3 ulin_vec3_sub(ulin_vec3 a, ulin_vec3 b);
ulin_vec3 ulin_vec3_scale(ulin_vec3 a, double s);
double ulin_vec3_dot(ulin_vec3 a, ulin_vec3 b);
double ulin_vec3_length(ulin_vec3 a);
ulin_vec3 ulin_vec3_normalize(ulin_vec3 a);
int ulin_vec3_almost_equal(ulin_vec3 a, ulin_vec3 b, double eps);
/* Right-hand-rule cross product. */
ulin_vec3 ulin_vec3_cross(ulin_vec3 a, ulin_vec3 b);

/* -----------------------------------------------------------------------
 * Vector4 (float64).
 * ----------------------------------------------------------------------- */

ulin_vec4 ulin_vec4_add(ulin_vec4 a, ulin_vec4 b);
ulin_vec4 ulin_vec4_sub(ulin_vec4 a, ulin_vec4 b);
ulin_vec4 ulin_vec4_scale(ulin_vec4 a, double s);
double ulin_vec4_dot(ulin_vec4 a, ulin_vec4 b);
double ulin_vec4_length(ulin_vec4 a);
ulin_vec4 ulin_vec4_normalize(ulin_vec4 a);

#ifdef __cplusplus
}
#endif

#endif /* UNILINALG_H */
