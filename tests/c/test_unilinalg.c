// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 lituus-lab
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "UniLinalg.h"

static int failures = 0;

static void check_d(const char *name, double got, double want, double eps) {
  if (fabs(got - want) > eps) {
    printf("FAIL %s: got %g want %g\n", name, got, want);
    failures++;
  } else {
    printf("ok   %s = %g\n", name, got);
  }
}

static void check_i(const char *name, int got, int want) {
  if (got != want) { printf("FAIL %s: got %d want %d\n", name, got, want); failures++; }
  else printf("ok   %s = %d\n", name, got);
}

static void check_str(const char *name, const char *got, const char *want) {
  if (strcmp(got, want) != 0) { printf("FAIL %s: got \"%s\" want \"%s\"\n", name, got, want); failures++; }
  else printf("ok   %s = \"%s\"\n", name, got);
}

int main(void) {
  ulin_init();
  check_str("version", ulin_version(), UNILINALG_VERSION);

  /* Matrix: solve a 3x3 system with a known solution (same case as
   * tests/test_linalg.nim): x+2y+z=8, 2x+y+3z=13, x+y+z=6 -> (1,2,3). */
  ulin_matrix a = ulin_matrix_create(3, 3);
  double avals[9] = {1,2,1, 2,1,3, 1,1,1};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      ulin_matrix_set(a, i, j, avals[i*3+j]);
  double b[3] = {8.0, 13.0, 6.0};
  double x[3];
  int n = ulin_matrix_lu_solve(a, b, 3, x, 3, false);
  check_i("lu_solve: elements written", n, 3);
  check_d("lu_solve: x[0]", x[0], 1.0, 1e-9);
  check_d("lu_solve: x[1]", x[1], 2.0, 1e-9);
  check_d("lu_solve: x[2]", x[2], 3.0, 1e-9);

  /* Same system: the plain solve above lands within 1e-9 but not bit-exact
   * (1.0000000000000007, 2.0, 2.9999999999999996) -- refine=true recovers
   * the exactly-rounded answer (ADR-0006). */
  double xr[3];
  int nr = ulin_matrix_lu_solve(a, b, 3, xr, 3, true);
  check_i("lu_solve refine: elements written", nr, 3);
  check_d("lu_solve refine: x[0]", xr[0], 1.0, 0.0);
  check_d("lu_solve refine: x[1]", xr[1], 2.0, 0.0);
  check_d("lu_solve refine: x[2]", xr[2], 3.0, 0.0);

  /* lu_solve failure reasons: each maps to its own negated ULIN_ERR_* code,
   * not a flat -1. */
  check_i("lu_solve: NULL handle", ulin_matrix_lu_solve(NULL, b, 3, x, 3, false),
          -ULIN_ERR_NULL_HANDLE);
  check_i("lu_solve: blen shape mismatch", ulin_matrix_lu_solve(a, b, 2, x, 3, false),
          -ULIN_ERR_SHAPE_MISMATCH);
  check_i("lu_solve: out_cap too small", ulin_matrix_lu_solve(a, b, 3, x, 2, false),
          -ULIN_ERR_BUFFER_TOO_SMALL);
  ulin_matrix singular = ulin_matrix_create(2, 2);
  ulin_matrix_set(singular, 0, 0, 1.0); ulin_matrix_set(singular, 0, 1, 2.0);
  ulin_matrix_set(singular, 1, 0, 2.0); ulin_matrix_set(singular, 1, 1, 4.0);
  double sb[2] = {1.0, 1.0};
  double sx[2];
  check_i("lu_solve: singular matrix", ulin_matrix_lu_solve(singular, sb, 2, sx, 2, false),
          -ULIN_ERR_SINGULAR);
  ulin_matrix_destroy(singular);

  /* Bulk buffer transfer: one copy in, one copy out, vs the same avals/a
   * already built above one element at a time. */
  ulin_matrix abuf = ulin_matrix_create_from_buffer(3, 3, avals, 9);
  check_d("create_from_buffer: [0][0]", ulin_matrix_get(abuf, 0, 0), 1.0, 1e-9);
  check_d("create_from_buffer: [1][2]", ulin_matrix_get(abuf, 1, 2), 3.0, 1e-9);
  check_d("create_from_buffer: [2][2]", ulin_matrix_get(abuf, 2, 2), 1.0, 1e-9);
  double around[9];
  int gn = ulin_matrix_get_buffer(a, around, 9);
  check_i("get_buffer: elements written", gn, 9);
  check_d("get_buffer: [0]", around[0], avals[0], 1e-9);
  check_d("get_buffer: [8]", around[8], avals[8], 1e-9);
  check_i("create_from_buffer: NULL on shape mismatch",
          ulin_matrix_create_from_buffer(3, 3, avals, 8) == NULL, 1);
  check_i("get_buffer: -ULIN_ERR_BUFFER_TOO_SMALL on too-small out_cap",
          ulin_matrix_get_buffer(a, around, 8), -ULIN_ERR_BUFFER_TOO_SMALL);
  check_i("get_buffer: -ULIN_ERR_NULL_HANDLE on a NULL handle",
          ulin_matrix_get_buffer(NULL, around, 9), -ULIN_ERR_NULL_HANDLE);
  ulin_matrix_destroy(abuf);

  /* rows*cols must fit in a returned cint count -- otherwise the count
   * silently wraps under -d:danger (range checks compiled away). 46341 *
   * 46346 = 2147719986, just past INT32_MAX (2147483647). */
  check_i("create: NULL when rows*cols overflows a cint count",
          ulin_matrix_create(46341, 46346) == NULL, 1);
  check_i("create_from_buffer: NULL when rows*cols overflows a cint count",
          ulin_matrix_create_from_buffer(46341, 46346, avals, 0) == NULL, 1);

  /* Determinant: known value -2.0 for [[1,2],[3,4]]. */
  ulin_matrix d2 = ulin_matrix_create(2, 2);
  ulin_matrix_set(d2, 0, 0, 1.0); ulin_matrix_set(d2, 0, 1, 2.0);
  ulin_matrix_set(d2, 1, 0, 3.0); ulin_matrix_set(d2, 1, 1, 4.0);
  int ok = 0;
  double det = ulin_matrix_determinant(d2, &ok);
  check_i("determinant: ok", ok, 1);
  check_d("determinant: value", det, -2.0, 1e-9);
  ulin_matrix_destroy(d2);

  /* Cholesky: [[4,2],[2,3]] = L L^T, L = [[2,0],[1,sqrt(2)]]. */
  ulin_matrix spd = ulin_matrix_create(2, 2);
  ulin_matrix_set(spd, 0, 0, 4.0); ulin_matrix_set(spd, 0, 1, 2.0);
  ulin_matrix_set(spd, 1, 0, 2.0); ulin_matrix_set(spd, 1, 1, 3.0);
  ulin_matrix l = ulin_matrix_cholesky(spd);
  check_d("cholesky: l[0][0]", ulin_matrix_get(l, 0, 0), 2.0, 1e-9);
  check_d("cholesky: l[1][0]", ulin_matrix_get(l, 1, 0), 1.0, 1e-9);
  check_d("cholesky: l[1][1]", ulin_matrix_get(l, 1, 1), sqrt(2.0), 1e-9);
  ulin_matrix_destroy(l);
  ulin_matrix_destroy(spd);

  /* Cholesky: NULL on a non-square matrix -- the underlying algorithm's own
   * shape guard is debug-only (compiled away under -d:danger, this build's
   * flag) and silently returns a wrong-but-plausible result otherwise. */
  ulin_matrix nonSquare = ulin_matrix_create(2, 3);
  check_i("cholesky: NULL on non-square", ulin_matrix_cholesky(nonSquare) == NULL, 1);
  ulin_matrix_destroy(nonSquare);

  /* QR: shape check on a valid 3x2 input. */
  ulin_matrix qa = ulin_matrix_create(3, 2);
  double qavals[6] = {1,0, 0,1, 1,1};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      ulin_matrix_set(qa, i, j, qavals[i*2+j]);
  ulin_matrix q = NULL, r = NULL;
  int qrStatus = ulin_matrix_qr(qa, &q, &r);
  check_i("qr: status", qrStatus, ULIN_OK);
  check_i("qr: q rows", ulin_matrix_rows(q), 3);
  check_i("qr: q cols", ulin_matrix_cols(q), 3);
  check_i("qr: r rows", ulin_matrix_rows(r), 3);
  check_i("qr: r cols", ulin_matrix_cols(r), 2);
  ulin_matrix_destroy(q);
  ulin_matrix_destroy(r);
  ulin_matrix_destroy(qa);

  /* SVD: diagonal matrix, singular values sorted descending. */
  ulin_matrix sv = ulin_matrix_create(2, 2);
  ulin_matrix_set(sv, 0, 0, 2.0);
  ulin_matrix_set(sv, 1, 1, 5.0);
  ulin_matrix u = NULL, v = NULL;
  double s[2];
  int svdStatus = ulin_matrix_svd(sv, &u, s, 2, &v);
  check_i("svd: status", svdStatus, ULIN_OK);
  check_d("svd: s[0]", s[0], 5.0, 1e-9);
  check_d("svd: s[1]", s[1], 2.0, 1e-9);
  ulin_matrix_destroy(u);
  ulin_matrix_destroy(v);
  ulin_matrix_destroy(sv);

  /* Symmetric eigen: [[2,1],[1,2]] has descending values 3,1. */
  double evals[2] = {0.0, 0.0};
  ulin_matrix evecs = NULL;
  ulin_matrix sym = ulin_matrix_create(2, 2);
  ulin_matrix_set(sym, 0, 0, 2.0); ulin_matrix_set(sym, 0, 1, 1.0);
  ulin_matrix_set(sym, 1, 0, 1.0); ulin_matrix_set(sym, 1, 1, 2.0);
  int eigenStatus = ulin_matrix_symmetric_eigen(
      sym, evals, 2, &evecs, 64, 1e-12);
  check_i("symmetric eigen: status", eigenStatus, ULIN_OK);
  check_d("symmetric eigen: value 0", evals[0], 3.0, 1e-11);
  check_d("symmetric eigen: value 1", evals[1], 1.0, 1e-11);
  check_i("symmetric eigen: vector rows", ulin_matrix_rows(evecs), 2);
  check_i("symmetric eigen: vector cols", ulin_matrix_cols(evecs), 2);
  ulin_matrix_destroy(evecs);
  ulin_matrix_destroy(sym);

  ulin_matrix asym = ulin_matrix_create(2, 2);
  ulin_matrix_set(asym, 0, 1, 1.0);
  ulin_matrix_set(asym, 1, 0, 2.0);
  evecs = NULL;
  check_i("symmetric eigen: asymmetric",
      ulin_matrix_symmetric_eigen(asym, evals, 2, &evecs, 64, 1e-12),
      ULIN_ERR_NOT_SYMMETRIC);
  check_i("symmetric eigen: no output on error", evecs == NULL, 1);
  check_i("symmetric eigen: short buffer",
      ulin_matrix_symmetric_eigen(asym, evals, 1, &evecs, 64, 1e-12),
      ULIN_ERR_BUFFER_TOO_SMALL);
  check_i("symmetric eigen: NULL handle",
      ulin_matrix_symmetric_eigen(NULL, evals, 2, &evecs, 64, 1e-12),
      ULIN_ERR_NULL_HANDLE);
  ulin_matrix_destroy(asym);

  /* Sparse round-trip: same matrix as tests/test_linalg.nim's CSR case. */
  ulin_matrix dense = ulin_matrix_create(3, 3);
  double dvals[9] = {5,0,0, 0,8,3, 0,6,0};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      ulin_matrix_set(dense, i, j, dvals[i*3+j]);
  ulin_sparse csr = ulin_sparse_from_dense(dense);
  check_i("sparse: nnz", ulin_sparse_nnz(csr), 4);
  double sv_in[3] = {1.0, 2.0, 3.0};
  double sv_out[3];
  int svn = ulin_sparse_matvec(csr, sv_in, 3, sv_out, 3);
  check_i("sparse matvec: elements written", svn, 3);
  check_d("sparse matvec: [0]", sv_out[0], 5.0, 1e-9);
  check_d("sparse matvec: [1]", sv_out[1], 25.0, 1e-9);
  check_d("sparse matvec: [2]", sv_out[2], 12.0, 1e-9);
  ulin_sparse_destroy(csr);
  ulin_matrix_destroy(dense);
  ulin_matrix_destroy(a);

  /* Vector3 cross product of unit axes, matching Vector[D,T]'s own test. */
  ulin_vec3 ux = {1.0, 0.0, 0.0};
  ulin_vec3 uy = {0.0, 1.0, 0.0};
  ulin_vec3 uz = ulin_vec3_cross(ux, uy);
  check_d("vec3 cross: x", uz.x, 0.0, 1e-9);
  check_d("vec3 cross: y", uz.y, 0.0, 1e-9);
  check_d("vec3 cross: z", uz.z, 1.0, 1e-9);

  /* Vector2 length: the same 3-4-5 case Vector.length() is tested against,
   * proving the C surface reaches the same UniMath.sqrtNewtonGeneric path. */
  ulin_vec2 v34 = {3.0, 4.0};
  check_d("vec2 length", ulin_vec2_length(v34), 5.0, 1e-9);

  ulin_cleanup();

  if (failures == 0) { printf("\nAll C ABI tests passed.\n"); return 0; }
  printf("\n%d C ABI test(s) FAILED.\n", failures);
  return 1;
}
