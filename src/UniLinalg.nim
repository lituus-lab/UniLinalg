# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg: Linear Algebra
# ==========================================================
#
# Dense and sparse matrices with the classic decompositions, written to be
# read: LU (Gaussian elimination with partial pivoting), Cholesky (SPD),
# QR (Householder reflections), SVD (one-sided Jacobi).
#
# Relocated from the original UniversalMath monorepo's UniLinalg package
# (1.0.0). Depends on UniMath (UniLinalg --> UniMath).

import ./UniLinalg/types/matrix
import ./UniLinalg/types/sparse
import ./UniLinalg/types/vector
import ./UniLinalg/types/tolerance
import ./UniLinalg/algorithms/refine
import ./UniLinalg/algorithms/lu
import ./UniLinalg/algorithms/cholesky
import ./UniLinalg/algorithms/qr
import ./UniLinalg/algorithms/svd

export matrix, sparse, vector, tolerance, refine, lu, cholesky, qr, svd

const UniLinalgVersion* = "1.0.0"
