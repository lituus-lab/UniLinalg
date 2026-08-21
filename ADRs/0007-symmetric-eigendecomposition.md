<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0007: Symmetric eigendecomposition

- Status: Accepted
- Date: 2026-08-21

## Decision

UniLinalg provides a Jacobi eigendecomposition for finite real symmetric
matrices. It returns eigenvalues in descending order and the corresponding
orthonormal eigenvectors as columns. Symmetry is checked with an explicit
absolute tolerance; non-square, non-finite, asymmetric, or non-convergent
inputs are rejected.

The implementation uses plane rotations directly on a private matrix copy.
It does not form normal equations and does not route through the general SVD.
Nim, C, and Python expose the same ordering and column convention.

## Consequences

Covariance and correlation matrices can be consumed by UniStatistics for
principal-component analysis without moving spectral algorithms into the
statistics layer. The Jacobi method is allocation-bounded and accurate for
small and medium dense matrices, but it is not a replacement for a blocked
LAPACK solver on large matrices.
