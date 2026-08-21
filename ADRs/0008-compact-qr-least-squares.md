# ADR-0008: Tall least-squares systems use compact Householder QR

## Status

Accepted.

## Context

`qrDecompose` intentionally exposes a full square Q matrix.  That is useful
when Q itself is requested, but wasteful for tall least-squares problems that
only need `Q^T b`: an m-by-n fit otherwise allocates m-by-m storage.

## Decision

Add `leastSquaresCompact`.  It retains only the mutable m-by-n R workspace,
one Householder vector, and a transformed copy of b.  Every reflector is
applied immediately to R and b; the resulting n-by-n triangular system is
then back-substituted with the same rank threshold as `qrSolve`.

The routine does not expose Q and does not perform iterative refinement.
Callers requiring either property continue to use `qrDecompose` and
`leastSquares(refine=true)`.

## Consequences

Tall regression designs use O(m*n) rather than O(m*m) memory while retaining
Householder stability.  Both APIs remain available and additive.
