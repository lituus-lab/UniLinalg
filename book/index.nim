# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniLinalg

nbInit(theme = useNimibook)
useLituus()
nb.title = "UniLinalg"

nbText: """
# UniLinalg

A linear algebra library for dense and sparse matrices, the classic
decompositions (LU, Cholesky, QR, SVD), and fixed-dimension vectors for
geometry and physics.

This page is a nimib book: every code block below is compiled and run when
the book is built, and the output shown is exactly what the code produced.
The exercises are real math and physics problems, the kind you'd meet in a
lycée or a first-year university course -- solved here with `UniLinalg`
instead of pencil and paper. Every function the library exports gets used
at least once along the way.
"""

nbSave
