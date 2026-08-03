# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib

nbInit
nb.title = "UniLinalg"

nbText: """
# UniLinalg

A linear algebra library: dense and sparse matrices with the classic
decompositions (LU, Cholesky, QR, SVD), and `Vector[D,T]`, a fixed-dimension
geometric/physical vector designed to be consumed by downstream geometry
and physics engines.

This page is a nimib book: every Nim block below is compiled and run when the
book is built, and the output shown is what the code actually produced. A
change that breaks the API breaks the docs build, so the two cannot drift
apart.

## Matrix: dense linear algebra

`Matrix[T]` is a pedagogical, dependency-free row-major matrix: no magic,
just the schoolbook algorithms written to be read.
"""

nbCode:
  import UniLinalg

  let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                                 2.0, 1.0, 3.0,
                                 1.0, 1.0, 1.0])
  echo "solve(a, b) = ", solve(a, [8.0, 13.0, 6.0])
  echo "det(a) = ", det(a)

nbText: """
Cholesky, for symmetric positive-definite matrices, is half the work of LU
and doubles as its own SPD certificate: a non-positive diagonal term during
the factorization means the matrix was never SPD.
"""

nbCode:
  let spd = matrix[float64](2, 2, [4.0, 2.0, 2.0, 3.0])
  let l = cholesky(spd)
  echo "cholesky(spd) = ", l
  echo "l * l^T == spd? ", almostEqual(l * transpose(l), spd, 1e-9)

nbText: """
QR (Householder reflections) and SVD (one-sided Jacobi) round out the
decompositions -- QR for least squares, SVD for rank and the singular
spectrum.
"""

nbCode:
  let qa = matrix[float64](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
  let qrResult = qrDecompose(qa)
  echo "Q is ", qrResult.q.rows, "x", qrResult.q.cols,
       ", R is ", qrResult.r.rows, "x", qrResult.r.cols

  let diag = matrix[float64](2, 2, [2.0, 0.0, 0.0, 5.0])
  echo "singular values of diag(2,5) = ", svdDecompose(diag).s

nbText: """
### References

- Wikipedia: [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
- Wikipedia: [Pivot element](https://en.wikipedia.org/wiki/Pivot_element) --
  partial pivoting, the numerical-stability strategy `lu.nim` uses.
- Wikipedia: [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)
- Wikipedia: [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
- Wikipedia: [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)
- Wikipedia: [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- Wikipedia: [Jacobi eigenvalue algorithm](https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm) --
  the one-sided variant `svd.nim` implements.
"""

nbText: """
## Vector: fixed-dimension geometric/physical vectors

`Vector[D: static[int], T: RealField]` is a different shape of problem than
`Matrix`: `D` is a compile-time constant, chosen for `Vector2d/3d/4d`
(float64) and `Vector2f/3f/4f` (float32) -- geometry and physics work with
2/3/4-component vectors, not arbitrary row/column counts. `RealField` (from
UniMath) also admits the exact scalars `Rational`/`Fixed`/`BigFloat`, not
just the two float aliases shown here.
"""

nbCode:
  let x = vec3(1.0, 0.0, 0.0)
  let y = vec3(0.0, 1.0, 0.0)
  echo "cross(x, y) = ", cross(x, y)
  echo "dot(x, y) = ", dot(x, y)

  let v = vec2(3.0, 4.0)
  echo "vec2(3, 4).length = ", v.length
  echo "vec2(3, 4).normalize = ", v.normalize

nbText: """
### References

- Wikipedia: [Euclidean vector](https://en.wikipedia.org/wiki/Euclidean_vector)
- Wikipedia: [Dot product](https://en.wikipedia.org/wiki/Dot_product)
- Wikipedia: [Cross product](https://en.wikipedia.org/wiki/Cross_product)
"""

nbText: """
## Why Matrix and Vector don't interoperate

There is no `Matrix[T] * Vector[D,T]` operator. This is a deliberate boundary,
not an oversight: `Matrix.rows`/`Matrix.cols` are runtime `int`s (a matrix's
shape is a value, decided when it is built), while `Vector[D,T]`'s `D` is a
`static[int]` (its dimension is a type, decided at compile time). A
matrix-vector product whose output dimension depends on a runtime value
cannot be typed as `Vector[D,T]` for any fixed `D` -- the two type families
solve genuinely different problems and are kept honestly separate rather
than forced into one API.

(`Matrix[T] * openArray[T]`, returning a plain `seq[T]`, already covers the
runtime-sized matrix-vector product -- see `solve` above, which uses exactly
that.)

## A real dependency on UniMath, not a decorative one

`UniLinalg.nimble` declares `UniLinalg --> UniMath`. That edge is backed by
an actual compiled call, not just a `requires` line nobody imports: before
this was wired, `length()` called `std/math.sqrt` directly.
"""

nbText: """
```nim
# before
func length*[D: static[int], T: SomeFloat](v: Vector[D, T]): T {.inline.} =
  sqrt(lengthSquared(v))

# after
import std/math
import UniMath
func length*[D: static[int], T: RealField](v: Vector[D, T]): T {.inline.} =
  sqrt(lengthSquared(v))
```

Same call, same signature shape -- only the constraint and the imports in
scope change. With both `std/math` and `UniMath` imported, unqualified
`sqrt` resolves by overload: `float32`/`float64` to the hardware root,
`BigFloat`/`Rational`/`Fixed` to their own UniMath `sqrt` (each `export`ed
through the umbrella). No `when` branch is needed -- the call site is
identical for every `RealField` instance. `normalize()` calls `length()`,
so it inherits the same path.

The Matrix decompositions (LU/Cholesky/QR/SVD) deliberately keep
`std/math.sqrt` -- they are mature, already-tested code relocated verbatim
from `UniversalMath/UniLinalg` 1.0.0, and rewriting their internals to route
through UniMath was not required to satisfy "wire a real dependency"; doing
so anyway would add regression risk to code that was ported, not authored,
here.

## The C ABI

Two different shapes for two different kinds of value. `Matrix`/`CsrMatrix`
are dynamically sized, so the C surface hands out an opaque handle
(`ulin_matrix`), pinned on the Nim side and freed by the caller via
`ulin_matrix_destroy` -- the same `pin`/`Of`/`unref` idiom `unimath_bigint`
already established for heap-sized values. `Vector[2/3/4, float64]` is a
compile-time-sized value type, so it crosses the boundary as a flat struct
(`ulin_vec2/3/4`), passed and returned by value -- no handle, no heap
allocation.

```c
ulin_matrix ulin_matrix_create(int rows, int cols);
void        ulin_matrix_destroy(ulin_matrix h);
int         ulin_matrix_lu_solve(ulin_matrix h, const double *b, size_t blen,
                                  double *out_buf, size_t out_cap);

typedef struct { double x, y, z; } ulin_vec3;
ulin_vec3 ulin_vec3_cross(ulin_vec3 a, ulin_vec3 b);
```

The C ABI never raises: buffer-writing operations (`lu_solve`, sparse
`matvec`) return the element count written or `-1`; multi-output operations
(`qr`, `svd`) return a `ULIN_OK`/`ULIN_ERR_*` status code; single-value
operations return `NULL`/`0.0` with an optional `*out_ok`. `tests/c` links
the hand-written header against the compiled library on every run, so a
renamed or retyped exported symbol fails to link there first.

## The Python surface

A Cython extension over the C ABI, shipped as a self-contained wheel: the
native library travels inside the package, so installing it needs neither
Nim nor a compiler.

```python
import unilinalg

a = unilinalg.Matrix.from_rows([[1.0, 2.0, 1.0],
                                 [2.0, 1.0, 3.0],
                                 [1.0, 1.0, 1.0]])
a.solve([8.0, 13.0, 6.0])   # [1.0000000000000007, 2.0, 2.9999999999999996]

unilinalg.Vec3(1.0, 0.0, 0.0).cross(unilinalg.Vec3(0.0, 1.0, 0.0))
```

Here the domain check returns, because Python has exceptions to carry it: a
non-square `determinant()`, a singular `solve()`, a non-SPD `cholesky()`, or
a rank-deficient `qr()`/`svd()` all raise `ValueError`; a non-numeric vector
component or a wrong-typed operand raises `TypeError`. Each surface expresses
the same contract in the terms its own callers expect -- a domain guard in
Nim, an error code in C, an exception in Python.

`py/notebooks/quickstart.ipynb` runs these calls against an installed wheel
and renders on GitHub directly.
"""

nbSave
