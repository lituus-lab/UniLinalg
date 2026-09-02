<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# unilinalg

Linear algebra for Python — dense and sparse matrices, the classic
decompositions (LU, Cholesky, QR, SVD, symmetric eigenpairs), and fixed-dimension geometric
vectors, backed by the native
[UniLinalg](https://github.com/lituus-lab/UniLinalg) library.

UniLinalg factors and solves through LU with partial pivoting, Cholesky for
symmetric positive-definite systems, Householder QR for least squares, and
one-sided Jacobi SVD. `Matrix.solve`'s optional `refine=True` runs one
UniAccurate-backed iterative refinement step to recover the last few ULP a
plain float64 solve can miss — the other three decompositions don't expose
this option through the Python API. CSR sparse matrices cover the
mostly-zero case; `Vec2`/`Vec3`/`Vec4` cover fixed-dimension geometry,
distinct from the runtime-sized `Matrix`.

## Install

```bash
pip install lituus-unilinalg
```

Prebuilt wheels include the native UniLinalg library for Linux, macOS, and
Windows on CPython 3.10–3.14. Installing a wheel needs neither Nim nor a C
compiler.

## Quick start

```python
import unilinalg

a = unilinalg.Matrix.from_rows([[1.0, 2.0, 1.0],
                                 [2.0, 1.0, 3.0],
                                 [1.0, 1.0, 1.0]])
a.solve([8.0, 13.0, 6.0])                # [1.0000000000000007, 2.0, 2.9999999999999996]
a.solve([8.0, 13.0, 6.0], refine=True)   # [1.0, 2.0, 3.0] -- exact with refinement

spd = unilinalg.Matrix.from_rows([[4.0, 2.0], [2.0, 3.0]])
spd.cholesky().to_rows()                 # [[2.0, 0.0], [1.0, 1.4142135623730951]]

diag = unilinalg.Matrix.from_rows([[2.0, 0.0], [0.0, 5.0]])
u, s, v = diag.svd()
s                                        # [5.0, 2.0]

dense = unilinalg.Matrix.from_rows([[5.0, 0.0, 0.0],
                                     [0.0, 8.0, 3.0],
                                     [0.0, 6.0, 0.0]])
sparse = dense.to_sparse()
sparse.nnz                               # 4
sparse.matvec([1.0, 2.0, 3.0])           # [5.0, 25.0, 12.0]

x, y = unilinalg.Vec3(1.0, 0.0, 0.0), unilinalg.Vec3(0.0, 1.0, 0.0)
x.cross(y)                               # Vec3(0.0, 0.0, 1.0)
unilinalg.Vec2(3.0, 4.0).length          # 5.0
```

## What's included

| Category | Python API |
|---|---|
| Dense matrix | `Matrix` -- construction, indexing, `+`/`-`/`@`/`*`, `transpose`, `almost_equal` |
| Linear solve | `Matrix.solve` (LU, partial pivoting), `Matrix.determinant` |
| Decompositions | `Matrix.cholesky`, `Matrix.qr`, `Matrix.svd`, `Matrix.symmetric_eigen` |
| Accurate refinement | `refine=True` on `solve` -- one UniAccurate-backed correction step |
| Sparse matrix | `Sparse` (CSR) -- `Matrix.to_sparse`, `Sparse.to_dense`, `Sparse.matvec`, `Sparse.nnz` |
| Fixed-dimension vectors | `Vec2`, `Vec3`, `Vec4` -- arithmetic, `dot`, `length`, `normalize`; `Vec3.cross`; `Vec2.cross2d`/`perp`/`perp_cw` |

A numeric-looking element is coerced through `float(...)`, same as a plain
Python `list` of numbers would be — but a singular matrix, a shape mismatch,
or a genuinely non-numeric component raises `ValueError`/`TypeError` rather
than returning a wrong-shaped or silently wrong result.

For an executable tour of the API, see the
[Python quickstart notebook](https://github.com/lituus-lab/UniLinalg/blob/main/py/notebooks/quickstart.ipynb).

## Links

- Source, Nim API, C ABI, and design records: <https://github.com/lituus-lab/UniLinalg>
- Issues: <https://github.com/lituus-lab/UniLinalg/issues>
- License: Apache-2.0

## Development

Building from source (contributing, or a platform without a prebuilt wheel)
needs a Nim toolchain.

```bash
nimble pyLib   # native lib for this platform
cd py
python3 setup.py build_ext --inplace   # build the Cython extension
python3 -m pytest -q                   # test
```

On Windows, use `python` instead of `python3` (PowerShell and `cmd.exe`
both resolve it; `python3` is a POSIX-only convention):

```powershell
nimble pyLib
cd py
python setup.py build_ext --inplace
python -m pytest -q
```

Run the build before the test in every case: `pytest` imports the
`unilinalg` package straight out of this checkout, so it only finds the
`_core` extension once `build_ext --inplace` has compiled it.
