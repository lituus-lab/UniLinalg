# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Author py/notebooks/quickstart.ipynb, then execute it so the committed file
carries real outputs for GitHub to render. Run from the repo root:

    python3 py/notebooks/build_quickstart.py

CI re-executes the notebook against an installed wheel; this script only
regenerates it after an API change."""
import os

import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "quickstart.ipynb")

CELLS = [
    ("md", """# UniLinalg — Python quickstart

`unilinalg` is a Cython extension over the UniLinalg C ABI, shipped as a
self-contained wheel: the native library travels inside the package, so
installing it needs neither Nim nor a compiler.

```
pip install unilinalg
```

CI installs the wheel the release actually publishes and executes this
notebook against it, so a change that breaks the API breaks the build -- but
only cell *execution* is checked, not that a printed value still matches
what's committed here."""),
    ("md", "## Matrix: solve a linear system"),
    ("code", """import unilinalg

unilinalg.version(), unilinalg.__version__"""),
    ("md", """`Matrix.solve` factors through LU with partial pivoting. The system
below has the hand-checkable solution (1, 2, 3)."""),
    ("code", """a = unilinalg.Matrix.from_rows([[1.0, 2.0, 1.0],
                                [2.0, 1.0, 3.0],
                                [1.0, 1.0, 1.0]])
a.solve([8.0, 13.0, 6.0])"""),
    ("md", """A singular matrix is a domain error, not a silently wrong answer."""),
    ("code", """try:
    unilinalg.Matrix.from_rows([[1.0, 2.0], [2.0, 4.0]]).solve([1.0, 1.0])
except ValueError as exc:
    print("ValueError:", exc)"""),
    ("md", "## Cholesky, QR, SVD"),
    ("code", """spd = unilinalg.Matrix.from_rows([[4.0, 2.0], [2.0, 3.0]])
l = spd.cholesky()
l.to_rows()"""),
    ("code", """try:
    unilinalg.Matrix.from_rows([[1.0, 2.0], [2.0, 1.0]]).cholesky()
except ValueError as exc:
    print("ValueError:", exc)"""),
    ("code", """diag = unilinalg.Matrix.from_rows([[2.0, 0.0], [0.0, 5.0]])
u, s, v = diag.svd()
s"""),
    ("md", """## Vector2/Vector3: fixed-dimension geometric vectors

`Vec2`/`Vec3` are distinct from `Matrix`: fixed dimension, not row/column
counts. `length()` routes through UniMath's `sqrtNewtonGeneric` on the Nim
side -- a real dependency, not a decorative one (see the book)."""),
    ("code", """x = unilinalg.Vec3(1.0, 0.0, 0.0)
y = unilinalg.Vec3(0.0, 1.0, 0.0)
x.cross(y)"""),
    ("code", "unilinalg.Vec2(3.0, 4.0).length"),
    ("md", "A non-numeric component is a type error, not a coercion."),
    ("code", """try:
    unilinalg.Vec2("x", 1.0)
except TypeError as exc:
    print("TypeError:", exc)"""),
    ("md", """## The C ABI underneath

The same entry points are reachable from anything that speaks C. There the
contract is expressed by NULL/negative-count/error-code returns instead of
raising -- an exception must never unwind across an ABI boundary:

```c
ulin_matrix_lu_solve(h, b, blen, out, out_cap);  /* -1 on singular/shape error */
ulin_matrix_cholesky(h);                         /* NULL if not SPD */
```

See `include/UniLinalg.h`, and the book for the full picture."""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    # Execute from the repo root, never from py/: there, `import unilinalg`
    # would resolve to the py/unilinalg source tree instead of the installed
    # package, and the notebook would stop testing what it claims to test.
    NotebookClient(nb, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": ROOT}}).execute()
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
