<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# unilinalg — Python binding

```bash
nimble pyLib                                    # native lib for this platform
(cd py && python3 setup.py build_ext --inplace) # build the Cython extension
(cd py && python3 -m pytest -q)                 # test
```

`nimble pyLib` builds the shared lib on Linux/macOS and the MSVC static lib on
Windows, so the same commands work everywhere a POSIX-compatible shell runs
them (bash/zsh on Linux/macOS, or WSL/Git Bash on Windows) -- the `( )`
subshells and `&&` chaining above are bash syntax, not PowerShell or
`cmd.exe`; adapt or use `python` instead of `python3` there. The subshells
keep your shell's cwd unchanged. Run the three commands in order: `pytest`
imports the `unilinalg` package straight out of this checkout, so it only
finds the `_core` extension `build_ext --inplace` just compiled if that step
already ran.

```python
import unilinalg

a = unilinalg.Matrix.from_rows([[1.0, 2.0, 1.0],
                                 [2.0, 1.0, 3.0],
                                 [1.0, 1.0, 1.0]])
a.solve([8.0, 13.0, 6.0])   # [1.0000000000000007, 2.0, 2.9999999999999996]
a.solve([8.0, 13.0, 6.0], refine=True)   # [1.0, 2.0, 3.0] -- exact with refinement

x = unilinalg.Vec3(1.0, 0.0, 0.0)
y = unilinalg.Vec3(0.0, 1.0, 0.0)
x.cross(y)                 # Vec3(0.0, 0.0, 1.0)

unilinalg.Vec2(3.0, 4.0).length   # 5.0
```
