# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — linear algebra (Nim + C-ABI + Python).

version       = "0.1.0"
author        = "lituus-lab"
description   = "Linear algebra: dense/sparse matrices, LU/Cholesky/QR/SVD, Vector[D,T] (Nim + C-ABI + Python)"
license       = "Apache-2.0"
srcDir        = "src"

requires "nim >= 2.0.0"
requires "https://github.com/lbartoletti/NimContracts#fix/generic-proc-support"
# Private repo today (see README): local dev needs `gh auth setup-git` or an
# SSH key with lituus-lab access for this to resolve via `nimble install`.
requires "https://github.com/lituus-lab/UniMath#main"

task lint, "Fail if nimpretty would reformat a source":
  exec "nim c -r --hints:off -o:build/lint_tool tools/lint.nim"

task checkVGraph, "Fail on an import that climbs the layers in vgraph.cfg":
  exec "nim c -r --hints:off -o:build/vgraph_tool tools/vgraph.nim"

task docsDeps, "Install the docs toolchain (nimib)":
  exec "nimble install -y nimib"

task book, "Build the nimib book (needs nimib)":
  # nimib compiles and runs the book's code blocks: a drift fails the build.
  exec "nim c -r --path:src --hints:off -o:build/book book/index.nim"

task docs, "API reference + book into pages/ — what CI publishes":
  rmDir "pages"
  exec "nim doc --index:on --outdir:pages/api --project --hints:off src/UniLinalg.nim"
  exec "nimble book"
  # The book is the landing page; the generated reference sits under api/.
  cpFile "book/index.html", "pages/index.html"

task test, "Nim tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_linalg tests/test_linalg.nim"
  exec "nim c -r --path:src -o:build/test_contracts tests/test_contracts.nim"
  exec "nim c -r --path:src -o:build/test_vector tests/test_vector.nim"

task testRelease, "Nim tests (-d:danger: bound/overflow checks off, contracts compiled away)":
  exec "nim c -r -d:danger --path:src -o:build/test_linalg_rel tests/test_linalg.nim"
  exec "nim c -r -d:danger --path:src -o:build/test_contracts_rel tests/test_contracts.nim"
  exec "nim c -r -d:danger --path:src -o:build/test_vector_rel tests/test_vector.nim"

task testCi, "Nim tests (CI subset, debug)":
  exec "nim c -r --path:src -o:build/test_linalg tests/test_linalg.nim"
  exec "nim c -r --path:src -o:build/test_contracts tests/test_contracts.nim"
  exec "nim c -r --path:src -o:build/test_vector tests/test_vector.nim"

task testCiRelease, "Nim tests (CI subset, -d:danger)":
  exec "nim c -r -d:danger --path:src -o:build/test_linalg_rel tests/test_linalg.nim"
  exec "nim c -r -d:danger --path:src -o:build/test_contracts_rel tests/test_contracts.nim"
  exec "nim c -r -d:danger --path:src -o:build/test_vector_rel tests/test_vector.nim"

task testAll, "debug + release + C ABI":
  exec "nimble test"
  exec "nimble testRelease"
  exec "nimble ctest"

task example, "Nim demo":
  exec "nim c -r --path:src -o:build/demo examples/demo.nim"

# `nimble <task> -- --foo` never reaches this task's `exec` line on its own --
# nimble's own argv (visible to `paramStr` here) carries the raw CLI tokens,
# but a hardcoded `exec` string doesn't forward them. Extracting everything
# after the task's own name is what makes `nimble bench -- --csv:...` from
# bench/README.md's Regression gate section actually work.
proc forwardedArgs(taskName: string): string =
  var found = false
  for i in 1 .. paramCount():
    let p = paramStr(i)
    if found:
      if p != "--": result &= " " & p
    elif p == taskName:
      found = true

task bench, "UniLinalg-only throughput benchmark (bench/bench.nim; forwards --csv:/--baseline:/--threshold:/--md: after --)":
  exec "nim c -r -d:danger --path:src -o:build/bench bench/bench.nim" & forwardedArgs("bench")

task benchReadme, "bench (+ vs LAPACK/Arraymancer if installed), splice into bench/README.md":
  exec "nim c -r -d:danger --path:src -o:build/bench bench/bench.nim --md:bench/.md_bench.md"
  let (lapackPath, lapackCode) = gorgeEx("nimble path nimlapack")
  if lapackCode == 0:
    exec "nim c -d:danger --path:src --path:" & lapackPath.strip() &
        " -o:build/bench_vs_lapack bench/compare/vs_lapack.nim"
    exec "./build/bench_vs_lapack > bench/.md_vs_lapack.txt"
  else:
    echo "benchReadme: nimlapack not installed -- skipping vs LAPACK"
  let (amPath, amCode) = gorgeEx("nimble path arraymancer")
  if amCode == 0:
    exec "nim c -d:danger --path:src --path:" & amPath.strip() &
        " -o:build/bench_vs_arraymancer bench/compare/vs_arraymancer.nim"
    exec "./build/bench_vs_arraymancer > bench/.md_vs_arraymancer.txt"
  else:
    echo "benchReadme: arraymancer not installed -- skipping vs Arraymancer"
  exec "nim c -r --path:src bench/export_readme.nim"

# Not `requires`d: nimlapack/arraymancer are dev-only comparison tooling, not
# real library dependencies (see bench/README.md) -- resolved at task-run time
# via `nimble path`, so no fixed version/hash is baked into UniLinalg.nimble.
task benchVsLapack, "Compare UniLinalg vs raw LAPACK (nimlapack)":
  let (lapackPath, code) = gorgeEx("nimble path nimlapack")
  if code != 0:
    quit("benchVsLapack: nimble path nimlapack failed -- " &
        "run `nimble install nimlapack` first", 1)
  exec "nim c -r -d:danger --path:src --path:" & lapackPath.strip() &
      " -o:build/bench_vs_lapack bench/compare/vs_lapack.nim"

task benchVsArraymancer, "Compare UniLinalg vs Arraymancer":
  let (amPath, code) = gorgeEx("nimble path arraymancer")
  if code != 0:
    quit("benchVsArraymancer: nimble path arraymancer failed -- " &
        "run `nimble install arraymancer` first", 1)
  exec "nim c -r -d:danger --path:src --path:" & amPath.strip() &
      " -o:build/bench_vs_arraymancer bench/compare/vs_arraymancer.nim"

# Nim takes `-o:` literally and appends no platform extension.
const
  sharedLib =
    when defined(windows): "libUniLinalg.dll"
    elif defined(macosx): "libUniLinalg.dylib"
    else: "libUniLinalg.so"
  staticLib = "libUniLinalg.a"  # MinGW `ar` on Windows, so `.a` everywhere.

  # @rpath install_name, so the copy bundled in the wheel is found at import.
  macArgs =
    when defined(macosx): " --passL:\"-Wl,-install_name,@rpath/" & sharedLib & "\""
    else: ""

task clib, "C shared library":
  # -d:danger: the C ABI is the shipped, performance-sensitive surface, and
  # every ulin_* entry point already validates handles/indices/lengths itself
  # (SECURITY.md) before touching Matrix internals, so Nim's own bound/
  # overflow checks are redundant belt-and-suspenders there, not a safety net
  # foreign callers rely on.
  exec "nim c --app:lib --noMain --mm:arc -d:danger -o:" & sharedLib & macArgs &
       " src/UniLinalg/c_api.nim"

task clibStatic, "C static library":
  exec "nim c --app:staticlib --noMain --mm:arc -d:danger -o:" & staticLib &
       " src/UniLinalg/c_api.nim"

task clibMsvc, "C static library, MSVC ABI (Windows Python extension)":
  # CPython on Windows is MSVC-built and cannot link MinGW output.
  exec "nim c --cc:vcc --app:staticlib --noMain --mm:arc -d:danger" &
       " -o:UniLinalg.lib src/UniLinalg/c_api.nim"

# Nim's MinGW toolchain names it mingw32-make.
let makeExe = if findExe("mingw32-make").len > 0: "mingw32-make" else: "make"

# `make -C`, not `cd dir && make`: nimble's exec runs no shell on Windows.
task ctest, "C ABI tests":
  exec "nimble clibStatic"
  exec makeExe & " -C tests/c"

task cexample, "C demo":
  exec "nimble clibStatic"
  exec makeExe & " -C examples/c"

task pyDeps, "Install Python build deps (setuptools, Cython, pytest) if missing":
  exec "python3 -m pip install --break-system-packages --quiet setuptools wheel \"Cython>=3.0.0\" pytest"

# The extension links the vcc static lib on Windows, the shared lib elsewhere.
task pyLib, "Build the library the Python extension links against":
  when defined(windows):
    exec "nimble clibMsvc"
  else:
    exec "nimble clib"

task pyNotebookDeps, "Install notebook build deps (nbformat, nbclient, ipykernel) if missing":
  exec "python3 -m pip install --break-system-packages --quiet nbformat nbclient ipykernel"

task buildCython, "Cython extension in-place":
  exec "nimble pyLib"
  exec "nimble pyDeps"
  # nimscript `cd` (lib/system/nimscript.nim) changes the VM cwd for the next
  # exec without a shell, so the task works under nimble's no-shell exec on Windows.
  cd "py"
  exec "python3 setup.py build_ext --inplace"
  cd ".."

task pyTest, "Cython extension + pytest":
  exec "nimble buildCython"
  cd "py"
  exec "python3 -m pytest -q"
  cd ".."

task pyWheel, "wheel":
  exec "nimble pyLib"
  exec "nimble pyDeps"
  cd "py"
  exec "python3 setup.py bdist_wheel"
  cd ".."

task coverage, "LCOV + HTML coverage report for the Nim sources (needs lcov)":
  # gcov and lcov driven directly, no coco. Linux and macOS only.
  # --debugger:native attributes lines to the .nim sources, not the generated C.
  # --include keeps stdlib out of the capture, where lcov 2.x aborts on Nim's
  # codegen. Together they leave nothing to suppress: no --ignore-errors here,
  # so a real problem still fails the build.
  let cache = "build/covcache"
  rmDir cache
  rmDir "coverage"
  exec "nim c --path:src --nimcache:" & cache &
       " --debugger:native --passC:--coverage --passL:--coverage" &
       " -o:build/test_coverage tests/test_linalg.nim"
  exec "./build/test_coverage"
  exec "nim c --path:src --nimcache:" & cache &
       " --debugger:native --passC:--coverage --passL:--coverage" &
       " -o:build/test_coverage2 tests/test_contracts.nim"
  exec "./build/test_coverage2"
  exec "nim c --path:src --nimcache:" & cache &
       " --debugger:native --passC:--coverage --passL:--coverage" &
       " -o:build/test_coverage3 tests/test_vector.nim"
  exec "./build/test_coverage3"
  exec "lcov --capture --directory " & cache & " --base-directory ." &
       " --include \"*/src/UniLinalg/*\" --output-file lcov.info --quiet"
  exec "genhtml lcov.info --output-directory coverage --legend --quiet"
  exec "lcov --summary lcov.info"
