# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniLinalg — linear algebra (Nim + C-ABI + Python).

version       = "1.1.0"
author        = "lituus-lab"
description   = "Linear algebra: dense/sparse matrices, LU/Cholesky/QR/SVD, Vector[D,T] (Nim + C-ABI + Python)"
license       = "Apache-2.0"
srcDir        = "src"

requires "nim >= 2.0.0"
requires "https://github.com/lbartoletti/NimContracts#main"
requires "https://github.com/lituus-lab/UniMath#main"

# nimble 0.22 exits 0 even when an `exec` inside a task fails, so a task's exit
# code says nothing about whether its body ran. Each task writes a marker as
# its last statement; `tools/gate.nim` removes the marker, runs the task, and
# fails if it is not there afterwards. `nimble canary` proves the gate still
# bites -- if that one ever passes, every other green result is worthless.
const gateExe =
  when defined(windows): "build/unigate.exe" else: "build/unigate"

template done(task: string) =
  mkDir "build/.gate"
  writeFile("build/.gate/" & task & ".ok", "")

proc gate(task: string): string =
  ## `exec gate("test")` -- builds the tool on first use.
  if not fileExists(gateExe):
    exec "nim c --hints:off -o:" & gateExe & " tools/gate.nim"
  gateExe & " " & task

task canary, "Must fail: proves the gate still catches a broken build":
  # No `done` here on purpose: the exec below raises, so the marker is never
  # written and the gate reports the failure nimble swallowed.
  exec "nim c -r --hints:off --path:src -o:build/canary tests/canary_broken.nim"

task lint, "Fail if nimpretty would reformat a source":
  exec "nim c -r --hints:off -o:build/lint_tool tools/lint.nim"
  done "lint"

task checkVGraph, "Fail on an import that climbs the layers in vgraph.cfg":
  exec "nim c -r --hints:off -o:build/vgraph_tool tools/vgraph.nim"
  done "checkVGraph"

# From the URL with a tag, not from the registry: the nimble registry lags
# upstream, and `nimble install nimibook` resolves 0.3.1, whose themes.nim does
# not compile against nimib 0.4.x. The theme is pinned for a different reason --
# several installs of one version is a resolution nimble cannot make.
const bookDeps = [
  "https://github.com/pietroppeter/nimib#v0.4.1",
  "https://github.com/pietroppeter/nimibook#v0.4.0",
  "https://github.com/lituus-lab/lituus-theme#v0.2.0",
]
taskRequires "docsDeps", bookDeps[0], bookDeps[1], bookDeps[2]
taskRequires "book", bookDeps[0], bookDeps[1], bookDeps[2]
taskRequires "docs", bookDeps[0], bookDeps[1], bookDeps[2]

task docsDeps, "Install the docs toolchain (nimib + nimibook + theme)":
  # This task's own `taskRequires` above is what fetches them.
  echo "nimib, nimibook and lituus-theme installed."
  done "docsDeps"

task bookInit, "Scaffold a chapter added to the table of contents":
  withDir "book":
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim init"
  done "bookInit"

task book, "Build the multi-chapter book (needs nimib + nimibook)":
  # The chapters compile and run their own code, so a drift in any of them
  # fails the build. Run from book/, because nimibook reads the nimib.toml of
  # the directory it starts in.
  withDir "book":
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim clean"
    # `init` before `build`, on every run: it is what creates `__site/assets`,
    # which is not tracked, so a fresh clone has none and every page ships
    # referencing a stylesheet and a script that are not there.
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim init"
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim build"
  done "book"

task docs, "API reference + book into pages/ — what CI publishes":
  rmDir "pages"
  exec gate("book")
  # The book *is* the site: its pages link to `assets/` and to each other as
  # siblings, so it is copied whole to the root rather than nested.
  cpDir "book/__site", "pages"
  # book.json is nimibook's build state -- no page fetches it -- and it carries
  # the absolute path of the machine that built it. It does not get published.
  rmFile "pages/book.json"
  exec "nim doc --index:on --outdir:pages/api --project --hints:off src/UniLinalg.nim"
  # ...and the reference wears the same theme. `nim doc` has no stylesheet
  # option, so the palette is appended to the one it just wrote.
  exec "nim c -r --hints:off --outdir:build tools/theme_api.nim " &
       "pages/api/nimdoc.out.css"
  done "docs"

task test, "Nim tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_linalg tests/test_linalg.nim"
  exec "nim c -r --path:src -o:build/test_contracts tests/test_contracts.nim"
  exec "nim c -r --path:src -o:build/test_vector tests/test_vector.nim"
  exec "nim c -r --path:src -o:build/test_version tests/test_version.nim"
  done "test"

task testRelease, "Nim tests (-d:danger: bound/overflow checks off, contracts compiled away)":
  exec "nim c -r -d:danger --path:src -o:build/test_linalg_rel tests/test_linalg.nim"
  exec "nim c -r -d:danger --path:src -o:build/test_contracts_rel tests/test_contracts.nim"
  exec "nim c -r -d:danger --path:src -o:build/test_vector_rel tests/test_vector.nim"
  done "testRelease"

task testCi, "Nim tests (CI subset, debug)":
  exec gate("test")
  done "testCi"

task testCiRelease, "Nim tests (CI subset, -d:danger)":
  exec gate("testRelease")
  done "testCiRelease"

task testAll, "debug + release + C ABI":
  exec gate("test")
  exec gate("testRelease")
  exec gate("ctest")
  done "testAll"

task example, "Nim demo":
  exec "nim c -r --path:src -o:build/demo examples/demo.nim"
  done "example"

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
  done "bench"

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
  done "benchReadme"

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
  done "benchVsLapack"

task benchVsArraymancer, "Compare UniLinalg vs Arraymancer":
  let (amPath, code) = gorgeEx("nimble path arraymancer")
  if code != 0:
    quit("benchVsArraymancer: nimble path arraymancer failed -- " &
        "run `nimble install arraymancer` first", 1)
  exec "nim c -r -d:danger --path:src --path:" & amPath.strip() &
      " -o:build/bench_vs_arraymancer bench/compare/vs_arraymancer.nim"
  done "benchVsArraymancer"

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
  done "clib"

task clibStatic, "C static library":
  exec "nim c --app:staticlib -d:staticNoAutoInit --noMain --mm:arc -d:danger -o:" & staticLib &
       " src/UniLinalg/c_api.nim"
  done "clibStatic"

task clibMsvc, "C static library, MSVC ABI (Windows Python extension)":
  # CPython on Windows is MSVC-built and cannot link MinGW output.
  exec "nim c --cc:vcc --app:staticlib -d:staticNoAutoInit --noMain --mm:arc -d:danger" &
       " -o:UniLinalg.lib src/UniLinalg/c_api.nim"
  done "clibMsvc"

# Nim's MinGW toolchain names it mingw32-make.
let makeExe = if findExe("mingw32-make").len > 0: "mingw32-make" else: "make"

# tests/c and examples/c are POSIX-portable Makefiles carrying no OS branch
# (GNU and BSD make share no conditional syntax), so the Windows names come
# from here as command-line assignments, which beat `?=` on every make flavor.
# `del` needs no `/q`: it is only ever handed a single name, never a wildcard.
proc winMakeVars(bin: string): string =
  when defined(windows):
    " CC=gcc BIN=" & bin & ".exe RUN=" & bin & ".exe RM_F=del"
  else:
    ""

# `make -C`, not `cd dir && make`: nimble's exec runs no shell on Windows.
task ctest, "C ABI tests":
  exec gate("clibStatic")
  exec makeExe & " -C tests/c" & winMakeVars("test_unilinalg")
  done "ctest"

task cexample, "C demo":
  exec gate("clibStatic")
  exec makeExe & " -C examples/c" & winMakeVars("demo")
  done "cexample"

task pyDeps, "Install Python build deps (setuptools, Cython, pytest) if missing":
  exec "python3 -m pip install --break-system-packages --quiet setuptools wheel \"Cython>=3.0.0\" pytest"
  done "pyDeps"

# The extension links the vcc static lib on Windows, the shared lib elsewhere.
task pyLib, "Build the library the Python extension links against":
  when defined(windows):
    exec gate("clibMsvc")
  else:
    exec gate("clib")
  done "pyLib"

task pyNotebookDeps, "Install notebook build deps (nbformat, nbclient, ipykernel) if missing":
  exec "python3 -m pip install --break-system-packages --quiet nbformat nbclient ipykernel"
  done "pyNotebookDeps"

task buildCython, "Cython extension in-place":
  exec gate("pyLib")
  exec gate("pyDeps")
  # nimscript `cd` (lib/system/nimscript.nim) changes the VM cwd for the next
  # exec without a shell, so the task works under nimble's no-shell exec on Windows.
  cd "py"
  exec "python3 setup.py build_ext --inplace"
  cd ".."
  done "buildCython"

task pyTest, "Cython extension + pytest":
  exec gate("buildCython")
  cd "py"
  exec "python3 -m pytest -q"
  cd ".."
  done "pyTest"

task pyWheel, "wheel":
  exec gate("pyLib")
  exec gate("pyDeps")
  cd "py"
  exec "python3 setup.py bdist_wheel"
  cd ".."
  done "pyWheel"

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
  done "coverage"
